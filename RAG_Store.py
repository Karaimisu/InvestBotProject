# RAG_Store.py
from __future__ import annotations
import os, json, re, time
from contextlib import suppress
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown as md

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"
KB_DIR = DATA_DIR / "knowledge"
RAG_DIR = DATA_DIR / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = RAG_DIR / "index.faiss"
META_PATH = RAG_DIR / "meta.json"
STORE_PATH = RAG_DIR / "store.jsonl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIM = 384

@dataclass
class Chunk:
    doc_id: str
    source: str
    title: str
    text: str

def _read_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if ext in {".md", ".markdown"}:
            html = md.markdown(path.read_text(encoding="utf-8", errors="ignore"))
            return BeautifulSoup(html, "html.parser").get_text(separator="\n")
        if ext in {".htm", ".html"}:
            html = path.read_text(encoding="utf-8", errors="ignore")
            return BeautifulSoup(html, "html.parser").get_text(separator="\n")
        if ext in {".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return ""

def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _chunk(text: str, max_tokens: int = 700, overlap: int = 100) -> List[str]:
    max_len = max_tokens * 4
    ov = overlap * 4
    text = _clean_text(text)
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_len)
        cut = text[i:j]
        last_break = cut.rfind("\n\n")
        if last_break > ov:
            j = i + last_break
            cut = text[i:j]
        chunks.append(cut.strip())
        i = max(i + max_len - ov, j)
    return [c for c in chunks if len(c) > 20]

def _title_from_path(p: Path) -> str:
    return p.stem.replace("_", " ").strip()

def _retry(times=5, delay=0.25):
    def deco(fn):
        def wrap(*a, **k):
            last = None
            for _ in range(times):
                try:
                    return fn(*a, **k)
                except (PermissionError, OSError) as e:
                    last = e
                    time.sleep(delay)
            if last:
                raise last
        return wrap
    return deco

@_retry(times=5, delay=0.25)
def _atomic_write_index(index, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    if tmp.exists() and tmp.stat().st_size > 0:
        with suppress(Exception):
            test = faiss.read_index(str(tmp))
            del test
        tmp.replace(path)
    else:
        raise RuntimeError("Failed to write FAISS index")

def _meta_write(obj: dict):
    META_PATH.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def rebuild_index() -> Tuple[int, int]:
    files = []
    for ext in ("*.pdf", "*.md", "*.markdown", "*.html", "*.htm", "*.txt"):
        files.extend(KB_DIR.rglob(ext))

    docs: List[Chunk] = []
    for path in files:
        raw = _read_text_from_file(path)
        if not raw:
            continue
        parts = _chunk(raw)
        title = _title_from_path(path)
        for k, part in enumerate(parts):
            docs.append(Chunk(
                doc_id=f"{path.name}#{k}",
                source=str(path.relative_to(REPO)),
                title=title,
                text=part
            ))

    model = SentenceTransformer(MODEL_NAME)

    if not docs:
        index = faiss.IndexFlatIP(EMB_DIM)
        _atomic_write_index(index, INDEX_PATH)
        _meta_write({"emb_model": MODEL_NAME, "count": 0})
        STORE_PATH.write_text("", encoding="utf-8")
        return 0, 0

    embs = model.encode([d.text for d in docs], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(embs)
    _atomic_write_index(index, INDEX_PATH)

    _meta_write({"emb_model": MODEL_NAME, "count": len(docs)})
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")

    return len(files), len(docs)

def _load_store():
    if (not INDEX_PATH.exists()) or INDEX_PATH.stat().st_size == 0:
        rebuild_index()
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if meta.get("emb_model") != MODEL_NAME:
            rebuild_index()
    except Exception:
        rebuild_index()

    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception:
        rebuild_index()
        index = faiss.read_index(str(INDEX_PATH))

    chunks: List[Chunk] = []
    if STORE_PATH.exists() and STORE_PATH.stat().st_size > 0:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    chunks.append(Chunk(**obj))
    return index, chunks

_MODEL_SINGLETON = None
def _get_model():
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = SentenceTransformer(MODEL_NAME)
    return _MODEL_SINGLETON

def _embed(texts: List[str]) -> np.ndarray:
    model = _get_model()
    X = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(X, dtype="float32")

def retrieve(query: str, k: int = 6) -> List[Dict]:
    try:
        index, store = _load_store()
    except Exception:
        rebuild_index()
        index, store = _load_store()

    if not store or index.ntotal == 0:
        return []

    q = _embed([query])
    try:
        D, I = index.search(q, min(k, index.ntotal))
    except Exception:
        rebuild_index()
        index, store = _load_store()
        if not store or index.ntotal == 0:
            return []
        D, I = index.search(q, min(k, index.ntotal))

    out = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        c = store[idx]
        out.append({
            "doc_id": c.doc_id,
            "source": c.source,
            "title": c.title,
            "text": c.text,
            "score": float(score)
        })
    return out

if __name__ == "__main__":
    files, chunks = rebuild_index()
    print(f"Indexed files: {files}, chunks: {chunks}")
