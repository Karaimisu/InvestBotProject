# RAG_Store.py
# Low-RAM RAG store: 384-dim MiniLM, small chunks, FAISS IndexHNSWFlat
import os, json, re
from pathlib import Path
from typing import List, Dict, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent / "data" / "rag"
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH  = DATA_DIR / "meta.json"

_MODEL = None
_INDEX = None
_META  = None

def _model():
    global _MODEL
    if _MODEL is None:
        # compact model (384 dims)
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL

def _encode(texts: List[str]) -> np.ndarray:
    m = _model()
    emb = m.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)

def _save(index, meta):
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def _load() -> Tuple[faiss.Index, Dict]:
    global _INDEX, _META
    if _INDEX is not None and _META is not None:
        return _INDEX, _META
    if INDEX_PATH.exists() and META_PATH.exists():
        try:
            idx = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            _INDEX, _META = idx, meta
            return _INDEX, _META
        except Exception:
            pass
    # empty index
    d = 384
    idx = faiss.IndexHNSWFlat(d, 32)
    idx.hnsw.efConstruction = 80
    idx.hnsw.efSearch = 64
    _INDEX, _META = idx, {"docs": []}
    return _INDEX, _META

def _chunk(text: str, max_tokens: int = 180) -> List[str]:
    # naïve splitter: paragraphs then size cap
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) < max_tokens*4:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur: out.append(cur); cur = ""
            out.append(p[:max_tokens*4])
    if cur: out.append(cur)
    return out[:200]  # cap

def rebuild_index() -> Tuple[int,int]:
    # Build from CSV knowledge if present
    KB = Path(__file__).resolve().parent / "data" / "kb"
    texts, metas = [], []
    files = list(KB.glob("*.txt")) + list(KB.glob("*.md")) + list(KB.glob("*.csv"))
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk(raw)
            for ch in chunks:
                texts.append(ch)
                metas.append({"title": fp.stem, "source": fp.name})
        except Exception:
            continue

    idx, meta = _load()
    # reset
    d = 384
    idx = faiss.IndexHNSWFlat(d, 32)
    idx.hnsw.efConstruction = 80
    idx.hnsw.efSearch = 64

    if not texts:
        _save(idx, {"docs": []})
        return 0, 0

    embs = _encode(texts)
    idx.add(embs)
    _save(idx, {"docs": metas})
    # replace singletons
    global _INDEX, _META
    _INDEX, _META = idx, {"docs": metas}
    return len(files), len(texts)

def retrieve(query: str, k: int = 6) -> List[Dict]:
    idx, meta = _load()
    if idx.ntotal == 0:
        return []
    q = _encode([query])
    D, I = idx.search(q, min(k, idx.ntotal))
    out = []
    docs = meta.get("docs", [])
    for rank, idxi in enumerate(I[0]):
        if idxi < 0 or idxi >= len(docs): continue
        m = docs[idxi]
        out.append({"rank": rank+1, "title": m["title"], "source": m["source"], "text": ""})  # keep light
    return out