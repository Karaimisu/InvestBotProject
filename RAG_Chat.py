# RAG_Chat.py
# Thai-only answerer built on top of retrieve()
import json
from typing import List, Dict, Any
from openai import OpenAI
from RAG_Store import retrieve

def _fmt_refs(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['title']} — {h['source']}")
    return "\n".join(lines)

def answer_investing_question(ai: OpenAI, model_id: str, question: str, k: int = 6) -> Dict[str, Any]:
    hits = retrieve(question, k=k)
    context_titles = "; ".join([f"{h['title']}" for h in hits]) if hits else "ไม่มี"
    sys = (
        "คุณเป็นผู้ช่วยอธิบายความรู้การลงทุน/ตลาดหุ้น สำหรับการศึกษาเท่านั้น "
        "ใช้เฉพาะข้อมูลจากบริบท ถ้าไม่พอ ให้บอกว่ายังไม่พบในคลังความรู้"
    )
    user = (
        f"คำถาม (ภาษาไทย): {question}\n"
        f"บริบทจากคลัง (หัวข้อ): {context_titles}\n"
        "ตอบเป็นภาษาไทยสั้น กระชับ เป็นกลาง จัดลำดับเป็นข้อ ถ้าไม่พบข้อมูลให้บอกตรงๆ"
    )
    try:
        resp = ai.chat.completions.create(
            model=model_id,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.3,
            max_tokens=280,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception:
        answer = "ยังไม่พบข้อมูลเพียงพอในคลังความรู้"

    return {"answer": answer, "refs": hits, "refs_label": "อ้างอิง"}