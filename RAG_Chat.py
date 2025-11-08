# RAG_Chat.py
import json
from typing import List, Dict, Any
from openai import OpenAI
from RAG_Store import retrieve

def _fmt_context(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h['title']} — {h['source']}"
        body = h["text"].strip()
        lines.append(head + "\n" + body)
    return "\n\n".join(lines)

def _safe_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

def answer_investing_question(ai: OpenAI, model_id: str, question: str, k: int = 6) -> Dict[str, Any]:
    hits = retrieve(question, k=k)
    context = _fmt_context(hits) if hits else "ไม่มีบริบทจากคลังความรู้"

    sys = (
        "คุณเป็นผู้ช่วยตอบคำถามด้านตลาดหุ้นและการลงทุน โดยอ้างอิงเฉพาะจากบริบทที่ให้มา "
        "ตอบเป็นภาษาไทย ใช้น้ำเสียงเป็นกลางทางวิชาการ "
        "ห้ามให้คำแนะนำการลงทุนเฉพาะบุคคล ให้ข้อมูลเชิงการศึกษาเท่านั้น "
        "ปิดท้ายด้วยรายการอ้างอิงในรูป [1], [2] อิงตามแหล่งที่มาที่แนบมา"
    )
    user = (
        f"คำถามของผู้ใช้:\n{question}\n\n"
        f"บริบทสำหรับค้นคืน (อาจตัดทอนบางส่วน):\n{context}\n\n"
        "จงสรุปให้กระชับ ชัดเจน ประเด็นสำคัญก่อน รายละเอียดทีหลัง "
        "ถ้าบริบทไม่มีข้อมูลเพียงพอ ให้บอกว่าไม่พบในคลังความรู้ และแนะนำคำสำคัญที่ควรค้นเพิ่ม"
    )

    resp = ai.chat.completions.create(
        model=model_id,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.4
    )
    answer = resp.choices[0].message.content.strip()

    refs = []
    for i, h in enumerate(hits, 1):
        refs.append({"n": i, "title": h["title"], "source": h["source"]})
    return {"answer": answer, "refs": refs}