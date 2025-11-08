# RAG_Chat.py
import json
from typing import List, Dict, Any
from openai import OpenAI
from RAG_Store import retrieve

# ---- Thai-only labels ----
TH_LABELS = {
    "no_kb": "ไม่พบข้อมูลที่เกี่ยวข้องในคลังความรู้ภายใน",
    "refs": "อ้างอิง",
    "style": "ตอบเป็นภาษาไทยที่กระชับ ชัดเจน ตรงประเด็น และเป็นกลางเชิงวิชาการ",
}

def _fmt_context(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        head = f"[{i}] {h['title']} — {h['source']}"
        body = h["text"].strip()
        lines.append(head + "\n" + body)
    return "\n\n".join(lines)

def answer_investing_question(ai: OpenAI, model_id: str, question: str, k: int = 6) -> Dict[str, Any]:
    # Always operate in Thai
    lang = "th"
    labels = TH_LABELS

    hits = retrieve(question, k=k)
    context = _fmt_context(hits) if hits else labels["no_kb"]

    # System prompt enforces Thai and educational tone
    sys = (
        "คุณเป็นผู้ช่วยด้านความรู้ตลาดหุ้นและการลงทุน เชิงการศึกษาเท่านั้น "
        "ใช้ข้อมูลจาก Context ที่ให้มาเท่านั้น ห้ามแต่งตัวเลขหรือรายละเอียดที่ไม่มีใน Context "
        "หลีกเลี่ยงคำแนะนำส่วนบุคคลและการชี้นำการลงทุนโดยตรง "
        "ตอบเป็นภาษาไทยเสมอ"
    )

    # User message provides instructions and context
    user = (
        f"{labels['style']}\n\n"
        f"คำถามของผู้ใช้:\n{question}\n\n"
        f"Context (ดึงจากคลังความรู้ภายใน อาจถูกตัดความยาว):\n{context}\n\n"
        "แนวทางการตอบ:\n"
        "- เริ่มด้วยประเด็นสำคัญแบบหัวข้อสั้น ๆ ก่อน แล้วค่อยขยายรายละเอียด\n"
        "- หาก Context ไม่เพียงพอต่อคำตอบ ให้ระบุอย่างชัดเจนว่า 'ข้อมูลไม่พอ' "
        "และแนะนำ 2–4 คำค้นที่ควรใช้ค้นหาเพิ่มในคลังความรู้\n"
        "- ปิดท้ายคำตอบด้วยการอ้างอิงแบบ [1], [2] ที่สอดคล้องกับรายการแหล่งข้อมูลด้านล่าง"
    )

    resp = ai.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.4,
    )
    answer = resp.choices[0].message.content.strip()

    refs = [{"n": i, "title": h["title"], "source": h["source"]} for i, h in enumerate(hits, 1)]
    return {"answer": answer, "refs": refs, "lang": lang, "refs_label": labels["refs"]}
