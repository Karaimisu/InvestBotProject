# Daily_News.py
import json
from typing import Optional, Dict, Any, List

def _safe_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

def generate_daily_news(ai_client, model_id: str, company: Optional[Dict[str, Any]] = None) -> List[dict]:
    """
    Returns a list of items: [{category, headline, blurb}]
    If company is provided: {"name": "...", "sector": "..."}
    Prompts in Thai. Keys stay English for stable parsing.
    """
    company_clause = ""
    if company and company.get("name"):
        company_clause = (
            f"รวมข่าว 1-2 หัวข้อที่เกี่ยวข้องกับบริษัทสมมติชื่อ '{company['name']}' "
            f"ซึ่งอยู่ในอุตสาหกรรม '{company.get('sector','')}'. "
        )

    system = (
        "คุณคือบรรณาธิการหนังสือพิมพ์รายวันในโลกสมมติ "
        "ให้ผลลัพธ์เป็น JSON เท่านั้น: {\"news\": [{\"category\": str, \"headline\": str, \"blurb\": str}, ...]} "
        "หลีกเลี่ยงการใส่ลิงก์ ตัวเลขที่เฉพาะเจาะจงมากเกินไป และชื่อบุคคลจริง"
    )
    user = (
        "สร้างเนื้อหาข่าววันนี้ 6-8 เรื่อง ครอบคลุมหมวด: โลก, ธุรกิจ, เทคโนโลยี, วิทยาศาสตร์, กีฬา, บันเทิง. "
        + company_clause +
        "หัวข้อข่าวสั้นกระชับ 6-12 คำ และมีคำโปรย 1 ประโยค "
        "ให้ภาษากระชับเหมือนหนังสือพิมพ์มืออาชีพ"
    )

    resp = ai_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.8
    )
    data = _safe_json(resp.choices[0].message.content)
    items = data.get("news") or []
    # normalize and clamp count
    out = []
    for it in items[:8]:
        out.append({
            "category": (it.get("category") or "General")[:20],
            "headline": (it.get("headline") or "").strip(),
            "blurb": (it.get("blurb") or "").strip()
        })
    return out