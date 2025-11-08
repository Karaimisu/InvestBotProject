# Market_Analysis.py
import json
from typing import List, Dict, Any

def _safe_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

def analyze_market_only(ai_client, model_id: str, news_items: List[Dict[str, Any]], company: Dict[str, Any] | None = None) -> dict:
    """
    Input: news_items like [{category, headline, blurb}]
    Output JSON:
    {
      "summary": str,
      "sentiment": "bullish"|"bearish"|"mixed",
      "themes": [str],
      "sectors": [str],
      "risks": [str],
      "opportunities": [str],
      "watchlist": [str]
    }
    Thai prompts. Keys in English for stable parsing.
    """
    # Pack news into compact lines for the model
    lines = [f"- [{it.get('category','')}] {it.get('headline','').strip()} — {it.get('blurb','').strip()}" for it in news_items]
    news_text = "\n".join(lines[:12])  # clamp size

    company_clause = ""
    if company and company.get("name"):
        company_clause = f" ให้กล่าวถึงบริษัทสมมติ '{company['name']}' ในกลุ่ม '{company.get('sector','')}' ถ้ามีความเชื่อมโยง. "

    system = (
        "คุณเป็นนักวิเคราะห์ตลาดทุน ทำงานกับข่าวสังเคราะห์เพื่อฝึกทักษะการอ่านตลาด."
        " ตอบกลับเป็น JSON เท่านั้น โดยใช้คีย์: summary, sentiment, themes, sectors, risks, opportunities, watchlist."
        " sentiment ต้องเป็นหนึ่งใน: bullish, bearish, mixed."
    )
    user = (
        "จงวิเคราะห์เฉพาะประเด็นที่เกี่ยวกับ 'ตลาดหุ้น' จากรายการข่าวต่อไปนี้ "
        "ข้ามข่าวที่ไม่ใช่หุ้น เช่น บันเทิงทั่วไป "
        + company_clause +
        "ให้สรุปภาพรวมตลาดแบบกระชับ ระบุธีมสำคัญ กลุ่มอุตสาหกรรมที่เด่น/เสี่ยง ความเสี่ยงเชิงมหภาค/จุลภาค "
        "และโอกาสที่น่าสนใจ พร้อมรายการ 'watchlist' เป็นชื่อธีมหรือสมมติชื่อหุ้นสั้นๆ 3-6 รายการ.\n\n"
        "รายการข่าว:\n" + news_text
    )

    resp = ai_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.6
    )
    data = _safe_json(resp.choices[0].message.content)
    # minimal normalization
    out = {
        "summary": (data.get("summary") or "").strip(),
        "sentiment": (data.get("sentiment") or "mixed").strip(),
        "themes": [str(x) for x in (data.get("themes") or [])][:8],
        "sectors": [str(x) for x in (data.get("sectors") or [])][:8],
        "risks": [str(x) for x in (data.get("risks") or [])][:8],
        "opportunities": [str(x) for x in (data.get("opportunities") or [])][:8],
        "watchlist": [str(x) for x in (data.get("watchlist") or [])][:8],
    }
    return out
