# Daily_News.py
# Lightweight daily news cache (keeps at most ~12 items/day)
import json, datetime
from pathlib import Path
from typing import List
from openai import OpenAI

DATA_ROOT = Path(__file__).resolve().parent / "data"
NEWS_DIR  = DATA_ROOT / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)

def _today_key() -> str:
    # Asia/Bangkok-ish without tz: use local date
    return datetime.date.today().isoformat()

def _file_for_today() -> Path:
    return NEWS_DIR / f"{_today_key()}.json"

def load_today_news(data_dir: Path) -> List[dict]:
    fp = _file_for_today()
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def seconds_until_next_refresh(hh: int, mm: int) -> int:
    now = datetime.datetime.now()
    tgt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if tgt <= now:
        tgt = tgt + datetime.timedelta(days=1)
    return int((tgt - now).total_seconds())

def refresh_daily_news_real(ai: OpenAI, model_id: str, data_dir: Path) -> List[dict]:
    # For low-RAM demo, we still synthesize short Thai summaries tagged as "live"
    sys = "สรุปข่าวตลาดหุ้นทั่วโลกวันนี้เป็นภาษาไทย แบบสั้น รายการหัวข้อ"
    user = (
        "สร้างหัวข้อข่าว 8-12 ชิ้น เกี่ยวกับตลาดหุ้น/ตราสารหนี้/สินค้าโภคภัณฑ์/อัตราดอกเบี้ย "
        "ตอบ JSON: [{'headline':..., 'blurb':..., 'source':'live', 'published':'YYYY-MM-DD'}]"
    )
    try:
        r = ai.chat.completions.create(
            model=model_id,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.5,
            max_tokens=700
        )
        import json as _json
        items = _json.loads(r.choices[0].message.content.replace("'", '"'))
        # cap and normalize
        key = _today_key()
        for it in items:
            it["published"] = key
            it["source"] = it.get("source","live")
        items = items[:12]
    except Exception:
        items = [{"headline":"ตลาดผันผวน นักลงทุนจับตาดอกเบี้ย","blurb":"แรงซื้อขายลดลงจากความไม่แน่นอน","source":"live","published":_today_key()}]

    fp = _file_for_today()
    fp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items

def generate_daily_news(ai: OpenAI, model_id: str) -> List[dict]:
    # Fallback synthetic
    return [{"headline":"สรุปภาพรวมตลาด","blurb":"ภาวะการลงทุนผสมผสานในหลายภูมิภาค","source":"syn","published":_today_key()}]