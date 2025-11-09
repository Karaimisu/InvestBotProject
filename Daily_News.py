# Daily_News.py
# Live real-world daily news for stocks/investing (Thai summaries) with synthetic fallback.
# Exposed API:
#   - refresh_daily_news_real(ai_client, model_id, data_dir: Path) -> List[dict]
#   - load_today_news(data_dir: Path) -> List[dict]
#   - seconds_until_next_refresh(hh: int, mm: int) -> int
#   - generate_daily_news(ai_client, model_id, company: Optional[dict] = None, n: int = 6) -> List[dict]  (synthetic fallback)

from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import feedparser

# ----------------------------------
# Config
# ----------------------------------
TZ = ZoneInfo("Asia/Bangkok")
MAX_ITEMS = 10
NEWS_DIRNAME = "news"
SOURCES = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",                 # WSJ Markets
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",  # Reuters Business
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",         # CNBC Top News & Analysis
    "https://finance.yahoo.com/rss/topstories",                       # Yahoo Finance Top
]

KEYWORDS = [
    # English
    "stock","stocks","equity","equities","market","markets","earnings","profit","revenue",
    "guidance","ipo","merger","acquisition","dividend","fed","interest rate","inflation",
    "bond","treasury","sector","index",
    # Thai
    "ตลาดหุ้น","หุ้น","กำไร","รายได้","ปันผล","เงินเฟ้อ","ดอกเบี้ย","กองทุน","ดัชนี","ภาวะตลาด",
]

# ----------------------------------
# Live pipeline
# ----------------------------------
def _now_bkk() -> datetime:
    return datetime.now(TZ)

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _match_investing(headline: str, summary: str) -> bool:
    text = f"{headline} {summary}".lower()
    return any(k in text for k in KEYWORDS)

def _fetch_all() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for url in SOURCES:
        try:
            parsed = feedparser.parse(url)
            src_name = _clean(parsed.feed.get("title", "")) or "RSS"
            for e in parsed.entries:
                title = _clean(getattr(e, "title", ""))
                link = _clean(getattr(e, "link", ""))
                summary = _clean(getattr(e, "summary", getattr(e, "description", "")))
                if not title or not link:
                    continue
                if not _match_investing(title, summary):
                    continue
                pub = None
                for k in ("published_parsed", "updated_parsed"):
                    t = getattr(e, k, None)
                    if t:
                        pub = datetime(*t[:6], tzinfo=timezone.utc)
                        break
                if pub is None:
                    pub = _now_bkk()
                out.append({
                    "title": title,
                    "url": link,
                    "source": src_name,
                    "published": _iso(pub),
                    "summary_raw": summary,
                })
        except Exception:
            continue
    # de-dup by URL or title
    dedup: Dict[str, Dict[str, Any]] = {}
    for it in out:
        key = _hash(it["url"])[:16]
        tkey = _hash(it["title"])[:16]
        if key in dedup or tkey in dedup:
            continue
        dedup[key] = it
    items = sorted(dedup.values(), key=lambda x: x["published"], reverse=True)
    return items[: 3 * MAX_ITEMS]

class NewsCache:
    def __init__(self, data_dir: Path):
        self.base = Path(data_dir) / NEWS_DIRNAME
        self.base.mkdir(parents=True, exist_ok=True)
    def _path_for_date(self, day: datetime) -> Path:
        return self.base / f"daily_{day.astimezone(TZ).date().isoformat()}.json"
    def load_day(self, day: datetime) -> List[Dict[str, Any]]:
        p = self._path_for_date(day)
        if p.exists() and p.stat().st_size > 0:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []
    def save_day(self, day: datetime, items: List[Dict[str, Any]]):
        p = self._path_for_date(day)
        p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def refresh_daily_news_real(ai_client, model_id: str, data_dir: Path) -> List[Dict[str, Any]]:
    """
    Pull live feeds, filter to investing, summarize to Thai with Typhoon, cache for today's date.
    """
    cache = NewsCache(data_dir)
    today = _now_bkk()
    raw = _fetch_all()
    if not raw:
        cache.save_day(today, [])
        return []

    items: List[Dict[str, Any]] = []
    for it in raw:
        if len(items) >= MAX_ITEMS:
            break
        prompt = (
            "สรุปข่าวตลาดทุนต่อไปนี้เป็นภาษาไทย 1–2 ประโยค "
            "เน้นใจความต่อภาวะตลาด หุ้น หรือปัจจัยกำไร โดยไม่ให้คำแนะนำลงทุน:\n\n"
            f"หัวข้อ: {it['title']}\n"
            f"ที่มา: {it['source']}\n"
            f"เนื้อหา: {it['summary_raw'][:800]}"
        )
        try:
            resp = ai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "คุณเป็นบรรณาธิการข่าวการเงิน สรุปให้สั้น กระชับ เป็นกลาง"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            blurb_th = resp.choices[0].message.content.strip()
        except Exception:
            blurb_th = it["summary_raw"][:200]

        items.append({
            "headline": it["title"],
            "blurb": blurb_th,
            "url": it["url"],
            "source": it["source"],
            "published": it["published"],
        })

    cache.save_day(today, items)
    return items

def load_today_news(data_dir: Path) -> List[Dict[str, Any]]:
    return NewsCache(data_dir).load_day(_now_bkk())

def seconds_until_next_refresh(hh: int, mm: int) -> int:
    now = _now_bkk()
    target_today = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target_today <= now:
        target_today = target_today + timedelta(days=1)
    return int((target_today - now).total_seconds())

# ----------------------------------
# Synthetic fallback (kept for compatibility)
# ----------------------------------
_DEFAULT_CATEGORIES = ["ตลาดหุ้น","ธุรกิจ","เทคโนโลยี","การเงิน","เศรษฐกิจ","พลังงาน","สุขภาพ","โลจิสติกส์"]

def _safe_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

def generate_daily_news(ai_client, model_id: str, company: Optional[Dict[str, Any]] = None, n: int = 6) -> List[dict]:
    """
    Synthetic Thai news. Used only if you explicitly call it.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    comp_clause = ""
    if company and company.get("name"):
        comp_clause = f"รวมข่าว 1–2 หัวข้อที่เกี่ยวข้องกับบริษัทสมมติชื่อ '{company['name']}' ในอุตสาหกรรม '{company.get('sector','')}'. "

    system = (
        "คุณคือบรรณาธิการข่าวการเงินในโลกสมมติ สร้างข่าวสั้นเกี่ยวกับตลาดทุน "
        "ห้ามลิงก์และชื่อบริษัทจริง ตอบเป็น JSON: {\"news\":[{\"category\":str,\"headline\":str,\"blurb\":str},...]}"
    )
    user = (
        f"วันที่: {today}\n"
        f"สร้างข่าวตลาดหุ้นและการลงทุนจำนวน {n} เรื่อง ภาษาไทย แนวหนังสือพิมพ์  "
        "หัวข้อ 6–12 คำ คำโปรย 1 ประโยค ไม่ซ้ำกันภายในชุด. " + comp_clause
    )

    try:
        resp = ai_client.chat.completions.create(
            model=model_id,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.8
        )
        data = _safe_json(resp.choices[0].message.content)
        items = []
        for it in (data.get("news") or [])[:n]:
            cat = (it.get("category") or "ตลาดหุ้น")[:20]
            head = (it.get("headline") or "").strip()
            blurb = (it.get("blurb") or "").strip()
            if head:
                items.append({"category":cat,"headline":head,"blurb":blurb})
        if items:
            return items
    except Exception:
        pass

    # Hardcoded minimal fallback
    samples = [
        ("ตลาดหุ้น","ดัชนีหุ้นสมมติปิดบวกตามกลุ่มเทคโนโลยี","แรงซื้อหุ้นเติบโตหนุนตลาดแม้ผันผวน"),
        ("การเงิน","กองทุนตราสารทุนจำลองรับเงินไหลเข้าเพิ่ม","นักลงทุนเลือกธีมเติบโตอย่างระมัดระวัง"),
    ]
    return [{"category":c,"headline":h,"blurb":b} for c,h,b in samples[:n]]