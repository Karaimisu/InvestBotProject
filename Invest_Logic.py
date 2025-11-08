# invest_logic.py
import json
import random
from dataclasses import dataclass
from openai import OpenAI  # type hint only

DIFFICULTY_MAX = 5

@dataclass
class Scenario:
    name: str
    sector: str
    tone: str            # 'bullish' | 'bearish' | 'mixed'
    summary: str
    difficulty: int
    stars: str

def stars(n: int) -> str:
    n = max(1, min(DIFFICULTY_MAX, int(n)))
    return "★" * n + "☆" * (DIFFICULTY_MAX - n)

def balanced_difficulty(money: int) -> int:
    base = 1 if money < 5_000 else 2 if money < 15_000 else 3 if money < 30_000 else 4
    r = random.random()
    if r < 0.10: delta = -1
    elif r < 0.35: delta = 0
    elif r < 0.75: delta = +1
    else: delta = +2
    return max(1, min(DIFFICULTY_MAX, base + delta))

def _safe_parse_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

def generate_scenario(ai_client: OpenAI, company_name: str, sector: str, money: int, model_id: str | None = None) -> Scenario:
    diff = balanced_difficulty(money)
    sys = (
        "คุณคือเอนจินจำลองตลาดหุ้นสำหรับเกมสอนการเทรด "
        "ตอบกลับเป็น JSON เท่านั้น โดยใช้คีย์: name, sector, tone, summary "
        "tone ต้องเป็นหนึ่งใน: bullish, bearish, mixed "
        "summary ไม่เกิน 2 ประโยค"
    )
    user = (
        f"สร้างสถานการณ์ของหุ้นสมมติชื่อ '{company_name}' อยู่ในกลุ่มอุตสาหกรรม '{sector}'. "
        f"เป้าความยาก {diff} จาก 1-5 ให้สะท้อนความกำกวมตามระดับความยาก "
        "ห้ามให้คำแนะนำการลงทุนโดยตรง"
    )
    resp = ai_client.chat.completions.create(
        model=model_id or "typhoon-v1",
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.7
    )
    data = _safe_parse_json(resp.choices[0].message.content)
    name = data.get("name") or company_name
    sec = data.get("sector") or sector
    tone = data.get("tone") if data.get("tone") in {"bullish","bearish","mixed"} else random.choice(["bullish","bearish","mixed"])
    summary = data.get("summary") or "บริบทตลาดที่ถูกสุ่มสร้างสำหรับการจำลอง"
    return Scenario(name=name, sector=sec, tone=tone, summary=summary, difficulty=diff, stars=stars(diff))

def generate_news(ai_client: OpenAI, scenario: Scenario, model_id: str | None = None):
    hint = 1 if scenario.difficulty <= 2 else 2 if scenario.difficulty == 3 else 3
    n_items = random.randint(3, 5)
    sys = (
        "คุณจะสร้างข่าวสั้นเชิงสมมติสำหรับบริบทตลาด "
        "ตอบเป็น JSON เท่านั้น รูปแบบ: {\"news\": [{\"headline\": str, \"blurb\": str}, ...]} "
        "ไม่ใส่ลิงก์ ไม่ใส่สัญลักษณ์ตลาด"
    )
    user = (
        f"สร้างข่าว {n_items} รายการสำหรับบริษัท '{scenario.name}' กลุ่ม '{scenario.sector}'. "
        f"โทนรวม: {scenario.tone}. "
        f"ระดับใบ้คำ {hint}: 1=ชัด 2=กลาง 3=คลุมเครือ. "
        "หัวข้อข่าวยาว 6-10 คำ ย่อหน้าอธิบาย 1 ประโยค "
        "ถ้าระดับ 3 ให้หลีกเลี่ยงตัวเลขที่เฉพาะเจาะจง"
    )
    resp = ai_client.chat.completions.create(
        model=model_id or "typhoon-v1",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.8
    )
    data = _safe_parse_json(resp.choices[0].message.content)
    items = (data.get("news") or [])[:n_items]
    return items

def simulate_outcome(choice: str, scenario: Scenario, bankroll: int):
    trade_size = max(100, int(bankroll * 0.10)) if bankroll != 0 else 100
    tone_mu = {"bullish": 0.012, "bearish": -0.012, "mixed": 0.0}[scenario.tone]
    vol = 0.02 + 0.01 * scenario.difficulty
    drift = random.gauss(tone_mu, vol)
    if choice == "Buy":
        pnl_pct = drift
    elif choice == "Sell":
        pnl_pct = -drift
    elif choice == "Hold":
        pnl_pct = random.gauss(0.0, vol * 0.5)
    else:
        pnl_pct = 0.0
    pnl = int(trade_size * pnl_pct)
    new_bankroll = bankroll + pnl  # อนุญาตให้ติดลบได้
    feedback_seed = {
        "Buy": "คุณตัดสินใจเข้าซื้อ",
        "Sell": "คุณเลือกขายชอร์ต",
        "Hold": "คุณรอดูทิศทาง",
        "Read News": "คุณเลือกอ่านข่าว"
    }[choice]
    return new_bankroll, pnl, trade_size, feedback_seed

def generate_feedback(ai_client: OpenAI, scenario: Scenario, choice: str, pnl: int, trade_size: int, model_id: str | None = None):
    direction = "กำไร" if pnl >= 0 else "ขาดทุน"
    sys = "คุณเป็นติวเตอร์การเทรดในเกม ตอบสั้น 2 ประโยค หลีกเลี่ยงคำแนะนำลงทุนแบบตรงไปตรงมา"
    user = (
        f"โทนสถานการณ์: {scenario.tone}. ความยาก: {scenario.difficulty}. "
        f"การกระทำผู้เล่น: {choice}. ผลลัพธ์: {direction} {abs(pnl)} ดอลลาร์ จากขนาด {trade_size} ดอลลาร์. "
        "ให้ข้อเสนอแนะแบบกระชับ เน้นคุณภาพการตัดสินใจและการควบคุมความเสี่ยง ห้ามบอกให้ทำอะไรต่อ"
    )
    resp = ai_client.chat.completions.create(
        model=model_id or "typhoon-v1",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.6
    )
    return resp.choices[0].message.content.strip()