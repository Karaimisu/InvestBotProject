# Invest_Logic.py
import json
import random
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI  # type hint only

ACTIONS = ("Buy", "Sell", "Hold")
DIFFICULTY_MAX = 5

@dataclass
class Scenario:
    name: str
    sector: str
    tone: str             # 'bullish' | 'bearish' | 'mixed'
    summary: str
    difficulty: int
    stars: str
    correct_action: str   # one of ACTIONS

def stars(n: int) -> str:
    n = max(1, min(DIFFICULTY_MAX, int(n)))
    return "★" * n + "☆" * (DIFFICULTY_MAX - n)

def balanced_difficulty(money: int) -> int:
    """
    Tuned so 'hard' occurs less often.
    Base by bankroll; random tilt biased toward same/softer difficulty.
    """
    base = 1 if money < 5_000 else 2 if money < 15_000 else 3 if money < 30_000 else 4
    r = random.random()
    # 60% stay at base, 25% easier (-1), 12% +1, 3% +2
    if r < 0.25: delta = -1
    elif r < 0.85: delta = 0
    elif r < 0.97: delta = +1
    else: delta = +2
    return max(1, min(DIFFICULTY_MAX, base + delta))

def pick_correct_action(tone: str, difficulty: int) -> str:
    """
    Fix one 'correct' action for this scenario.
    Low difficulty aligns with tone; higher difficulty allows contrarian outcomes.
    """
    contrarian_p = 0.05 * difficulty  # 5%..25%
    if tone == "bullish":
        return random.choices(["Buy", "Hold", "Sell"], weights=[1-contrarian_p, contrarian_p*0.6, contrarian_p*0.4])[0]
    if tone == "bearish":
        return random.choices(["Sell", "Hold", "Buy"], weights=[1-contrarian_p, contrarian_p*0.6, contrarian_p*0.4])[0]
    # mixed
    return random.choices(["Hold", "Buy", "Sell"], weights=[0.7, 0.15, 0.15])[0]

def _safe_parse_json(txt: str) -> dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

def generate_scenario(ai_client: OpenAI, company_name: str, sector: str, money: int, model_id: Optional[str] = None) -> Scenario:
    diff = balanced_difficulty(money)
    sys = (
        "คุณคือเอนจินจำลองตลาดหุ้นสำหรับเกมสอนการเทรด "
        "ตอบกลับเป็น JSON เท่านั้น โดยใช้คีย์: name, sector, tone, summary "
        "tone ต้องเป็นหนึ่งใน: bullish, bearish, mixed "
        "summary ไม่เกิน 2 ประโยค ให้สะท้อนความยาก (ความกำกวม/ปัจจัยขัดแย้งเพิ่มเมื่อยากขึ้น)"
    )
    user = (
        f"สร้างสถานการณ์ของหุ้นสมมติชื่อ '{company_name}' ในอุตสาหกรรม '{sector}'. "
        f"เป้าความยาก {diff} จาก 1-5. "
        "ห้ามให้คำแนะนำลงทุนตรงๆ"
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
    summary = data.get("summary") or "บริบทตลาดที่สุ่มสร้าง"
    correct = pick_correct_action(tone, diff)
    return Scenario(
        name=name, sector=sec, tone=tone, summary=summary,
        difficulty=diff, stars=stars(diff), correct_action=correct
    )

def generate_news(ai_client: OpenAI, scenario: Scenario, model_id: Optional[str] = None):
    """
    Decision-oriented news: include Pros/Cons/Signals to help choose.
    """
    hint = 1 if scenario.difficulty <= 2 else 2 if scenario.difficulty == 3 else 3
    n_items = 4  # keep tight and useful
    sys = (
        "คุณจะสร้างข่าวสั้นเพื่อช่วยผู้เล่นตัดสินใจในสถานการณ์หุ้น "
        "ตอบเป็น JSON: {"
        "\"news\": [{\"headline\": str, \"blurb\": str}], "
        "\"decision_brief\": {"
        "\"pros\": [str], \"cons\": [str], "
        "\"signals\": [str], \"bias\": \"bullish|bearish|mixed\" } } "
        "หลีกเลี่ยงตัวเลขละเอียดและลิงก์"
    )
    user = (
        f"บริษัท: '{scenario.name}', อุตสาหกรรม: '{scenario.sector}'. "
        f"โทนสถานการณ์: {scenario.tone}. ระดับใบ้คำ {hint} (3=กำกวม). "
        "หัวข้อข่าว 6-10 คำ, คำโปรย 1 ประโยค. "
        "ใน decision_brief ให้ระบุ pros/cons ที่ชี้ไปสู่ Buy/Sell/Hold และสัญญาณสำคัญสั้นๆ"
    )
    resp = ai_client.chat.completions.create(
        model=model_id or "typhoon-v1",
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0.7
    )
    data = _safe_parse_json(resp.choices[0].message.content)
    items = (data.get("news") or [])[:n_items]
    brief = data.get("decision_brief") or {}
    # sanitize brief
    brief.setdefault("pros", [])
    brief.setdefault("cons", [])
    brief.setdefault("signals", [])
    brief["bias"] = brief.get("bias") if brief.get("bias") in {"bullish","bearish","mixed"} else "mixed"
    return {"items": items, "brief": brief}

def _max_pct_by_bankroll(bankroll: int) -> float:
    """
    Max position size as % of bankroll, grows with bankroll.
    """
    b = abs(bankroll)
    if b < 5_000: return 0.08
    if b < 20_000: return 0.12
    if b < 100_000: return 0.18
    return 0.25

def _difficulty_vol_multiplier(difficulty: int) -> float:
    # impacts volatility and payout range
    return 1.0 + 0.15 * (difficulty - 1)   # 1.0 .. 1.6

def _floor_min_by_difficulty(difficulty: int) -> int:
    # min absolute dollar swing (100..500)
    return min(500, 100 + (difficulty - 1) * 100)

def compute_trade_size(request_amount: Optional[int], bankroll: int, difficulty: int) -> int:
    """
    Applies floors/caps. If bankroll < 0, cap to amount needed to return to >= 0.
    """
    # hard floor 100
    floor_abs = 100
    # soft cap by bankroll
    cap_pct = _max_pct_by_bankroll(bankroll)
    cap_abs = int(max(200, abs(bankroll) * cap_pct))
    # negative-balance rule: cannot invest more than needed to get back to >= 0
    if bankroll < 0:
        cap_abs = min(cap_abs, abs(bankroll))
    # if user provided, clamp it
    if request_amount and request_amount > 0:
        size = max(floor_abs, min(request_amount, cap_abs))
    else:
        # default: 10% of bankroll, clamped
        size = int(abs(bankroll) * 0.10)
        size = max(floor_abs, min(size, cap_abs))
    return size

def simulate_outcome(choice: str, scenario: Scenario, bankroll: int, requested_amount: Optional[int] = None):
    """
    One fixed correct answer. P&L scales with difficulty and bankroll.
    """
    size = compute_trade_size(requested_amount, bankroll, scenario.difficulty)

    # base drift from scenario tone
    base_mu = {"bullish": 0.010, "bearish": -0.010, "mixed": 0.0}[scenario.tone]
    vol_mult = _difficulty_vol_multiplier(scenario.difficulty)
    sigma = 0.02 * vol_mult  # 2% .. 3.2%

    # outcome shaping by correctness
    # correct: tilt in your favor; wrong: tilt against you; hold: small variance around zero
    if choice == scenario.correct_action:
        drift = random.gauss(base_mu * 1.5, sigma)
    elif choice == "Hold":
        drift = random.gauss(0.0, sigma * 0.5)
    else:
        drift = random.gauss(-base_mu * 1.2, sigma)

    raw_pnl = int(size * drift)

    # ensure min absolute swing based on difficulty
    min_abs = _floor_min_by_difficulty(scenario.difficulty)
    if raw_pnl >= 0:
        pnl = max(raw_pnl, min_abs)
    else:
        pnl = min(raw_pnl, -min_abs)

    new_bankroll = bankroll + pnl  # allow negatives

    feedback_seed = {
        "Buy": "คุณตัดสินใจเข้าซื้อ",
        "Sell": "คุณเลือกขายชอร์ต",
        "Hold": "คุณรอดูทิศทาง",
        "Read News": "คุณเลือกอ่านข่าว"
    }[choice]

    return new_bankroll, pnl, size, feedback_seed