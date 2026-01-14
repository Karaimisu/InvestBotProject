# Invest_Logic.py
from dataclasses import dataclass
import random
from typing import Dict, Tuple
from openai import OpenAI


@dataclass
class Scenario:
    name: str
    sector: str
    difficulty: int       # 1..5
    stars: str            # "★☆☆☆☆"
    tone: str             # bullish/bearish/mixed
    summary: str
    correct_action: str   # Buy/Sell/Hold


def _stars(n: int) -> str:
    n = max(1, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)


def _pick_difficulty(bankroll: int) -> int:
    """
    Bias toward easier scenarios but still allow hard ones even for new players.
    """
    base = 1 if bankroll < 5_000 else 2 if bankroll < 20_000 else 3
    weights = {
        1: [0.45, 0.30, 0.15, 0.07, 0.03],
        2: [0.25, 0.35, 0.20, 0.12, 0.08],
        3: [0.15, 0.25, 0.30, 0.20, 0.10],
    }[base]
    return random.choices([1, 2, 3, 4, 5], weights=weights, k=1)[0]


def compute_trade_size(max_cap: int, bankroll: int, difficulty: int) -> int:
    """
    Cap the trade size based on bankroll and difficulty.
    Uses abs(bankroll) so it still works if user money is negative.
    """
    difficulty = max(1, min(5, difficulty))
    scale = {1: 0.15, 2: 0.20, 3: 0.25, 4: 0.30, 5: 0.35}[difficulty]
    return max(100, min(int(abs(bankroll) * scale), max_cap))


def generate_scenario(
    ai: OpenAI,
    company: str,
    sector: str,
    bankroll: int,
    *,
    model_id: str,
) -> Scenario:
    """
    Generate scenario with tone and summary via Typhoon.
    Correct action is derived from tone (so it’s not always Buy).
    """
    difficulty = _pick_difficulty(bankroll)

    sys = "คุณเป็นผู้ช่วยสร้างสถานการณ์ลงทุนให้ผู้ใช้ฝึกตัดสินใจ หลีกเลี่ยงคำแนะนำเชิงส่วนบุคคล"
    user = (
        f"สร้างสถานการณ์จำลองหุ้นเป็นภาษาไทยแบบละเอียด 4-6 บรรทัด\n"
        f"- ชื่อบริษัท: {company}, กลุ่ม: {sector}\n"
        f"- ระดับความยาก (1-5): {difficulty}\n\n"
        "ข้อกำหนดสำคัญ:\n"
        "1. สร้างเหตุการณ์เฉพาะเจาะจง เช่น: ประกาศผลประกอบการ, ซื้อกิจการ, เปิดตัวผลิตภัณฑ์, "
        "ปรับโครงสร้าง, ฟ้องร้อง, ได้สัญญาใหม่, ปัญหาซัพพลายเชน ฯลฯ\n"
        "2. ใส่ตัวเลขจริง เช่น: กำไรเพิ่มขึ้น 23%, ราคาหุ้นร่วง 15%, สัญญามูลค่า 500 ล้านบาท\n"
        "3. อธิบายบริบทตลาดโดยรอบ เช่น: คู่แข่งทำอะไร, สภาพเศรษฐกิจ, นโยบายรัฐ\n"
        "4. ห้ามบอกตรงๆ ว่าควร Buy/Sell/Hold แต่ให้ข้อมูลเพียงพอที่ผู้เล่นจะตัดสินใจได้\n"
        "5. ใช้โทนเล่าเรื่องแบบข่าว ไม่ใช่รายงานแห้งๆ\n\n"
        "ตอบ JSON: {'tone':'bullish'|'bearish'|'mixed', 'summary':'...(สถานการณ์ละเอียด)...'}"
    )

    try:
        resp = ai.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.4,
            max_tokens=220,
        )
        txt = resp.choices[0].message.content.strip()
    except Exception:
        txt = "{'tone':'mixed','summary':'ภาวะไม่ชัดเจน ราคาผันผวน'}"

    tone, summary = "mixed", "ภาวะไม่ชัดเจน ราคาผันผวน"
    try:
        import json
        j = json.loads(txt.replace("'", '"'))
        tone = (j.get("tone") or "mixed").lower()
        summary = j.get("summary", "ภาวะไม่ชัดเจน ราคาผันผวน")
    except Exception:
        pass

    # Derive correct action from tone + randomness (so it’s not always Buy)
    tone_key = tone.lower()
    if "bull" in tone_key:
        actions = ["Buy", "Hold", "Sell"]
        weights = [0.6, 0.3, 0.1]
    elif "bear" in tone_key:
        actions = ["Sell", "Hold", "Buy"]
        weights = [0.6, 0.3, 0.1]
    else:  # mixed / sideways
        actions = ["Hold", "Buy", "Sell"]
        weights = [0.5, 0.25, 0.25]

    correct_action = random.choices(actions, weights=weights, k=1)[0]

    return Scenario(
        name=company,
        sector=sector,
        difficulty=difficulty,
        stars=_stars(difficulty),
        tone=tone,
        summary=summary,
        correct_action=correct_action,
    )


def generate_news(ai: OpenAI, scenario: Scenario, *, model_id: str) -> Dict:
    """
    News focuses on 'company decisions' and clarity depends on difficulty:

    - difficulty 1–2 (easy):
        Every blurb clearly states if investors/public see it as positive or negative.
    - difficulty 3 (medium):
        Mix of clearly positive/negative and 'mixed/divided' reactions.
    - difficulty 4–5 (hard):
        Several items are just factual; reaction is ambiguous or conflicting.
    """
    d = scenario.difficulty
    if d <= 2:
        reaction_rule = (
            "ระดับความยากต่ำ (ง่าย): ให้ทุกข่าวระบุชัดใน blurb ว่าข่าวนี้ "
            "เป็นบวกหรือเป็นลบต่อมุมมองนักลงทุน เช่น "
            "'โดยภาพรวมตลาดตอบรับเชิงบวก' หรือ 'นักลงทุนส่วนใหญ่กังวลและมองเชิงลบ'"
        )
    elif d == 3:
        reaction_rule = (
            "ระดับความยากกลาง: ให้บางข่าวระบุชัดว่าเป็นบวกหรือลบ, "
            "บางข่าวให้บอกว่า 'มุมมองนักลงทุนยังแบ่งฝ่าย/ผสม' อย่างชัดเจน"
        )
    else:
        reaction_rule = (
            "ระดับความยากสูง: หลายข่าวเล่าเป็นข้อเท็จจริงเฉยๆ โดยไม่ฟันธงชัดว่าเป็นบวกหรือลบ, "
            "บางข่าวอาจมีมุมมองขัดแย้งกัน ให้ผู้เล่นต้องตีความเอง"
        )

    sys = "คุณช่วยสร้างพาดหัวข่าวสั้นเกี่ยวกับการตัดสินใจของบริษัทในตลาดหุ้น เพื่อให้ผู้เล่นฝึกอ่านข่าว"
    user = (
        f"บริษัท {scenario.name} ({scenario.sector}) โทนตลาดภาพรวม: {scenario.tone}.\n"
        "สร้างข่าวภาษาไทย 6-8 ชิ้น โดยแต่ละข่าวต้องมี:\n"
        "- headline: อธิบาย 'การตัดสินใจของบริษัท' ให้เห็นชัด "
        "เช่น เปิดตัวผลิตภัณฑ์ใหม่, ปรับลดต้นทุน, ซื้อกิจการ, ปรับโครงสร้าง, ขยายตลาด ฯลฯ\n"
        "- blurb: สรุปสั้น 1 ประโยคว่าเกิดอะไรขึ้น และมุมมองของตลาด/นักลงทุนเป็นอย่างไร "
        "(ถ้าตามกติกาความยากต้องพูดถึง)\n"
        f"- {reaction_rule}\n\n"
        "อย่าใช้ศัพท์เทคนิคเยอะเกินไป ให้คนเริ่มต้นอ่านรู้เรื่อง\n"
        "ตอบ JSON: {\n"
        "  'items': [\n"
        "    {'headline': '...', 'blurb': '...'}, ...\n"
        "  ],\n"
        "  'brief': {\n"
        "    'bias': 'bullish' | 'bearish' | 'mixed',\n"
        "    'pros': ['มุมบวกสำคัญ 1-3 ข้อ'],\n"
        "    'cons': ['มุมเสี่ยง 1-3 ข้อ'],\n"
        "    'signals': ['ตัวชี้วัดหรือสิ่งที่ควรจับตา 1-3 ข้อ']\n"
        "  }\n"
        "}"
    )

    try:
        r = ai.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.6,
            max_tokens=420,
        )
        txt = r.choices[0].message.content
        import json
        j = json.loads(txt.replace("'", '"'))
        items = j.get("items", [])
        brief = j.get("brief", {})
    except Exception:
        items = [
            {
                "headline": "บริษัทประกาศกลยุทธ์ใหม่",
                "blurb": "บริษัทเปลี่ยนทิศทางธุรกิจครั้งใหญ่ ทำให้นักลงทุนต้องจับตาปฏิกิริยาตลาดอย่างใกล้ชิด",
            }
        ]
        brief = {"bias": "mixed", "pros": [], "cons": [], "signals": []}

    return {"items": items[:8], "brief": brief}


def simulate_outcome(
    choice: str,
    scenario: Scenario,
    money: int,
    *,
    requested_amount: int,
) -> Tuple[int, int, int, Dict]:
    """
    Profit/loss is based on the invested amount AND the scenario's tone.

    The scenario's tone (bullish/bearish/mixed) now directly influences
    the underlying "market movement" direction:
    - Bullish: Market tends to go UP. Buying is advantageous.
    - Bearish: Market tends to go DOWN. Selling is advantageous.
    - Mixed: Market is unpredictable. Holding is safer.

    The player's choice then interacts with this movement:
    - Buy in an up-market = profit. Buy in a down-market = loss.
    - Sell in a down-market = profit. Sell in an up-market = loss.
    - Hold = small P&L based on market direction.

    Difficulty scales the volatility (magnitude of moves).
    """
    base = max(1, int(requested_amount))

    vol_map = {
        1: (0.03, 0.10),
        2: (0.04, 0.14),
        3: (0.05, 0.18),
        4: (0.06, 0.22),
        5: (0.07, 0.25),
    }
    low, high = vol_map[max(1, min(5, scenario.difficulty))]

    # --- Determine market direction based on scenario tone ---
    tone_key = scenario.tone.lower()
    if "bull" in tone_key:
        # Bullish: market moves up on average
        market_move = random.uniform(low * 0.5, high)
    elif "bear" in tone_key:
        # Bearish: market moves down on average
        market_move = random.uniform(-high, -low * 0.5)
    else:
        # Mixed/sideways: small, unpredictable move
        market_move = random.uniform(-low, low)

    # --- Determine P&L based on player's choice vs market direction ---
    choice_upper = choice.upper()
    if choice_upper == "BUY":
        # Profit if market goes up, loss if market goes down
        pct = market_move
    elif choice_upper == "SELL":
        # Profit if market goes down (short position), loss if market goes up
        pct = -market_move
    else:  # HOLD
        # Small fraction of the move (you're not fully exposed)
        pct = market_move * random.uniform(0.1, 0.3)

    # --- Bonus for choosing the "correct" action (educational reward) ---
    if choice == scenario.correct_action:
        bonus = random.uniform(low * 0.2, low * 0.5)
        pct += bonus if pct >= 0 else -bonus  # amplify the result

    pnl = int(base * pct)
    new_money = money + pnl

    return new_money, pnl, base, {"pct": pct, "market_move": market_move}


def generate_tip_and_reason(
    ai: OpenAI,
    scenario: Scenario,
    action: str,
    pnl: int,
    size: int,
    trend: str,
    *,
    model_id: str,
) -> str:
    sys = "คุณอธิบายเหตุผลราคาหุ้นและให้คำแนะนำเพื่อการเรียนรู้แบบสั้น ชัด ไม่สั่งให้ทำจริง"
    user = (
        f"บริษัท: {scenario.name} ({scenario.sector}) | โทน: {scenario.tone} | แนวโน้มกราฟ: {trend}\n"
        f"ผู้เล่นเลือก: {action} | ผลลัพธ์ P&L: {pnl} จากขนาด {size}\n"
        "อธิบายสั้นว่าทำไมราคาจึงขึ้น/ลง/แกว่ง จากมุมมองข่าว/เทคนิค แล้วให้ TIP 1-2 ข้อ\n"
        "ตอบเป็นไทย 3-5 บรรทัด"
    )
    try:
        r = ai.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.5,
            max_tokens=350,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return "ราคาขยับตามกระแสข่าวและสัญญาณเทคนิค TIP: รอการยืนยันด้วยปริมาณก่อนเพิ่มน้ำหนัก"