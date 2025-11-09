# main.py
from dotenv import load_dotenv
import os, json, asyncio
from io import BytesIO
from pathlib import Path
import pandas as pd
import discord
from discord.ext import commands
from discord import Interaction, File, Embed, app_commands
from discord.errors import NotFound, HTTPException
from openai import OpenAI

# --- local modules ---
from User_Data import load_or_create_excel, get_user_money, set_user_money
from Invest_Logic import (
    generate_scenario, generate_news, simulate_outcome,
    compute_trade_size, Scenario, balanced_difficulty, generate_tip_and_reason
)
from GenGraph import generate_stock_graph
from Model_Util import resolve_model
from RAG_Chat import answer_investing_question
from RAG_Store import rebuild_index
from Daily_News import (
    refresh_daily_news_real, load_today_news, seconds_until_next_refresh,  # live
    generate_daily_news  # synthetic fallback
)

# ------------------- env -------------------
load_dotenv()
TOKEN = os.getenv("Bot_Token")
TYPHOON_KEY = os.getenv("Typhoon_Key")
TYPHOON_MODEL_ENV = os.getenv("Typhoon_Model")
NEWS_REFRESH_HHMM = os.getenv("NEWS_REFRESH_HHMM", "08:00")
NEWS_HH, NEWS_MM = map(int, NEWS_REFRESH_HHMM.split(":"))

AI = OpenAI(api_key=TYPHOON_KEY, base_url="https://api.opentyphoon.ai/v1")
MODEL_ID = resolve_model(AI, TYPHOON_MODEL_ENV)
excel_path = load_or_create_excel()

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LAST_COMPANY_PATH = DATA_DIR / "last_company.json"

# ------------------- discord -------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="=", intents=intents)

# ------------------- helpers -------------------
async def safe_defer(interaction: Interaction, *, ephemeral: bool = False, thinking: bool = False):
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=ephemeral, thinking=thinking)
    except (NotFound, HTTPException):
        pass

async def ensure_button_ack(interaction: Interaction, *, ephemeral: bool = True):
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=ephemeral, thinking=False)
    except (NotFound, HTTPException):
        pass

async def safe_followup(interaction: Interaction, **kwargs):
    try:
        return await interaction.followup.send(**kwargs)
    except NotFound:
        kwargs.pop("ephemeral", None)
        return await interaction.channel.send(**kwargs)
    except HTTPException:
        return None

def _trend_from_csv(csv_path: str) -> str:
    try:
        df = pd.read_csv(csv_path)
        df = df.tail(60).reset_index(drop=True)
        if "Close" not in df.columns:
            return "sideways"
        y = df["Close"].values
        x = pd.Series(range(len(y))).values
        xm = x.mean(); ym = y.mean()
        denom = ((x - xm) ** 2).sum() or 1.0
        slope = ((x - xm) * (y - ym)).sum() / denom
        sma_fast = pd.Series(y).rolling(10).mean().iloc[-1]
        sma_slow = pd.Series(y).rolling(30).mean().iloc[-1]
        if pd.isna(sma_fast) or pd.isna(sma_slow):
            sma_fast, sma_slow = y[-1], y[-1]
        if slope > 0 and sma_fast >= sma_slow:
            return "uptrend"
        if slope < 0 and sma_fast <= sma_slow:
            return "downtrend"
        return "sideways"
    except Exception:
        return "sideways"

def _news_analysis_cost_by_difficulty(difficulty: int, bankroll: int) -> int:
    difficulty = max(1, min(5, int(difficulty)))
    rate_table = {1: 0.005, 2: 0.01, 3: 0.015, 4: 0.02, 5: 0.025}
    rate = rate_table[difficulty]
    return max(1, int(abs(bankroll) * rate))

def _file_from_path(path: str) -> File | None:
    try:
        with open(path, "rb") as f:
            buf = BytesIO(f.read())
        buf.seek(0)
        return File(buf, filename=os.path.basename(path))
    except Exception:
        return None

# ------------------- background news refresh -------------------
async def news_refresher_loop():
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            refresh_daily_news_real(AI, MODEL_ID, DATA_DIR)
        except Exception:
            pass
        delay = seconds_until_next_refresh(NEWS_HH, NEWS_MM)
        await asyncio.sleep(max(60, delay))

# ------------------- views -------------------
class InvestView(discord.ui.View):
    def __init__(self, ai_client, model_id: str, user_id: int, username: str, money: int, scenario: Scenario,
                 excel_path: str, requested_amount: int, trend: str, clamped_note: str | None):
        super().__init__(timeout=300)
        self.ai = ai_client
        self.model_id = model_id
        self.user_id = user_id
        self.username = username
        self.money = money
        self.scenario = scenario
        self.excel_path = excel_path
        self.requested_amount = requested_amount
        self.trend = trend
        self.clamped_note = clamped_note

        self.round_finished = False
        self.news_shown = False
        self.analysis_done = False
        self._cached_news = None  # {items, brief}

    def _lock_main_buttons(self):
        for child in self.children:
            if isinstance(child, discord.ui.Button) and child.custom_id != "read_news":
                child.disabled = True

    async def _send_news_embed_with_button(self, interaction: Interaction, news: dict):
        items = news.get("items", [])
        shown = items[:7] if len(items) >= 5 else items
        desc_items = "\n".join([
            f"• **{it.get('headline','')}** — {it.get('blurb','')}"
            for it in shown
        ]) or "ไม่มีข่าว"

        cost = _news_analysis_cost_by_difficulty(self.scenario.difficulty, self.money)
        pct_map = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5}
        pct = pct_map.get(int(self.scenario.difficulty), 1.0)

        disclaimer = (
            f"\n\n⚠️ **หมายเหตุ**: หากกดปุ่ม '**วิเคราะห์ & คำแนะนำ**' "
            f"ระบบจะหักเงิน **{pct:.1f}%** ของยอดเงินปัจจุบัน ≈ **${cost:,}** "
            f"ตามระดับความยาก เพื่อแสดงสรุปเชิงลึกและคำแนะนำเชิงการเรียนรู้"
        )

        embed = Embed(
            title="📰 ข่าวเพื่อช่วยตัดสินใจ (โฟกัสบริษัทจำลอง)",
            description=desc_items + disclaimer,
            color=discord.Color.dark_teal()
        )
        embed.set_footer(text="นี่คือสรุปข่าวเท่านั้น ยังไม่รวมการวิเคราะห์เชิงลึก")

        view = NewsAnalysisView(self)
        await safe_followup(interaction, embed=embed, view=view, ephemeral=True)

    async def _handle_choice(self, interaction: Interaction, choice: str):
        if interaction.user.id != self.user_id:
            await safe_followup(interaction, content="This session belongs to another user.", ephemeral=True)
            return

        if choice == "Read News":
            if not self.news_shown:
                batches = []
                for _ in range(3):
                    try:
                        batches.append(generate_news(self.ai, self.scenario, model_id=self.model_id))
                    except Exception:
                        pass
                if not batches:
                    batches = [{"items": [], "brief": {"bias": "mixed", "pros": [], "cons": [], "signals": []}}]

                seen = set()
                merged = []
                company = self.scenario.name.split(" — ")[0] if " — " in self.scenario.name else self.scenario.name
                company_low = company.lower()

                def score_item(it):
                    h = (it.get("headline") or "").lower()
                    b = (it.get("blurb") or "").lower()
                    s = 0
                    if company_low in h: s += 3
                    if company_low in b: s += 2
                    if self.scenario.sector and (self.scenario.sector.lower() in h or self.scenario.sector.lower() in b):
                        s += 1
                    return s

                for batch in batches:
                    for it in batch.get("items", []):
                        h = (it.get("headline") or "").strip()
                        if not h or h in seen:
                            continue
                        seen.add(h)
                        blurb = (it.get("blurb") or "").strip()
                        if company_low not in blurb.lower():
                            it["blurb"] = f"{blurb} (เกี่ยวข้องกับ {company})" if blurb else f"อัปเดตที่เกี่ยวข้องกับ {company}"
                        merged.append(it)

                merged.sort(key=score_item, reverse=True)
                if len(merged) < 5:
                    try:
                        extra = generate_news(self.ai, self.scenario, model_id=self.model_id).get("items", [])
                        for it in extra:
                            h = (it.get("headline") or "").strip()
                            if h and h not in seen:
                                seen.add(h)
                                blurb = (it.get("blurb") or "").strip()
                                if company_low not in blurb.lower():
                                    it["blurb"] = f"{blurb} (เกี่ยวข้องกับ {company})" if blurb else f"อัปเดตที่เกี่ยวข้องกับ {company}"
                                merged.append(it)
                    except Exception:
                        pass
                    merged.sort(key=score_item, reverse=True)

                target_n = 7 if len(merged) >= 7 else max(5, len(merged))
                merged = merged[:target_n]

                base_brief = {}
                for b in batches:
                    if b.get("brief"):
                        base_brief = b["brief"]; break
                self._cached_news = {"items": merged, "brief": base_brief or {"bias": "mixed", "pros": [], "cons": [], "signals": []}}
                self.news_shown = True

            await self._send_news_embed_with_button(interaction, self._cached_news)
            return

        if self.round_finished:
            await safe_followup(interaction, content="คุณได้เลือกไปแล้ว โปรดเริ่มรอบใหม่ด้วย /invest", ephemeral=True)
            return

        # Simulate
        new_money, pnl, size, _ = simulate_outcome(
            choice, self.scenario, self.money, requested_amount=self.requested_amount
        )
        set_user_money(self.user_id, self.excel_path, new_money)
        self.money = new_money
        self.round_finished = True
        self._lock_main_buttons()
        try:
            if interaction.message:
                await interaction.message.edit(view=self)
        except (NotFound, HTTPException):
            pass

        tip_reason = generate_tip_and_reason(self.ai, self.scenario, choice, pnl, size, self.trend, model_id=self.model_id)

        invested = max(1, int(self.requested_amount))
        pnl_pct = (pnl / invested) * 100.0
        sign = "+" if pnl >= 0 else "-"
        pnl_pct_str = f"{sign}{abs(pnl_pct):.2f}%"
        pnl_abs_str = f"{sign}${abs(pnl):,}"

        desc = [f"คำตอบที่ถูกต้องคือ: **{self.scenario.correct_action}**", tip_reason]
        if self.clamped_note:
            desc.append(self.clamped_note)

        embed = Embed(
            title=f"📊 ผลลัพธ์: {choice}",
            description="\n\n".join(desc),
            color=discord.Color.green() if pnl >= 0 else discord.Color.red()
        )
        embed.add_field(name="P&L (เทียบเงินที่ลงทุน)", value=pnl_pct_str, inline=True)
        embed.add_field(name="P&L มูลค่า", value=pnl_abs_str, inline=True)
        embed.add_field(name="ขนาดสถานะ", value=f"${size:,}", inline=True)
        embed.add_field(name="ยอดเงินใหม่", value=f"${new_money:,}", inline=True)
        await safe_followup(interaction, embed=embed, ephemeral=True)

    async def run_paid_analysis(self, interaction: Interaction):
        if interaction.user.id != self.user_id:
            await safe_followup(interaction, content="This session belongs to another user.", ephemeral=True)
            return
        if not self.news_shown or not self._cached_news:
            await safe_followup(interaction, content="ยังไม่มีข่าวสำหรับการวิเคราะห์ โปรดกด Read News ก่อน", ephemeral=True)
            return
        if self.analysis_done:
            await safe_followup(interaction, content="คุณได้กดวิเคราะห์ไปแล้วสำหรับรอบนี้", ephemeral=True)
            return

        cost = _news_analysis_cost_by_difficulty(self.scenario.difficulty, self.money)
        new_money = self.money - cost
        set_user_money(self.user_id, self.excel_path, new_money)
        self.money = new_money
        self.analysis_done = True

        brief = self._cached_news.get("brief", {})
        pros = brief.get("pros", [])[:5]
        cons = brief.get("cons", [])[:5]
        sigs = brief.get("signals", [])[:5]
        bias = brief.get("bias", "mixed")

        tip_reason = generate_tip_and_reason(
            self.ai, self.scenario, "Analyze", -cost, cost, self.trend, model_id=self.model_id
        )

        rec_hint = {
            "bullish": "แนวโน้มข่าวเอียงเชิงบวก: ฝึกพิจารณา Buy หรือรอจังหวะย่อที่ยืนยันด้วยปริมาณ",
            "bearish": "แนวโน้มข่าวเอียงเชิงลบ: ฝึกพิจารณา Sell/หลีกเลี่ยง จนกว่าจะมีสัญญาณกลับตัวชัด",
            "mixed": "แนวโน้มข่าวผสม: ฝึก Hold/ลดขนาดสถานะ รอสัญญาณชัดขึ้น"
        }.get(bias, "แนวโน้มไม่ชัด: ฝึกรอการยืนยันจากราคา/ปริมาณเพิ่มเติม")

        sections = []
        if pros:
            sections.append("**ปัจจัยหนุนที่ตรวจพบ**\n" + "\n".join(f"• {p}" for p in pros))
        if cons:
            sections.append("**ปัจจัยกดดันที่ตรวจพบ**\n" + "\n".join(f"• {c}" for c in cons))
        if sigs:
            sections.append("**ตัวชี้วัด/สัญญาณที่ควรจับตา**\n" + "\n".join(f"• {s}" for s in sigs))

        sections.append(f"**สรุปแนวโน้มข่าวรวม:** {bias}")
        sections.append(f"**แนวทางเชิงการเรียนรู้:** {rec_hint}")
        sections.append("**คำอธิบายเหตุผลของทิศทาง (TIP & REASON)**\n" + tip_reason)
        sections.append(
            f"\n💸 ค่าบริการการวิเคราะห์ถูกหักแล้ว: **-${cost:,}**  | ยอดเงินคงเหลือ: **${self.money:,}**\n"
            "🛑 เนื้อหานี้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุนจริง"
        )

        em = Embed(
            title="🔍 วิเคราะห์เชิงลึก & คำแนะนำจากข่าว",
            description="\n\n".join(sections),
            color=discord.Color.dark_orange()
        )
        await safe_followup(interaction, embed=em, ephemeral=True)

    @discord.ui.button(label="Buy", style=discord.ButtonStyle.primary)
    async def buy(self, interaction: Interaction, button: discord.ui.Button):
        await ensure_button_ack(interaction, ephemeral=True)
        await self._handle_choice(interaction, "Buy")

    @discord.ui.button(label="Sell", style=discord.ButtonStyle.primary)
    async def sell(self, interaction: Interaction, button: discord.ui.Button):
        await ensure_button_ack(interaction, ephemeral=True)
        await self._handle_choice(interaction, "Sell")

    @discord.ui.button(label="Hold", style=discord.ButtonStyle.secondary)
    async def hold(self, interaction: Interaction, button: discord.ui.Button):
        await ensure_button_ack(interaction, ephemeral=True)
        await self._handle_choice(interaction, "Hold")

    @discord.ui.button(label="Read News", style=discord.ButtonStyle.success, custom_id="read_news")
    async def news(self, interaction: Interaction, button: discord.ui.Button):
        await ensure_button_ack(interaction, ephemeral=True)
        await self._handle_choice(interaction, "Read News")


class NewsAnalysisView(discord.ui.View):
    def __init__(self, parent: InvestView):
        super().__init__(timeout=180)
        self.parent = parent

    @discord.ui.button(
        label="วิเคราะห์ & คำแนะนำ (เสียเงินตามความยาก)",
        style=discord.ButtonStyle.danger
    )
    async def analyze(self, interaction: Interaction, button: discord.ui.Button):
        await ensure_button_ack(interaction, ephemeral=True)
        await self.parent.run_paid_analysis(interaction)
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        try:
            if interaction.message:
                await interaction.message.edit(view=self)
        except (NotFound, HTTPException):
            pass

# ------------------- lifecycle -------------------
@bot.event
async def on_ready():
    await bot.tree.sync()
    if not getattr(bot, "_news_loop_started", False):
        bot._news_loop_started = True
        bot.loop.create_task(news_refresher_loop())
    print("------- Bot Started -------")

# ------------------- commands -------------------
@bot.tree.command(name="invest", description="เริ่มสถานการณ์การลงทุน (ต้องระบุจำนวนเงินลงทุน)")
@app_commands.describe(amount="จำนวนเงินที่ต้องการลงทุน")
async def invest(interaction: Interaction, amount: int):
    # Do NOT defer here. Build first, then send exactly once.

    user_id = interaction.user.id
    username = interaction.user.name
    money = get_user_money(user_id, username, excel_path)

    # Generate graph and read PNG into memory to avoid file-handle issues
    graph = generate_stock_graph()
    company, sector = graph["company"], graph["sector"]
    png_path = graph["png_path"]; csv_path = graph["csv_path"]
    trend = _trend_from_csv(csv_path)

    # Read file into memory buffer so it remains open during send
    file_obj = None
    try:
        with open(png_path, "rb") as f:
            from io import BytesIO
            buf = BytesIO(f.read())
            buf.seek(0)
            file_obj = File(buf, filename=os.path.basename(png_path))
    except Exception:
        file_obj = None

    with open(LAST_COMPANY_PATH, "w", encoding="utf-8") as f:
        json.dump({"name": company, "sector": sector}, f, ensure_ascii=False)

    scenario = generate_scenario(AI, company, sector, money, model_id=MODEL_ID)
    cap = compute_trade_size(10**9, money, scenario.difficulty)
    invest_amount = max(100, min(amount, cap))
    clamped_note = None
    if invest_amount != amount:
        clamped_note = f"หมายเหตุ: จำกัดการลงทุนสูงสุดไว้ที่ ${invest_amount:,}"

    em = Embed(
        title=f"📈 {scenario.name} — {scenario.sector}",
        description=scenario.summary,
        color=discord.Color.blue()
    )
    em.add_field(name="ความยาก", value=scenario.stars, inline=True)
    em.add_field(name="โทนตลาด", value=scenario.tone.capitalize(), inline=True)
    em.add_field(name="ยอดเงิน", value=f"${money:,}", inline=True)
    em.add_field(name="เพดานลงทุน", value=f"${cap:,}", inline=True)
    em.add_field(name="ลงทุนรอบนี้", value=f"${invest_amount:,}", inline=True)

    view = InvestView(AI, MODEL_ID, user_id, username, money, scenario, excel_path, invest_amount, trend, clamped_note)

    # Send exactly once using response.send_message
    if file_obj:
        em.set_image(url=f"attachment://{file_obj.filename}")
        await interaction.response.send_message(embed=em, view=view, file=file_obj)
    else:
        await interaction.response.send_message(embed=em, view=view)
        
@bot.tree.command(name="dailynews", description="ข่าวตลาดหุ้นจริงแบบอัปเดตรายวัน (สรุปไทย)")
async def dailynews(interaction: Interaction):
    await safe_defer(interaction, ephemeral=False, thinking=False)
    items = load_today_news(DATA_DIR)
    if not items:
        try:
            items = refresh_daily_news_real(AI, MODEL_ID, DATA_DIR)
        except Exception:
            items = []
    if not items:
        await safe_followup(interaction, content="วันนี้ยังไม่มีข่าวที่ดึงมาได้ ลองใหม่ภายหลัง")
        return
    lines = [f"• **{it['headline']}** — {it['blurb']}  \n{it['source']} • {it['published'][:10]}" for it in items]
    embed = Embed(
        title="🗞️ ข่าวตลาดหุ้นวันนี้ (สรุปไทย)",
        description="\n\n".join(lines[:10]),
        color=discord.Color.dark_gold()
    )
    embed.set_footer(text=f"อัปเดตเวลา {NEWS_HH:02d}:{NEWS_MM:02d} น. (Asia/Bangkok) • ไม่แสดงซ้ำภายในวัน")
    await safe_followup(interaction, embed=embed)

@bot.tree.command(name="rebuildkb", description="สร้างดัชนีคลังความรู้ใหม่ (ผู้ดูแล)")
async def rebuildkb(interaction: Interaction):
    await safe_defer(interaction, ephemeral=True, thinking=False)
    files, chunks = rebuild_index()
    await safe_followup(interaction, content=f"สร้างดัชนีแล้ว: ไฟล์ {files} ชิ้นส่วน {chunks}", ephemeral=True)

@bot.tree.command(name="askinvest", description="ถามเรื่องตลาดหุ้น/การลงทุนจากคลังความรู้ภายใน")
@app_commands.describe(question="คำถามของคุณ (ตอบเป็นภาษาไทย)")
async def askinvest(interaction: Interaction, question: str):
    await safe_defer(interaction, ephemeral=False, thinking=False)
    result = answer_investing_question(AI, MODEL_ID, question, k=6)
    answer = result["answer"]; refs = result["refs"]; label = result.get("refs_label", "อ้างอิง")
    em = Embed(
        title="💬 ที่ปรึกษาการลงทุน (ฐานความรู้ภายใน)",
        description=answer[:4000],
        color=discord.Color.blurple()
    )
    if refs:
        ref_lines = [f"[{r['n']}] {r['title']} — {r['source']}" for r in refs[:10]]
        em.add_field(name=label, value="\n".join(ref_lines)[:1024], inline=False)
    await safe_followup(interaction, embed=em)

# ------------------- run -------------------
if __name__ == "__main__":
    if not TOKEN: raise SystemExit("Missing Bot_Token in environment.")
    if not TYPHOON_KEY: raise SystemExit("Missing Typhoon_Key in environment.")
    bot.run(TOKEN)