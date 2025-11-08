# main.py
from dotenv import load_dotenv
import os, json, asyncio
from pathlib import Path
import pandas as pd
import discord
from discord.ext import commands
from discord import Interaction, File, Embed, app_commands
from discord.errors import NotFound, HTTPException
from openai import OpenAI

from User_Data import load_or_create_excel, get_user_money, set_user_money
from Invest_Logic import (
    generate_scenario, generate_news, simulate_outcome,
    compute_trade_size, Scenario, balanced_difficulty, generate_tip_and_reason
)
from GenGraph import generate_stock_graph
from Model_Util import resolve_model
from Daily_News import generate_daily_news
from Market_Analysis import analyze_market_only
from RAG_Chat import answer_investing_question
from RAG_Store import rebuild_index

load_dotenv()

TOKEN = os.getenv("Bot_Token")
TYPHOON_KEY = os.getenv("Typhoon_Key")
TYPHOON_MODEL_ENV = os.getenv("Typhoon_Model")

AI = OpenAI(api_key=TYPHOON_KEY, base_url="https://api.opentyphoon.ai/v1")
MODEL_ID = resolve_model(AI, TYPHOON_MODEL_ENV)
excel_path = load_or_create_excel()

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LAST_COMPANY_PATH = DATA_DIR / "last_company.json"
LAST_NEWS_PATH = DATA_DIR / "last_daily_news.json"

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="=", intents=intents)

# --- safe defer/send helpers to avoid Unknown interaction 10062 ---
async def safe_defer(interaction: Interaction, *, ephemeral: bool = True, thinking: bool = True):
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=ephemeral, thinking=thinking)
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

# --- chart trend detector ---
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

# --- interactive view ---
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

    async def _ensure_deferred(self, interaction: Interaction, ephemeral: bool = True):
        await safe_defer(interaction, ephemeral=ephemeral, thinking=True)

    def _lock_buttons(self):
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True

    async def _handle_choice(self, interaction: Interaction, choice: str):
        if interaction.user.id != self.user_id:
            await safe_followup(interaction, content="This session belongs to another user.", ephemeral=True)
            return

        await self._ensure_deferred(interaction, ephemeral=True)

        if choice == "Read News":
            news = generate_news(self.ai, self.scenario, model_id=self.model_id)
            items = news["items"]
            brief = news["brief"]
            desc_items = "\n".join([f"• **{it.get('headline','')}** — {it.get('blurb','')}" for it in items]) or "ไม่มีข่าว"
            pros = "\n".join(f"• {p}" for p in brief.get("pros", [])[:5]) or "—"
            cons = "\n".join(f"• {c}" for c in brief.get("cons", [])[:5]) or "—"
            sigs = "\n".join(f"• {s}" for s in brief.get("signals", [])[:5]) or "—"
            embed = Embed(title="📰 ข่าวเพื่อช่วยตัดสินใจ", description=desc_items, color=discord.Color.dark_teal())
            embed.add_field(name="สัญญาณสนับสนุน (Pros)", value=pros, inline=False)
            embed.add_field(name="สัญญาณต้าน (Cons)", value=cons, inline=False)
            embed.add_field(name="ตัวชี้วัด/เบาะแส", value=sigs, inline=False)
            embed.set_footer(text=f"อคติรวมของข่าว: {brief.get('bias','mixed')}")
            await safe_followup(interaction, embed=embed, ephemeral=True)
            return

        if self.round_finished:
            await safe_followup(interaction, content="คุณได้เลือกไปแล้ว โปรดเริ่มรอบใหม่ด้วย /invest", ephemeral=True)
            return

        new_money, pnl, size, _ = simulate_outcome(
            choice, self.scenario, self.money, requested_amount=self.requested_amount
        )
        set_user_money(self.user_id, self.excel_path, new_money)
        self.money = new_money
        self.round_finished = True
        self._lock_buttons()
        try:
            if interaction.message:
                await interaction.message.edit(view=self)
        except (NotFound, HTTPException):
            pass

        tip_reason = generate_tip_and_reason(self.ai, self.scenario, choice, pnl, size, self.trend, model_id=self.model_id)

        sign = "+" if pnl >= 0 else "-"
        pnl_str = f"{sign}${abs(pnl):,}"
        desc = [f"คำตอบที่ถูกต้องคือ: **{self.scenario.correct_action}**", tip_reason]
        if self.clamped_note:
            desc.append(self.clamped_note)

        embed = Embed(
            title=f"📊 ผลลัพธ์: {choice}",
            description="\n\n".join(desc),
            color=discord.Color.green() if pnl >= 0 else discord.Color.red()
        )
        embed.add_field(name="P&L", value=pnl_str, inline=True)
        embed.add_field(name="ขนาดสถานะ", value=f"${size:,}", inline=True)
        embed.add_field(name="ยอดเงินใหม่", value=f"${new_money:,}", inline=True)

        await safe_followup(interaction, embed=embed, ephemeral=True)

    @discord.ui.button(label="Buy", style=discord.ButtonStyle.primary)
    async def buy(self, interaction: Interaction, button: discord.ui.Button):
        await self._handle_choice(interaction, "Buy")

    @discord.ui.button(label="Sell", style=discord.ButtonStyle.primary)
    async def sell(self, interaction: Interaction, button: discord.ui.Button):
        await self._handle_choice(interaction, "Sell")

    @discord.ui.button(label="Hold", style=discord.ButtonStyle.secondary)
    async def hold(self, interaction: Interaction, button: discord.ui.Button):
        await self._handle_choice(interaction, "Hold")

    @discord.ui.button(label="Read News", style=discord.ButtonStyle.success)
    async def news(self, interaction: Interaction, button: discord.ui.Button):
        await self._handle_choice(interaction, "Read News")

@bot.event
async def on_ready():
    await bot.tree.sync()
    print("------- Bot Started -------")

@bot.tree.command(name="invest", description="เริ่มสถานการณ์การลงทุน (ต้องระบุจำนวนเงินลงทุน)")
@app_commands.describe(amount="จำนวนเงินที่ต้องการลงทุน (พิมพ์หรือเลือกจากรายการ)")
async def invest(interaction: Interaction, amount: int):
    await safe_defer(interaction, ephemeral=False, thinking=True)

    user_id = interaction.user.id
    username = interaction.user.name
    money = get_user_money(user_id, username, excel_path)

    graph = generate_stock_graph()
    company, sector = graph["company"], graph["sector"]
    png_path = graph["png_path"]
    csv_path = graph["csv_path"]
    trend = _trend_from_csv(csv_path)

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

    if os.path.exists(png_path):
        file = File(png_path, filename=os.path.basename(png_path))
        em.set_image(url=f"attachment://{os.path.basename(png_path)}")
        await safe_followup(interaction, embed=em, view=view, file=file)
    else:
        await safe_followup(interaction, embed=em, view=view)

@invest.autocomplete("amount")
async def amount_autocomplete(interaction: Interaction, current: str):
    try:
        money = get_user_money(interaction.user.id, interaction.user.name, excel_path)
        diff_guess = balanced_difficulty(money)
        cap_guess = compute_trade_size(10**9, money, diff_guess)
        return [
            app_commands.Choice(name=f"สูงสุด ~ ${cap_guess:,}", value=cap_guess),
            app_commands.Choice(name=f"ครึ่งหนึ่ง ~ ${cap_guess//2:,}", value=max(100, cap_guess//2)),
            app_commands.Choice(name="ขั้นต่ำ $100", value=100),
        ]
    except Exception:
        return [app_commands.Choice(name="ขั้นต่ำ $100", value=100)]

@bot.tree.command(name="dailynews", description="ข่าวประจำวันแนวหนังสือพิมพ์ พร้อมพาดพิงบริษัทสมมติล่าสุด")
async def dailynews(interaction: Interaction):
    await safe_defer(interaction, ephemeral=False, thinking=True)

    company_info = None
    try:
        if LAST_COMPANY_PATH.exists():
            with open(LAST_COMPANY_PATH, "r", encoding="utf-8") as f:
                company_info = json.load(f)
    except Exception:
        company_info = None

    items = generate_daily_news(AI, MODEL_ID, company=company_info)
    if not items:
        await safe_followup(interaction, content="ยังสร้างข่าวไม่สำเร็จ ลองอีกครั้ง")
        return

    try:
        with open(LAST_NEWS_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
    except Exception:
        pass

    lines = [f"• **[{it.get('category','')}] {it.get('headline','')}** — {it.get('blurb','')}" for it in items]
    embed = Embed(
        title="🗞️ ข่าวประจำวัน",
        description="\n".join(lines),
        color=discord.Color.dark_gold()
    )
    if company_info and company_info.get("name"):
        embed.set_footer(text=f"รวมข่าวเกี่ยวกับ: {company_info['name']} ({company_info.get('sector','')})")
    await safe_followup(interaction, embed=embed)

@bot.tree.command(name="marketanalysis", description="วิเคราะห์เฉพาะประเด็นตลาดหุ้นจากข่าวประจำวัน")
async def marketanalysis(interaction: Interaction):
    await safe_defer(interaction, ephemeral=False, thinking=True)

    company_info = None
    try:
        if LAST_COMPANY_PATH.exists():
            with open(LAST_COMPANY_PATH, "r", encoding="utf-8") as f:
                company_info = json.load(f)
    except Exception:
        company_info = None

    news_items = None
    try:
        if LAST_NEWS_PATH.exists():
            with open(LAST_NEWS_PATH, "r", encoding="utf-8") as f:
                news_items = json.load(f)
    except Exception:
        news_items = None

    if not news_items:
        news_items = generate_daily_news(AI, MODEL_ID, company=company_info)
        try:
            with open(LAST_NEWS_PATH, "w", encoding="utf-8") as f:
                json.dump(news_items, f, ensure_ascii=False)
        except Exception:
            pass

    if not news_items:
        await safe_followup(interaction, content="ไม่มีข่าวให้วิเคราะห์ ลองรัน /dailynews ก่อน")
        return

    analysis = analyze_market_only(AI, MODEL_ID, news_items, company_info)
    sent = analysis.get("sentiment", "mixed").lower()
    color = discord.Color.gold()
    if sent == "bullish":
        color = discord.Color.green()
    elif sent == "bearish":
        color = discord.Color.red()

    em = Embed(
        title="📊 วิเคราะห์ตลาดหุ้นจากข่าววันนี้",
        description=analysis.get("summary", "") or "ไม่มีสรุป",
        color=color
    )
    if analysis.get("themes"):
        em.add_field(name="ธีมเด่น", value="• " + "\n• ".join(analysis["themes"]), inline=False)
    if analysis.get("sectors"):
        em.add_field(name="กลุ่มอุตสาหกรรม", value="• " + "\n• ".join(analysis["sectors"]), inline=False)
    if analysis.get("risks"):
        em.add_field(name="ความเสี่ยง", value="• " + "\n• ".join(analysis["risks"]), inline=False)
    if analysis.get("opportunities"):
        em.add_field(name="โอกาส", value="• " + "\n• ".join(analysis["opportunities"]), inline=False)
    if analysis.get("watchlist"):
        em.add_field(name="Watchlist", value="• " + "\n• ".join(analysis["watchlist"]), inline=False)

    if company_info and company_info.get("name"):
        em.set_footer(text=f"พาดพิงบริษัท: {company_info['name']} ({company_info.get('sector','')})")

    await safe_followup(interaction, embed=em)

@bot.tree.command(name="rebuildkb", description="สร้างดัชนีคลังความรู้ใหม่ (ผู้ดูแล)")
async def rebuildkb(interaction: Interaction):
    await safe_defer(interaction, ephemeral=True, thinking=True)
    files, chunks = rebuild_index()
    await safe_followup(interaction, content=f"สร้างดัชนีแล้ว: ไฟล์ {files} ชิ้นส่วน {chunks}", ephemeral=True)

@bot.tree.command(name="askinvest", description="ถามเรื่องตลาดหุ้น/การลงทุนจากคลังความรู้ภายใน")
@app_commands.describe(question="คำถามของคุณ")
async def askinvest(interaction: Interaction, question: str):
    await safe_defer(interaction, ephemeral=False, thinking=True)
    result = answer_investing_question(AI, MODEL_ID, question, k=6)
    answer = result["answer"]
    refs = result["refs"]
    label = result.get("refs_label", "References")

    em = Embed(
        title="💬 ที่ปรึกษาการลงทุน (ฐานความรู้ภายใน)",
        description=answer[:4000],
        color=discord.Color.blurple()
    )
    if refs:
        ref_lines = [f"[{r['n']}] {r['title']} — {r['source']}" for r in refs[:10]]
        em.add_field(name=label, value="\n".join(ref_lines)[:1024], inline=False)

    await safe_followup(interaction, embed=em)

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Missing Bot_Token in environment.")
    if not TYPHOON_KEY:
        raise SystemExit("Missing Typhoon_Key in environment.")
    bot.run(TOKEN)
