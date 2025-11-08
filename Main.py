# main.py
from dotenv import load_dotenv
import os
import discord
from discord.ext import commands
from discord import Interaction, File, Embed
from openai import OpenAI

from User_Data import load_or_create_excel, get_user_money, set_user_money
from Invest_Logic import generate_scenario, generate_news, simulate_outcome, generate_feedback
from GenGraph import generate_stock_graph
from Model_Util import resolve_model

load_dotenv()

TOKEN = os.getenv("Bot_Token")
TYPHOON_KEY = os.getenv("Typhoon_Key")
TYPHOON_MODEL_ENV = os.getenv("Typhoon_Model")

AI = OpenAI(api_key=TYPHOON_KEY, base_url="https://api.opentyphoon.ai/v1")
MODEL_ID = resolve_model(AI, TYPHOON_MODEL_ENV)
excel_path = load_or_create_excel()

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="=", intents=intents)

class InvestView(discord.ui.View):
    def __init__(self, ai_client, model_id: str, user_id: int, username: str, money: int, scenario, excel_path: str):
        super().__init__(timeout=300)
        self.ai = ai_client
        self.model_id = model_id
        self.user_id = user_id
        self.username = username
        self.money = money
        self.scenario = scenario
        self.excel_path = excel_path

    async def _ensure_deferred(self, interaction: Interaction, ephemeral: bool = True):
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=ephemeral, thinking=True)

    async def _handle_choice(self, interaction: Interaction, choice: str):
        if interaction.user.id != self.user_id:
            if not interaction.response.is_done():
                await interaction.response.send_message("This session belongs to another user.", ephemeral=True)
            else:
                await interaction.followup.send("This session belongs to another user.", ephemeral=True)
            return

        await self._ensure_deferred(interaction, ephemeral=True)

        if choice == "Read News":
            items = generate_news(self.ai, self.scenario, model_id=self.model_id)
            desc = "\n".join([f"• **{it.get('headline','')}** — {it.get('blurb','')}" for it in items]) or "ไม่มีข่าวที่สร้างขึ้น"
            embed = Embed(title="ข่าวสถานการณ์", description=desc, color=discord.Color.dark_teal())
            await interaction.followup.send(embed=embed, ephemeral=True)
            return

        new_money, pnl, size, _ = simulate_outcome(choice, self.scenario, self.money)
        fb = generate_feedback(self.ai, self.scenario, choice, pnl, size, model_id=self.model_id)
        set_user_money(self.user_id, self.excel_path, new_money)
        self.money = new_money

        sign = "+" if pnl >= 0 else "-"
        pnl_str = f"{sign}${abs(pnl):,}"
        embed = Embed(
            title=f"ผลลัพธ์: {choice}",
            description=fb,
            color=discord.Color.green() if pnl >= 0 else discord.Color.red()
        )
        embed.add_field(name="P&L", value=pnl_str, inline=True)
        embed.add_field(name="ขนาดสถานะ", value=f"${size:,}", inline=True)
        embed.add_field(name="ยอดเงินใหม่", value=f"${new_money:,}", inline=True)
        await interaction.followup.send(embed=embed, ephemeral=True)

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
    try:
        _ = AI.models.retrieve(MODEL_ID)
        print(f"[Typhoon] model OK: {MODEL_ID}")
    except Exception as e:
        print(f"[Typhoon] model check failed for '{MODEL_ID}': {e}")
    await bot.tree.sync()
    print("-------Started-------")

@bot.tree.command(name="invest", description="เริ่มสถานการณ์การลงทุน")
async def invest(interaction: Interaction):
    await interaction.response.defer(thinking=True)

    user_id = interaction.user.id
    username = interaction.user.name
    money = get_user_money(user_id, username, excel_path)

    graph = generate_stock_graph()
    company, sector = graph["company"], graph["sector"]
    png_path = graph["png_path"]

    scenario = generate_scenario(AI, company, sector, money, model_id=MODEL_ID)

    em = Embed(
        title=f"📈 {scenario.name} — {scenario.sector}",
        description=scenario.summary,
        color=discord.Color.blue()
    )
    em.add_field(name="ความยาก", value=scenario.stars, inline=True)
    em.add_field(name="โทนตลาด", value=scenario.tone.capitalize(), inline=True)
    em.add_field(name="ยอดเงิน", value=f"${money:,}", inline=True)

    view = InvestView(AI, MODEL_ID, user_id, username, money, scenario, excel_path)

    if os.path.exists(png_path):
        file = File(png_path, filename=os.path.basename(png_path))
        em.set_image(url=f"attachment://{os.path.basename(png_path)}")
        await interaction.followup.send(embed=em, view=view, file=file)
    else:
        await interaction.followup.send(embed=em, view=view)

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Missing Bot_Token in environment.")
    if not TYPHOON_KEY:
        raise SystemExit("Missing Typhoon_Key in environment.")
    bot.run(TOKEN)