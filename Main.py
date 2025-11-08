from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import random, string, time
from openai import OpenAI

import discord
from discord.ui import View, Select
from discord import SelectOption
from discord.ext import commands
from discord import app_commands

import GenGraph

load_dotenv()

token = os.getenv("Bot_Token")
api = os.getenv("Typhoon_Key")
filepath = os.getenv("Excel_File")
PF = "="

AIclient = OpenAI(
    api_key=api,
    base_url="https://api.opentyphoon.ai/v1"
)

intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PF, intents=intents)
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

@bot.event
async def on_ready():
    await bot.tree.sync()
    print("-------Started-------")