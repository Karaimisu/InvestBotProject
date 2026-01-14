# GenGraph.py
import os
import numpy as np
from datetime import datetime, timedelta
import random, string, time
import matplotlib
matplotlib.use("Agg")  # headless, low RAM
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "graphs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

seed = int(time.time() * 1000) % 2**32
np.random.seed(seed)
random.seed(seed)

PREFIXES = ["Nova","Luna","Apex","Vertex","Quantum","Neo","Omega","Atlas","Vera","Zenix","Orion","Titan","Horizon","Helix","Pioneer","Ardent","Skyline","Solaris","Aurora","Equinox","Momentum","Eclipse","Polaris","Zenith","Fusion","Summit","Ignite","Evolve","Nimbus","Cascade","Cobalt","Falcon","Crimson","Obsidian","Sierra","Catalyst","Infinite","Vortex","Sable"]
SUFFIXES = ["Corp","Holdings","Industries","Systems","Group","Enterprises","Labs","Capital","Technologies","Partners","Resources","Networks","Logistics","Dynamics","Solutions","Power","Media","Energy","Pharma","Finance"]
SECTORS  = ["Energy","Tech","Pharma","Finance","Aerospace","Retail","Automotive","Materials","Telecom","Biotech","Crypto","AI","Food","Construction","Defense"]

def random_stock_name():
    name = f"{random.choice(PREFIXES)} {random.choice(SUFFIXES)}"
    symbol = ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 4)))
    sector = random.choice(SECTORS)
    return f"{name} ({symbol})", sector

def trading_days(days=252):
    start_date = datetime.utcnow().date() - timedelta(days=int(days * 1.5))
    dates = []
    cur = np.datetime64(start_date)
    while len(dates) < days:
        weekday = (cur.astype('datetime64[D]').astype(int) + 4) % 7  # 0=Mon
        if weekday < 5:
            dates.append(cur)
        cur = cur + np.timedelta64(1, 'D')
    return np.array(dates[-days:], dtype='datetime64[D]')

def simulate_stock(start_price=100.0, days=252, mu=0.10, sigma=0.25,
                   jump_prob=0.02, jump_mu=-0.02, jump_sigma=0.08):
    dt = 1 / 252
    mu_d = mu * dt
    sigma_d = sigma * np.sqrt(dt)

    # Vectorized random components
    shocks = np.random.randn(days) * sigma_d
    jumps = np.where(
        np.random.rand(days) < jump_prob,
        np.random.normal(jump_mu, jump_sigma, days),
        0.0
    )

    # Pre-compute base log returns (without momentum)
    base_log_r = (mu_d - 0.5 * sigma_d**2) + shocks + jumps

    # Momentum requires sequential computation, but we can still optimize
    prices = np.empty(days, dtype=np.float32)
    prices[0] = start_price
    last_ret = 0.0
    for t in range(1, days):
        momentum = 0.25 * last_ret
        if np.random.rand() < 0.05:
            momentum *= -1
        log_r = base_log_r[t] + momentum
        prices[t] = prices[t-1] * np.exp(log_r)
        last_ret = log_r

    return prices

def generate_stock_graph(days=252):
    company_name, sector = random_stock_name()
    start_price = float(np.random.uniform(20, 500))
    mu = float(np.random.uniform(0.05, 0.15))
    sigma = float(np.random.uniform(0.15, 0.35))
    jump_prob = float(np.random.uniform(0.01, 0.04))
    jump_mu = float(np.random.uniform(-0.04, 0.01))
    jump_sigma = float(np.random.uniform(0.05, 0.1))

    prices = simulate_stock(start_price, days, mu, sigma, jump_prob, jump_mu, jump_sigma).astype(np.float32)
    dates = trading_days(days)

    # lightweight rolling means on last window only for speed
    s = prices
    sma20 = np.convolve(s, np.ones(20)/20, mode="same")
    sma50 = np.convolve(s, np.ones(50)/50, mode="same")

    base = DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    base_name = company_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    csv_path = str(base / f"{base_name}.csv")
    png_path = str(base / f"{base_name}.png")

    # write CSV (Date,Close) only to keep file tiny
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Date,Close\n")
        for d, p in zip(dates, prices):
            f.write(f"{np.datetime_as_string(d, unit='D')},{p:.4f}\n")

    # plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    ax.plot(dates, prices, linewidth=1.6, label="Close")
    ax.plot(dates, sma20, linestyle="--", linewidth=1.0, label="SMA20")
    ax.plot(dates, sma50, linestyle="--", linewidth=1.0, label="SMA50")
    ax.set_title(f"{company_name} — {sector}")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)  # release RAM

    return {"company": company_name, "sector": sector, "csv_path": csv_path, "png_path": png_path}

def trend_from_csv(csv_path: str) -> str:
    try:
        # read last ~200 rows only
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-201:]  # header + 200
        if len(lines) < 3:
            return "sideways"
        closes = []
        for ln in lines[1:]:
            parts = ln.strip().split(",")
            if len(parts) >= 2:
                try:
                    closes.append(float(parts[1]))
                except:
                    pass
        arr = np.array(closes, dtype=np.float32)
        if arr.size < 10:
            return "sideways"
        x = np.arange(arr.size, dtype=np.float32)
        xm = x.mean(); ym = arr.mean()
        denom = np.square(x - xm).sum() or 1.0
        slope = ((x - xm) * (arr - ym)).sum() / denom
        sma_fast = np.convolve(arr, np.ones(10)/10, mode="valid")[-1]
        sma_slow = np.convolve(arr, np.ones(30)/30, mode="valid")[-1] if arr.size >= 30 else arr[-1]
        if slope > 0 and sma_fast >= sma_slow:
            return "uptrend"
        if slope < 0 and sma_fast <= sma_slow:
            return "downtrend"
        return "sideways"
    except Exception:
        return "sideways"