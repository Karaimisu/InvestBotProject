# GenGraph.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random, string, time

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = REPO_ROOT / "data" / "graphs"

seed = int(time.time() * 1000) % 2**32
np.random.seed(seed)
random.seed(seed)

def random_stock_name():
    prefixes = [
        "Nova","Luna","Apex","Vertex","Quantum","Neo","Omega","Atlas","Vera","Zenix",
        "Orion","Titan","Horizon","Helix","Pioneer","Ardent","Skyline","Solaris","Aurora","Equinox",
        "Momentum","Vertex","Eclipse","Polaris","Zenith","Fusion","Summit","Ignite","Evolve","Nimbus",
        "Cascade","Cobalt","Falcon","Crimson","Obsidian","Sierra","Catalyst","Infinite","Vortex","Sable"
    ]
    suffixes = [
        "Corp","Holdings","Industries","Systems","Group","Enterprises","Labs","Capital","Technologies",
        "Partners","Resources","Networks","Logistics","Dynamics","Solutions","Power","Media","Energy","Pharma","Finance"
    ]
    sectors = [
        "Energy","Tech","Pharma","Finance","Aerospace","Retail","Automotive","Materials",
        "Telecom","Biotech","Crypto","AI","Food","Construction","Defense"
    ]
    name = f"{random.choice(prefixes)} {random.choice(suffixes)}"
    symbol = ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 4)))
    sector = random.choice(sectors)
    return f"{name} ({symbol})", sector

def trading_days(start_date=None, days=252):
    if start_date is None:
        start_date = datetime.now().date() - timedelta(days=int(days * 1.5))
    dates = []
    cur = pd.Timestamp(start_date)
    while len(dates) < days:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += pd.Timedelta(days=1)
    return pd.DatetimeIndex(dates[-days:])

def simulate_stock(start_price=100.0, days=252, mu=0.10, sigma=0.25,
                   jump_prob=0.02, jump_mu=-0.02, jump_sigma=0.08,
                   vol_clustering=True, momentum_strength=0.3,
                   mean_reversion_prob=0.05):
    dt = 1 / 252
    mu_d = mu * dt
    sigma_d = sigma * np.sqrt(dt)
    prices = np.empty(days)
    prices[0] = start_price
    recent_var = sigma_d ** 2
    alpha = 0.05
    last_return = 0.0
    for t in range(1, days):
        sigma_t = np.sqrt(recent_var) * (1 + np.random.randn() * 0.1) if vol_clustering else sigma_d
        sigma_t = max(1e-4, sigma_t)
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * t / 252)
        shock = np.random.randn() * sigma_t
        jump = np.random.normal(loc=jump_mu, scale=jump_sigma) if np.random.rand() < jump_prob else 0.0
        momentum = momentum_strength * last_return
        if np.random.rand() < mean_reversion_prob:
            momentum *= -1
        log_return = (mu_d * seasonal_factor - 0.5 * sigma_t**2) + shock + jump + momentum
        prices[t] = prices[t-1] * np.exp(log_return)
        recent_var = (1 - alpha) * recent_var + alpha * shock**2
        last_return = log_return
    return prices

def simulate_volume(days, avg_volume=2e6):
    base = np.random.lognormal(mean=np.log(avg_volume), sigma=0.2, size=days)
    noise = np.random.normal(1.0, 0.1, size=days)
    volume = base * noise
    spikes = int(days * 0.03)
    if spikes > 0:
        for i in np.random.choice(range(days), size=spikes, replace=False):
            volume[i] *= np.random.uniform(1.5, 4.0)
    return volume.round().astype(int)

def generate_stock_graph(out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    company_name, sector = random_stock_name()
    days = 252
    start_price = random.uniform(20, 500)
    mu = random.uniform(0.05, 0.15)
    sigma = random.uniform(0.15, 0.35)
    jump_prob = random.uniform(0.01, 0.04)
    jump_mu = random.uniform(-0.04, 0.01)
    jump_sigma = random.uniform(0.05, 0.1)

    prices = simulate_stock(start_price, days, mu, sigma, jump_prob, jump_mu, jump_sigma)
    dates = trading_days(days=days)
    df = pd.DataFrame({"Date": dates, "Close": prices}).set_index("Date")
    df["Return"] = df["Close"].pct_change()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["Vol20"] = df["Return"].rolling(20).std() * np.sqrt(252)
    df["Volume"] = simulate_volume(days)
    df["Sector"] = sector

    filename_base = (
        company_name.replace(" ", "_")
        .replace("(", "").replace(")", "").replace("/", "")
    )
    csv_path = out_dir / f"{filename_base}.csv"
    png_path = out_dir / f"{filename_base}.png"

    df.to_csv(csv_path)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close", linewidth=1.8)
    plt.plot(df.index, df["SMA20"], label="SMA20", linestyle="--", linewidth=1.2)
    plt.plot(df.index, df["SMA50"], label="SMA50", linestyle="--", linewidth=1.2)
    plt.title(f"{company_name} — {sector} Sector")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    return {
        "company": company_name,
        "sector": sector,
        "csv_path": str(csv_path),
        "png_path": str(png_path),
    }