# User_Data.py
# Lightweight CSV store with basic stats

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "users.csv"

HEADER = [
    "user_id",
    "username",
    "money",
    "rounds",
    "total_gain",
    "total_loss",
    "wins",
    "losses",
]

def load_or_create_store() -> str:
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
    return str(CSV_PATH)

def _read_all() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    uid = int(row.get("user_id", "0"))
                except ValueError:
                    continue
                def _iv(key, default=0):
                    try:
                        return int(row.get(key, default))
                    except (TypeError, ValueError):
                        return default
                rows[uid] = {
                    "username": row.get("username", "") or "",
                    "money": _iv("money", 0),
                    "rounds": _iv("rounds", 0),
                    "total_gain": _iv("total_gain", 0),
                    "total_loss": _iv("total_loss", 0),
                    "wins": _iv("wins", 0),
                    "losses": _iv("losses", 0),
                }
    return rows

def _write_all(rows: dict[int, dict]):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for uid, data in rows.items():
            w.writerow([
                uid,
                data.get("username", ""),
                int(data.get("money", 0)),
                int(data.get("rounds", 0)),
                int(data.get("total_gain", 0)),
                int(data.get("total_loss", 0)),
                int(data.get("wins", 0)),
                int(data.get("losses", 0)),
            ])

def _ensure_row(rows: dict[int, dict], user_id: int, username: str) -> dict:
    if user_id not in rows:
        rows[user_id] = {
            "username": username,
            "money": 10000,
            "rounds": 0,
            "total_gain": 0,
            "total_loss": 0,
            "wins": 0,
            "losses": 0,
        }
    else:
        if username and rows[user_id].get("username") != username:
            rows[user_id]["username"] = username
    return rows[user_id]

def get_user_money(user_id: int, username: str, _store_path: str) -> int:
    rows = _read_all()
    row = _ensure_row(rows, user_id, username)
    _write_all(rows)
    return int(row.get("money", 0))

def set_user_money(user_id: int, _store_path: str, money: int):
    rows = _read_all()
    row = _ensure_row(rows, user_id, row_username := rows.get(user_id, {}).get("username", ""))
    # if user doesn't exist yet, row_username will be "", but that's ok
    row["money"] = int(money)
    rows[user_id] = row
    _write_all(rows)

def update_stats(user_id: int, username: str, pnl: int, _store_path: str):
    rows = _read_all()
    row = _ensure_row(rows, user_id, username)
    row["rounds"] = int(row.get("rounds", 0)) + 1
    if pnl > 0:
        row["total_gain"] = int(row.get("total_gain", 0)) + pnl
        row["wins"] = int(row.get("wins", 0)) + 1
    elif pnl < 0:
        row["total_loss"] = int(row.get("total_loss", 0)) + abs(pnl)
        row["losses"] = int(row.get("losses", 0)) + 1
    rows[user_id] = row
    _write_all(rows)

def get_user_stats(user_id: int, username: str, _store_path: str) -> dict:
    # Ensure user exists
    _ = get_user_money(user_id, username, _store_path)
    rows = _read_all()
    row = rows.get(user_id)
    if not row:
        return {
            "username": username,
            "money": 10000,
            "rounds": 0,
            "total_gain": 0,
            "total_loss": 0,
            "wins": 0,
            "losses": 0,
        }
    return row

def get_all_users(_store_path: str) -> dict[int, dict]:
    return _read_all()