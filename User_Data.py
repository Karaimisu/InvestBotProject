# User_Data.py
import time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH = DATA_DIR / "users.xlsx"
START_MONEY = 10_000

def _retry_io(func, *args, **kwargs):
    last = None
    for _ in range(5):
        try:
            return func(*args, **kwargs)
        except PermissionError as e:
            last = e
            time.sleep(0.25)
    raise last

def _write_df(df: pd.DataFrame):
    df = df.copy()
    df["UserID"] = df["UserID"].astype(str)
    df["Username"] = df["Username"].astype(str)
    df["Money"] = df["Money"].astype(int)
    _retry_io(df.to_excel, EXCEL_PATH, index=False)

def load_or_create_excel():
    if not EXCEL_PATH.exists():
        df = pd.DataFrame(columns=["UserID", "Username", "Money"])
        _write_df(df)
    else:
        df = _retry_io(pd.read_excel, EXCEL_PATH, dtype={"UserID": str, "Username": str, "Money": "Int64"})
        if "UserID" not in df.columns:
            df = pd.DataFrame(columns=["UserID", "Username", "Money"])
        else:
            df["UserID"] = df["UserID"].astype(str)
            if "Money" in df.columns:
                df["Money"] = df["Money"].fillna(START_MONEY).astype(int)
            else:
                df["Money"] = START_MONEY
            if "Username" not in df.columns:
                df["Username"] = ""
        _write_df(df)
    return str(EXCEL_PATH)

def _read_df():
    return _retry_io(pd.read_excel, EXCEL_PATH, dtype={"UserID": str, "Username": str, "Money": int})

def get_user_money(user_id, username, filepath):
    uid = str(user_id)
    df = _read_df()
    row = df[df["UserID"] == uid]
    if row.empty:
        df.loc[len(df)] = [uid, str(username), START_MONEY]
        _write_df(df)
        return START_MONEY
    return int(row.iloc[0]["Money"])

def set_user_money(user_id, filepath, new_money):
    uid = str(user_id)
    df = _read_df()
    if (df["UserID"] == uid).any():
        df.loc[df["UserID"] == uid, "Money"] = int(new_money)
    else:
        df.loc[len(df)] = [uid, "Unknown", int(new_money)]
    _write_df(df)