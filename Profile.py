# Profile.py
from __future__ import annotations
import json, time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORE = DATA_DIR / "users.json"

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _today_str() -> str:
    return date.today().isoformat()

def _load() -> Dict[str, Any]:
    if not STORE.exists() or STORE.stat().st_size == 0:
        return {}
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save(obj: Dict[str, Any]) -> None:
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STORE)

def ensure_profile(user_id: int, username: str) -> Dict[str, Any]:
    db = _load()
    uid = str(user_id)
    if uid not in db:
        db[uid] = {
            "username": username,
            "created_at": _now_iso(),
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_pnl": 0,
            "streak": 0,
            "last_play_date": None,
            "last_actions": []  # list of dict (most recent first)
        }
        _save(db)
    else:
        # Keep latest username
        if db[uid].get("username") != username:
            db[uid]["username"] = username
            _save(db)
    return db[uid]

def get_profile(user_id: int, username: str) -> Dict[str, Any]:
    ensure_profile(user_id, username)
    return _load()[str(user_id)]

def _update_streak(profile: Dict[str, Any]) -> None:
    today = _today_str()
    last = profile.get("last_play_date")
    if last == today:
        # already counted today; do not change
        return
    # simple daily streak: +1 if consecutive day, else reset to 1
    if last:
        try:
            # if yesterday
            y = (date.today()).fromisoformat(today).toordinal()
            l = (date.today()).fromisoformat(last).toordinal()
            if y - l == 1:
                profile["streak"] = int(profile.get("streak", 0)) + 1
            else:
                profile["streak"] = 1
        except Exception:
            profile["streak"] = 1
    else:
        profile["streak"] = 1
    profile["last_play_date"] = today

def record_trade(
    user_id: int,
    username: str,
    *,
    action: str,
    pnl: int,
    size: int,
    correct_action: str,
    company: str,
    sector: str,
    tone: str,
    difficulty: int
) -> Dict[str, Any]:
    """
    Update counters and append action history.
    Returns the updated profile dict.
    """
    db = _load()
    uid = str(user_id)
    prof = db.get(uid) or ensure_profile(user_id, username)
    prof = db[str(user_id)]  # re-read to be safe

    prof["trades"] = int(prof.get("trades", 0)) + 1
    prof["total_pnl"] = int(prof.get("total_pnl", 0)) + int(pnl)

    if pnl > 0:
        prof["wins"] = int(prof.get("wins", 0)) + 1
    elif pnl < 0:
        prof["losses"] = int(prof.get("losses", 0)) + 1
    else:
        prof["ties"] = int(prof.get("ties", 0)) + 1

    # daily streak progression
    _update_streak(prof)

    entry = {
        "ts": _now_iso(),
        "action": action,
        "pnl": int(pnl),
        "size": int(size),
        "correct": (action == correct_action),
        "company": company,
        "sector": sector,
        "tone": tone,
        "difficulty": int(difficulty)
    }
    hist: List[Dict[str, Any]] = list(prof.get("last_actions") or [])
    hist.insert(0, entry)
    prof["last_actions"] = hist[:20]  # keep last 20

    db[uid] = prof
    _save(db)
    return prof

def summarize_profile(prof: Dict[str, Any]) -> Dict[str, Any]:
    trades = int(prof.get("trades", 0))
    wins = int(prof.get("wins", 0))
    losses = int(prof.get("losses", 0))
    ties = int(prof.get("ties", 0))
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0
    total_pnl = int(prof.get("total_pnl", 0))
    streak = int(prof.get("streak", 0))
    created_at = prof.get("created_at")

    recent = []
    for it in (prof.get("last_actions") or [])[:5]:
        sgn = "+" if it["pnl"] >= 0 else "-"
        pnl_txt = f"{sgn}${abs(int(it['pnl'])):,}"
        recent.append(
            f"[{it['ts']}] {it['company']} · {it['action']} · {pnl_txt} · diff {it['difficulty']}"
        )

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "winrate": round(winrate, 1),
        "total_pnl": total_pnl,
        "streak": streak,
        "created_at": created_at,
        "recent": recent
    }
