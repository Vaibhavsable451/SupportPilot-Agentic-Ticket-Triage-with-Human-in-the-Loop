"""SQLite: tickets, decisions, human-review outcomes, analytics."""
import sqlite3
import time
import pandas as pd
from .config import settings

COLS = ["ts", "text", "category", "category_conf", "priority", "priority_conf", "sentiment", "sentiment_score",
        "draft", "final_reply", "groundedness", "decision", "reasons", "status", "flags", "sources",
        "latency_ms", "tokens_in", "tokens_out", "reviewer_note"]
REAL = {"category_conf", "priority_conf", "sentiment_score", "groundedness", "latency_ms", "tokens_in", "tokens_out"}


def _conn():
    c = sqlite3.connect(settings.db_path)
    c.execute("CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY AUTOINCREMENT, " +
              ", ".join(f"{k} {'REAL' if k in REAL else 'TEXT'}" for k in COLS) + ")")
    return c


def insert(row: dict) -> int:
    row = {**row, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with _conn() as c:
        cur = c.execute(f"INSERT INTO tickets ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                        [row.get(k) for k in COLS])
        return cur.lastrowid


def fetch(status: str | None = None, limit: int = 200) -> pd.DataFrame:
    q, args = "SELECT * FROM tickets WHERE 1=1", []
    if status:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return pd.read_sql(q, c, params=args)


def review(ticket_id: int, status: str, final_reply: str | None, note: str | None) -> bool:
    with _conn() as c:
        cur = c.execute("UPDATE tickets SET status=?, final_reply=?, reviewer_note=? WHERE id=? AND status='pending'",
                        (status, final_reply, note, ticket_id))
        return cur.rowcount > 0


def _m(series, nd=3):
    v = series.dropna()
    return round(float(v.mean()), nd) if len(v) else None


def summary() -> dict:
    df = fetch(limit=5000)
    if df.empty:
        return {"tickets": 0}
    n = len(df)
    auto = int((df.decision == "auto_send").sum())
    return {"tickets": n, "auto_send": auto, "human_review": n - auto, "auto_rate": round(auto / n, 3),
            "pending": int((df.status == "pending").sum()), "approved": int((df.status == "approved").sum()),
            "rejected": int((df.status == "rejected").sum()),
            "avg_groundedness": _m(df.groundedness), "avg_latency_ms": _m(df.latency_ms, 1),
            "tokens_in": int(df.tokens_in.fillna(0).sum()), "tokens_out": int(df.tokens_out.fillna(0).sum()),
            "by_category": df.category.value_counts().to_dict(), "by_priority": df.priority.value_counts().to_dict(),
            "by_sentiment": df.sentiment.value_counts().to_dict()}
