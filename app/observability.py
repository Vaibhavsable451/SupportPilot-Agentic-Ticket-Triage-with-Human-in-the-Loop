"""Spans + JSONL traces."""
import json
import time
from contextlib import contextmanager
from .config import settings


@contextmanager
def span(trace: list, name: str):
    rec, t0 = {"span": name}, time.perf_counter()
    try:
        yield rec
    finally:
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        trace.append(rec)


def write_trace(ticket_id: int, decision: str, trace: list, usage: dict) -> None:
    with open(settings.trace_file, "a") as f:
        f.write(json.dumps({"ticket_id": ticket_id, "decision": decision, "total_ms": round(sum(s["ms"] for s in trace), 1),
                            "usage": usage, "trace": trace}) + "\n")
