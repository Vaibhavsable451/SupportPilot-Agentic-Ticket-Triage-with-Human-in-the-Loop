"""AI governance: PII redaction, injection screening, human-in-the-loop policy, audit trail."""
import hashlib
import json
import re
import time
from .config import settings

PII = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}(?!\d)"),
}
INJECTION = [r"ignore (all |any )?(previous|prior|above) (instructions|rules)", r"reveal (your )?(system )?prompt",
             r"disregard (the )?(system|previous) prompt", r"you are now", r"jailbreak"]
# Actions an AI must never promise on its own: always routed to a human.
SENSITIVE = re.compile(r"\b(refund|chargeback|delete my account|lawsuit|lawyer|legal action|sue|gdpr|data breach)\b", re.I)


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    for label, rx in PII.items():
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[REDACTED_{label}]", text)
    return text, found


def screen(text: str) -> dict:
    clean, pii = redact(text)
    flags = [f"PII_{p}" for p in pii]
    blocked = any(re.search(p, text.lower()) for p in INJECTION)
    if blocked:
        flags.append("PROMPT_INJECTION")
    if SENSITIVE.search(text):
        flags.append("SENSITIVE_ACTION")
    return {"clean": clean, "flags": flags, "blocked": blocked}


def decide(s: dict) -> tuple[str, list[str]]:
    """Return ('auto_send' | 'human_review', reasons). Any single reason forces human review."""
    reasons = []
    if s.get("blocked"):
        reasons.append("Possible prompt injection in ticket")
    if "SENSITIVE_ACTION" in s.get("flags", []):
        reasons.append("Sensitive/irreversible action (refund, deletion, legal) needs a human")
    if s.get("category_conf", 1) < settings.low_conf:
        reasons.append(f"Low classifier confidence ({s.get('category_conf')})")
    if s.get("priority") == "high" and s.get("sentiment_score", 0) <= -0.6:
        reasons.append("High-priority ticket from an upset customer")
    if not s.get("blocked"):
        if not s.get("chunks"):
            reasons.append("No supporting knowledge-base content found")
        elif s.get("groundedness") is None or s["groundedness"] < settings.grounded_threshold:
            reasons.append(f"Draft not sufficiently grounded ({s.get('groundedness')})")
    return ("human_review" if reasons else "auto_send"), reasons


def audit(rec: dict) -> None:
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "ticket_sha256": hashlib.sha256(rec["text"].encode()).hexdigest(), "model": settings.groq_model,
             "category": rec.get("category"), "priority": rec.get("priority"), "decision": rec.get("decision"),
             "reasons": rec.get("reasons"), "flags": rec.get("flags"), "groundedness": rec.get("groundedness")}
    with open(settings.audit_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
