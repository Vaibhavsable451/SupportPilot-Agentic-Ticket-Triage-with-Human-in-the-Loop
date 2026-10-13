"""Hallucination check: share of factual claims in the draft entailed by KB/order data.
Pleasantries (apologies, thanks, sign-offs) are not factual claims and are skipped."""
import re
from . import nli

PLEASANTRY = re.compile(r"\b(sorry|apolog|thank|happy to help|hope this helps|please (let us know|reach out|contact)|"
                        r"best regards|kind regards|sincerely|understand|frustrat|specialist will|team will)\b", re.I)


def check(draft: str, chunks: list[dict]) -> dict:
    claims = [s for s in nli.sentences(draft) if not PLEASANTRY.search(s)]
    if not chunks:
        return {"groundedness": 0.0, "unsupported": claims}
    if not claims:
        return {"groundedness": 1.0, "unsupported": []}
    sup = nli.support(claims, [c["text"] for c in chunks])
    return {"groundedness": round(sum(s >= 0.5 for s in sup) / len(sup), 3),
            "unsupported": [c for c, s in zip(claims, sup) if s < 0.5]}
