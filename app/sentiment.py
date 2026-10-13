"""Customer sentiment with a transformers pipeline."""
from functools import lru_cache
from .config import settings


@lru_cache(maxsize=1)
def _pipe():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=settings.sentiment_model)


def score(text: str) -> dict:
    r = _pipe()(text[:1500])[0]
    s = r["score"] if r["label"].upper().startswith("POS") else -r["score"]
    label = "negative" if s <= -0.6 else "positive" if s >= 0.6 else "neutral"
    return {"label": label, "score": round(float(s), 3)}
