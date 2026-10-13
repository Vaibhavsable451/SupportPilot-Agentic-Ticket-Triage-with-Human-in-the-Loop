"""NLI entailment (transformers cross-encoder) used to verify drafts against the KB."""
import re
from functools import lru_cache
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .config import settings


@lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(settings.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model).eval()
    ent = next(i for i, l in model.config.id2label.items() if l.lower().startswith("entail"))
    return tok, model, ent


def sentences(text: str) -> list[str]:
    text = re.sub(r"\[[\w-]+\]", "", text)
    return [p.strip(" -*") for p in re.split(r"(?<=[.!?])\s+|\n+", text) if len(p.strip()) > 15]


@torch.no_grad()
def support(hypotheses: list[str], premises: list[str]) -> list[float]:
    if not hypotheses or not premises:
        return [0.0] * len(hypotheses)
    tok, model, ent = _load()
    pairs = [(p, h) for h in hypotheses for p in premises]
    probs = []
    for i in range(0, len(pairs), 16):
        b = pairs[i:i + 16]
        enc = tok([p for p, _ in b], [h for _, h in b], truncation="only_first",
                  max_length=512, padding=True, return_tensors="pt")
        probs += torch.softmax(model(**enc).logits, dim=-1)[:, ent].tolist()
    n = len(premises)
    return [max(probs[i * n:(i + 1) * n]) for i in range(len(hypotheses))]
