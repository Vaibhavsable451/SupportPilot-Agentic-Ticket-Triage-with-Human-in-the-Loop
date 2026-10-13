"""Classic ML: TF-IDF + logistic regression predicting ticket category and priority."""
import json
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .config import settings

TARGETS = ("category", "priority")
_cache = {}


def _pipe():
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                     ("clf", LogisticRegression(max_iter=1000, C=5.0))])


def train() -> dict:
    df = pd.read_csv(settings.dataset_path)
    models, report = {}, {}
    for t in TARGETS:
        Xtr, Xte, ytr, yte = train_test_split(df["text"], df[t], test_size=0.25, random_state=42, stratify=df[t])
        p = _pipe().fit(Xtr, ytr)
        rep = classification_report(yte, p.predict(Xte), output_dict=True, zero_division=0)
        report[t] = {"accuracy": round(rep["accuracy"], 3), "macro_f1": round(rep["macro avg"]["f1-score"], 3),
                     "per_class_f1": {k: round(v["f1-score"], 3) for k, v in rep.items()
                                      if k not in ("accuracy", "macro avg", "weighted avg")}}
        models[t] = _pipe().fit(df["text"], df[t])  # final model on all data
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    joblib.dump(models, settings.model_path)
    _cache["models"] = models
    report["rows"] = len(df)
    with open(settings.model_path + ".json", "w") as f:
        json.dump(report, f)
    return report


def info() -> dict:
    p = settings.model_path + ".json"
    return json.load(open(p)) if os.path.exists(p) else {"trained": False}


def _models():
    if "models" not in _cache:
        if not os.path.exists(settings.model_path):
            train()
        else:
            _cache["models"] = joblib.load(settings.model_path)
    return _cache["models"]


def predict(text: str) -> dict:
    m, out = _models(), {}
    for t in TARGETS:
        proba = m[t].predict_proba([text])[0]
        i = int(proba.argmax())
        out[t] = str(m[t].classes_[i])
        out[f"{t}_conf"] = round(float(proba[i]), 3)
    return out
