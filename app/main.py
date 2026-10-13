import json
import os
import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from . import classifier, graph, ingestion, store
from .config import settings

app = FastAPI(title="SupportPilot", version="1.0.0")
for d in ("data/uploads", "models", "logs"):
    os.makedirs(d, exist_ok=True)


class Ticket(BaseModel):
    text: str = Field(min_length=3, max_length=5000)


class Review(BaseModel):
    approve: bool
    final_reply: str | None = None
    note: str | None = None


def _records(df):
    return json.loads(df.to_json(orient="records"))


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.groq_model, "index": settings.pinecone_index}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    paths = []
    for f in files:
        dest = os.path.join("data/uploads", os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    try:
        return ingestion.ingest_paths(paths)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.post("/tickets")
def triage(t: Ticket):
    try:
        return graph.run(t.text)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")


@app.get("/tickets")
def list_tickets(status: str | None = None, limit: int = 100):
    return _records(store.fetch(status=status, limit=limit))


@app.post("/tickets/{ticket_id}/review")
def review(ticket_id: int, r: Review):
    ok = store.review(ticket_id, "approved" if r.approve else "rejected", r.final_reply, r.note)
    if not ok:
        raise HTTPException(404, "Ticket not found or not pending")
    return {"ticket_id": ticket_id, "status": "approved" if r.approve else "rejected"}


@app.post("/ml/train")
def ml_train():
    return classifier.train()


@app.get("/ml/info")
def ml_info():
    return classifier.info()


@app.post("/ml/predict")
def ml_predict(t: Ticket):
    return classifier.predict(t.text)


@app.get("/metrics")
def metrics():
    return store.summary()


@app.get("/audit")
def get_audit(limit: int = 50):
    if not os.path.exists(settings.audit_file):
        return []
    with open(settings.audit_file) as f:
        return [json.loads(l) for l in f.readlines()[-limit:]]
