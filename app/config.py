import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "supportpilot")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_dim: int = 384
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    sentiment_model: str = os.getenv("SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    context_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "2500"))
    grounded_threshold: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.75"))
    low_conf: float = float(os.getenv("LOW_CONFIDENCE", "0.45"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "1"))
    top_k: int = 4
    min_score: float = 0.2
    db_path: str = "data/supportpilot.db"
    model_path: str = "models/triage.joblib"
    dataset_path: str = "data/tickets.csv"
    audit_file: str = "logs/audit.jsonl"
    trace_file: str = "logs/traces.jsonl"


settings = Settings()
