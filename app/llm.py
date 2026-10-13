from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from .config import settings


def chat(system: str, user: str, temperature: float = 0.2) -> tuple[str, dict]:
    llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=temperature)
    r = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    u = getattr(r, "usage_metadata", None) or {}
    return r.content, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
