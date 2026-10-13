"""Context-window management: relevance-ordered token-budget packing."""
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(t: str) -> int:
    return len(_enc.encode(t))


def build_context(chunks: list[dict], budget: int) -> tuple[list[dict], str, int]:
    kept, parts, used = [], [], 0
    for c in sorted(chunks, key=lambda x: -x["score"]):
        header = f"[{len(kept) + 1}] (source: {c['source']})\n"
        cost = count_tokens(header + c["text"])
        if used + cost > budget:
            continue
        parts.append(header + c["text"])
        kept.append(c)
        used += cost
    return kept, "\n\n".join(parts), used
