"""LlamaIndex loading + chunking of help-center articles -> Pinecone."""
import hashlib
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from . import vectorstore


def ingest_paths(paths: list[str], chunk_size: int = 300, overlap: int = 40) -> dict:
    docs = SimpleDirectoryReader(input_files=paths).load_data()
    nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap).get_nodes_from_documents(docs)
    chunks = []
    for n in nodes:
        text = n.get_content().strip()
        if len(text) < 50:
            continue
        src = n.metadata.get("file_name", "unknown")
        chunks.append({"id": hashlib.md5(f"{src}:{text}".encode()).hexdigest(), "text": text, "source": src})
    return {"documents": len(docs), "chunks": vectorstore.upsert(chunks) if chunks else 0}
