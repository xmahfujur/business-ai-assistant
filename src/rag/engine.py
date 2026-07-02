"""Shared RAG engine: lazy index load, rebuild on demand, query with sources."""

from __future__ import annotations

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    GEMINI_API_KEY,
    LLM_MODEL,
    TOP_K,
)
from src.rag.vector_store import build_vector_index, get_collection_count  # noqa: F401 — re-exported

_query_engine = None


def _configure_settings() -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )
    Settings.llm = GoogleGenAI(api_key=GEMINI_API_KEY, model=LLM_MODEL)


def _connect_vector_store() -> ChromaVectorStore:
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = db.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


def rebuild_index() -> int:
    """Rebuild the vector index from files in the upload directory."""
    global _query_engine
    _configure_settings()
    build_vector_index(rebuild=True)
    _query_engine = None
    return get_collection_count()


def get_query_engine():
    """Return a cached query engine, building the index only when needed."""
    global _query_engine
    _configure_settings()

    if _query_engine is None:
        if get_collection_count() == 0:
            build_vector_index(rebuild=True)

        vector_store = _connect_vector_store()
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        _query_engine = index.as_query_engine(similarity_top_k=TOP_K)

    return _query_engine


def query_documents(question: str) -> dict:
    """Run a RAG query and return the answer plus source snippets."""
    engine = get_query_engine()
    response = engine.query(question)

    sources = []
    for node in response.source_nodes:
        meta = node.metadata or {}
        sources.append(
            {
                "file": meta.get("file_name", "unknown"),
                "score": round(node.score, 3) if node.score is not None else None,
                "excerpt": node.text[:300] + ("..." if len(node.text) > 300 else ""),
            }
        )

    return {"answer": str(response), "sources": sources}
