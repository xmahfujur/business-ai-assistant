"""Backward-compatible entry point. Prefer src.rag.engine for new code."""

from src.rag.engine import get_query_engine, query_documents, rebuild_index

query_engine = None


def _ensure_engine():
    global query_engine
    if query_engine is None:
        query_engine = get_query_engine()
    return query_engine


def __getattr__(name):
    if name == "query_engine":
        return _ensure_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
