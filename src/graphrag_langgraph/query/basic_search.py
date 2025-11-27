"""Baseline RAG search over text units."""
from __future__ import annotations

from typing import List

from .state import QueryState

_DEF_VECTOR_SIZE = 24


def _hash_embedding(text: str) -> List[float]:
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    chunk = h[:_DEF_VECTOR_SIZE]
    return [(b - 128) / 128 for b in chunk]


def _embed_question(state: QueryState) -> List[float]:
    if state.index_store.embedding_fn:
        return state.index_store.embedding_fn([state.question])[0]
    return _hash_embedding(state.question)


def basic_search_node(state: QueryState) -> QueryState:
    query_vector = _embed_question(state)
    state.retrieved_text_units = state.index_store.search_text_units(
        query_vector, k=state.config.top_k_text_units
    )
    snippets = []
    for res in state.retrieved_text_units:
        tu = state.index_store.text_units.get(res.id)
        if tu:
            snippets.append(tu.text)
    state.context = "\n".join(snippets)
    return state


__all__ = ["basic_search_node"]
