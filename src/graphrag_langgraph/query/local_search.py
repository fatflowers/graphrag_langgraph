"""Local search node."""
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


def local_search_node(state: QueryState) -> QueryState:
    """Retrieve entities and neighborhood context for local questions."""

    query_vector = _embed_question(state)
    state.retrieved_entities = state.index_store.search_entities(
        query_vector, k=state.config.top_k_entities
    )
    context_parts: List[str] = []
    seen_text_units = []
    for ent_res in state.retrieved_entities:
        ent = state.index_store.entities.get(ent_res.id)
        if not ent:
            continue
        context_parts.append(f"Entity {ent.title}: {ent.description}")
        # explore neighbors
        neighbors = state.index_store.get_neighbors(ent.id, depth=1)
        neighbor_names = [
            state.index_store.entities[nid].title for nid in neighbors if nid in state.index_store.entities and nid != ent.id
        ]
        if neighbor_names:
            context_parts.append("Neighbors: " + ", ".join(neighbor_names))
        if ent.text_unit_ids:
            for tu_id in ent.text_unit_ids:
                tu = state.index_store.text_units.get(tu_id)
                if tu:
                    seen_text_units.append(tu.text)
    if seen_text_units:
        context_parts.append("Evidence: " + " | ".join(seen_text_units[: state.config.top_k_text_units]))
    state.context = "\n".join(context_parts)
    return state


__all__ = ["local_search_node"]
