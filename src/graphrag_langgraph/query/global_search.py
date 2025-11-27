"""Global search nodes."""
from __future__ import annotations

from typing import List

from ..types import TextUnit
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


def global_search_node(state: QueryState) -> QueryState:
    """Retrieve communities and supporting text for global questions."""

    query_vector = _embed_question(state)
    state.retrieved_communities = state.index_store.search_communities(
        query_vector, k=state.config.top_k_communities
    )

    context_parts: List[str] = []
    for comm_res in state.retrieved_communities:
        summary = state.index_store.community_summaries.get(comm_res.id)
        if summary:
            context_parts.append(f"Community {comm_res.id}: {summary.summary_text}")
            if summary.full_content:
                context_parts.append(f"Report: {summary.full_content}")
        comm = state.index_store.communities.get(comm_res.id)
        if not comm:
            continue
        # gather text units for community entities
        text_units: List[TextUnit] = []
        for ent_id in comm.member_entity_ids:
            ent = state.index_store.entities.get(ent_id)
            if ent and ent.text_unit_ids:
                for tu_id in ent.text_unit_ids:
                    tu = state.index_store.text_units.get(tu_id)
                    if tu:
                        text_units.append(tu)
        snippet = " | ".join(tu.text[:200] for tu in text_units[: state.config.top_k_text_units])
        if snippet:
            context_parts.append(f"Evidence: {snippet}")
    state.context = "\n".join(context_parts)
    return state


__all__ = ["global_search_node"]
