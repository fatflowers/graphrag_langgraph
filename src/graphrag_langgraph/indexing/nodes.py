"""Node functions for the indexing graph."""
from __future__ import annotations

import hashlib
import itertools
from typing import Iterable, List

from ..graph_store import GraphIndexStore
from ..types import Claim, Community, CommunitySummary, Document, Entity, Relation, TextUnit
from .state import IndexState


_DEF_VECTOR_SIZE = 24


def _hash_embedding_fn(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        chunk = h[:_DEF_VECTOR_SIZE]
        vec = [(b - 128) / 128 for b in chunk]
        vectors.append(vec)
    return vectors


def _chunk_block(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[tuple[str, int]]:
    words = text.split()
    if not words:
        return []
    stride = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        yield " ".join(window), len(window)
        if start + chunk_size >= len(words):
            break
        start += stride


def split_into_text_units(state: IndexState) -> IndexState:
    """Split raw documents into text units using configurable chunking."""

    text_units: List[TextUnit] = []
    unit_id_counter = itertools.count()
    for doc in state.raw_docs:
        position_counter = itertools.count()
        blocks = [b.strip() for b in doc.text.split("\n\n") if b.strip()]
        for block in blocks:
            for chunk, token_count in _chunk_block(block, state.config.chunk_size, state.config.chunk_overlap):
                uid = f"tu-{next(unit_id_counter)}"
                pos = next(position_counter)
                text_units.append(
                    TextUnit(
                        id=uid,
                        document_ids=[doc.id],
                        source_doc_id=doc.id,
                        position=pos,
                        text=chunk,
                        n_tokens=token_count,
                        attributes={"title": doc.title},
                    )
                )
                doc.text_unit_ids.append(uid)
    state.text_units = text_units
    return state


def extract_entities_relations_claims(state: IndexState) -> IndexState:
    """Extract entities, relations, and claims using a simple heuristic or LLM stub."""

    entities: List[Entity] = []
    relations: List[Relation] = []
    claims: List[Claim] = []
    previous_entity_id: str | None = None
    last_doc_id: str | None = None

    for tu in state.text_units:
        current_doc = tu.document_ids[0] if tu.document_ids else tu.source_doc_id
        if current_doc != last_doc_id:
            previous_entity_id = None
        tokens = tu.text.split()
        name = next((t.strip(",.;") for t in tokens if t[:1].isupper()), tokens[0] if tokens else "Unknown")
        ent_id = f"ent-{tu.id}"
        entity = Entity(
            id=ent_id,
            title=name,
            type="heuristic",
            description=f"Entity mentioned in {tu.id}",
            text_unit_ids=[tu.id],
        )
        entities.append(entity)
        tu.entity_ids.append(ent_id)

        claim = Claim(
            id=f"claim-{tu.id}",
            subject_id=ent_id,
            text=tu.text[:200],
            text_unit_ids=[tu.id],
        )
        claims.append(claim)

        if previous_entity_id:
            rel_id = f"rel-{tu.id}"
            relations.append(
                Relation(
                    id=rel_id,
                    source=previous_entity_id,
                    target=ent_id,
                    relation_type="co-occurs",
                    description="Heuristic co-occurrence in adjacent text units",
                    text_unit_ids=[tu.id],
                )
            )
            tu.relationship_ids.append(rel_id)
        previous_entity_id = ent_id
        last_doc_id = current_doc

    state.entities = entities
    state.relations = relations
    state.claims = claims
    return state


def build_knowledge_graph(state: IndexState) -> IndexState:
    store = state.index_store or GraphIndexStore(embedding_fn=_hash_embedding_fn)
    store.add_documents(state.raw_docs)
    store.add_text_units(state.text_units)
    store.add_entities(state.entities)
    store.add_relations(state.relations)
    store.add_claims(state.claims)
    state.index_store = store
    return state


def detect_communities(state: IndexState) -> IndexState:
    graph = state.index_store.graph if state.index_store else None
    undirected = graph.to_undirected() if graph else None
    communities: List[Community] = []
    components = undirected.connected_components() if undirected else []
    for idx, component in enumerate(components):
        entity_ids = list(component)
        relationship_ids = [
            rel.id for rel in state.relations if rel.source in component or rel.target in component
        ]
        text_unit_ids: List[str] = []
        for eid in entity_ids:
            ent = state.index_store.entities.get(eid) if state.index_store else None
            if ent:
                text_unit_ids.extend(ent.text_unit_ids)
        covariate_ids = {"claim": [c.id for c in state.claims if c.subject_id in component]} if state.claims else {}
        communities.append(
            Community(
                id=f"comm-{idx}",
                title=f"Community {idx}",
                level=0,
                parent=None,
                children=[],
                entity_ids=entity_ids,
                relationship_ids=relationship_ids,
                text_unit_ids=text_unit_ids,
                covariate_ids=covariate_ids,
                size=len(text_unit_ids) or len(entity_ids),
            )
        )
    state.communities = communities
    if state.index_store:
        state.index_store.add_communities(communities)
    return state


def summarize_communities(state: IndexState) -> IndexState:
    summaries: List[CommunitySummary] = []
    for comm in state.communities:
        names = [
            state.index_store.entities[eid].title
            for eid in comm.member_entity_ids
            if state.index_store and eid in state.index_store.entities
        ]
        summary_text = "; ".join(names) if names else "Community of related entities"
        referenced_text_units: List[str] = []
        if state.index_store:
            for tu_id in comm.text_unit_ids:
                if tu_id in state.index_store.text_units:
                    referenced_text_units.append(tu_id)
        max_units = max(1, state.config.max_summary_tokens // 80)
        full_content = (
            " | ".join(
                state.index_store.text_units[tid].text
                for tid in referenced_text_units[:max_units]
            )
            if state.index_store
            else summary_text
        )
        summaries.append(
            CommunitySummary(
                id=f"{comm.id}-report",
                community_id=comm.id,
                title=comm.title,
                level=comm.level,
                summary=summary_text,
                full_content=full_content or summary_text,
                referenced_entities=comm.member_entity_ids,
                referenced_text_units=referenced_text_units,
            )
        )
    state.community_summaries = summaries
    if state.index_store:
        state.index_store.add_community_summaries(summaries)
    return state


def create_embeddings_and_vector_stores(state: IndexState) -> IndexState:
    if state.index_store is None:
        state.index_store = GraphIndexStore(embedding_fn=_hash_embedding_fn)
    entity_ids = list(state.index_store.entities.keys())
    text_unit_ids = list(state.index_store.text_units.keys())
    community_ids = [summary.community_id for summary in state.community_summaries]
    state.index_store.embed_and_store_entities(entity_ids)
    state.index_store.embed_and_store_text_units(text_unit_ids)
    state.index_store.embed_and_store_communities(community_ids)
    return state


def persist_index(state: IndexState) -> IndexState:
    if state.config.persist_graph and state.index_store:
        state.index_store.save(state.config.vector_store_dir)
    return state


__all__ = [
    "split_into_text_units",
    "extract_entities_relations_claims",
    "build_knowledge_graph",
    "detect_communities",
    "summarize_communities",
    "create_embeddings_and_vector_stores",
    "persist_index",
]
