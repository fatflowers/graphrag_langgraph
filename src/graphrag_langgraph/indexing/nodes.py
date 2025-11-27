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


def split_into_text_units(state: IndexState) -> IndexState:
    """Split raw documents into text units."""

    text_units: List[TextUnit] = []
    unit_id_counter = itertools.count()
    for doc in state.raw_docs:
        blocks = [b.strip() for b in doc.text.split("\n\n") if b.strip()]
        for pos, block in enumerate(blocks):
            uid = f"tu-{next(unit_id_counter)}"
            text_units.append(
                TextUnit(
                    id=uid,
                    source_doc_id=doc.id,
                    position=pos,
                    text=block,
                    token_count=len(block.split()),
                    metadata={"title": doc.title},
                )
            )
    state.text_units = text_units
    return state


def extract_entities_relations_claims(state: IndexState) -> IndexState:
    """Extract entities, relations, and claims using a simple heuristic or LLM stub."""

    entities: List[Entity] = []
    relations: List[Relation] = []
    claims: List[Claim] = []

    for tu in state.text_units:
        # naive entity: first capitalized word fallback to first token
        tokens = tu.text.split()
        name = next((t.strip(",.;") for t in tokens if t[:1].isupper()), tokens[0] if tokens else "Unknown")
        ent_id = f"ent-{tu.id}"
        entities.append(
            Entity(
                id=ent_id,
                name=name,
                type="heuristic",
                description=f"Entity mentioned in {tu.id}",
                text_unit_id=tu.id,
            )
        )
        claims.append(
            Claim(
                id=f"claim-{tu.id}",
                text=tu.text[:200],
                entity_ids=[ent_id],
                text_unit_ids=[tu.id],
            )
        )
        # simplistic relation to previous entity if exists
        if len(entities) > 1:
            prev_entity = entities[-2]
            relations.append(
                Relation(
                    id=f"rel-{tu.id}",
                    head_entity=prev_entity.id,
                    relation_type="co-occurs",
                    tail_entity=ent_id,
                    evidence_text_unit_ids=[tu.id, prev_entity.text_unit_id or tu.id],
                )
            )

    state.entities = entities
    state.relations = relations
    state.claims = claims
    return state


def build_knowledge_graph(state: IndexState) -> IndexState:
    store = state.index_store or GraphIndexStore(embedding_fn=_hash_embedding_fn)
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
    if undirected:
        components = undirected.connected_components()
    else:
        components = []
    for idx, component in enumerate(components):
        communities.append(
            Community(
                id=f"comm-{idx}",
                level=0,
                member_entity_ids=list(component),
                parent_community_id=None,
            )
        )
    state.communities = communities
    if state.index_store:
        state.index_store.add_communities(communities)
    return state


def summarize_communities(state: IndexState) -> IndexState:
    summaries: List[CommunitySummary] = []
    for comm in state.communities:
        names = [state.index_store.entities[eid].name for eid in comm.member_entity_ids if state.index_store and eid in state.index_store.entities]
        summary_text = "; ".join(names) if names else "Community of related entities"
        summaries.append(
            CommunitySummary(
                community_id=comm.id,
                level=comm.level,
                summary_text=summary_text,
                report_text=summary_text,
                referenced_entities=comm.member_entity_ids,
                referenced_text_units=[],
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
    community_ids = list(state.index_store.communities.keys())
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
