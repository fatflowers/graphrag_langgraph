"""Knowledge graph and vector store abstractions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import math

from .types import Claim, Community, CommunitySummary, Document, Entity, Relation, RetrievalResult, TextUnit


class BasicGraph:
    """Lightweight directed multigraph replacement for environments without networkx."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def add_node(self, node_id: str, **attrs: Any) -> None:
        existing = self.nodes.get(node_id, {})
        existing.update(attrs)
        self.nodes[node_id] = existing

    def add_edge(self, source: str, target: str, key: str, **attrs: Any) -> None:
        self.edges.setdefault(source, {}).setdefault(target, {})[key] = attrs

    def predecessors(self, node_id: str):
        for src, targets in self.edges.items():
            if node_id in targets:
                yield src

    def successors(self, node_id: str):
        return self.edges.get(node_id, {}).keys()

    def to_undirected(self) -> "BasicGraph":
        undirected = BasicGraph()
        undirected.nodes = dict(self.nodes)
        for src, targets in self.edges.items():
            for tgt, keyed in targets.items():
                for key, attrs in keyed.items():
                    undirected.add_edge(src, tgt, key, **attrs)
                    undirected.add_edge(tgt, src, key, **attrs)
        return undirected

    def connected_components(self):
        visited = set()
        for node in self.nodes:
            if node in visited:
                continue
            comp = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current in comp:
                    continue
                comp.add(current)
                neighbors = set(self.successors(current)) | set(self.predecessors(current))
                stack.extend(neighbors - comp)
            visited.update(comp)
            yield comp


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors represented as lists."""

    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return float(dot / denom)


class SimpleVectorStore:
    """A tiny in-memory vector store for embeddings."""

    def __init__(self) -> None:
        self.items: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    def add(self, item_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        self.items[item_id] = (list(vector), metadata or {})

    def add_many(
        self, ids: Iterable[str], vectors: Iterable[List[float]], metadatas: Iterable[Optional[Dict[str, Any]]]
    ) -> None:
        for item_id, vector, meta in zip(ids, vectors, metadatas):
            self.add(item_id, vector, meta)

    def search(self, query: List[float], k: int = 4) -> List[RetrievalResult]:
        scores = [
            RetrievalResult(id=item_id, score=cosine_similarity(query, vec), metadata=meta)
            for item_id, (vec, meta) in self.items.items()
        ]
        scores.sort(key=lambda r: r.score, reverse=True)
        return scores[:k]

    def serialize(self) -> Dict[str, Any]:
        return {
            "items": [
                {"id": item_id, "vector": list(vec), "metadata": meta}
                for item_id, (vec, meta) in self.items.items()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleVectorStore":
        store = cls()
        for item in data.get("items", []):
            store.add(item["id"], item.get("vector", []), item.get("metadata", {}))
        return store


class GraphIndexStore:
    """Holds the knowledge graph, extracted artifacts, and vector stores."""

    def __init__(self, embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None) -> None:
        self.embedding_fn = embedding_fn
        self.graph = BasicGraph()
        self.documents: Dict[str, Document] = {}
        self.text_units: Dict[str, TextUnit] = {}
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.claims: Dict[str, Claim] = {}
        self.communities: Dict[str, Community] = {}
        self.community_summaries: Dict[str, CommunitySummary] = {}

        self.text_unit_store = SimpleVectorStore()
        self.entity_store = SimpleVectorStore()
        self.community_store = SimpleVectorStore()

    # --------------------- adders ---------------------
    def add_documents(self, documents: Iterable[Document]) -> None:
        for doc in documents:
            self.documents[doc.id] = doc

    def add_text_units(self, text_units: Iterable[TextUnit]) -> None:
        for tu in text_units:
            self.text_units[tu.id] = tu

    def add_entities(self, entities: Iterable[Entity]) -> None:
        for ent in entities:
            self.entities[ent.id] = ent
            self.graph.add_node(ent.id, **ent.dict())

    def add_relations(self, relations: Iterable[Relation]) -> None:
        for rel in relations:
            self.relations[rel.id] = rel
            self.graph.add_edge(rel.source, rel.target, key=rel.id, **rel.dict())

    def add_claims(self, claims: Iterable[Claim]) -> None:
        for claim in claims:
            self.claims[claim.id] = claim

    def add_communities(self, communities: Iterable[Community]) -> None:
        for comm in communities:
            self.communities[comm.id] = comm

    def add_community_summaries(self, summaries: Iterable[CommunitySummary]) -> None:
        for summary in summaries:
            self.community_summaries[summary.community_id] = summary

    # ----------------- embeddings ------------------
    def embed_and_store_entities(self, entity_ids: List[str]) -> None:
        if self.embedding_fn is None:
            return
        texts: List[str] = []
        ids: List[str] = []
        for eid in entity_ids:
            ent = self.entities.get(eid)
            if ent:
                ids.append(eid)
                texts.append(f"{ent.title}\n{ent.description}")
        if not ids:
            return
        vectors = self.embedding_fn(texts)
        self.entity_store.add_many(ids, vectors, [{} for _ in ids])

    def embed_and_store_text_units(self, text_unit_ids: List[str]) -> None:
        if self.embedding_fn is None:
            return
        texts: List[str] = []
        ids: List[str] = []
        for tid in text_unit_ids:
            tu = self.text_units.get(tid)
            if tu:
                ids.append(tid)
                texts.append(tu.text)
        if not ids:
            return
        vectors = self.embedding_fn(texts)
        self.text_unit_store.add_many(ids, vectors, [{} for _ in ids])

    def embed_and_store_communities(self, community_ids: List[str]) -> None:
        if self.embedding_fn is None:
            return
        texts: List[str] = []
        ids: List[str] = []
        for cid in community_ids:
            summary = self.community_summaries.get(cid)
            if summary:
                ids.append(cid)
                texts.append(summary.full_content or summary.summary or summary.title)
        if not ids:
            return
        vectors = self.embedding_fn(texts)
        self.community_store.add_many(ids, vectors, [{} for _ in ids])

    # ----------------- retrieval ------------------
    def search_entities(self, query_embedding: List[float], k: int = 5) -> List[RetrievalResult]:
        return self.entity_store.search(query_embedding, k)

    def search_text_units(self, query_embedding: List[float], k: int = 5) -> List[RetrievalResult]:
        return self.text_unit_store.search(query_embedding, k)

    def search_communities(self, query_embedding: List[float], k: int = 5) -> List[RetrievalResult]:
        return self.community_store.search(query_embedding, k)

    def get_neighbors(self, entity_id: str, depth: int = 1) -> List[str]:
        visited = {entity_id}
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.predecessors(node))
                next_frontier.update(self.graph.successors(node))
            frontier = next_frontier - visited
            visited.update(frontier)
        return list(visited)

    # ----------------- persistence ------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "documents.json").open("w", encoding="utf-8") as f:
            json.dump([doc.dict() for doc in self.documents.values()], f, indent=2)
        with (path / "text_units.json").open("w", encoding="utf-8") as f:
            json.dump([tu.dict() for tu in self.text_units.values()], f, indent=2)
        with (path / "entities.json").open("w", encoding="utf-8") as f:
            json.dump([ent.dict() for ent in self.entities.values()], f, indent=2)
        with (path / "relations.json").open("w", encoding="utf-8") as f:
            json.dump([rel.dict() for rel in self.relations.values()], f, indent=2)
        with (path / "claims.json").open("w", encoding="utf-8") as f:
            json.dump([cl.dict() for cl in self.claims.values()], f, indent=2)
        with (path / "communities.json").open("w", encoding="utf-8") as f:
            json.dump([c.dict() for c in self.communities.values()], f, indent=2)
        with (path / "community_summaries.json").open("w", encoding="utf-8") as f:
            json.dump([cs.dict() for cs in self.community_summaries.values()], f, indent=2)

        with (path / "text_unit_store.json").open("w", encoding="utf-8") as f:
            json.dump(self.text_unit_store.serialize(), f)
        with (path / "entity_store.json").open("w", encoding="utf-8") as f:
            json.dump(self.entity_store.serialize(), f)
        with (path / "community_store.json").open("w", encoding="utf-8") as f:
            json.dump(self.community_store.serialize(), f)

    @classmethod
    def load(cls, path: Path, embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None) -> "GraphIndexStore":
        store = cls(embedding_fn=embedding_fn)
        documents_path = path / "documents.json"
        if documents_path.exists():
            with documents_path.open("r", encoding="utf-8") as f:
                store.add_documents(Document(**item) for item in json.load(f))
        with (path / "text_units.json").open("r", encoding="utf-8") as f:
            store.add_text_units(TextUnit(**item) for item in json.load(f))
        with (path / "entities.json").open("r", encoding="utf-8") as f:
            store.add_entities(Entity(**item) for item in json.load(f))
        with (path / "relations.json").open("r", encoding="utf-8") as f:
            store.add_relations(Relation(**item) for item in json.load(f))
        with (path / "claims.json").open("r", encoding="utf-8") as f:
            store.add_claims(Claim(**item) for item in json.load(f))
        with (path / "communities.json").open("r", encoding="utf-8") as f:
            store.add_communities(Community(**item) for item in json.load(f))
        with (path / "community_summaries.json").open("r", encoding="utf-8") as f:
            store.add_community_summaries(CommunitySummary(**item) for item in json.load(f))

        with (path / "text_unit_store.json").open("r", encoding="utf-8") as f:
            store.text_unit_store = SimpleVectorStore.from_dict(json.load(f))
        with (path / "entity_store.json").open("r", encoding="utf-8") as f:
            store.entity_store = SimpleVectorStore.from_dict(json.load(f))
        with (path / "community_store.json").open("r", encoding="utf-8") as f:
            store.community_store = SimpleVectorStore.from_dict(json.load(f))

        return store


__all__ = ["GraphIndexStore", "SimpleVectorStore"]
