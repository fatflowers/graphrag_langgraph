"""High-level API for running GraphRAG with LangGraph."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Optional

from .config import IndexConfig, QueryConfig
from .graph_store import GraphIndexStore
from .indexing.pipeline import build_index_graph, run_indexing
from .query.pipeline import build_query_graph, run_query
from .types import Document


class GraphRAGEngine:
    """User-facing facade that orchestrates indexing and querying."""

    def __init__(
        self,
        index_config: Optional[IndexConfig] = None,
        query_config: Optional[QueryConfig] = None,
        index_store: Optional[GraphIndexStore] = None,
    ) -> None:
        self.index_config = index_config or IndexConfig()
        self.query_config = query_config or QueryConfig()
        self.index_store = index_store

    @classmethod
    def from_config_file(cls, path: str | Path) -> "GraphRAGEngine":
        path = Path(path)
        import yaml  # optional dependency; imported lazily

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        index_config = IndexConfig(**data.get("index", {}))
        query_config = QueryConfig(**data.get("query", {}))
        return cls(index_config=index_config, query_config=query_config)

    def index(self, corpus: Iterable[Document]) -> None:
        return self.index_with_llm(corpus)

    def index_with_llm(self, corpus: Iterable[Document], llm: Optional[Callable[[str], str]] = None) -> None:
        documents = [doc if isinstance(doc, Document) else Document(**doc) for doc in corpus]
        index_graph = build_index_graph(self.index_config)
        final_state = run_indexing(index_graph, documents, self.index_config, llm=llm)
        self.index_store = final_state.index_store

    def load_index(self, root_path: str | Path) -> None:
        self.index_store = GraphIndexStore.load(Path(root_path))

    def answer(self, question: str, mode: str = "auto") -> str:
        if self.index_store is None:
            raise ValueError("Index not built or loaded.")
        config = deepcopy(self.query_config)
        config.default_mode = mode  # type: ignore
        query_graph = build_query_graph(config)
        result_state = run_query(query_graph, question, config, self.index_store)
        return result_state.answer or ""


__all__ = ["GraphRAGEngine"]
