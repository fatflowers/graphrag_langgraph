"""
High-level engine that exposes GraphRAG functionality via LangGraph graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from graphrag.config.models.graph_rag_config import GraphRagConfig

from .config import load_graph_rag_config
from .index_graph import build_index_graph
from .query_graph import build_query_graph
from .state import IndexState, QueryState


class GraphRAGEngine:
    """
    Thin façade that loads upstream GraphRAG configuration and executes LangGraph
    graphs for indexing and querying.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: GraphRagConfig | None = None,
        root: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ):
        self.config = config or load_graph_rag_config(
            config_path, root=root, overrides=overrides
        )
        self.index_graph = build_index_graph(self.config)
        self.query_graph = build_query_graph(self.config)

    def index(self, **kwargs: Any) -> IndexState:
        """
        Run the LangGraph indexing pipeline.

        Returns a final `IndexState`. Node implementations will fill in artifacts,
        pipeline outputs, and any diagnostics.
        """
        state: IndexState = {"config": self.config, **kwargs}
        return self.index_graph.invoke(state)

    def query(self, question: str, mode: str = "auto", **kwargs: Any) -> QueryState:
        """
        Run a query through the LangGraph query pipeline.

        Parameters
        ----------
        question:
            The user question to answer.
        mode:
            Desired search mode; defaults to "auto" (router to be implemented).
        """
        state: QueryState = {
            "config": self.config,
            "question": question,
            "mode": mode,
            **kwargs,
        }
        return self.query_graph.invoke(state)

    def global_search(self, question: str, **kwargs: Any) -> QueryState:
        """Convenience helper for global mode."""
        return self.query(question=question, mode="global", **kwargs)

    def local_search(self, question: str, **kwargs: Any) -> QueryState:
        """Convenience helper for local mode."""
        return self.query(question=question, mode="local", **kwargs)

    def basic_search(self, question: str, **kwargs: Any) -> QueryState:
        """Convenience helper for basic mode."""
        return self.query(question=question, mode="basic", **kwargs)

    def drift_search(self, question: str, **kwargs: Any) -> QueryState:
        """Convenience helper for drift mode."""
        return self.query(question=question, mode="drift", **kwargs)
