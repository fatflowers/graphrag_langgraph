"""
High-level engine that exposes GraphRAG functionality via LangGraph graphs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig

from .config import load_graph_rag_config
from .index_graph import (
    build_fast_index_graph,
    build_fast_update_index_graph,
    build_standard_index_graph,
    build_standard_update_index_graph,
)
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
        self.index_graphs = {
            IndexingMethod.Standard.value: build_standard_index_graph(),
            IndexingMethod.Fast.value: build_fast_index_graph(),
            IndexingMethod.StandardUpdate.value: build_standard_update_index_graph(),
            IndexingMethod.FastUpdate.value: build_fast_update_index_graph(),
        }
        self.query_graph = build_query_graph(self.config)

    def index(self, **kwargs: Any) -> IndexState:
        """
        Run the LangGraph indexing pipeline.

        Returns a final `IndexState`. Node implementations will fill in artifacts,
        pipeline outputs, and any diagnostics.
        """
        method = kwargs.get("pipeline_method", IndexingMethod.Standard)
        is_update = bool(kwargs.get("is_update_run", False))
        method_val = _normalize_indexing_method(method, is_update)
        graph = self.index_graphs.get(method_val)
        if graph is None:
            msg = f"Unsupported indexing method: {method}"
            raise ValueError(msg)
        state: IndexState = {"config": self.config, **kwargs}
        return graph.invoke(state)

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


def _normalize_indexing_method(
    method: IndexingMethod | str, is_update: bool
) -> str:
    """Map method/is_update to the registered pipeline key."""
    if isinstance(method, IndexingMethod):
        base = method.value
    else:
        base = str(method)
    if is_update:
        if base == IndexingMethod.Standard.value:
            return IndexingMethod.StandardUpdate.value
        if base == IndexingMethod.Fast.value:
            return IndexingMethod.FastUpdate.value
        if base.endswith("update"):
            return base
    return base
