"""
Shared LangGraph state definitions.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from graphrag.config.models.graph_rag_config import GraphRagConfig


class IndexState(TypedDict, total=False):
    """Mutable state carried through the LangGraph indexing graph."""

    config: GraphRagConfig
    pipeline_method: str
    additional_context: dict[str, Any]
    input_documents: Any
    pipeline_outputs: list[Any]
    artifacts: dict[str, Any]
    errors: list[Exception]
    progress: dict[str, Any]


class QueryState(TypedDict, total=False):
    """Mutable state carried through the LangGraph query graph."""

    config: GraphRagConfig
    question: str
    mode: Literal["auto", "global", "local", "basic", "drift"]
    query_context: dict[str, Any]
    intermediate: dict[str, Any]
    answer: Any
    debug: dict[str, Any]
    errors: list[Exception]
