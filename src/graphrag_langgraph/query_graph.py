"""
LangGraph representation of GraphRAG query flows (global/local/basic/drift).

Each upstream query API (global_search, local_search, etc.) is mapped to its own
LangGraph node. The engine prepares config and index artifacts; the graph only
handles mode routing and delegating to the appropriate upstream implementation.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from graphrag.api import (
    basic_search,
    drift_search,
    global_search,
    local_search,
    multi_index_basic_search,
    multi_index_drift_search,
    multi_index_global_search,
    multi_index_local_search,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from langgraph.graph import END, StateGraph

from .state import QueryState


class CompiledQueryGraph(Protocol):
    """Minimal protocol for compiled LangGraph query graphs."""

    async def ainvoke(self, state: QueryState) -> QueryState:  # pragma: no cover - protocol
        ...


def _normalize_mode(mode: str | None) -> Literal["global", "local", "basic", "drift"]:
    """Normalize a user-provided mode string."""
    m = (mode or "auto").lower()
    if m == "auto":
        return "global"
    if m in {"global", "local", "basic", "drift"}:
        return m  # type: ignore[return-value]
    return "global"


def _set_mode(state: QueryState) -> QueryState:
    """Normalize and set the mode in-place."""
    state["mode"] = _normalize_mode(state.get("mode"))  # type: ignore[index]
    return state


def _route_next(state: QueryState) -> str:
    """Return the node name corresponding to the current mode."""
    mode = _normalize_mode(state.get("mode"))
    return f"{mode}_query"


async def _global_query(state: QueryState) -> QueryState:
    """Delegate to single- or multi-index global search."""
    config: GraphRagConfig = state["config"]
    question = state["question"]
    artifacts: dict[str, Any] = state["artifacts"]
    response_type = state.get("response_type", "multiple paragraphs")
    community_level = state.get("community_level")
    dynamic_selection = bool(state.get("dynamic_community_selection", False))
    callbacks = state.get("callbacks")
    verbose = bool(state.get("verbose", False))

    if artifacts.get("multi-index"):
        answer, context = await multi_index_global_search(
            config=config,
            entities_list=artifacts["entities_list"],
            communities_list=artifacts["communities_list"],
            community_reports_list=artifacts["community_reports_list"],
            index_names=artifacts["index_names"],
            community_level=community_level,
            dynamic_community_selection=dynamic_selection,
            response_type=response_type,
            streaming=False,
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        answer, context = await global_search(
            config=config,
            entities=artifacts["entities"],
            communities=artifacts["communities"],
            community_reports=artifacts["community_reports"],
            community_level=community_level,
            dynamic_community_selection=dynamic_selection,
            response_type=response_type,
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )

    state["answer"] = answer  # type: ignore[index]
    state["query_context"] = {"artifacts": artifacts}  # type: ignore[index]
    state.setdefault("debug", {})["context_data"] = context  # type: ignore[index]
    return state


async def _local_query(state: QueryState) -> QueryState:
    """Delegate to single- or multi-index local search."""
    config: GraphRagConfig = state["config"]
    question = state["question"]
    artifacts: dict[str, Any] = state["artifacts"]
    response_type = state.get("response_type", "multiple paragraphs")
    community_level = state.get("community_level") or 0
    callbacks = state.get("callbacks")
    verbose = bool(state.get("verbose", False))

    if artifacts.get("multi-index"):
        answer, context = await multi_index_local_search(
            config=config,
            entities_list=artifacts["entities_list"],
            communities_list=artifacts["communities_list"],
            community_reports_list=artifacts["community_reports_list"],
            text_units_list=artifacts["text_units_list"],
            relationships_list=artifacts["relationships_list"],
            covariates_list=artifacts.get("covariates_list"),
            index_names=artifacts["index_names"],
            community_level=community_level,
            response_type=response_type,
            streaming=False,
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        answer, context = await local_search(
            config=config,
            communities=artifacts["communities"],
            community_reports=artifacts["community_reports"],
            text_units=artifacts["text_units"],
            relationships=artifacts["relationships"],
            entities=artifacts["entities"],
            covariates=artifacts.get("covariates"),
            community_level=community_level,
            response_type=response_type,
            streaming=False,
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )

    state["answer"] = answer  # type: ignore[index]
    state["query_context"] = {"artifacts": artifacts}  # type: ignore[index]
    state.setdefault("debug", {})["context_data"] = context  # type: ignore[index]
    return state


async def _basic_query(state: QueryState) -> QueryState:
    """Delegate to single- or multi-index basic search."""
    config: GraphRagConfig = state["config"]
    question = state["question"]
    artifacts: dict[str, Any] = state["artifacts"]
    response_type = state.get("response_type", "multiple paragraphs")
    callbacks = state.get("callbacks")
    verbose = bool(state.get("verbose", False))

    if artifacts.get("multi-index"):
        answer, context = await multi_index_basic_search(
            config=config,
            text_units_list=artifacts["text_units_list"],
            index_names=artifacts["index_names"],
            response_type=response_type,
            streaming=False,
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        answer, context = await basic_search(
            config=config,
            text_units=artifacts["text_units"],
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )

    state["answer"] = answer  # type: ignore[index]
    state["query_context"] = {"artifacts": artifacts}  # type: ignore[index]
    state.setdefault("debug", {})["context_data"] = context  # type: ignore[index]
    return state


async def _drift_query(state: QueryState) -> QueryState:
    """Delegate to single- or multi-index DRIFT search."""
    config: GraphRagConfig = state["config"]
    question = state["question"]
    artifacts: dict[str, Any] = state["artifacts"]
    response_type = state.get("response_type", "multiple paragraphs")
    community_level = state.get("community_level") or 0
    callbacks = state.get("callbacks")
    verbose = bool(state.get("verbose", False))

    if artifacts.get("multi-index"):
        answer, context = await multi_index_drift_search(
            config=config,
            entities_list=artifacts["entities_list"],
            communities_list=artifacts["communities_list"],
            community_reports_list=artifacts["community_reports_list"],
            text_units_list=artifacts["text_units_list"],
            relationships_list=artifacts["relationships_list"],
            index_names=artifacts["index_names"],
            community_level=community_level,
            response_type=response_type,
            streaming=False,
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )
    else:
        answer, context = await drift_search(
            config=config,
            entities=artifacts["entities"],
            communities=artifacts["communities"],
            community_reports=artifacts["community_reports"],
            text_units=artifacts["text_units"],
            relationships=artifacts["relationships"],
            community_level=community_level,
            response_type=response_type,
            query=question,
            callbacks=callbacks,
            verbose=verbose,
        )

    state["answer"] = answer  # type: ignore[index]
    state["query_context"] = {"artifacts": artifacts}  # type: ignore[index]
    state.setdefault("debug", {})["context_data"] = context  # type: ignore[index]
    return state


def build_query_graph(
    config: GraphRagConfig | dict[str, Any] | None = None,  # kept for API symmetry
) -> CompiledQueryGraph:
    """
    Build and compile the query LangGraph.

    The engine is responsible for loading index artifacts and attaching them to
    the state as `artifacts`.
    """
    graph = StateGraph(QueryState)
    graph.add_node("set_mode", _set_mode)
    graph.add_node("global_query", _global_query)
    graph.add_node("local_query", _local_query)
    graph.add_node("basic_query", _basic_query)
    graph.add_node("drift_query", _drift_query)

    graph.set_entry_point("set_mode")
    graph.add_conditional_edges(
        "set_mode",
        _route_next,
        {
            "global_query": "global_query",
            "local_query": "local_query",
            "basic_query": "basic_query",
            "drift_query": "drift_query",
        },
    )

    graph.add_edge("global_query", END)
    graph.add_edge("local_query", END)
    graph.add_edge("basic_query", END)
    graph.add_edge("drift_query", END)

    return graph.compile()

