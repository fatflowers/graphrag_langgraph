"""
LangGraph representation of GraphRAG query flows (global/local/basic/drift).

Nodes are responsible for loading index artifacts, selecting a query mode, and
delegating to the upstream search implementations.
"""

from __future__ import annotations

from typing import Any, Iterable

from langgraph.graph import END, StateGraph

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
from graphrag.utils.api import create_storage_from_config
from graphrag.utils.storage import load_table_from_storage, storage_has_table

from .config import load_graph_rag_config
from .state import QueryState


def _load_config(state: QueryState) -> QueryState:
    """Load or normalize the GraphRAG config into the state."""
    if isinstance(state.get("config"), GraphRagConfig):
        return state
    cfg_path = state.get("config_path")
    root = state.get("root")
    overrides = state.get("config_overrides", {})
    loaded = load_graph_rag_config(cfg_path, root=root, overrides=overrides)
    return {**state, "config": loaded}


def _route_mode(state: QueryState) -> QueryState:
    """
    Placeholder router that will later classify and route query modes.

    For now this simply normalizes the requested mode (defaulting to global).
    """
    mode = state.get("mode", "auto") or "auto"
    normalized = (mode or "auto").lower()
    if normalized == "auto":
        normalized = "global"
    state["mode"] = normalized  # type: ignore
    return state


async def _execute_query(state: QueryState) -> QueryState:
    """
    Placeholder LangGraph node that will invoke the upstream query engines.

    Delegates to the official GraphRAG query APIs to preserve behaviour.
    """
    config = state["config"]
    mode = state.get("mode", "global") or "global"
    question = state["question"]
    response_type = state.get("response_type", "multiple paragraphs")
    community_level = state.get("community_level")
    dynamic_selection = bool(state.get("dynamic_community_selection", False))
    verbose = bool(state.get("verbose", False))
    callbacks = state.get("callbacks")

    try:
        if mode == "local":
            artifacts = await _load_outputs(
                config,
                required=["communities", "community_reports", "text_units", "relationships", "entities"],
                optional=["covariates"],
            )
            if artifacts["multi-index"]:
                answer, context = await multi_index_local_search(
                    config=config,
                    entities_list=artifacts["entities"],
                    communities_list=artifacts["communities"],
                    community_reports_list=artifacts["community_reports"],
                    text_units_list=artifacts["text_units"],
                    relationships_list=artifacts["relationships"],
                    covariates_list=artifacts.get("covariates"),
                    index_names=list(artifacts["index_names"]),
                    community_level=community_level or 0,
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
                    community_level=community_level or 0,
                    response_type=response_type,
                    streaming=False,
                    query=question,
                    callbacks=callbacks,
                    verbose=verbose,
                )
        elif mode == "drift":
            artifacts = await _load_outputs(
                config,
                required=[
                    "entities",
                    "communities",
                    "community_reports",
                    "text_units",
                    "relationships",
                ],
            )
            if artifacts["multi-index"]:
                answer, context = await multi_index_drift_search(
                    config=config,
                    entities_list=artifacts["entities"],
                    communities_list=artifacts["communities"],
                    community_reports_list=artifacts["community_reports"],
                    text_units_list=artifacts["text_units"],
                    relationships_list=artifacts["relationships"],
                    index_names=list(artifacts["index_names"]),
                    community_level=community_level or 0,
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
                    community_level=community_level or 0,
                    response_type=response_type,
                    query=question,
                    callbacks=callbacks,
                    verbose=verbose,
                )
        elif mode == "basic":
            artifacts = await _load_outputs(config, required=["text_units"])
            if artifacts["multi-index"]:
                answer, context = await multi_index_basic_search(
                    config=config,
                    text_units_list=artifacts["text_units"],
                    index_names=list(artifacts["index_names"]),
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
        else:
            # default / global
            artifacts = await _load_outputs(
                config,
                required=["entities", "communities", "community_reports"],
            )
            if artifacts["multi-index"]:
                answer, context = await multi_index_global_search(
                    config=config,
                    entities_list=artifacts["entities"],
                    communities_list=artifacts["communities"],
                    community_reports_list=artifacts["community_reports"],
                    index_names=list(artifacts["index_names"]),
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
        return {
            **state,
            "answer": answer,
            "query_context": {"artifacts": artifacts},
            "debug": {"context_data": context},
        }
    except Exception as exc:  # pragma: no cover - preserved for graph state visibility
        errors = list(state.get("errors", []))
        errors.append(exc)
        return {**state, "errors": errors}


async def _load_outputs(
    config: GraphRagConfig,
    required: Iterable[str],
    optional: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Load required/optional parquet tables from storage, handling multi-index configs.
    """
    optional = optional or []
    data: dict[str, Any] = {}
    if config.outputs:
        data["multi-index"] = True
        data["num_indexes"] = len(config.outputs)
        data["index_names"] = list(config.outputs.keys())
        for name, output in config.outputs.items():
            storage = create_storage_from_config(output)
            for req in required:
                data.setdefault(req, [])
                df = await load_table_from_storage(req, storage)
                data[req].append(df)
            for opt in optional:
                data.setdefault(opt, [])
                exists = await storage_has_table(opt, storage)
                if exists:
                    df = await load_table_from_storage(opt, storage)
                    data[opt].append(df)
        for opt in optional:
            if len(data[opt]) != len(config.outputs):
                data[opt] = None
        return data

    # single-index
    storage = create_storage_from_config(config.output)
    data["multi-index"] = False
    for req in required:
        data[req] = await load_table_from_storage(req, storage)
    for opt in optional:
        exists = await storage_has_table(opt, storage)
        if exists:
            data[opt] = await load_table_from_storage(opt, storage)
        else:
            data[opt] = None
    return data


def build_query_graph(
    config: GraphRagConfig | dict[str, Any] | None = None,
) -> Any:
    """
    Build and compile the query LangGraph.

    Parameters
    ----------
    config:
        Optional pre-loaded `GraphRagConfig` or mapping compatible with the upstream
        configuration loader.
    """
    graph = StateGraph(QueryState)
    graph.add_node("load_config", _load_config)
    graph.add_node("route_mode", _route_mode)
    graph.add_node("execute_query", _execute_query)

    graph.set_entry_point("load_config")
    graph.add_edge("load_config", "route_mode")
    graph.add_edge("route_mode", "execute_query")
    graph.add_edge("execute_query", END)
    return graph.compile()
