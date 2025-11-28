"""
LangGraph representation of the GraphRAG indexing pipeline.

This module keeps the upstream indexing logic intact and swaps in LangGraph for
orchestration. Nodes will call into `graphrag.index` workflows and utilities.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from graphrag.api import build_index
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig

from .config import load_graph_rag_config
from .state import IndexState


def _load_config(state: IndexState) -> IndexState:
    """Load or normalize the GraphRAG config into the state."""
    if isinstance(state.get("config"), GraphRagConfig):
        return state
    cfg_path = state.get("config_path")
    root = state.get("root")
    overrides = state.get("config_overrides", {})
    loaded = load_graph_rag_config(cfg_path, root=root, overrides=overrides)
    return {**state, "config": loaded}


async def _run_official_pipeline(state: IndexState) -> IndexState:
    """
    Placeholder LangGraph node that will invoke the upstream indexing pipeline.

    This delegates to `graphrag.api.build_index` to preserve upstream behaviour.
    """
    config = state["config"]
    method = state.get("pipeline_method", IndexingMethod.Standard)
    is_update = bool(state.get("is_update_run", False))
    additional_context = state.get("additional_context")
    input_documents = state.get("input_documents")
    verbose = bool(state.get("verbose", False))
    memory_profile = bool(state.get("memory_profile", False))
    callbacks = state.get("callbacks")

    outputs = await build_index(
        config=config,
        method=method,
        is_update_run=is_update,
        memory_profile=memory_profile,
        callbacks=callbacks,
        additional_context=additional_context,
        verbose=verbose,
        input_documents=input_documents,
    )
    return {
        **state,
        "pipeline_outputs": outputs,
        "artifacts": {"output_storage": config.output, "outputs": outputs},
    }


def build_index_graph(
    config: GraphRagConfig | dict[str, Any] | None = None,
) -> Any:
    """
    Build and compile the indexing LangGraph.

    Parameters
    ----------
    config:
        Optional pre-loaded `GraphRagConfig` or mapping compatible with the upstream
        configuration loader.
    """
    graph = StateGraph(IndexState)
    graph.add_node("load_config", _load_config)
    graph.add_node("run_pipeline", _run_official_pipeline)

    graph.set_entry_point("load_config")
    graph.add_edge("load_config", "run_pipeline")
    graph.add_edge("run_pipeline", END)
    return graph.compile()
