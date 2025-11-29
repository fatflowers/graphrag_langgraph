"""
LangGraph representation of the GraphRAG indexing pipelines (standard/fast + update variants).

Each upstream workflow function becomes its own LangGraph node; the graph itself is a
simple linear chain of those workflow nodes. All I/O and context construction are
handled outside the graph in the engine.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Iterable, Protocol

from langgraph.graph import END, StateGraph

from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index import workflows as wf
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.index.typing.workflow import WorkflowFunctionOutput

from .state import IndexState

logger = logging.getLogger(__name__)


class CompiledIndexGraph(Protocol):
    """Minimal protocol for compiled LangGraph index graphs."""

    async def ainvoke(self, state: IndexState) -> IndexState:  # pragma: no cover - protocol
        ...


# Map workflow names to upstream run_workflow callables
WORKFLOW_FNS: dict[
    str, Callable[[GraphRagConfig, PipelineRunContext], Awaitable[WorkflowFunctionOutput]],
] = {
    "load_input_documents": wf.run_load_input_documents,
    "load_update_documents": wf.run_load_update_documents,
    "create_base_text_units": wf.run_create_base_text_units,
    "create_final_documents": wf.run_create_final_documents,
    "extract_graph": wf.run_extract_graph,
    "extract_graph_nlp": wf.run_extract_graph_nlp,
    "prune_graph": wf.run_prune_graph,
    "finalize_graph": wf.run_finalize_graph,
    "extract_covariates": wf.run_extract_covariates,
    "create_communities": wf.run_create_communities,
    "create_final_text_units": wf.run_create_final_text_units,
    "create_community_reports": wf.run_create_community_reports,
    "create_community_reports_text": wf.run_create_community_reports_text,
    "generate_text_embeddings": wf.run_generate_text_embeddings,
    "update_final_documents": wf.run_update_final_documents,
    "update_entities_relationships": wf.run_update_entities_relationships,
    "update_text_units": wf.run_update_text_units,
    "update_covariates": wf.run_update_covariates,
    "update_communities": wf.run_update_communities,
    "update_community_reports": wf.run_update_community_reports,
    "update_text_embeddings": wf.run_update_text_embeddings,
    "update_clean_state": wf.run_update_clean_state,
}


async def _finalize(state: IndexState) -> IndexState:
    """
    Finalization node.

    The engine is responsible for calling pipeline_end and writing stats/context.
    Here we simply return the state unchanged.
    """
    return state


WorkflowNode = Callable[[IndexState], Awaitable[IndexState]]


def _workflow_node(
    name: str,
    fn: Callable[[GraphRagConfig, PipelineRunContext], Awaitable[WorkflowFunctionOutput]],
) -> WorkflowNode:
    """Wrap an upstream workflow function into a LangGraph node."""

    async def _runner(state: IndexState) -> IndexState:
        context: PipelineRunContext = state["context"]
        config: GraphRagConfig = state["config"]
        callbacks = context.callbacks
        callbacks.workflow_start(name, None)
        try:
            result = await fn(config, context)
            callbacks.workflow_end(name, result)
            outputs = list(state.get("pipeline_outputs", []))
            outputs.append(
                PipelineRunResult(
                    workflow=name,
                    result=result.result,
                    state=context.state,
                    errors=None,
                )
            )
            state["pipeline_outputs"] = outputs  # type: ignore[index]
            if result.stop:
                state["stop"] = True  # type: ignore[index]
        except Exception as exc:  # pragma: no cover - parity with upstream runner
            callbacks.workflow_end(name, None)
            outputs = list(state.get("pipeline_outputs", []))
            outputs.append(
                PipelineRunResult(
                    workflow=name,
                    result=None,
                    state=context.state,
                    errors=[exc],
                )
            )
            state["pipeline_outputs"] = outputs  # type: ignore[index]
            errors = list(state.get("errors", []))
            errors.append(exc)
            state["errors"] = errors  # type: ignore[index]
            state["stop"] = True  # type: ignore[index]
        return state

    return _runner


def _linear_graph(workflow_names: Iterable[str]) -> CompiledIndexGraph:
    """Compile a LangGraph with explicit linear edges for the given workflow order."""
    names = tuple(workflow_names)
    if not names:
        msg = "Index graph requires at least one workflow."
        raise ValueError(msg)

    graph = StateGraph(IndexState)

    # workflow nodes
    for name in names:
        graph.add_node(name, _workflow_node(name, WORKFLOW_FNS[name]))

    graph.add_node("finalize", _finalize)

    # wiring (explicit linear chain)
    graph.set_entry_point(names[0])
    prev = names[0]
    for wf_name in names[1:]:
        graph.add_edge(prev, wf_name)
        prev = wf_name
    graph.add_edge(prev, "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def build_standard_index_graph() -> CompiledIndexGraph:
    """Standard pipeline graph (LLM-heavy)."""
    return _linear_graph(
        (
            "load_input_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph",
            "finalize_graph",
            "extract_covariates",
            "create_communities",
            "create_final_text_units",
            "create_community_reports",
            "generate_text_embeddings",
        ),
    )


def build_fast_index_graph() -> CompiledIndexGraph:
    """Fast pipeline graph (NLP + LLM)."""
    return _linear_graph(
        (
            "load_input_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph_nlp",
            "prune_graph",
            "finalize_graph",
            "create_communities",
            "create_final_text_units",
            "create_community_reports_text",
            "generate_text_embeddings",
        ),
    )


def build_standard_update_index_graph() -> CompiledIndexGraph:
    """Standard update pipeline graph."""
    return _linear_graph(
        (
            "load_update_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph",
            "finalize_graph",
            "extract_covariates",
            "create_communities",
            "create_final_text_units",
            "create_community_reports",
            "generate_text_embeddings",
            "update_final_documents",
            "update_entities_relationships",
            "update_text_units",
            "update_covariates",
            "update_communities",
            "update_community_reports",
            "update_text_embeddings",
            "update_clean_state",
        ),
    )


def build_fast_update_index_graph() -> CompiledIndexGraph:
    """Fast update pipeline graph."""
    return _linear_graph(
        (
            "load_update_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph_nlp",
            "prune_graph",
            "finalize_graph",
            "create_communities",
            "create_final_text_units",
            "create_community_reports_text",
            "generate_text_embeddings",
            "update_final_documents",
            "update_entities_relationships",
            "update_text_units",
            "update_covariates",
            "update_communities",
            "update_community_reports",
            "update_text_embeddings",
            "update_clean_state",
        ),
    )


def build_index_graph(method: IndexingMethod | str = IndexingMethod.Standard) -> CompiledIndexGraph:
    """
    Convenience selector returning the compiled graph for a given indexing method.
    """
    method_val = method.value if isinstance(method, IndexingMethod) else method
    if method_val == IndexingMethod.Standard.value:
        return build_standard_index_graph()
    if method_val == IndexingMethod.Fast.value:
        return build_fast_index_graph()
    if method_val == IndexingMethod.StandardUpdate.value:
        return build_standard_update_index_graph()
    if method_val == IndexingMethod.FastUpdate.value:
        return build_fast_update_index_graph()
    msg = f"Unsupported indexing method: {method}"
    raise ValueError(msg)

