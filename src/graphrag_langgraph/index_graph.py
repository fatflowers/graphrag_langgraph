"""
LangGraph representation of the GraphRAG indexing pipelines (standard/fast + update variants).

Each upstream workflow function becomes its own LangGraph node; there is no single
“prepare” or routing node. The graph is a fixed linear chain per pipeline variant,
mirroring the official workflow order.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Awaitable, Callable, Iterable, Protocol, TypedDict, cast

from langgraph.graph import END, StateGraph

from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index import workflows as wf
from graphrag.index.run.run_pipeline import _copy_previous_output, _dump_json
from graphrag.index.run.utils import create_run_context
from graphrag.cache.pipeline_cache import PipelineCache
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from graphrag.index.typing.context import PipelineRunContext
from graphrag.storage.pipeline_storage import PipelineStorage
from graphrag.index.typing.state import PipelineState
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.api import create_cache_from_config, create_storage_from_config
from graphrag.utils.storage import write_table_to_storage

from .config import load_graph_rag_config
from .state import IndexState

logger = logging.getLogger(__name__)

class CompiledIndexGraph(Protocol):
    """Minimal protocol for compiled LangGraph objects used here."""

    def invoke(self, state: IndexState) -> IndexState:
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


# ---- Graph node implementations ------------------------------------------------

class _IOState(TypedDict, total=False):
    input_storage: PipelineStorage
    output_storage: PipelineStorage
    previous_storage: PipelineStorage
    cache: PipelineCache


def _load_config(state: IndexState) -> IndexState:
    """Load or normalize the GraphRAG config into the state."""
    if isinstance(state.get("config"), GraphRagConfig):
        return state
    cfg_path = state.get("config_path")
    root = state.get("root")
    overrides = state.get("config_overrides", {})
    loaded = load_graph_rag_config(cfg_path, root=root, overrides=overrides)
    return {**state, "config": loaded}


def _init_io(state: IndexState) -> IndexState:
    """Create storages and cache (no IO side-effects)."""
    config = state["config"]
    input_storage = create_storage_from_config(config.input.storage)
    output_storage = create_storage_from_config(config.output)
    cache = create_cache_from_config(config.cache, config.root_dir)
    io_state: _IOState = {
        "input_storage": input_storage,
        "output_storage": output_storage,
        "cache": cache,
    }
    return {**state, "io": io_state}


async def _load_context_state(state: IndexState) -> IndexState:
    """Load context.json (if any) and merge additional_context."""
    io = cast(_IOState, state["io"])
    output_storage = io["output_storage"]
    raw = await output_storage.get("context.json")
    ctx_state = json.loads(raw) if raw else {}
    additional_context = state.get("additional_context")
    if additional_context:
        ctx_state.setdefault("additional_context", {}).update(additional_context)
    return {**state, "ctx_state": ctx_state}


async def _prepare_update(state: IndexState) -> IndexState:
    """Handle update-mode storages and backups when is_update_run=True."""
    if not state.get("is_update_run"):
        return state
    config = state["config"]
    io = cast(_IOState, state["io"])
    output_storage = io["output_storage"]
    update_storage = create_storage_from_config(config.update_index_output)
    update_timestamp = time.strftime("%Y%m%d-%H%M%S")
    timestamped_storage = update_storage.child(update_timestamp)
    delta_storage = timestamped_storage.child("delta")
    previous_storage = timestamped_storage.child("previous")
    await _copy_previous_output(output_storage, previous_storage)
    io["output_storage"] = delta_storage
    io["previous_storage"] = previous_storage
    state["ctx_state"]["update_timestamp"] = update_timestamp  # type: ignore
    return state


async def _write_input_documents_if_any(state: IndexState) -> IndexState:
    """If input_documents provided, persist them for downstream workflows."""
    input_docs = state.get("input_documents")
    if input_docs is None:
        return state
    io = cast(_IOState, state["io"])
    await write_table_to_storage(input_docs, "documents", io["output_storage"])
    return state


def _build_run_context(state: IndexState) -> IndexState:
    """Instantiate PipelineRunContext."""
    io = cast(_IOState, state["io"])
    callbacks = state.get("callbacks") or NoopWorkflowCallbacks()
    run_context: PipelineRunContext = create_run_context(
        input_storage=io["input_storage"],
        output_storage=io["output_storage"],
        previous_storage=io.get("previous_storage"),
        cache=io["cache"],
        callbacks=callbacks,
        state=cast(PipelineState, state.get("ctx_state", {})),
    )
    return {
        **state,
        "context": run_context,
        "artifacts": {"output_storage": run_context.output_storage},
    }


def _start_pipeline(state: IndexState) -> IndexState:
    """Invoke pipeline_start callback and prime run stats."""
    workflows = state["pipeline_names"]
    context: PipelineRunContext = state["context"]
    context.callbacks.pipeline_start(list(workflows))
    state["start_time"] = time.time()
    state["pipeline_outputs"] = []
    return state


WorkflowNode = Callable[[IndexState], Awaitable[IndexState]]


def _workflow_node(
    name: str,
    fn: Callable[[GraphRagConfig, PipelineRunContext], Awaitable[WorkflowFunctionOutput]],
) -> WorkflowNode:
    """Wrap an upstream workflow function into a LangGraph node."""

    async def _runner(state: IndexState) -> IndexState:
        context = state["context"]
        config = state["config"]
        callbacks = context.callbacks
        start = time.time()
        callbacks.workflow_start(name, None)
        try:
            result = await fn(config, context)
            callbacks.workflow_end(name, result)
            context.stats.workflows[name] = {"overall": time.time() - start}
            outputs = list(state.get("pipeline_outputs", []))
            outputs.append(
                PipelineRunResult(
                    workflow=name,
                    result=result.result,
                    state=context.state,
                    errors=None,
                )
            )
            state["pipeline_outputs"] = outputs  # type: ignore
            if result.stop:
                state["stop"] = True  # type: ignore
        except Exception as exc:  # pragma: no cover - parity with upstream runner
            callbacks.workflow_end(name, None)
            context.stats.workflows[name] = {"overall": time.time() - start}
            errors = list(state.get("errors", []))
            errors.append(exc)
            outputs = list(state.get("pipeline_outputs", []))
            outputs.append(
                PipelineRunResult(
                    workflow=name,
                    result=None,
                    state=context.state,
                    errors=[exc],
                )
            )
            state["errors"] = errors  # type: ignore
            state["stop"] = True  # type: ignore
        return state

    return _runner


async def _finalize(state: IndexState) -> IndexState:
    """Finalize stats/state persistence and run pipeline_end callbacks."""
    context: PipelineRunContext = state["context"]
    callbacks = context.callbacks or NoopWorkflowCallbacks()
    context.stats.total_runtime = time.time() - state.get("start_time", time.time())
    await _dump_json(context)
    callbacks.pipeline_end(state.get("pipeline_outputs", []))
    return state


# ---- Graph builders -----------------------------------------------------------

def _linear_graph(workflow_names: Iterable[str], is_update_run: bool) -> CompiledIndexGraph:
    """Compile a LangGraph with explicit linear edges for the given workflow order."""
    names = tuple(workflow_names)
    graph = StateGraph(IndexState)

    # prep nodes
    graph.add_node("load_config", _load_config)
    graph.add_node("init_io", _init_io)
    graph.add_node("load_context_state", _load_context_state)
    graph.add_node("prepare_update", _prepare_update)
    graph.add_node("write_input_documents", _write_input_documents_if_any)
    graph.add_node("build_run_context", _build_run_context)
    graph.add_node("start_pipeline", _start_pipeline)

    # workflow nodes
    for name in names:
        graph.add_node(name, _workflow_node(name, WORKFLOW_FNS[name]))

    graph.add_node("finalize", _finalize)

    # wiring (explicit linear chain)
    graph.set_entry_point("load_config")
    graph.add_edge("load_config", "init_io")
    graph.add_edge("init_io", "load_context_state")
    if is_update_run:
        graph.add_edge("load_context_state", "prepare_update")
        graph.add_edge("prepare_update", "write_input_documents")
    else:
        graph.add_edge("load_context_state", "write_input_documents")
    graph.add_edge("write_input_documents", "build_run_context")
    graph.add_edge("build_run_context", "start_pipeline")

    # workflow edges
    prev = "start_pipeline"
    for wf_name in names:
        graph.add_edge(prev, wf_name)
        prev = wf_name
    graph.add_edge(prev, "finalize")
    graph.add_edge("finalize", END)

    # set pipeline names for later nodes
    def _set_pipeline_names(state: IndexState) -> IndexState:
        return {
            **state,
            "pipeline_names": list(names),
            "is_update_run": is_update_run,
        }

    graph.add_node("set_pipeline_names", _set_pipeline_names)
    graph.add_edge("load_config", "set_pipeline_names")
    graph.add_edge("set_pipeline_names", "init_io")

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
        is_update_run=False,
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
        is_update_run=False,
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
        is_update_run=True,
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
        is_update_run=True,
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
