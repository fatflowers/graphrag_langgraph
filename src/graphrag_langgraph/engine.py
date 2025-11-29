"""
High-level engine that exposes GraphRAG functionality via LangGraph graphs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping

from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.enums import IndexingMethod
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.run.run_pipeline import _copy_previous_output, _dump_json
from graphrag.index.run.utils import create_run_context
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.state import PipelineState
from graphrag.logger.standard_logging import init_loggers
from graphrag.utils.api import create_cache_from_config, create_storage_from_config
from graphrag.utils.storage import write_table_to_storage

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
        verbose = bool(kwargs.get("verbose", False))
        additional_context = kwargs.get("additional_context")
        input_documents = kwargs.get("input_documents")
        memory_profile = bool(kwargs.get("memory_profile", False))
        callbacks = kwargs.get("callbacks") or NoopWorkflowCallbacks()

        if memory_profile:
            msg = "Memory profiling is not supported in the LangGraph pipeline."
            raise NotImplementedError(msg)

        # Initialize logging consistent with the upstream implementation
        init_loggers(config=self.config, verbose=verbose)

        method_val = _normalize_indexing_method(method, is_update)
        graph = self.index_graphs.get(method_val)
        if graph is None:
            msg = f"Unsupported indexing method: {method}"
            raise ValueError(msg)

        # Build PipelineRunContext (I/O + cache + state)
        config = self.config
        input_storage = create_storage_from_config(config.input.storage)
        output_storage = create_storage_from_config(config.output)
        cache = create_cache_from_config(config.cache, config.root_dir)

        pipeline_state: PipelineState = {}
        if additional_context:
            pipeline_state["additional_context"] = additional_context  # type: ignore[index]

        if is_update:
            # For now, reuse the same output path; full update semantics can be layered later.
            logger = __import__("logging").getLogger(__name__)
            logger.warning("Update mode is not fully supported in LangGraph engine; running against primary output.")

        if input_documents is not None:
            asyncio.run(
                write_table_to_storage(input_documents, "documents", output_storage)
            )

        run_context = create_run_context(
            input_storage=input_storage,
            output_storage=output_storage,
            cache=cache,
            callbacks=callbacks,
            state=pipeline_state,
        )

        # Fire pipeline_start before graph execution using known workflow order
        from .index_graph import WORKFLOW_FNS

        workflow_names = list(WORKFLOW_FNS.keys())
        run_context.callbacks.pipeline_start(workflow_names)

        state: IndexState = {
            "config": self.config,
            "context": run_context,
            "pipeline_outputs": [],
        }
        final_state = asyncio.run(graph.ainvoke(state))

        # Finalize: write stats/context and pipeline_end
        asyncio.run(_dump_json(run_context))
        run_context.callbacks.pipeline_end(final_state.get("pipeline_outputs", []))
        return final_state

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
        verbose = bool(kwargs.get("verbose", False))
        community_level = kwargs.get("community_level")

        # Initialize logging consistent with the upstream query API
        init_loggers(config=self.config, verbose=verbose, filename="query.log")

        artifacts = _load_query_artifacts_sync(
            self.config,
            mode=mode,
            community_level=community_level,
        )

        state: QueryState = {
            "config": self.config,
            "question": question,
            "mode": mode,
            "artifacts": artifacts,
            **kwargs,
        }
        return asyncio.run(self.query_graph.ainvoke(state))

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


def _load_query_artifacts_sync(
    config: GraphRagConfig,
    mode: str,
    community_level: int | None,
) -> dict[str, Any]:
    """
    Synchronously load index artifacts required for the selected query mode.
    """
    from graphrag.utils.api import create_storage_from_config
    from graphrag.utils.storage import load_table_from_storage, storage_has_table
    import asyncio

    m = (mode or "auto").lower()
    if m == "auto":
        m = "global"

    async def _load_single() -> dict[str, Any]:
        data: dict[str, Any] = {}
        # Determine which tables are required
        if m == "local":
            required = [
                "communities",
                "community_reports",
                "text_units",
                "relationships",
                "entities",
            ]
            optional = ["covariates"]
        elif m == "basic":
            required, optional = ["text_units"], []
        elif m == "drift":
            required = [
                "entities",
                "communities",
                "community_reports",
                "text_units",
                "relationships",
            ]
            optional = []
        else:  # global
            required, optional = ["entities", "communities", "community_reports"], []

        if config.outputs:
            data["multi-index"] = True
            data["index_names"] = list(config.outputs.keys())
            data["num_indexes"] = len(config.outputs)
            # multi-index lists
            for name, out_cfg in config.outputs.items():
                storage = create_storage_from_config(out_cfg)
                for tbl in required:
                    key = f"{tbl}_list"
                    data.setdefault(key, [])
                    df = await load_table_from_storage(tbl, storage)
                    data[key].append(df)
                for opt in optional:
                    key = f"{opt}_list"
                    data.setdefault(key, [])
                    if await storage_has_table(opt, storage):
                        df = await load_table_from_storage(opt, storage)
                        data[key].append(df)
            return data

        # single-index
        storage = create_storage_from_config(config.output)
        data["multi-index"] = False
        for tbl in required:
            data[tbl] = await load_table_from_storage(tbl, storage)
        for opt in optional:
            if await storage_has_table(opt, storage):
                data[opt] = await load_table_from_storage(opt, storage)
            else:
                data[opt] = None
        return data

    return asyncio.run(_load_single())
