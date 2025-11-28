"""
LangGraph orchestration layer for the official GraphRAG package.

- Internal map of the upstream implementation (for feature parity):
- Public entry points: the `graphrag` CLI (`graphrag.cli.main`) exposes `init`, `index`, `update`, `prompt-tune`, and `query` subcommands; programmatic APIs live in `graphrag.api` (`build_index`, `global_search`/`local_search`/`basic_search`/`drift_search`, and streaming variants).
- Indexing wiring: `PipelineFactory` (`graphrag.index.workflows.factory`) builds ordered workflow lists (load_input_documents -> create_base_text_units -> create_final_documents -> extract_graph/extract_graph_nlp -> prune/finalize -> covariates/communities/final_text_units/community_reports -> generate_text_embeddings, plus update-specific steps). Orchestration is handled by `graphrag.index.run.run_pipeline`, with storage/cache creation in `graphrag.index.run.utils.create_run_context`.
- Query wiring: CLI helpers (`graphrag.cli.query`) load parquet artifacts via `graphrag.utils.storage`, then delegate to `graphrag.api.query`, which adapts index outputs using `graphrag.query.indexer_adapters`, loads prompts via `graphrag.utils.api.load_search_prompt`, and instantiates search engines via `graphrag.query.factory` (global/local/basic/drift) backed by implementations in `graphrag.query.structured_search.*`.
- Prompt templates: indexing prompts live in `graphrag/prompts/index/*` (graph/entity extraction, claims, community reports, description summarization); query prompts live in `graphrag/prompts/query/*`; prompt tuning assets are under `graphrag/prompt_tune`.

This package keeps the upstream domain logic and prompts, replacing only the workflow/orchestration layer with LangGraph graphs.
"""

from .engine import GraphRAGEngine

__all__ = ["GraphRAGEngine"]
