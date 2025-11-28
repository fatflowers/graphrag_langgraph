"""
Thin wrappers around the upstream GraphRAG configuration loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

__all__ = ["GraphRagConfig", "load_graph_rag_config"]


def load_graph_rag_config(
    config_path: str | Path | None = None,
    root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> GraphRagConfig:
    """
    Load a `GraphRagConfig` using the official loader.

    Parameters
    ----------
    config_path:
        Optional path to a YAML configuration file (graphrag.yml). If omitted,
        defaults mirror `graphrag.cli` semantics.
    root:
        Project root; falls back to the current working directory when not provided.
    overrides:
        Optional dot-delimited overrides (e.g., {"output.base_dir": "/tmp/out"}).
    """
    root_path = Path(root) if root is not None else Path()
    cfg_path = Path(config_path) if config_path is not None else None
    return load_config(root_path, cfg_path, overrides or {})
