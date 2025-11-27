"""Configuration dataclasses for GraphRAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class IndexConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    community_resolution: float = 1.0
    community_levels: int = 1
    vector_store_dir: Path = Path(".graph_index")
    max_summary_tokens: int = 512
    persist_graph: bool = True


@dataclass
class QueryConfig:
    top_k_communities: int = 5
    top_k_entities: int = 8
    top_k_text_units: int = 8
    context_token_budget: int = 2048
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    default_mode: Literal["auto", "global", "local", "basic"] = "auto"
    vector_store_dir: Optional[Path] = None
    temperature: float = 0.1
    max_answer_tokens: int = 512
