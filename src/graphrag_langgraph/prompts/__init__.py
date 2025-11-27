"""Lightweight prompt templates aligned with the official GraphRAG repo."""

from .index_prompts import (
    build_claims_prompt,
    build_community_report_prompt,
    build_graph_extraction_prompt,
)

__all__ = [
    "build_graph_extraction_prompt",
    "build_claims_prompt",
    "build_community_report_prompt",
]
