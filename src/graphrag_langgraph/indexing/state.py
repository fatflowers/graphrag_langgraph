"""Dataclass for indexing state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import IndexConfig
from ..graph_store import GraphIndexStore
from ..types import Claim, Community, CommunitySummary, Document, Entity, Relation, TextUnit


@dataclass
class IndexState:
    raw_docs: List[Document]
    config: IndexConfig
    text_units: List[TextUnit] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    communities: List[Community] = field(default_factory=list)
    community_summaries: List[CommunitySummary] = field(default_factory=list)
    index_store: Optional[GraphIndexStore] = None
    graph_extraction_prompts: List[str] = field(default_factory=list)
    claim_extraction_prompts: List[str] = field(default_factory=list)
    community_report_prompts: List[str] = field(default_factory=list)
