"""State for the query pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from ..config import QueryConfig
from ..graph_store import GraphIndexStore
from ..types import RetrievalResult


@dataclass
class QueryState:
    question: str
    config: QueryConfig
    index_store: GraphIndexStore
    mode: Literal["auto", "global", "local", "basic"] = "auto"
    retrieved_communities: List[RetrievalResult] = field(default_factory=list)
    retrieved_entities: List[RetrievalResult] = field(default_factory=list)
    retrieved_text_units: List[RetrievalResult] = field(default_factory=list)
    context: str = ""
    answer: Optional[str] = None
