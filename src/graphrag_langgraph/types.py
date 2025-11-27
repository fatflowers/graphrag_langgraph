"""Shared domain models for GraphRAG."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RecordBase:
    def dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Document(RecordBase):
    id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextUnit(RecordBase):
    id: str
    source_doc_id: str
    position: int
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity(RecordBase):
    id: str
    name: str
    type: str = "unknown"
    description: str = ""
    text_unit_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation(RecordBase):
    id: str
    head_entity: str
    relation_type: str
    tail_entity: str
    evidence_text_unit_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Claim(RecordBase):
    id: str
    text: str
    entity_ids: List[str] = field(default_factory=list)
    text_unit_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Community(RecordBase):
    id: str
    level: int
    member_entity_ids: List[str]
    parent_community_id: Optional[str] = None


@dataclass
class CommunitySummary(RecordBase):
    community_id: str
    level: int
    summary_text: str
    report_text: Optional[str] = None
    referenced_entities: List[str] = field(default_factory=list)
    referenced_text_units: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    id: str
    score: float
    metadata: Dict[str, Any]
