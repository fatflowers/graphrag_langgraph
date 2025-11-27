"""Shared domain models for GraphRAG."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RecordBase:
    id: str
    short_id: Optional[str] = None

    def dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Document(RecordBase):
    title: str
    text: str
    type: str = "text"
    text_unit_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # keep metadata/attributes in sync for compatibility with the official model
        if self.metadata and not self.attributes:
            self.attributes = dict(self.metadata)
        elif self.attributes and not self.metadata:
            self.metadata = dict(self.attributes)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        title_key: str = "title",
        text_key: str = "text",
        type_key: str = "type",
        text_unit_ids_key: str = "text_unit_ids",
        attributes_key: str = "attributes",
    ) -> "Document":
        return cls(
            id=data[id_key],
            short_id=data.get(short_id_key),
            title=data[title_key],
            text=data[text_key],
            type=data.get(type_key, "text"),
            text_unit_ids=data.get(text_unit_ids_key, []),
            attributes=data.get(attributes_key, {}),
        )


@dataclass
class TextUnit(RecordBase):
    text: str
    document_ids: List[str] = field(default_factory=list)
    position: int = 0
    n_tokens: int = 0
    entity_ids: List[str] = field(default_factory=list)
    relationship_ids: List[str] = field(default_factory=list)
    covariate_ids: Dict[str, List[str]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.metadata and not self.attributes:
            self.attributes = dict(self.metadata)
        elif self.attributes and not self.metadata:
            self.metadata = dict(self.attributes)
        if self.source_doc_id and self.source_doc_id not in self.document_ids:
            self.document_ids.append(self.source_doc_id)

    @property
    def token_count(self) -> int:
        return self.n_tokens

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        text_key: str = "text",
        document_ids_key: str = "document_ids",
        position_key: str = "position",
        n_tokens_key: str = "n_tokens",
        entity_ids_key: str = "entity_ids",
        relationship_ids_key: str = "relationship_ids",
        covariate_ids_key: str = "covariate_ids",
        attributes_key: str = "attributes",
    ) -> "TextUnit":
        return cls(
            id=data[id_key],
            short_id=data.get(short_id_key),
            text=data[text_key],
            document_ids=data.get(document_ids_key, []),
            position=data.get(position_key, 0),
            n_tokens=data.get(n_tokens_key, 0),
            entity_ids=data.get(entity_ids_key, []),
            relationship_ids=data.get(relationship_ids_key, []),
            covariate_ids=data.get(covariate_ids_key, {}),
            attributes=data.get(attributes_key, {}),
        )


@dataclass
class Entity(RecordBase):
    title: str
    type: str = "unknown"
    description: str = ""
    community_ids: List[str] = field(default_factory=list)
    text_unit_ids: List[str] = field(default_factory=list)
    rank: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata and not self.attributes:
            self.attributes = dict(self.metadata)
        elif self.attributes and not self.metadata:
            self.metadata = dict(self.attributes)

    @property
    def name(self) -> str:
        return self.title

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        id_key: str = "id",
        title_key: str = "title",
        short_id_key: str = "human_readable_id",
        type_key: str = "type",
        description_key: str = "description",
        community_ids_key: str = "community",
        text_unit_ids_key: str = "text_unit_ids",
        rank_key: str = "degree",
        attributes_key: str = "attributes",
    ) -> "Entity":
        return cls(
            id=data[id_key],
            title=data[title_key],
            short_id=data.get(short_id_key),
            type=data.get(type_key, "unknown"),
            description=data.get(description_key, ""),
            community_ids=data.get(community_ids_key, []),
            text_unit_ids=data.get(text_unit_ids_key, []),
            rank=data.get(rank_key, 1),
            attributes=data.get(attributes_key, {}),
        )


@dataclass
class Relation(RecordBase):
    source: str
    target: str
    description: str = ""
    relation_type: str = "related_to"
    weight: float = 1.0
    text_unit_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata and not self.attributes:
            self.attributes = dict(self.metadata)
        elif self.attributes and not self.metadata:
            self.metadata = dict(self.attributes)

    @property
    def head_entity(self) -> str:
        return self.source

    @property
    def tail_entity(self) -> str:
        return self.target

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        id_key: str = "id",
        short_id_key: str = "human_readable_id",
        source_key: str = "source",
        target_key: str = "target",
        description_key: str = "description",
        weight_key: str = "weight",
        relation_type_key: str = "relation_type",
        text_unit_ids_key: str = "text_unit_ids",
        attributes_key: str = "attributes",
    ) -> "Relation":
        return cls(
            id=data[id_key],
            short_id=data.get(short_id_key),
            source=data[source_key],
            target=data[target_key],
            description=data.get(description_key, ""),
            weight=data.get(weight_key, 1.0),
            relation_type=data.get(relation_type_key, "related_to"),
            text_unit_ids=data.get(text_unit_ids_key, []),
            attributes=data.get(attributes_key, {}),
        )


@dataclass
class Claim(RecordBase):
    subject_id: str
    text: str
    subject_type: str = "entity"
    covariate_type: str = "claim"
    text_unit_ids: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def entity_ids(self) -> List[str]:
        # Compatibility alias with the original stubbed implementation.
        return [self.subject_id]


@dataclass
class Community(RecordBase):
    title: str
    level: int
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    relationship_ids: List[str] = field(default_factory=list)
    text_unit_ids: List[str] = field(default_factory=list)
    covariate_ids: Dict[str, List[str]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    size: Optional[int] = None
    period: Optional[str] = None

    @property
    def member_entity_ids(self) -> List[str]:
        return self.entity_ids


@dataclass
class CommunitySummary(RecordBase):
    community_id: str
    title: str
    summary: str = ""
    full_content: str = ""
    level: Optional[int] = None
    referenced_entities: List[str] = field(default_factory=list)
    referenced_text_units: List[str] = field(default_factory=list)
    rank: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary_text(self) -> str:
        return self.summary

    @property
    def report_text(self) -> str:
        return self.full_content or self.summary


@dataclass
class RetrievalResult:
    id: str
    score: float
    metadata: Dict[str, Any]
