"""Configuration models for the GraphRAG-LangGraph pipeline."""

from enum import Enum
from typing import List, Optional

from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, Field

from graphrag_langgraph2.models.loader import LoaderConfig


class ChunkerType(str, Enum):
    """Supported text splitter types."""

    RECURSIVE_CHARACTER = "recursive_character"
    CHARACTER = "character"
    TOKEN = "token"


class ChunkerConfig(BaseModel):
    """Configuration for document chunking."""

    splitter_type: ChunkerType = Field(
        default=ChunkerType.RECURSIVE_CHARACTER,
        description="The type of text splitter to use.",
    )
    # Common parameters
    chunk_size: int = Field(default=1000, description="Target size of each chunk.")
    chunk_overlap: int = Field(
        default=200,
        description="Number of overlapping characters/tokens between chunks.",
    )
    # Character-based options
    separators: Optional[List[str]] = Field(
        default=None,
        description="For recursive splitter: priority-ordered separators. "
        + "For character splitter: first element is used as the separator.",
    )
    keep_separator: bool = Field(
        default=False,
        description="Whether to keep the separator in the resulting chunks (character splitter).",
    )
    strip_whitespace: bool = Field(
        default=True,
        description="Whether to strip leading/trailing whitespace from chunks (character/recursive).",
    )
    # Token-based options
    encoding_name: Optional[str] = Field(
        default="cl100k_base",
        description="Encoding name used by the token splitter. Ignored if model_name is provided.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name to infer tokenizer configuration for the token splitter.",
    )


class GraphRagConfig(BaseModel):
    """Top-level configuration container for the GraphRAG-LangGraph pipeline."""

    loader_config: LoaderConfig
    chunker_config: ChunkerConfig = Field(
        default_factory=ChunkerConfig,
        description="Document chunking configuration.",
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "GraphRagConfig":
        """Create a GraphRagConfig from a RunnableConfig."""
        cfg_dict = config.get("configurable", {})
        if cfg_dict is None:
            raise ValueError("No configurable fields found in the RunnableConfig.")
        return cls(**cfg_dict)
