from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime

from graphrag_langgraph2.models.storage import StorageConfig

class InputFileType(str, Enum):
    """The input file type for the pipeline."""

    csv = "csv"
    """The CSV input type."""
    text = "text"
    """The text input type."""
    json = "json"
    """The JSON input type."""

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'

# class InputDocument(BaseModel):
#     """Pydantic model representing the output structure of create_final_documents function."""

#     id: str = Field(..., description="Unique document identifier")
#     human_readable_id: int = Field(
#         ..., description="Human readable ID (integer index)"
#     )
#     title: Optional[str] = Field(None, description="Document title")
#     text: Optional[str] = Field(None, description="Document text content")
#     text_unit_ids: Optional[list[str]] = Field(
#         default_factory=list, description="List of associated text unit IDs"
#     )
#     creation_date: Optional[datetime] = Field(None, description="Document creation date")
#     metadata: Optional[dict[str, Any]] = Field(
#         default_factory=dict, description="Additional document metadata"
#     )

class InputConfig(BaseModel):
    storage: StorageConfig = Field(
        description="The storage configuration to use for reading input documents.",
        default=StorageConfig(
            base_dir="input",
        ),
    )
    file_type: InputFileType = Field(
        description="The input file type to use.",
        default=InputFileType.text,
    )
    encoding: str = Field(
        description="The input file encoding to use.",
        default="utf-8",
    )