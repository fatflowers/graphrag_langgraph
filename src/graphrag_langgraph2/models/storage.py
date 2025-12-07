from enum import Enum

class StorageType(str, Enum):
    """The storage type for the pipeline."""

    file = "file"
    """The file storage type."""
    memory = "memory"
    """The memory storage type."""

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.enums import StorageType


class StorageConfig(BaseModel):
    """The default configuration section for storage."""

    type: StorageType | str = Field(
        description="The storage type to use.",
        default=StorageType.file,
    )
    base_dir: str = Field(
        description="The base directory for the output.",
        default="output",
    )

    # Validate the base dir for multiple OS (use Path)
    # if not using a cloud storage type.
    @field_validator("base_dir", mode="before")
    @classmethod
    def validate_base_dir(cls, value, info):
        """Ensure that base_dir is a valid filesystem path when using local storage."""
        # info.data contains other field values, including 'type'
        if info.data.get("type") != StorageType.file:
            return value
        return str(Path(value))