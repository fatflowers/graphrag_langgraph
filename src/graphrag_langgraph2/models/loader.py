"""Loader models and validation for the GraphRAG-LangGraph pipeline."""

from pathlib import Path

from pydantic import BaseModel, Field


class LoaderConfig(BaseModel):
    """The loader configuration for the pipeline."""

    s3_bucket: str = Field(
        description="The S3 bucket to use.",
        default=None,
    )
    s3_key: str = Field(
        description="The S3 key to use. If s3_dir is provided, will omit this.",
        default=None,
    )
    s3_dir: str = Field(
        description="The S3 directory to use. If provided, "
        + "will load all files in the directory.",
        default=None,
    )
    file_path: Path = Field(
        description="The file path to use. If file_dir is provided, will omit this.",
        default=None,
    )
    file_dir: Path = Field(
        description="The file directory to use. If file_path is provided, "
        + "will load all files in the directory.",
        default=None,
    )
