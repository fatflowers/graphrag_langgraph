"""
Loader node for the GraphRAG-LangGraph pipeline.
"""

from langchain_community.document_loaders import (
    DirectoryLoader,
    S3DirectoryLoader,
    S3FileLoader,
    UnstructuredFileLoader,
)
from langgraph.graph.state import RunnableConfig

from graphrag_langgraph2.models.config import GraphRagConfig
from graphrag_langgraph2.models.index_graph_state import IndexGraphState


def load_input(_: IndexGraphState, config: RunnableConfig):
    """Load input documents from the configured source."""
    loader_config = GraphRagConfig.from_runnable_config(config).loader_config

    if loader_config.s3_dir:
        loader = S3DirectoryLoader(
            bucket=loader_config.s3_bucket, prefix=loader_config.s3_dir
        )
    elif loader_config.s3_key:
        loader = S3FileLoader(bucket=loader_config.s3_bucket, key=loader_config.s3_key)
    elif loader_config.file_path:
        loader = UnstructuredFileLoader(loader_config.file_path)
    elif loader_config.file_dir:
        loader = DirectoryLoader(loader_config.file_dir)
    else:
        raise ValueError("No loader configuration provided.")

    return {"input_documents": loader.load()}
