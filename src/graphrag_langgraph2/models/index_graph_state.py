from typing import TypedDict

from langchain_core.documents import Document

from graphrag_langgraph2.models.config import GraphRagConfig


class IndexGraphState(TypedDict, total=False):
    """State for the index graph."""

    config: GraphRagConfig
    input_documents: list[Document]
    chunked_documents: list[Document]
