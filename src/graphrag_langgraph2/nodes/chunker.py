"""
Chunker node for the GraphRAG-LangGraph pipeline.
"""

from typing import List

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document
from langgraph.graph.state import RunnableConfig

from graphrag_langgraph2.models.config import (
    ChunkerConfig,
    ChunkerType,
    GraphRagConfig,
)
from graphrag_langgraph2.models.index_graph_state import IndexGraphState


def _build_text_splitter(chunker_config: ChunkerConfig):
    """Instantiate the configured text splitter."""
    if chunker_config.splitter_type == ChunkerType.RECURSIVE_CHARACTER:
        separators = (
            chunker_config.separators
            if chunker_config.separators is not None
            else ["\n\n", "\n", " ", ""]
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=chunker_config.chunk_size,
            chunk_overlap=chunker_config.chunk_overlap,
            separators=separators,
            keep_separator=chunker_config.keep_separator,
            strip_whitespace=chunker_config.strip_whitespace,
        )

    if chunker_config.splitter_type == ChunkerType.CHARACTER:
        separator = (
            chunker_config.separators[0] if chunker_config.separators else "\n\n"
        )
        return CharacterTextSplitter(
            chunk_size=chunker_config.chunk_size,
            chunk_overlap=chunker_config.chunk_overlap,
            separator=separator,
            keep_separator=chunker_config.keep_separator,
            strip_whitespace=chunker_config.strip_whitespace,
        )

    if chunker_config.splitter_type == ChunkerType.TOKEN:
        return TokenTextSplitter(
            chunk_size=chunker_config.chunk_size,
            chunk_overlap=chunker_config.chunk_overlap,
            model_name=chunker_config.model_name,
            encoding_name=(
                chunker_config.encoding_name
                if chunker_config.model_name is None
                else None
            ),
        )

    raise ValueError(f"Unsupported splitter type: {chunker_config.splitter_type}")


def chunk_input(state: IndexGraphState, config: RunnableConfig):
    """Chunk input documents into chunks."""
    rag_config: GraphRagConfig = GraphRagConfig.from_runnable_config(config)
    splitter = _build_text_splitter(rag_config.chunker_config)

    input_docs: List[Document] = state.get("input_documents", []) or []
    if not input_docs:
        return {"chunked_documents": []}

    chunked: List[Document] = splitter.split_documents(input_docs)
    return {"chunked_documents": chunked}
