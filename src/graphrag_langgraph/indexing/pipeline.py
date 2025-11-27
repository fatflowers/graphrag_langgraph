"""LangGraph pipeline for indexing."""
from __future__ import annotations

from typing import Callable, List, Optional

from langgraph.graph import END, StateGraph

from ..config import IndexConfig
from ..types import Document
from .nodes import (
    build_knowledge_graph,
    create_embeddings_and_vector_stores,
    detect_communities,
    extract_entities_relations_claims,
    persist_index,
    split_into_text_units,
    summarize_communities,
)
from .state import IndexState


CompiledGraph = Callable[[IndexState], IndexState]


def build_index_graph(config: IndexConfig) -> StateGraph:
    """Create the LangGraph state machine for indexing."""

    graph = StateGraph(IndexState)
    graph.add_node("split", split_into_text_units)
    graph.add_node("extract", extract_entities_relations_claims)
    graph.add_node("build_graph", build_knowledge_graph)
    graph.add_node("communities", detect_communities)
    graph.add_node("summaries", summarize_communities)
    graph.add_node("embed", create_embeddings_and_vector_stores)
    graph.add_node("persist", persist_index)

    graph.set_entry_point("split")
    graph.add_edge("split", "extract")
    graph.add_edge("extract", "build_graph")
    graph.add_edge("build_graph", "communities")
    graph.add_edge("communities", "summaries")
    graph.add_edge("summaries", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)

    return graph


def run_indexing(graph: StateGraph, documents: List[Document], config: IndexConfig) -> IndexState:
    """Convenience wrapper to run the compiled graph."""

    initial_state = IndexState(raw_docs=documents, config=config)
    compiled = graph.compile()
    final_state = compiled.invoke(initial_state)
    return final_state


__all__ = ["build_index_graph", "run_indexing"]
