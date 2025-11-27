"""Query pipeline powered by LangGraph."""
from __future__ import annotations

from typing import Callable

from langgraph.graph import END, StateGraph

from ..config import QueryConfig
from ..types import Document
from .basic_search import basic_search_node
from .global_search import global_search_node
from .local_search import local_search_node
from .routing import decide_mode
from .state import QueryState


def compose_prompt(state: QueryState) -> QueryState:
    """Compose the final prompt/context string."""

    state.context = state.context.strip()
    return state


def llm_answer_node(state: QueryState) -> QueryState:
    """Synthesize an answer from the assembled context.

    This implementation keeps things offline-friendly by using a deterministic
    template rather than making network calls. Users can swap this with an
    actual LLM call in downstream applications.
    """

    context = state.context or "No context available."
    state.answer = f"Question: {state.question}\nContext: {context}\nAnswer: This is a synthesized response based on the retrieved context."
    return state


CompiledQueryGraph = Callable[[QueryState], QueryState]


def build_query_graph(config: QueryConfig) -> StateGraph:
    graph = StateGraph(QueryState)
    graph.add_node("route", decide_mode)
    graph.add_node("global_search", global_search_node)
    graph.add_node("local_search", local_search_node)
    graph.add_node("basic_search", basic_search_node)
    graph.add_node("compose", compose_prompt)
    graph.add_node("answer", llm_answer_node)

    graph.set_entry_point("route")

    def _route(state: QueryState) -> str:
        return state.mode

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "global": "global_search",
            "local": "local_search",
            "basic": "basic_search",
            "auto": "basic_search",
        },
    )
    graph.add_edge("global_search", "compose")
    graph.add_edge("local_search", "compose")
    graph.add_edge("basic_search", "compose")
    graph.add_edge("compose", "answer")
    graph.add_edge("answer", END)

    return graph


def run_query(graph: StateGraph, question: str, config: QueryConfig, index_store) -> QueryState:
    initial = QueryState(
        question=question,
        mode=config.default_mode,
        config=config,
        index_store=index_store,
    )
    compiled = graph.compile()
    result = compiled.invoke(initial)
    return result


__all__ = ["build_query_graph", "run_query", "compose_prompt", "llm_answer_node"]
