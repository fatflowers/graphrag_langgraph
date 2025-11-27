"""Minimal stub of langgraph.graph for offline testing."""
from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Optional

END = "__end__"


class CompiledGraph:
    def __init__(self, graph: "StateGraph") -> None:
        self.graph = graph

    def invoke(self, state: Any) -> Any:
        current = self.graph.entry_point
        while current != END:
            node_fn = self.graph.nodes[current]
            state = node_fn(state)
            if current in self.graph.conditional_edges:
                router, mapping = self.graph.conditional_edges[current]
                key = router(state)
                current = mapping.get(key, self.graph.default_next.get(current, END))
            else:
                current = self.graph.default_next.get(current, END)
        return state


class StateGraph:
    def __init__(self, state_type: Any) -> None:
        self.state_type = state_type
        self.nodes: Dict[str, Callable[[Any], Any]] = {}
        self.default_next: Dict[str, str] = {}
        self.entry_point: Optional[str] = None
        self.conditional_edges: Dict[str, tuple[Callable[[Any], Hashable], Dict[Hashable, str]]] = {}

    def add_node(self, name: str, func: Callable[[Any], Any]) -> None:
        self.nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def add_edge(self, source: str, target: str) -> None:
        self.default_next[source] = target

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[Any], Hashable],
        mapping: Dict[Hashable, str],
    ) -> None:
        self.conditional_edges[source] = (router, mapping)

    def compile(self) -> CompiledGraph:
        return CompiledGraph(self)


__all__ = ["StateGraph", "END"]
