"""Routing logic for deciding query mode."""
from __future__ import annotations

from .state import QueryState


GLOBAL_HINTS = {"overview", "summary", "broad", "global"}
LOCAL_HINTS = {"who", "what", "where", "entity", "relationship"}


def decide_mode(state: QueryState) -> QueryState:
    """Choose the query mode using lightweight heuristics."""

    if state.mode != "auto":
        return state
    question_lower = state.question.lower()
    if any(h in question_lower for h in GLOBAL_HINTS):
        state.mode = "global"
    elif any(h in question_lower for h in LOCAL_HINTS):
        state.mode = "local"
    else:
        state.mode = "basic"
    return state


__all__ = ["decide_mode"]
