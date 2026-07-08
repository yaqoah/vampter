"""
workflow.edges
==============
Conditional edge routing functions for the LangGraph StateGraph.

The router node sets ``state["route"]`` to either ``"vector"`` or ``"graph"``.
The edge function ``route_after_router`` reads this value and returns the
name of the next node to execute.
"""

from __future__ import annotations

from typing import Literal

from workflow.state import AuditState


def route_after_router(
    state: AuditState,
) -> Literal["vector_node", "graph_node"]:
    """
    Determine which retrieval node to invoke based on router classification.

    Parameters
    ----------
    state:
        Current ``AuditState`` after the router node has run.

    Returns
    -------
    Literal["vector_node", "graph_node"]
        The name of the next node to execute.
    """
    route = state.get("route", "vector")
    if route == "graph":
        return "graph_node"
    return "vector_node"
