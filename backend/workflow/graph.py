"""
workflow.graph
==============
LangGraph StateGraph assembly and compilation.

Execution DAG
-------------

    START
      │
      ▼
  router_node          (intent classification — "vector" or "graph")
      │
      ├──[route="vector"]──► vector_node   (Qdrant dense retrieval)
      │                          │
      └──[route="graph"]──► graph_node    (Neo4j Cypher walk)
                                 │
                        (both paths merge here)
                                 │
                                 ▼
                          compress_node    (LLMLingua boilerplate removal)
                                 │
                                 ▼
                          generate_node   (Poolside AI → AuditReport)
                                 │
                                 ▼
                               END

Public API
----------
- ``build_audit_graph()`` — Returns a compiled, directly-invokable graph.
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END  # type: ignore[import]

from workflow.state import AuditState
from workflow.edges import route_after_router
from workflow.nodes.router_node import router_node
from workflow.nodes.vector_node import vector_node
from workflow.nodes.graph_node import graph_node
from workflow.nodes.compress_node import compress_node
from workflow.nodes.generate_node import generate_node

logger = logging.getLogger(__name__)

# Module-level compiled graph cache — built once per process.
_COMPILED_GRAPH = None


def build_audit_graph():
    """
    Assemble and compile the Vampter LangGraph audit orchestration graph.

    The graph is compiled once and cached at the module level.  Subsequent
    calls return the cached instance with no overhead.

    Returns
    -------
    CompiledStateGraph
        A LangGraph compiled graph ready to be invoked via
        ``await graph.ainvoke(state)``.
    """
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH

    logger.info("Assembling LangGraph audit orchestration graph ...")

    builder = StateGraph(AuditState)

    # ── Register nodes ──────────────────────────────────────────────────────
    builder.add_node("router_node", router_node)
    builder.add_node("vector_node", vector_node)
    builder.add_node("graph_node", graph_node)
    builder.add_node("compress_node", compress_node)
    builder.add_node("generate_node", generate_node)

    # ── Edges ────────────────────────────────────────────────────────────────
    # Entry point
    builder.add_edge(START, "router_node")

    # Router → retrieval branch (conditional)
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "vector_node": "vector_node",
            "graph_node": "graph_node",
        },
    )

    # Both retrieval branches converge on compress_node
    builder.add_edge("vector_node", "compress_node")
    builder.add_edge("graph_node", "compress_node")

    # Linear tail: compress → generate → END
    builder.add_edge("compress_node", "generate_node")
    builder.add_edge("generate_node", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    _COMPILED_GRAPH = builder.compile()
    logger.info("LangGraph audit graph compiled successfully.")
    return _COMPILED_GRAPH
