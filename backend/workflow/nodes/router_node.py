"""
workflow.nodes.router_node
==========================
Intent-classification router node.

Responsibility
--------------
Makes a fast, low-cost Gemini Flash call to categorise the incoming query
into one of two execution branches:

- ``"vector"`` — free-form semantic question, best answered by dense
  Qdrant retrieval over chunk embeddings.
- ``"graph"`` — structural, multi-revision, or contradiction-detection
  question, best answered by a multi-hop Neo4j Cypher walk.

The node writes ``state["route"]`` and returns the modified state.

Requirements
------------
At least one LLM provider must be configured (Gemini or Groq).
If no provider is available, an HTTPException is raised.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import HTTPException, status

from config import settings
from workflow.state import AuditState

logger = logging.getLogger(__name__)

_ROUTER_PROMPT_TEMPLATE = """
You are a query-routing engine for a legal policy document auditing system.

Given the user query below, classify it into EXACTLY ONE of the following categories:
- "vector"  : The question is semantic / free-form and benefits from dense similarity search.
- "graph"   : The question requires multi-hop structural traversal (e.g. comparing revisions,
              detecting clause contradictions across document versions, or policy change timelines).

Respond with ONLY the word "vector" or "graph". Nothing else.

User query: {query}
Company context: {company_name}
Active intents: {intents}
""".strip()


async def router_node(state: AuditState) -> AuditState:
    """
    Classify the query and set ``state["route"]``.

    Uses Gemini or Groq to make a routing decision. No fallback is provided
    if no LLM provider is available - this ensures the system fails explicitly
    when misconfigured.

    Parameters
    ----------
    state:
        Current ``AuditState`` containing ``query``, ``company_name``,
        and ``intents``.

    Returns
    -------
    AuditState
        Updated state with ``route`` set to ``"vector"`` or ``"graph"``.
    """
    query = state.get("query", "")
    company_name = state.get("company_name", "")
    intents = state.get("intents", [])

    gemini_api_key = (
        settings.gemini_api_key.get_secret_value()
        if settings.gemini_api_key
        else None
    )
    groq_api_key = (
        settings.groq_api_key.get_secret_value()
        if settings.groq_api_key
        else None
    )

    # Build prompt for LLM routing
    prompt = _ROUTER_PROMPT_TEMPLATE.format(
        query=query,
        company_name=company_name,
        intents=", ".join(intents) if intents else "none",
    )

    # ── Primary: Gemini ──────────────────────────────────────────────────
    if gemini_api_key:
        try:
            import google.genai as genai  # type: ignore[import]

            client = genai.Client(api_key=gemini_api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            raw = response.text.strip().lower()
            if raw in ("vector", "graph"):
                route = raw
                logger.info("Router decision (Gemini): route='%s'  query='%s'", route, query[:60])
                return {**state, "route": route}
            logger.warning(
                "Router Gemini returned unexpected value '%s'; raising error.", raw
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Router LLM returned invalid value: {raw}"
            )
        except HTTPException:
            raise
        except Exception as gemini_exc:
            logger.warning(
                "Router Gemini call failed (%s: %s) — pivoting to Groq failover.",
                type(gemini_exc).__name__,
                gemini_exc,
            )

    # ── Failover: Groq ───────────────────────────────────────────────────
    if groq_api_key:
        try:
            from groq import Groq  # type: ignore[import]

            groq_client = Groq(api_key=groq_api_key)
            groq_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            raw = groq_response.choices[0].message.content.strip().lower()
            if raw in ("vector", "graph"):
                logger.info("Router decision (Groq): route='%s'  query='%s'", raw, query[:60])
                return {**state, "route": raw}
            logger.warning(
                "Groq router returned unexpected value '%s'; raising error.", raw
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Router LLM returned invalid value: {raw}"
            )
        except HTTPException:
            raise
        except Exception as groq_exc:
            logger.error(
                "Groq failover router also failed (%s: %s).",
                type(groq_exc).__name__,
                groq_exc,
            )

    # ── No LLM provider available ────────────────────────────────────────
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="LLM_PROVIDER_MISSING"
    )

