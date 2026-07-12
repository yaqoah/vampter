"""
workflow.nodes.router_node
===========================
Intent-classification router node.

Responsibility
--------------
Makes a Mistral AI call to categorise the incoming query
into one of two execution branches:

- ``"vector"`` — free-form semantic question, best answered by dense
  Qdrant retrieval over chunk embeddings.
- ``"graph"`` — structural, multi-revision, or contradiction-detection
  question, best answered by a multi-hop Neo4j Cypher walk.

The node writes ``state["route"]`` and returns the modified state.
Uses mistral-large-latest for reliable output.
"""

from __future__ import annotations

import asyncio
import logging
import random

from openai import AsyncOpenAI, RateLimitError
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

# Global semaphore for rate limiting (Mistral free tier: 1 RPS / 30 RPM)
_rate_limit_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the global rate limit semaphore."""
    global _rate_limit_semaphore
    if _rate_limit_semaphore is None:
        _rate_limit_semaphore = asyncio.Semaphore(1)
    return _rate_limit_semaphore


async def _route_with_retry(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """
    Generate routing decision with exponential backoff retry for rate limiting.

    Catches 429 RateLimitError and retries with jitter.
    """
    semaphore = _get_semaphore()

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10,
                )
                # Buffer delay between requests (Mistral free tier protection)
                await asyncio.sleep(1.1)
                return response.choices[0].message.content or ""

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                # Exponential backoff with jitter for rate limiting
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Rate limit hit (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but return empty string as fallback
        return ""


async def router_node(state: AuditState) -> AuditState:
    """
    Classify the query and set ``state["route"]``.

    Uses Mistral AI to make a routing decision. No fallback is provided
    if Mistral is unavailable - this ensures the system fails explicitly
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

    mistral_api_key = (
        settings.mistral_api_key.get_secret_value()
        if settings.mistral_api_key
        else None
    )

    if not mistral_api_key:
        logger.error("MISTRAL_API_KEY not configured — cannot route query.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MISTRAL_API_KEY not configured. Set the API key in your environment."
        )

    # Build prompt for LLM routing
    prompt = _ROUTER_PROMPT_TEMPLATE.format(
        query=query,
        company_name=company_name,
        intents=", ".join(intents) if intents else "none",
    )

    try:
        client = AsyncOpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=mistral_api_key,
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        response_text = await _route_with_retry(
            client=client,
            model=settings.llm_model,
            messages=messages,
        )

        # Extract the route from the response
        raw = response_text.strip().lower().split()[-1] if response_text.strip() else ""
        route = raw if raw in ("vector", "graph") else None

        if route:
            logger.info("Router decision (Mistral): route='%s'  query='%s'", route, query[:60])
            return {**state, "route": route}

        logger.warning(
            "Router Mistral returned unexpected value '%s'. Response was: %s", raw, response_text[:200]
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Router LLM returned invalid value: {raw}"
        )

    except HTTPException:
        raise
    except Exception as mistral_exc:
        logger.error(
            "Mistral router call failed (%s: %s).",
            type(mistral_exc).__name__,
            mistral_exc,
            exc_info=mistral_exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mistral router error: {mistral_exc}"
        ) from mistral_exc