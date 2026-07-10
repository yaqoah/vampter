"""
app.routers.audit
=================
POST /api/v1/audit — Primary audit endpoint.

Request / Response
------------------
POST body (JSON):
    {
        "company_name": "Android Auto",
        "query":        "What data retention clauses changed in the latest revision?",
        "intents":      ["policy_change", "vulnerability_scan"]
    }

Response (JSON):
    AuditReport — fully validated Pydantic model serialised to JSON.

Pipeline Flow
-------------
1. Embed the user query → query vector.
2. Check ``SemanticCache`` — if similarity >= 0.92, return cached report immediately.
3. Otherwise, invoke the LangGraph ``audit_graph`` asynchronously.
4. Store the result in ``SemanticCache`` for future requests.
5. Return the ``AuditReport`` JSON response.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cache.semantic_cache import SemanticCache
from config import settings
from workflow.graph import build_audit_graph
from workflow.state import AuditReport

logger = logging.getLogger(__name__)

router = APIRouter()
_cache = SemanticCache(threshold=0.92)


# ---------------------------------------------------------------------------
# Platform list / discovery endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/platforms",
    summary="Get tracked platforms",
    description="Fetch a comprehensive index of currently tracked service/platform records."
)
async def get_platforms() -> list[dict]:
    """
    Fetch a list of tracked platforms by utilizing the refactored
    OpenTermsArchiveClient fetch_services call dynamically.
    """
    from ingestion.api_client import OpenTermsArchiveClient
    try:
        async with OpenTermsArchiveClient() as client:
            services = await client.fetch_services()
            # Transform to clean format: [{"id": s.get("id"), "name": s.get("name")} for s in services]
            results = []
            for service in services:
                service_id = service.get("id")
                service_name = service.get("name", service_id)
                if service_id:
                    results.append({"id": service_id, "name": service_name})
            if not results:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="PLATFORMS_FETCH_FAILED"
                )
            return results
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to dynamically fetch platforms: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PLATFORMS_FETCH_FAILED"
        )


@router.get(
    "/platforms/search",
    summary="Search platforms by name",
    description="Search for platforms by name using fuzzy matching against the Neo4j database."
)
async def search_platforms(q: str = Query(..., min_length=1, max_length=100)) -> list[dict]:
    """
    Search for platforms by name using fuzzy matching against the Neo4j database.

    Parameters
    ----------
    q:
        The search query string (minimum 1 character, maximum 100 characters).

    Returns
    -------
    list[dict]
        List of matching platforms with their names and IDs.
    """
    from neo4j import AsyncGraphDatabase

    try:
        # Connect to Neo4j
        auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
        async with AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
            async with driver.session() as session:
                # Execute Cypher query with case-insensitive matching
                # Prioritizes prefix matches first, then substring matches
                # Uses COLLATE for proper case-insensitive comparison
                query = """
                MATCH (p:Platform)
                WHERE toLower(p.name) CONTAINS toLower($query)
                RETURN p.id AS id, p.name AS name,
                       CASE WHEN toLower(p.name) STARTS WITH toLower($query) THEN 0 ELSE 1 END AS sort_priority
                ORDER BY sort_priority, p.name
                LIMIT 10
                """
                result = await session.run(query, query=q)
                records = await result.data()

                if not records:
                    logger.warning("SEARCH_FAILURE: Neo4j database returned 0 Platform nodes.")
                    return []

                # Transform results to list of dicts with id and name
                platforms = [{"id": rec.get("id", ""), "name": rec.get("name", "")} for rec in records]
                return platforms

    except Exception as exc:
        logger.error("Failed to search platforms: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PLATFORMS_SEARCH_FAILED"
        )


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class AuditRequest(BaseModel):
    """Incoming payload for the /api/v1/audit endpoint."""

    company_name: str = Field(
        description="Target company or platform name to audit.",
        min_length=1,
        max_length=200,
    )
    query: str = Field(
        default="Summarise the latest policy changes and risk profile.",
        description="Free-form question or audit directive.",
        max_length=2000,
    )
    intents: List[str] = Field(
        default_factory=list,
        description="Active audit intent toggles e.g. ['policy_change', 'vulnerability_scan'].",
    )
    bypass_cache: Optional[bool] = Field(
        default=None,
        description="If true, bypass semantic cache and force fresh pipeline execution. Alternatively, set DEV_BYPASS_CACHE=true environment variable.",
    )


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


def _embed_query(query: str) -> Optional[List[float]]:
    """
    Produce a dense embedding for ``query`` using the model registered on
    ``LlamaSettings``, or return ``None`` if embedding is unavailable.
    """
    try:
        from llama_index.core import Settings as LlamaSettings
        model = LlamaSettings.embed_model
        return model.get_text_embedding(query)
    except Exception as exc:
        logger.warning("Embedding failed: %s — cache will be bypassed.", exc)
        return None


def _build_cache_key_text(company_name: str, query: str, intents: List[str]) -> str:
    """
    Build a composite text string for cache embedding that includes context.
    This prevents collisions between queries for different companies/intents.
    """
    intents_str = ','.join(intents) if intents else ''
    return f"Company: {company_name} | Query: {query} | Intents: {intents_str}"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/audit",
    response_model=AuditReport,
    summary="Run a full policy audit for a target company",
    description=(
        "Accepts a company name, optional free-form query, and active intent toggles. "
        "Returns a structured ``AuditReport`` with vulnerability scoring, policy insights, "
        "timeline trends, and graph visualisation data."
    ),
    responses={
        200: {"description": "Audit report generated successfully."},
        500: {"description": "Internal pipeline error."},
    },
)
async def run_audit(request: AuditRequest) -> AuditReport:
    """
    Execute the full Vampter AI audit pipeline.

    Flow:
    - Semantic cache lookup (< 5 ms hot path)
    - LangGraph orchestration: router → [vector|graph] → compress → generate
    - Cache store on miss
    - Return validated ``AuditReport``
    """
    logger.info(
        "Audit request — company='%s'  query='%s'  intents=%s",
        request.company_name,
        request.query[:80],
        request.intents,
    )

    # ── DEVELOPMENT BYPASS: Hardcoded override for testing ─────────────────────
    BYPASS_CACHE = True
    if BYPASS_CACHE:
        logger.warning(
            "DEV BYPASS ACTIVE — Cache will be skipped for company='%s'.",
            request.company_name
        )

    # ── Step 1: Embed query for cache lookup ───────────────────────────────────
    # Build composite cache key text to prevent cross-company collisions
    cache_key_text = _build_cache_key_text(
        request.company_name,
        request.query,
        request.intents
    )
    query_vector = _embed_query(cache_key_text)

    # ── Step 2: Semantic cache check ────────────────────────────────────────
    if not BYPASS_CACHE and query_vector is not None:
        cached = await _cache.lookup(
            company_name=request.company_name,
            query_vector=query_vector,
        )
        if cached is not None:
            logger.info(
                "Serving cached report for company='%s'.", request.company_name
            )
            return AuditReport(**cached)

    # ── Step 3: Invoke LangGraph orchestration ──────────────────────────────
    try:
        graph = build_audit_graph()

        initial_state = {
            "company_name": request.company_name,
            "query": request.query,
            "intents": request.intents,
        }

        final_state = await graph.ainvoke(initial_state)
        report: AuditReport = final_state.get("report")

        if report is None:
            raise ValueError("LangGraph pipeline produced no report.")

    except Exception as exc:
        logger.error("Audit pipeline failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audit pipeline error: {exc}",
        )

    # ── Step 4: Store result in cache ───────────────────────────────────────
    if query_vector is not None:
        try:
            await _cache.store(
                company_name=request.company_name,
                query_vector=query_vector,
                payload=report.model_dump(),
            )
        except Exception as exc:
            # Non-fatal: log and continue — we still return the report.
            logger.warning("Cache store failed: %s", exc)

    # ── Step 5: Return validated report ─────────────────────────────────────
    logger.info(
        "Audit complete — company='%s'  score=%.1f  threat=%s",
        report.company_name,
        report.vulnerability_score,
        report.threat_level,
    )
    return report