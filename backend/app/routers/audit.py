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
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
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
                # Plausible fallback defaults if services index is offline or unreachable
                results = [
                    {"id": "amazon", "name": "Amazon"},
                    {"id": "google", "name": "Google"},
                    {"id": "netflix", "name": "Netflix"},
                    {"id": "meta", "name": "Meta"},
                    {"id": "zoom", "name": "Zoom"}
                ]
            return results
    except Exception as exc:
        logger.error("Failed to dynamically fetch platforms: %s", exc, exc_info=True)
        return [
            {"id": "amazon", "name": "Amazon"},
            {"id": "google", "name": "Google"},
            {"id": "netflix", "name": "Netflix"},
            {"id": "meta", "name": "Meta"},
            {"id": "zoom", "name": "Zoom"}
        ]


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

    # ── Step 1: Embed query for cache lookup ────────────────────────────────
    query_vector = _embed_query(request.query)

    # ── Step 2: Semantic cache check ────────────────────────────────────────
    import os
    bypass_cache = os.getenv("BYPASS_CACHE", "false").lower() == "true"
    if query_vector is not None and not bypass_cache:
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
