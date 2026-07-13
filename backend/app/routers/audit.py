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
# Disable cache for local development if REDIS_DISABLED is set
if os.getenv("REDIS_DISABLED", "false").lower() in ("true", "1", "yes"):
    _cache = None
else:
    _cache = SemanticCache(threshold=0.92)


# ---------------------------------------------------------------------------
# Platform list / discovery endpoint - ALL platforms from OTA
# ---------------------------------------------------------------------------

@router.get(
    "/platforms",
    summary="Get all tracked platforms",
    description="Fetch a comprehensive index of all platforms available from Open Terms Archive."
)
async def get_platforms() -> list[dict]:
    """
    Fetch a list of all tracked platforms from OTA GitHub repos.

    This endpoint returns ALL available platforms for the main dropdown.
    Use /platforms/quick-select for only pre-seeded platforms.
    """
    try:
        from ingestion.api_client import OpenTermsArchiveClient
        client = OpenTermsArchiveClient()
        platforms = client.fetch_services_sync() or []
        return platforms
    except Exception as exc:
        logger.warning("OTA fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Quick Select endpoint - only platforms with ingested data
# ---------------------------------------------------------------------------

@router.get(
    "/platforms/quick-select",
    summary="Get pre-seeded platforms",
    description="Fetch platforms that have active structural data in Neo4j (pre-seeded or previously ingested)."
)
async def get_quick_select_platforms() -> list[dict]:
    """
    Fetch a list of platforms with ingested policy data for Quick Select.
    Only returns platforms that have actual data in Neo4j.
    """
    try:
        from neo4j import AsyncGraphDatabase
        
        auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
        async with AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
            async with driver.session() as session:
                # Check for nodes with text content (LlamaIndex format)
                cypher_query = """
                MATCH (n) WHERE n.text IS NOT NULL
                RETURN DISTINCT n.name AS name
                ORDER BY n.name
                LIMIT 100
                """
                result = await session.run(cypher_query)
                records = await result.data()
                
                if records:
                    platforms = [{"id": rec.get("name", ""), "name": rec.get("name", "")} for rec in records]
                    return platforms
                
                # Fallback: check Platform nodes (seed data format)
                result = await session.run(
                    "MATCH (p:Platform) RETURN p.id AS id, p.name AS name ORDER BY p.id"
                )
                records = await result.data()
                if records:
                    platforms = [{"id": rec.get("id", ""), "name": rec.get("name", rec.get("id", ""))} for rec in records]
                    return platforms
    except Exception as exc:
        logger.warning("Neo4j unavailable for quick-select: %s", exc)

    return []


@router.get(
    "/platforms/search",
    summary="Search platforms by name",
    description="Search for platforms by name using fuzzy matching against all available platforms."
)
async def search_platforms(q: str = Query(..., min_length=1, max_length=100)) -> list[dict]:
    """
    Search for platforms by name against all available platforms from OTA.

    This searches ALL platforms, not just those with ingested data.
    Use /platforms/quick-select for only platforms with data.
    """
    try:
        from ingestion.api_client import OpenTermsArchiveClient
        client = OpenTermsArchiveClient()
        all_platforms = client.fetch_services_sync() or []
        
        # Case-insensitive substring match
        q_lower = q.lower()
        filtered = [
            {"id": p["id"], "name": p["name"]}
            for p in all_platforms
            if q_lower in p.get("name", "").lower() or q_lower in p.get("id", "").lower()
        ][:10]
        return filtered
    except Exception as exc:
        logger.warning("Platform search failed: %s", exc)
    
    return []


@router.get(
    "/debug/neo4j",
    summary="Debug Neo4j node structure",
    description="Inspect actual Neo4j node labels and properties for debugging."
)
async def debug_neo4j() -> dict:
    """
    Debug endpoint to inspect Neo4j node structure.
    """
    try:
        from neo4j import AsyncGraphDatabase
        
        auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
        async with AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
            async with driver.session() as session:
                # Get all unique labels
                labels_result = await session.run(
                    "CALL db.labels() YIELD label RETURN label"
                )
                labels = [rec["label"] for rec in await labels_result.data()]
                
                # Get sample nodes with properties
                sample_result = await session.run(
                    "MATCH (n) RETURN n.name AS name, n.text AS text, n._properties AS props, labels(n) AS labels LIMIT 5"
                )
                samples = []
                async for rec in sample_result:
                    sample = {
                        "name": rec.get("name"),
                        "text_preview": rec.get("text", "")[:100] if rec.get("text") else None,
                        "labels": list(rec.get("labels", [])) if rec.get("labels") else [],
                        "props_keys": list(rec.get("props", {}).keys()) if rec.get("props") else [],
                    }
                    samples.append(sample)
                
                # Get relationship types
                rel_result = await session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                relationships = [rec["relationshipType"] for rec in await rel_result.data()]
        
        return {
            "labels": labels,
            "relationships": relationships,
            "sample_nodes": samples,
            "node_count": len(samples),
        }
    except Exception as exc:
        return {"error": str(exc)}


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


async def _ensure_company_ingested(company_name: str) -> bool:
    """
    Check if company has ingested data in Neo4j or Qdrant.
    
    No longer triggers on-demand ingestion - only checks for existing data.
    If no data exists, the audit will return an informative message.
    
    Returns True if data exists, False otherwise.
    """
    from neo4j import AsyncGraphDatabase
    
    # Check Neo4j for any nodes with text content (LlamaIndex PropertyGraphStore format)
    try:
        auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
        async with AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
            async with driver.session() as session:
                # Check for any nodes that might contain platform-related text
                result = await session.run(
                    "MATCH (n) WHERE (n.text IS NOT NULL OR n.name IS NOT NULL) "
                    "AND toLower(toString(coalesce(n.text, n.name))) CONTAINS toLower($company) "
                    "RETURN count(n) AS count LIMIT 1",
                    company=company_name
                )
                record = await result.single()
                if record and record["count"] > 0:
                    logger.info("Found %d existing nodes for company='%s'", record["count"], company_name)
                    return True
    except Exception as exc:
        logger.warning("Neo4j check failed: %s", exc)
    
    # Also check Qdrant for vector data
    try:
        from qdrant_client import AsyncQdrantClient
        qc = AsyncQdrantClient(
            url=settings.qdrant_url if settings.qdrant_url else None,
            host=settings.qdrant_host if not settings.qdrant_url else None,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        )
        collections = await qc.get_collections()
        if any(c.name == "vampter_docs" for c in collections.collections):
            count_result = await qc.count(collection_name="vampter_docs")
            if count_result.count > 0:
                logger.info("Found %d vectors in Qdrant", count_result.count)
                return True
            else:
                logger.info("Qdrant collection exists but is empty")
    except Exception as exc:
        logger.warning("Qdrant check failed: %s", exc)
    
    return False


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

    # ── Step 1: Ensure company data exists (lazy load if needed) ───────────────
    data_ready = await _ensure_company_ingested(request.company_name)
    if not data_ready:
        logger.warning("No data available for company='%s' after ingestion attempt", request.company_name)

    # ── Determine cache bypass status ───────────────────────────────────
    # Respect request parameter, or fall back to environment variable, or default to False
    bypass_cache = request.bypass_cache
    if bypass_cache is None:
        bypass_cache = os.getenv("BYPASS_CACHE", "false").lower() in ("true", "1", "yes")
    if bypass_cache:
        logger.warning(
            "Cache bypass active — Cache will be skipped for company='%s'.",
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
    if _cache and not bypass_cache and query_vector is not None:
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
    # Only cache non-empty reports to avoid caching empty results
    if _cache and query_vector is not None and report.vulnerability_score > 0.0:
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