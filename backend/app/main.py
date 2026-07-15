"""
app.main
========
FastAPI application factory for the Vampter backend.

Startup
-------
Run from the project root with the virtual environment activated:

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Or via the config-driven entrypoint:

    python -m app.main
"""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import audit as audit_router
from config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global LlamaIndex embedding binding
# ---------------------------------------------------------------------------
# Configure the global LlamaIndex Settings singleton to use Mistral cloud embeddings.
# This MUST run before any node, cache, or graph module is imported so that
# LlamaSettings.embed_model is always populated.

from llama_index.core import Settings as LlamaSettings                  # noqa: E402
from llama_index.embeddings.mistralai import MistralAIEmbedding           # noqa: E402

# Use Mistral embed model (cloud API, no local model download required)
# The API key is read from MISTRAL_API_KEY environment variable
LlamaSettings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
logger.info(
    "LlamaIndex global embed_model set → MistralAIEmbedding(mistral-embed)"
)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Vampter — AI Policy Document Auditor",
    description=(
        "Asynchronous AI backend for auditing Open Terms Archive software update "
        "policy documents. Combines Redis semantic caching, LangGraph orchestration, "
        "Qdrant vector retrieval, Neo4j graph traversal, and Poolside AI generation "
        "to deliver structured audit reports."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
# In production, replace allow_origins with your actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(
    audit_router.router,
    prefix="/api/v1",
    tags=["Audit"],
)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    """Pre-warm the LangGraph compiled graph."""
    from workflow.graph import build_audit_graph
    
    build_audit_graph()
    logger.info("Vampter backend started — LangGraph graph pre-compiled.")
    # Platform cache is lazily initialized in audit.py when needed


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Vampter backend shutting down.")


# ---------------------------------------------------------------------------
# Mangum handler for Vercel serverless deployment
# ---------------------------------------------------------------------------

from mangum import Mangum

# Create AWS Lambda/ALB handler for serverless deployment
# This wraps the FastAPI app for Vercel's serverless functions
handler = Mangum(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health() -> dict:
    """Returns 200 OK when the service is running."""
    return {"status": "ok", "service": "vampter-backend", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
