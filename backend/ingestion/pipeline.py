"""
ingestion.pipeline
==================
Top-level async orchestrator for the Vampter ingestion pipeline.

Orchestration sequence
-----------------------

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────┐
    │                   run_pipeline(settings)                    │
    └───────────────────────────┬─────────────────────────────────┘
                                │
             ┌──────────────────▼──────────────────┐
             │  1. async_fetch_api_documents()      │
             │     API JSON payloads                │
             └──────────────────┬──────────────────┘
                                │  List[Document]
             ┌──────────────────▼──────────────────┐
             │  2. parse_documents_to_nodes()       │
             │     SentenceSplitter                 │
             │     + BAAI/bge-small-en-v1.5 embeds  │
             └──────────────────┬──────────────────┘
                                │  List[BaseNode]
    ┌───────────────────────────▼─────────────────────────────────┐
    │  3. PropertyGraphIndex.from_documents()                     │
    │     ┌──────────────────────────────────────────────────┐   │
    │     │ SchemaLLMPathExtractor (Gemini 1.5 Pro)           │   │
    │     │   (Platform)-[TRACKS_POLICY]->(Document)         │   │
    │     │   (Document)-[HAS_REVISION_VERSION]->(Revision)  │   │
    │     │   (Revision)-[CONTAINS_CLAUSE]->(Clause)         │   │
    │     └────────────┬─────────────────────────────────────┘   │
    │                  │  writes                                  │
    │       ┌──────────┴──────────┐                              │
    │       │                     │                              │
    │  ┌────▼──────┐    ┌─────────▼────────┐                    │
    │  │  Qdrant   │    │     Neo4j        │                    │
    │  │  (vectors)│    │  (triples/graph) │                    │
    │  └───────────┘    └──────────────────┘                    │
    └─────────────────────────────────────────────────────────────┘
                                │
             ┌──────────────────▼──────────────────┐
             │  4. Persist StorageContext to disk   │
             │     ./storage/                       │
             └──────────────────┬──────────────────┘
                                │
             ┌──────────────────▼──────────────────┐
             │  5. Return IngestionResult           │
             └─────────────────────────────────────┘

LLM selection
-------------
The pipeline uses ``google.genai`` (Gemini) as the primary LLM for
triple extraction via LlamaIndex's Gemini integration.  If
``GEMINI_API_KEY`` is not set, the pipeline falls back to a
``MockLLM`` for offline testing and development.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llama_index.core import (
    PropertyGraphIndex,
    Settings as LlamaSettings,
    StorageContext,
)
from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from config import AppSettings
from ingestion.graph_extractor import build_schema_extractor
from ingestion.api_client import async_fetch_api_documents
from ingestion.parser import parse_documents_to_nodes
from ingestion.stores import init_neo4j_store, init_qdrant_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """
    Aggregated statistics and references returned after a successful
    pipeline run.
    """

    documents_loaded: int = 0
    nodes_parsed: int = 0
    elapsed_seconds: float = 0.0
    index: Optional[PropertyGraphIndex] = field(default=None, repr=False)
    storage_path: Optional[Path] = None

    def summary(self) -> str:
        """Human-readable single-line summary string."""
        return (
            f"IngestionResult("
            f"docs={self.documents_loaded}, "
            f"nodes={self.nodes_parsed}, "
            f"elapsed={self.elapsed_seconds:.1f}s, "
            f"storage='{self.storage_path}')"
        )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _build_llm(settings: AppSettings):
    """
    Construct the LLM instance used for triple extraction.

    Prefers Gemini 1.5 Pro when ``GEMINI_API_KEY`` is configured.
    Falls back to LlamaIndex's ``MockLLM`` for fully offline operation
    (no triples will be extracted, but vector ingestion proceeds normally).
    """
    api_key = (
        settings.gemini_api_key.get_secret_value()
        if settings.gemini_api_key
        else None
    )

    if api_key:
        try:
            # Use LlamaIndex's built-in Gemini integration.
            from llama_index.llms.gemini import Gemini  # type: ignore[import]

            llm = Gemini(
                api_key=api_key,
                model_name=settings.llm_model,
            )
            logger.info("LLM: Gemini  model=%s", settings.llm_model)
            return llm
        except ImportError:
            logger.warning(
                "llama-index-llms-gemini not installed.  "
                "Attempting OpenAI-compatible endpoint …"
            )

    # Fallback: MockLLM — triples won't be real but pipeline is testable.
    from llama_index.core.llms import MockLLM  # type: ignore[import]

    logger.warning(
        "GEMINI_API_KEY not set or Gemini LLM unavailable.  "
        "Using MockLLM — graph extraction will be a no-op."
    )
    return MockLLM(max_tokens=512)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(
    settings: AppSettings,
    *,
    api_url: str = "https://api.example-ota.org/v1/documents",
    storage_dir: str = "storage",
    embed_model_uri: str = "local:BAAI/bge-small-en-v1.5",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    dry_run: bool = False,
) -> IngestionResult:
    """
    Execute the full asynchronous API ingestion pipeline.

    This is the single public entry-point for Phase-2 ingestion.  It
    orchestrates document fetching from the OTA API, node parsing, dual-store 
    indexing, and optional persistence to disk.

    Parameters
    ----------
    settings:
        Pydantic ``AppSettings`` instance (from ``config.py``).
    api_url:
        The target API endpoint to pull JSON data from.
    storage_dir:
        Local directory where the ``StorageContext`` will be persisted.
        Relative to the current working directory.
    embed_model_uri:
        Local embedding model URI accepted by ``resolve_embed_model``.
    chunk_size:
        Token budget per text chunk for the ``SentenceSplitter``.
    chunk_overlap:
        Sliding overlap window in tokens.
    dry_run:
        When ``True``, load and parse documents but skip store writes
        and index construction.  Useful for validating the loader without
        running live infrastructure.

    Returns
    -------
    IngestionResult
        Statistics and references for the completed ingestion run.
    """
    t_start = time.perf_counter()
    result = IngestionResult()

    # ── Step 1: Fetch ──────────────────────────────────────────────────────
    logger.info("=== VAMPTER INGESTION PIPELINE ===")
    logger.info("Step 1/4 - Fetching API documents ...")

    documents = await async_fetch_api_documents(api_url=api_url)
    result.documents_loaded = len(documents)

    if not documents:
        logger.warning("No documents loaded — pipeline aborted early.")
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    # ── Step 2: Parse ──────────────────────────────────────────────────────
    logger.info("Step 2/4 - Parsing nodes (chunk_size=%d) ...", chunk_size)

    nodes = await parse_documents_to_nodes(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_uri=embed_model_uri,
    )
    result.nodes_parsed = len(nodes)

    if dry_run:
        logger.info("Dry-run mode active — skipping store writes.")
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    # ── Step 3: Initialise stores ──────────────────────────────────────────
    logger.info("Step 3/4 - Initialising dual stores ...")

    qdrant_api_key: Optional[str] = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key
        else None
    )

    vector_store: BasePydanticVectorStore = init_qdrant_store(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_port_grpc,
        api_key=qdrant_api_key,
    )

    graph_store: PropertyGraphStore = init_neo4j_store(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        property_graph_store=graph_store,
    )

    # ── Step 4: Build PropertyGraphIndex ──────────────────────────────────
    logger.info("Step 4/4 - Building PropertyGraphIndex ...")

    llm = _build_llm(settings)
    LlamaSettings.llm = llm

    kg_extractor = build_schema_extractor(llm=llm)

    # ``PropertyGraphIndex.from_documents`` is synchronous internally but
    # may dispatch async sub-tasks; we run it in the default executor to
    # avoid blocking the event loop during LLM calls.
    loop = asyncio.get_event_loop()
    index: PropertyGraphIndex = await loop.run_in_executor(
        None,
        lambda: PropertyGraphIndex.from_documents(
            documents,
            kg_extractors=[kg_extractor],
            storage_context=storage_context,
            show_progress=True,
        ),
    )

    # Persist the storage context for downstream query / audit engines.
    storage_path = Path(storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_path))

    result.index = index
    result.storage_path = storage_path
    result.elapsed_seconds = time.perf_counter() - t_start

    logger.info("=== INGESTION COMPLETE ===")
    logger.info(result.summary())
    return result
