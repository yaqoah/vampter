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
    │     │ SchemaLLMPathExtractor (Mistral)                  │   │
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
The pipeline uses Mistral AI for triple extraction via
OpenAI-compatible API.  ``MISTRAL_API_KEY`` must be configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from llama_index.core import (
    PropertyGraphIndex,
    Settings as LlamaSettings,
    StorageContext,
)
from llama_index.core.graph_stores.types import PropertyGraphStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from config import AppSettings, settings as global_settings
from ingestion.graph_extractor import build_schema_extractor
from ingestion.api_client import OpenTermsArchiveClient, OTASettings
from ingestion.parser import parse_documents_to_nodes
from ingestion.stores import init_neo4j_store, init_qdrant_store, run_neo4j_schema_migration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = Path("ingestion_checkpoint.json")


def load_checkpoint() -> dict:
    """Load ingestion progress checkpoint."""
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text())
        except Exception:
            pass
    return {"completed_platforms": [], "total_platforms": 441, "start_time": None}


def save_checkpoint(data: dict) -> None:
    """Save ingestion progress checkpoint."""
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2))
    logger.info("Checkpoint saved: %d completed platforms", len(data.get("completed_platforms", [])))


def get_platforms_to_skip(resume: bool) -> set:
    """Get platform IDs that should be skipped when resuming."""
    checkpoint = load_checkpoint()
    if resume and checkpoint.get("completed_platforms"):
        return set(checkpoint["completed_platforms"])
    return set()

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
    neo4j_rows_inserted: int = 0
    qdrant_vectors_stored: int = 0
    elapsed_seconds: float = 0.0
    index: Optional[PropertyGraphIndex] = field(default=None, repr=False)
    storage_path: Optional[Path] = None

    def summary(self) -> str:
        """Human-readable single-line summary string."""
        return (
            f"IngestionResult("
            f"docs={self.documents_loaded}, "
            f"nodes={self.nodes_parsed}, "
            f"neo4j_rows={self.neo4j_rows_inserted}, "
            f"qdrant_vectors={self.qdrant_vectors_stored}, "
            f"elapsed={self.elapsed_seconds:.1f}s, "
            f"storage='{self.storage_path}')"
        )


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm(settings: AppSettings):
    """
    Construct the LLM instance used for triple extraction.

    Uses Mistral AI when ``MISTRAL_API_KEY`` is configured.
    No mock fallback is provided - real LLM is required for graph extraction.

    Parameters
    ----------
    settings:
        Pydantic ``AppSettings`` instance.

    Returns
    -------
    LLM instance (OpenAI-compatible).

    Raises
    ------
    RuntimeError
        If the API key is not configured or the package is unavailable.
    """
    api_key = (
        settings.mistral_api_key.get_secret_value()
        if settings.mistral_api_key
        else None
    )

    if api_key:
        try:
            from llama_index.llms.openai import OpenAI as LlamaOpenAI  # type: ignore[import]

            llm = LlamaOpenAI(
                api_key=api_key,
                base_url="https://api.mistral.ai/v1",
                model_name=settings.llm_model,
                default_headers={"Content-Type": "application/json"},
            )
            logger.info("LLM: Mistral model=%s", settings.llm_model)
            return llm
        except ImportError as import_err:
            raise RuntimeError(
                "llama-index-llms-openai not installed. "
                "Install it with: pip install llama-index-llms-openai"
            ) from import_err

    raise RuntimeError(
        "MISTRAL_API_KEY not set. Graph extraction requires a configured LLM. "
        "Set MISTRAL_API_KEY in your environment."
    )


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
    run_schema_migration: bool = True,
    incremental: bool = False,
    max_platforms: Optional[int] = None,
    clear_all: bool = False,
    clear_platform: Optional[str] = None,
    resume: bool = False,
    batch_size: int = 50,
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
    run_schema_migration:
        When ``True`` (default), run Neo4j schema migration before indexing.
    incremental:
        When ``True``, skip platforms that already have Document data in Neo4j.
    max_platforms:
        Limit ingestion to first N platforms (useful for testing or batching).

    Returns
    -------
    IngestionResult
        Statistics and references for the completed ingestion run.
    """
    t_start = time.perf_counter()
    result = IngestionResult()

    # ── Step 1: Clear (if requested) ───────────────────────────────────────
    if clear_all:
        logger.info("CLEAR ALL: Removing all data from Neo4j and Qdrant...")
        import qdrant_client
        try:
            # Clear Qdrant
            from qdrant_client.http.models import CollectionStatus
            qc = qdrant_client.QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            if qc.collection_exists("vampter_docs"):
                qc.delete_collection("vampter_docs")
                logger.info("Cleared Qdrant collection: vampter_docs")
            # Recreate collection
            qc.create_collection(
                collection_name="vampter_docs",
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=384,
                    distance=qdrant_client.http.models.Distance.COSINE
                )
            )
        except Exception as exc:
            logger.warning("Could not clear Qdrant: %s", exc)
        
        try:
            # Clear Neo4j
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
            )
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared all Neo4j nodes")
            driver.close()
        except Exception as exc:
            logger.warning("Could not clear Neo4j: %s", exc)
    
    if clear_platform:
        logger.info("CLEAR PLATFORM: Removing data for '%s'...", clear_platform)
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
            )
            with driver.session() as session:
                session.run(
                    "MATCH (p:Platform {id: $platform_id}) DETACH DELETE p",
                    platform_id=clear_platform.lower()
                )
                session.run(
                    "MATCH (d:Document {platform: $platform}) DETACH DELETE d",
                    platform=clear_platform.lower()
                )
                logger.info("Cleared platform '%s' from Neo4j", clear_platform)
            driver.close()
        except Exception as exc:
            logger.warning("Could not clear platform from Neo4j: %s", exc)
        
        try:
            import qdrant_client
            qc = qdrant_client.QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            if qc.collection_exists("vampter_docs"):
                qc.delete(
                    collection_name="vampter_docs",
                    points_selector=qdrant_client.http.models.Filter(
                        must=[qdrant_client.http.models.FieldCondition(
                            key="platform",
                            match=qdrant_client.http.models.MatchText(text=clear_platform.lower())
                        )]
                    )
                )
                logger.info("Cleared platform '%s' from Qdrant", clear_platform)
        except Exception as exc:
            logger.warning("Could not clear platform from Qdrant: %s", exc)
    
    if clear_all or clear_platform:
        # Continue without incremental - we just cleared data
        incremental = False

    # ── Step 1: Fetch ──────────────────────────────────────────────────────
    logger.info("=== VAMPTER INGESTION PIPELINE ===")
    logger.info("Step 1/4 - Fetching API documents ...")

    # Resolve client base URL from api_url if customized, otherwise defaults.
    ota_settings = OTASettings()
    if api_url and "api.example-ota.org" not in api_url:
        # If a custom URL is provided, strip "/documents" or similar suffixes if present to get the base URL
        base_url = api_url
        if base_url.endswith("/documents"):
            base_url = base_url.removesuffix("/documents")
        ota_settings = OTASettings(base_url=base_url)

    async with OpenTermsArchiveClient(settings=ota_settings) as client:
        # Use the hierarchical pipeline pattern with dynamic discovery to eliminate 404 validation loops
        documents = await client.fetch_all_documents()

    # Load checkpoint and filter out completed platforms
    if resume or incremental:
        completed = get_platforms_to_skip(resume)
        if completed:
            original_count = len(documents)
            documents = [doc for doc in documents 
                       if doc.metadata.get("platform") not in completed]
            logger.info("Resume/Incremental mode: skipped %d documents for %d completed platforms", 
                       original_count - len(documents), len(completed))
    
    # Apply max_platforms limit if specified
    if max_platforms and max_platforms > 0:
        original_count = len(documents)
        # Group documents by platform and limit
        platform_docs: dict[str, List[Document]] = {}
        for doc in documents:
            platform = doc.metadata.get("platform", "unknown")
            if platform not in platform_docs:
                platform_docs[platform] = []
            platform_docs[platform].append(doc)
        
        # Take only first N platforms that haven't been completed
        limited_platforms = [p for p in platform_docs.keys() if p not in completed][:max_platforms]
        documents = []
        for platform in limited_platforms:
            documents.extend(platform_docs[platform])
        
        logger.info("Limited to %d platforms: %d documents selected (from %d total)", 
                   max_platforms, len(documents), original_count)
    if incremental:
        logger.info("Incremental mode: checking for existing documents in Neo4j...")
        try:
            from neo4j import AsyncGraphDatabase
            auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
            async with AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
                async with driver.session() as session:
                    # Get platforms that already have documents
                    result_q = await session.run(
                        "MATCH (d:Document) RETURN DISTINCT d.platform AS platform"
                    )
                    existing_platforms = {rec["platform"] for rec in (await result_q.data()) if rec.get("platform")}
                    
            if existing_platforms:
                original_count = len(documents)
                documents = [doc for doc in documents 
                           if doc.metadata.get("platform") not in existing_platforms]
                logger.info("Incremental mode: skipped %d documents for %d existing platforms", 
                           original_count - len(documents), len(existing_platforms))
        except Exception as exc:
            logger.warning("Could not check existing platforms (continuing without filter): %s", exc)
    
    # Apply max_platforms limit if specified
    if max_platforms and max_platforms > 0:
        original_count = len(documents)
        # Group documents by platform and limit
        platform_docs: dict[str, List[Document]] = {}
        for doc in documents:
            platform = doc.metadata.get("platform", "unknown")
            if platform not in platform_docs:
                platform_docs[platform] = []
            platform_docs[platform].append(doc)
        
        # Take only first N platforms
        limited_platforms = list(platform_docs.keys())[:max_platforms]
        documents = []
        for platform in limited_platforms:
            documents.extend(platform_docs[platform])
        
        logger.info("Limited to %d platforms: %d documents selected (from %d total)", 
                   max_platforms, len(documents), original_count)

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

    # Run Neo4j schema migration first
    if run_schema_migration:
        logger.info("Running Neo4j schema migration...")
        run_neo4j_schema_migration(
            url=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password.get_secret_value(),
        )

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

    # Log explicit counts for Neo4j and Qdrant
    # Get vector count from Qdrant
    try:
        qdrant_count = vector_store.client.count(
            collection_name=vector_store.collection_name
        )
        result.qdrant_vectors_stored = qdrant_count.count
        logger.info("QDRANT VECTORS STORED: %d", result.qdrant_vectors_stored)
    except Exception as exc:
        logger.warning("Could not get Qdrant vector count: %s", exc)
        result.qdrant_vectors_stored = len(nodes)  # Fallback to node count

    # Get graph count from Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value())
        )
        with driver.session() as session:
            neo4j_count = session.run("MATCH (n) RETURN count(n) as count").single()
            result.neo4j_rows_inserted = neo4j_count["count"] if neo4j_count else 0
            logger.info("NEO4J ROWS INSERTED: %d", result.neo4j_rows_inserted)
        driver.close()
    except Exception as exc:
        logger.warning("Could not get Neo4j row count: %s", exc)
        result.neo4j_rows_inserted = 0

    # Persist the storage context for downstream query / audit engines.
    storage_path = Path(storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_path))

    # Save checkpoint with completed platforms
    if resume or incremental:
        checkpoint = load_checkpoint()
        if checkpoint.get("start_time") is None:
            checkpoint["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Add completed platform IDs
        for doc in documents:
            platform = doc.metadata.get("platform")
            if platform and platform not in checkpoint.get("completed_platforms", []):
                checkpoint.setdefault("completed_platforms", []).append(platform)
        
        checkpoint["total_platforms"] = 441
        checkpoint["last_run_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        checkpoint["elapsed_seconds"] = result.elapsed_seconds
        save_checkpoint(checkpoint)
        
        completed_count = len(checkpoint.get("completed_platforms", []))
        logger.info("CHECKPOINT: %d/%d platforms completed", completed_count, checkpoint["total_platforms"])

    result.index = index
    result.storage_path = storage_path
    result.elapsed_seconds = time.perf_counter() - t_start

    logger.info("=== INGESTION COMPLETE ===")
    logger.info(result.summary())
    return result


async def ingest_company_on_demand(
    company_name: str,
    *,
    storage_dir: str = "storage",
) -> bool:
    """
    Trigger on-demand ingestion for a specific company/platform.
    
    This function is called when an audit request is made for a platform
    that has no existing data in Neo4j. It fetches documents from OTA,
    parses them, and writes to both Qdrant and Neo4j.
    
    Parameters
    ----------
    company_name:
        Target company name to ingest (e.g., "Netflix", "Spotify").
    storage_dir:
        Local directory for storage persistence.
        
    Returns
    -------
    bool
        True if ingestion succeeded, False otherwise.
    """
    logger.info("=== ON-DEMAND INGESTION: %s ===", company_name)
    
    # Use global settings with Mistral API key
    app_settings = global_settings
    
    # Ensure LLM config is available for graph extraction
    if not app_settings.mistral_api_key:
        logger.warning("No MISTRAL_API_KEY - skipping graph extraction, will use vector-only")
    
    try:
        ota_settings = OTASettings()
        client = OpenTermsArchiveClient(settings=ota_settings)
        
        # Normalize company name to platform ID format (e.g., "Netflix" -> "netflix")
        service_id = company_name.lower().replace(" ", "-").replace("_", "-")
        
        # Fetch ONLY documents for this specific platform
        filtered_docs = await client.fetch_platform_documents(service_id=service_id)
        
        if not filtered_docs:
            # Try alternate lookup - maybe the company_name is partial match
            # Try searching services for partial match
            from ingestion.api_client import OpenTermsArchiveClient
            async with OpenTermsArchiveClient() as search_client:
                services = await search_client.fetch_services()
                for service in services:
                    if company_name.lower() in service.get("name", "").lower():
                        service_id = service.get("id", "")
                        filtered_docs = await client.fetch_platform_documents(service_id=service_id)
                        if filtered_docs:
                            break
        
        if not filtered_docs:
            logger.warning("No documents found for company='%s'", company_name)
            return False
            
        logger.info("Found %d documents for company='%s'", len(filtered_docs), company_name)
        
        # Parse documents into nodes
        nodes = await parse_documents_to_nodes(
            filtered_docs,
            chunk_size=512,
            chunk_overlap=64,
            embed_model_uri="local:BAAI/bge-small-en-v1.5",
        )
        
        # Initialize stores
        run_neo4j_schema_migration(
            url=app_settings.neo4j_uri,
            username=app_settings.neo4j_user,
            password=app_settings.neo4j_password.get_secret_value(),
        )
        
        qdrant_api_key: Optional[str] = (
            app_settings.qdrant_api_key.get_secret_value()
            if app_settings.qdrant_api_key
            else None
        )
        
        vector_store: BasePydanticVectorStore = init_qdrant_store(
            host=app_settings.qdrant_host,
            port=app_settings.qdrant_port,
            grpc_port=app_settings.qdrant_port_grpc,
            api_key=qdrant_api_key,
        )
        
        graph_store: PropertyGraphStore = init_neo4j_store(
            url=app_settings.neo4j_uri,
            username=app_settings.neo4j_user,
            password=app_settings.neo4j_password.get_secret_value(),
        )
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            property_graph_store=graph_store,
        )
        
        # Build index - use ImplicitPathExtractor if no LLM, else use schema extractor
        loop = asyncio.get_event_loop()
        
        if app_settings.mistral_api_key:
            llm = _build_llm(app_settings)
            LlamaSettings.llm = llm
            kg_extractor = build_schema_extractor(llm=llm)
            index: PropertyGraphIndex = await loop.run_in_executor(
                None,
                lambda: PropertyGraphIndex.from_documents(
                    filtered_docs,
                    kg_extractors=[kg_extractor],
                    storage_context=storage_context,
                    show_progress=True,
                ),
            )
        else:
            from llama_index.core.indices.property_graph import ImplicitPathExtractor
            index = await loop.run_in_executor(
                None,
                lambda: PropertyGraphIndex.from_documents(
                    filtered_docs,
                    kg_extractors=[ImplicitPathExtractor()],
                    storage_context=storage_context,
                    show_progress=True,
                ),
            )
        
        # Persist storage
        storage_path = Path(storage_dir).resolve()
        storage_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(storage_path))
        
        logger.info("=== ON-DEMAND INGESTION COMPLETE: %s ===", company_name)
        return True
        
    except Exception as exc:
        logger.error("On-demand ingestion failed for '%s': %s", company_name, exc)
        return False