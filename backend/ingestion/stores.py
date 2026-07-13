"""
ingestion.stores
================
Store initialisation helpers for the dual-backend indexing strategy.

Dual-store design
-----------------
Vampter's ``PropertyGraphIndex`` writes to **two** complementary backends
simultaneously:

1. **Qdrant** (Vector Store)
   Dense embedding vectors for every parsed text chunk are stored in a
   Qdrant collection named ``vampter_docs``.  This enables fast
   approximate-nearest-neighbour semantic similarity retrieval during
   audit queries.

2. **Neo4j** (Property Graph Store)
   Structured legal knowledge-graph triples extracted by the
   ``SchemaLLMPathExtractor`` are persisted as labelled nodes and typed
   directed edges.  This enables Cypher-based traversal, clause-level
   fact checking, and cross-document relationship discovery.

Connection handling
-------------------
Both client objects are constructed synchronously.  Actual network I/O
(collection creation, schema assertion) is deferred until the index is
first used, so these helpers are lightweight and safe to call at import
time.
"""

from __future__ import annotations

import logging
import ssl
import time
from typing import Optional

import qdrant_client
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from neo4j.exceptions import SessionExpired, ServiceUnavailable, TransientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default Qdrant collection name for Vampter document embeddings.
QDRANT_COLLECTION_NAME: str = "vampter_docs"


# ---------------------------------------------------------------------------
# Retry utilities
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0  # seconds

# Errors that indicate transient connection issues worth retrying
RETRIABLE_EXCEPTIONS = (
    SessionExpired,
    ServiceUnavailable,
    TransientError,
    ConnectionError,
    OSError,
    ssl.SSLEOFError,
    ssl.SSLError,
)


def check_neo4j_connection(
    url: Optional[str] = None,
    username: str = "neo4j",
    password: str = "vampter_neo4j_password",
    database: Optional[str] = None,
) -> bool:
    """
    Check if Neo4j connection is healthy and reconnect if needed.
    
    This function pings the Neo4j server to verify connectivity,
    particularly useful before long-running operations that may
    cause connection timeouts in GitHub Actions.
    
    Returns True if connection is healthy, False otherwise.
    """
    if not url:
        logger.warning("Neo4j URL not configured - cannot check connection")
        return False
    
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            url,
            auth=(username, password),
            keep_alive=True,
            max_connection_lifetime=30 * 60,
            max_connection_pool_size=50,
        )
        
        db_arg = {"database": database} if database else {}
        
        with driver.session(**db_arg) as session:
            result = session.run("RETURN 1 as health")
            _ = result.single()
        
        driver.close()
        logger.info("Neo4j connection health check passed")
        return True
    except RETRIABLE_EXCEPTIONS as exc:
        logger.warning("Neo4j connection health check failed: %s", exc)
        return False
    except Exception as exc:
        logger.warning("Neo4j connection health check error: %s", exc)
        return False


def _with_retry(func, *args, **kwargs):
    """
    Execute a function with exponential backoff retry logic.
    
    Retries on Neo4j transient errors (SessionExpired, ServiceUnavailable, etc.)
    """
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except RETRIABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    "Neo4j operation failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, exc
                )
                time.sleep(delay)
            else:
                logger.error("Neo4j operation failed after %d attempts", MAX_RETRIES)
                raise
    raise last_exception


# ---------------------------------------------------------------------------
# Schema Migration
# ---------------------------------------------------------------------------

def run_neo4j_schema_migration(
    url: Optional[str] = None,
    username: str = "neo4j",
    password: str = "vampter_neo4j_password",
) -> Optional[str]:
    """
    Run Neo4j schema migration to create uniqueness constraints.
    
    Returns the detected/created database name, or None if migration failed.
    Uses retry logic with exponential backoff for transient connection errors.
    """
    from neo4j import GraphDatabase
    
    logger.info("Running Neo4j schema migration...")
    
    if not url:
        logger.warning("Neo4j URL not configured - skipping schema migration")
        return None
    
    # Configure driver with appropriate timeouts for cloud connections
    driver = GraphDatabase.driver(
        url,
        auth=(username, password),
        # Neo4j Aura connections can timeout after 60s idle - use shorter keepalive
        keep_alive=True,
        max_connection_lifetime=30 * 60,  # 30 minutes
        max_connection_pool_size=50,
    )
    
    constraints = [
        "CREATE CONSTRAINT platform_name_unique IF NOT EXISTS FOR (p:Platform) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT revision_id_unique IF NOT EXISTS FOR (r:Revision) REQUIRE r.id IS UNIQUE",
    ]
    
    working_db = None
    
    # List available databases (use implicit default - Aura routes to working DB automatically)
    available_dbs = []
    try:
        # First try connecting without specifying database - uses implicit default
        with driver.session() as session:
            # This query works in the implicitly routed database
            result = session.run("SHOW DATABASES YIELD name")
            available_dbs = [rec["name"] for rec in result.data()]
            logger.info("Available databases: %s", available_dbs)
            # If we can list databases, we're already connected to the working one
            # Return the first non-system database as the working one
            for db in available_dbs:
                if db != "system":
                    # Found a working database via implicit routing - run constraints now
                    working_db = db
                    logger.info("Using database for schema migration (via implicit routing): %s", working_db)
                    for constraint in constraints:
                        try:
                            session.run(constraint)
                            logger.info("Executed: %s", constraint)
                        except Exception as exc:
                            logger.warning("Constraint failed: %s - %s", constraint, exc)
                    logger.info("Successfully ran constraints on database: %s", working_db)
                    break
    except Exception as exc:
        logger.warning("Cannot list databases with implicit session: %s", exc)
    
    # If implicit didn't work, try explicit database names
    aura_db_id_candidates = []
    for db_name in available_dbs:
        if len(db_name) == 8 and db_name.isalnum() and db_name not in ["system", "neo4j"]:
            aura_db_id_candidates.append(db_name)
    
    # Build priority list: 
    # 1. If neo4j exists, use it (standard default)
    # 2. Otherwise, use first discovered non-system database
    if "neo4j" in available_dbs:
        db_priority = ["neo4j"] + [d for d in available_dbs if d not in ["neo4j", "system"]]
    elif available_dbs and aura_db_id_candidates:
        # Neo4j doesn't exist - try Aura ID databases first, then any other
        db_priority = aura_db_id_candidates + [d for d in available_dbs if d not in ["system", "neo4j"] + aura_db_id_candidates]
    else:
        # No databases discovered - try implicit connection (no database param)
        logger.info("No databases discovered - will try implicit default connection")
        db_priority = []
    
    for db_name in db_priority:
        try:
            with driver.session(database=db_name) as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logger.info("Executed on %s: %s", db_name, constraint)
                    except Exception as exc:
                        logger.warning("Constraint failed on %s: %s - %s", db_name, constraint, exc)
                logger.info("Successfully connected to database: %s", db_name)
                working_db = db_name
                break
        except Exception as db_exc:
            logger.warning("Database '%s' not accessible: %s", db_name, db_exc)
    
    driver.close()
    logger.info("Neo4j schema migration completed. Working DB: %s", working_db or "implicit")
    return working_db


# ---------------------------------------------------------------------------
# Qdrant Vector Store
# ---------------------------------------------------------------------------

def init_qdrant_store(
    host: str = "localhost",
    port: int = 6333,
    grpc_port: int = 6334,
    api_key: Optional[str] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    *,
    prefer_grpc: bool = True,
    url: Optional[str] = None,  # Qdrant Cloud URL (overrides host/port)
) -> QdrantVectorStore:
    """
    Initialise and return a ``QdrantVectorStore`` connected to the local
    Qdrant instance or Qdrant Cloud.

    The collection *collection_name* will be created automatically by
    LlamaIndex on first index build if it does not already exist.

    Parameters
    ----------
    host:
        Qdrant server hostname (for local instances).
    port:
        Qdrant REST API port (default: 6333).
    grpc_port:
        Qdrant gRPC port for high-throughput upserts (default: 6334).
    api_key:
        Optional Qdrant API key for authenticated clusters.
    collection_name:
        Name of the Qdrant collection that will hold document embeddings.
    prefer_grpc:
        Use gRPC transport for batch upsert operations (faster than REST
        for large ingestion payloads).
    url:
        Full Qdrant Cloud URL (e.g., https://xyzabc.cloud.qdrant.io). 
        Overrides host/port when provided.

    Returns
    -------
    QdrantVectorStore
        A configured vector-store ready to be passed to
        ``PropertyGraphIndex``.
    """
    logger.info(
        "Connecting to Qdrant — host=%s  url=%s  collection=%s",
        host,
        url or "local",
        collection_name,
    )

    client_kwargs: dict = {}
    
    # Qdrant Cloud uses URL, local uses host/port
    if url:
        client_kwargs["url"] = url
        # Use REST for cloud (gRPC can have connectivity issues)
        client_kwargs["prefer_grpc"] = False
    else:
        client_kwargs["host"] = host or "localhost"
        client_kwargs["port"] = port
        client_kwargs["grpc_port"] = grpc_port
        client_kwargs["prefer_grpc"] = prefer_grpc
    
    if api_key:
        client_kwargs["api_key"] = api_key

    qdrant_client_obj = qdrant_client.QdrantClient(**client_kwargs)

    vector_store = QdrantVectorStore(
        client=qdrant_client_obj,
        collection_name=collection_name,
    )

    logger.info(
        "QdrantVectorStore initialised  collection='%s'", collection_name
    )
    return vector_store


# ---------------------------------------------------------------------------
# Neo4j Property Graph Store
# ---------------------------------------------------------------------------

def init_neo4j_store(
    url: Optional[str] = None,
    username: str = "neo4j",
    password: str = "vampter_neo4j_password",
    *,
    database: Optional[str] = None,
    refresh_schema: bool = True,
) -> Neo4jPropertyGraphStore:
    """
    Initialise and return a ``Neo4jPropertyGraphStore`` connected to the
    local Neo4j instance or Neo4j Aura cloud.

    The store will assert the required graph schema (node labels, relation
    types, and uniqueness constraints) on first connection when
    *refresh_schema* is ``True``.

    Uses retry logic with exponential backoff for transient connection errors.

    Parameters
    ----------
    url:
        Bolt URI for the Neo4j server.
    username:
        Neo4j authentication username.
    password:
        Neo4j authentication password.
    database:
        Optional target database name.  If not specified, uses 'neo4j' which
        is the default in Neo4j Aura.
    refresh_schema:
        When ``True`` (default), the store refreshes its internal schema
        cache after connecting.  Required the first time after data is
        written to ensure Cypher completions are accurate.

    Returns
    -------
    Neo4jPropertyGraphStore
        A configured graph-store ready to be passed to
        ``PropertyGraphIndex``.
    """
    logger.info(
        "Connecting to Neo4j — url=%s  user=%s  database=%s",
        url or "(not configured)",
        username,
        database or "neo4j (default)",
    )

    if not url:
        raise RuntimeError(
            "NEO4J_URI is not configured. Set NEO4J_URI environment variable."
        )

    # LlamaIndex Neo4jPropertyGraphStore doesn't expose driver config directly,
    # but we can set certain parameters for better connection handling
    store_kwargs: dict = {
        "username": username,
        "password": password,
        "url": url,
        "refresh_schema": refresh_schema,
    }
    # If database is None, let LlamaIndex use its implicit default (don't pass database param)
    # This allows Neo4j to route to the correct database automatically
    if database:
        store_kwargs["database"] = database
    else:
        # Log that we're using implicit default
        logger.info("Using Neo4j implicit default database (will use connected DB routing)")

    # Pass Neo4j driver configuration via neo4j_kwargs for better connection handling.
    # keep_alive prevents connection from being closed due to inactivity.
    # Note: These are passed to the neo4j driver via **neo4j_kwargs
    neo4j_driver_kwargs = {
        "keep_alive": True,
        "max_connection_lifetime": 30 * 60,  # 30 minutes before refreshing connections
        "max_connection_pool_size": 50,
    }
    # Add driver config to store_kwargs - these go to **neo4j_kwargs
    store_kwargs.update(neo4j_driver_kwargs)

    # Try to create the graph store with retries for transient errors
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            graph_store = Neo4jPropertyGraphStore(**store_kwargs)
            break
        except RETRIABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.warning(
                    "Neo4j store initialization failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, exc
                )
                time.sleep(delay)
            else:
                logger.error("Neo4j store initialization failed after %d attempts", MAX_RETRIES)
                raise

    logger.info("Neo4jPropertyGraphStore initialised  url='%s'", url)
    return graph_store