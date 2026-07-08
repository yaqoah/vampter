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
from typing import Optional

import qdrant_client
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default Qdrant collection name for Vampter document embeddings.
QDRANT_COLLECTION_NAME: str = "vampter_docs"


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
) -> QdrantVectorStore:
    """
    Initialise and return a ``QdrantVectorStore`` connected to the local
    Qdrant instance.

    The collection *collection_name* will be created automatically by
    LlamaIndex on first index build if it does not already exist.

    Parameters
    ----------
    host:
        Qdrant server hostname.
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

    Returns
    -------
    QdrantVectorStore
        A configured vector-store ready to be passed to
        ``PropertyGraphIndex``.
    """
    logger.info(
        "Connecting to Qdrant — host=%s  port=%d  grpc=%d  collection=%s",
        host,
        port,
        grpc_port,
        collection_name,
    )

    client_kwargs: dict = {
        "host": host,
        "port": port,
        "grpc_port": grpc_port,
        "prefer_grpc": prefer_grpc,
    }
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
    url: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = "vampter_neo4j_password",
    *,
    database: Optional[str] = None,
    refresh_schema: bool = True,
) -> Neo4jPropertyGraphStore:
    """
    Initialise and return a ``Neo4jPropertyGraphStore`` connected to the
    local Neo4j instance.

    The store will assert the required graph schema (node labels, relation
    types, and uniqueness constraints) on first connection when
    *refresh_schema* is ``True``.

    Parameters
    ----------
    url:
        Bolt URI for the Neo4j server.
    username:
        Neo4j authentication username.
    password:
        Neo4j authentication password.
    database:
        Optional target database name.  Defaults to the configured default
        database (usually ``neo4j``).
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
        url,
        username,
        database or "<default>",
    )

    store_kwargs: dict = {
        "username": username,
        "password": password,
        "url": url,
        "refresh_schema": refresh_schema,
    }
    if database:
        store_kwargs["database"] = database

    graph_store = Neo4jPropertyGraphStore(**store_kwargs)

    logger.info("Neo4jPropertyGraphStore initialised  url='%s'", url)
    return graph_store
