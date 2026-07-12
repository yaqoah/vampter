"""
workflow.nodes.vector_node
==========================
Dense vector retrieval node (Qdrant path).

Responsibility
--------------
Performs a top-k similarity search against the ``vampter_docs`` Qdrant
collection to retrieve the most semantically relevant text passages
for the incoming user query.

This node is invoked when the router classifies the query as ``"vector"``.
No mock fallback is provided - Qdrant must be available for this node
to function.
"""

from __future__ import annotations

import logging
from typing import List

from config import settings
from workflow.state import AuditState

logger = logging.getLogger(__name__)


async def vector_node(state: AuditState) -> AuditState:
    """
    Retrieve top-k text passages from Qdrant via dense similarity search.

    Parameters
    ----------
    state:
        Current ``AuditState`` with ``query`` and ``company_name``.

    Returns
    -------
    AuditState
        Updated state with ``retrieved_passages`` populated.
    """
    query = state.get("query", "")
    company_name = state.get("company_name", "")
    passages: List[str] = []

    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import]
        from qdrant_client.http.exceptions import UnexpectedResponse  # type: ignore[import]
        from qdrant_client.models import (  # type: ignore[import]
            Filter, FieldCondition, MatchText, Distance, VectorParams, CollectionStatus
        )

        # Build the query embedding using the same local model the ingestion
        # pipeline registered on LlamaSettings.
        from llama_index.core import Settings as LlamaSettings
        embed_model = LlamaSettings.embed_model
        query_embedding = embed_model.get_text_embedding(query)

        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_port_grpc,
            prefer_grpc=True,
        )

        # Ensure collection exists - auto-create if missing
        # BAAI/bge-small-en-v1.5 embeddings: 384 dimensions, Cosine distance
        try:
            collection_info = await client.get_collection(collection_name="vampter_docs")
            if collection_info.status != CollectionStatus.GREEN:
                logger.warning("Collection exists but not ready: %s", collection_info.status)
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info("Creating Qdrant collection 'vampter_docs' with 384 dimensions...")
            await client.create_collection(
                collection_name="vampter_docs",
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

        # Optional: filter by company / platform metadata if present.
        search_filter = None
        if company_name:
            # First, try to resolve company_name to platform ID for filtering
            # by querying Neo4j to map name to ID
            try:
                from neo4j import AsyncGraphDatabase
                auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
                async with AsyncGraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
                    async with driver.session() as session:
                        # Try finding Platform node first (full ingestion)
                        result = await session.run(
                            "MATCH (p:Platform) WHERE toLower(p.name) CONTAINS toLower($name) RETURN p.id AS id LIMIT 1",
                            name=company_name
                        )
                        record = await result.single()
                        if record and record["id"]:
                            platform_id = record["id"]
                            search_filter = Filter(
                                should=[
                                    FieldCondition(
                                        key="metadata.platform",
                                        match=MatchText(text=platform_id),
                                    ),
                                    FieldCondition(
                                        key="properties.platform",
                                        match=MatchText(text=platform_id),
                                    ),
                                ]
                            )
            except Exception:
                pass
            
            # If no Platform node found, try direct filter (seed data compatibility)
            if not search_filter:
                search_filter = Filter(
                    should=[
                        FieldCondition(
                            key="metadata.platform",
                            match=MatchText(text=company_name.lower()),
                        ),
                        FieldCondition(
                            key="properties.platform",
                            match=MatchText(text=company_name.lower()),
                        ),
                    ]
                )

        # Use query_points for modern Qdrant client API (v1.7.0+)
        response = await client.query_points(
            collection_name="vampter_docs",
            query=query_embedding,
            limit=8,
            query_filter=search_filter,
        )

        passages = []
        for point in response.points:
            if point.payload:
                text = None
                
                # Handle _node_content (LlamaIndex JSON payload format)
                if "_node_content" in point.payload:
                    try:
                        import json
                        node_data = json.loads(point.payload["_node_content"])
                        if isinstance(node_data, dict):
                            text = node_data.get("text") or node_data.get("content") or str(node_data)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                
                # Try other possible text field names
                if not text:
                    for key in ["content", "text", "node_content", "chunk_text"]:
                        if key in point.payload:
                            val = point.payload[key]
                            if isinstance(val, str) and val.strip():
                                text = val
                                break
                            elif isinstance(val, dict):
                                text = val.get("text") or val.get("content") or str(val)
                                if text and str(text).strip():
                                    break
                
                # Final fallback: use any string value in payload that looks like text
                if not text:
                    for key, val in point.payload.items():
                        if isinstance(val, str) and len(val) > 50 and key not in ["id", "doc_id"]:
                            text = val
                            break
                
                if text and str(text).strip():
                    passages.append(str(text))
        
        # Log detailed passage info for debugging
        logger.info(
            "Vector node: %d points, %d non-empty passages, platform filter=%s for query='%s'",
            len(response.points) if response.points else 0,
            len(passages),
            "yes" if search_filter else "no",
            query[:60],
        )
        
        # DEBUG: Log payload keys to understand structure
        if response.points and response.points[0].payload:
            payload_keys = list(response.points[0].payload.keys())
            logger.info(f"DEBUG: First point payload keys: {payload_keys}")
            # Log full payload for debugging
            import json
            payload_str = json.dumps(response.points[0].payload, indent=2, default=str)[:500]
            logger.info(f"DEBUG: Full payload: {payload_str}")
        
        # DEBUG: Log actual passage content for troubleshooting
        if passages:
            logger.info("Vector node: first passage preview (100 chars): %s", passages[0][:100].replace('\n', ' '))
        else:
            logger.warning("Vector node: NO PASSAGES RETRIEVED for query='%s'", query[:60])
        
        await client.close()

    except Exception as exc:
        logger.error("Vector node error: %s — raising exception.", exc)
        raise RuntimeError(
            f"Qdrant vector search failed: {exc}. "
            "Ensure Qdrant is running and the 'vampter_docs' collection is populated."
        ) from exc

    return {**state, "retrieved_passages": passages}
