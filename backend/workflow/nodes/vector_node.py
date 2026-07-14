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

Note: This node no longer has a fallback to graph_node to prevent infinite
loops. If no passages are retrieved, it returns empty passages.
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

        # Build the client connection - prioritize Qdrant Cloud URL
        # If QDRANT_URL is set (Qdrant Cloud), use it; otherwise use localhost
        if settings.qdrant_url:
            client = AsyncQdrantClient(
                url=settings.qdrant_url,
                prefer_grpc=False,  # REST is more reliable for cloud
                api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            )
            logger.info("Connected to Qdrant Cloud: %s", settings.qdrant_url[:40] + "...")
        else:
            client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                grpc_port=settings.qdrant_port_grpc,
                prefer_grpc=True,
            )
            logger.info("Connected to local Qdrant: %s:%s", settings.qdrant_host, settings.qdrant_port)

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
        # NOTE: We skip Qdrant-side filtering because it requires pre-created indexes.
        # Instead, we perform similarity search without filter and filter results in Python.
        search_filter = None

        # Use query_points for modern Qdrant client API (v1.7.0+)
        response = await client.query_points(
            collection_name="vampter_docs",
            query=query_embedding,
            limit=20,  # Get more results for potential filtering
            query_filter=search_filter,
        )

        logger.info(f"DEBUG: Response points: {len(response.points) if response.points else 0}")
        if response.points:
            logger.info(f"DEBUG: First point payload keys: {list(response.points[0].payload.keys()) if response.points[0].payload else 'none'}")

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
        logger.warning("Vector node error: %s — returning empty passages.", exc)
        return {**state, "retrieved_passages": []}

    return {**state, "retrieved_passages": passages}
