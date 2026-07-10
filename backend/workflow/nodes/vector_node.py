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

_COLLECTION_NAME = "vampter_docs"
_TOP_K = 8


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
        from qdrant_client.models import Filter, FieldCondition, MatchText  # type: ignore[import]

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

        # Optional: filter by company / platform metadata if present.
        search_filter = None
        if company_name:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.platform",
                        match=MatchText(text=company_name),
                    )
                ]
            )

        # Use query_points for modern Qdrant client API (v1.7.0+)
        query_response = await client.query_points(
            collection_name=_COLLECTION_NAME,
            query=query_embedding,
            limit=_TOP_K,
            query_filter=search_filter,
            with_payload=True,
        )

        passages = [
            point.payload.get("text", "")
            for point in query_response.points
            if point.payload
        ]

        logger.info(
            "Vector node retrieved %d passage(s) for query='%s'",
            len(passages),
            query[:60],
        )
        await client.close()

    except Exception as exc:
        logger.error("Vector node error: %s — raising exception.", exc)
        raise RuntimeError(
            f"Qdrant vector search failed: {exc}. "
            "Ensure Qdrant is running and the 'vampter_docs' collection is populated."
        ) from exc

    return {**state, "retrieved_passages": passages}
