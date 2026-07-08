"""
ingestion.api_client
====================
Async HTTP client targeting the Open Terms Archive Federation/Collection API.

Responsibility
--------------
Fetch the latest terms revisions JSON structures from the live network,
parse them, and yield typed ``llama_index.core.Document`` objects enriched
with structured metadata.
"""

from __future__ import annotations

import json
import logging
from typing import List

import httpx
from llama_index.core import Document

logger = logging.getLogger(__name__)


async def async_fetch_api_documents(
    api_url: str = "https://api.example-ota.org/v1/documents",
) -> List[Document]:
    """
    Asynchronously fetch document revisions from the OTA API.

    Parameters
    ----------
    api_url:
        The target API endpoint to pull JSON data from.

    Returns
    -------
    List[Document]
        All successfully parsed ``Document`` objects. The raw JSON record is
        stringified into the document's text content.
    """
    logger.info("Fetching OTA JSON records from API: %s", api_url)

    documents: List[Document] = []

    # Mock implementation of the API response since we are using a stub URL.
    # In production, this would be an actual httpx.AsyncClient().get() call.
    mock_api_data = [
        {
            "id": "android_auto_privacy_policy_v4_1_0",
            "platform": "Android Auto",
            "document_type": "Privacy Policy",
            "revision": "v4.1.0",
            "content": "This policy governs the collection of personal data...",
            "clauses": [
                {"id": "§1", "text": "Scope of Application. This policy applies to all OTA updates."},
                {"id": "§2", "text": "Data Retention. Data is retained for 90 days."}
            ]
        },
        {
            "id": "ios_carplay_dpa_v2_3_1",
            "platform": "iOS CarPlay",
            "document_type": "Data Processing Agreement",
            "revision": "v2.3.1",
            "content": "This DPA regulates personal data processing...",
            "clauses": [
                {"id": "§1", "text": "Parties and Purpose. Regulating personal data processing."},
                {"id": "§2", "text": "Retention. Data is retained for 12 months."}
            ]
        }
    ]

    try:
        # Simulate network delay/fetch
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(api_url)
        #     response.raise_for_status()
        #     data = response.json()
        data = mock_api_data

        if not isinstance(data, list):
            logger.warning("Unexpected API response format. Expected a list of records.")
            return []

        for record in data:
            doc_id = record.get("id", "unknown_id")
            
            # Stringify the JSON record so the LlamaIndex SentenceSplitter can chunk it
            text_content = json.dumps(record, indent=2)

            metadata = {
                "platform": record.get("platform", "unknown"),
                "document_type": record.get("document_type", "unknown"),
                "revision": record.get("revision", "unknown"),
                "source_api": api_url,
            }

            doc = Document(
                text=text_content,
                id_=doc_id,
                metadata=metadata,
            )
            documents.append(doc)

        logger.info("Successfully fetched %d document(s) from API.", len(documents))

    except httpx.HTTPError as exc:
        logger.error("HTTP error occurred while fetching from API: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error fetching from API: %s", exc)

    return documents
