"""
ingestion.api_client
====================
Async HTTP client targeting the Open Terms Archive (OTA) REST API schema.

Responsibility
--------------
Fetch the latest terms revisions JSON structures from the live network,
parse them, and yield typed ``llama_index.core.Document`` objects enriched
with structured metadata.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import List, Optional

import httpx
from llama_index.core import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Settings for OTA API Configuration
# ---------------------------------------------------------------------------

class OTASettings(BaseModel):
    """
    Configuration for Open Terms Archive API client.
    
    Loaded from environment variables via AppSettings or direct instantiation.
    """
    base_url: str = Field(
        default="https://api.opentermsarchive.org/v1",
        description="Base URL for the Open Terms Archive REST API"
    )
    timeout: float = Field(
        default=60.0,
        description="HTTP request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for failed requests"
    )


# ---------------------------------------------------------------------------
# OpenTermsArchiveClient
# ---------------------------------------------------------------------------

class OpenTermsArchiveClient:
    """
    Asynchronous client for the Open Terms Archive REST API.
    
    Implements the hierarchical pipeline pattern:
    1. Fetch services index via /services
    2. For each service, fetch document versions via /services/{serviceId}/documents/{documentId}/versions
    
    The client produces clean LlamaIndex Document objects with:
    - text: Pure legal text content (no JSON formatting)
    - metadata: Structural identifiers (platform, document_type, revision, snapshot_time)
    """

    def __init__(self, settings: Optional[OTASettings] = None):
        """
        Initialize the OTA client with connection configuration.
        
        Parameters
        ----------
        settings:
            Pydantic OTASettings instance. If None, uses defaults.
        """
        self._settings = settings or OTASettings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Lazily initialize and return the httpx.AsyncClient.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.base_url,
                timeout=self._settings.timeout,
            )
        return self._client

    async def close(self) -> None:
        """
        Close the HTTP client connection.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OpenTermsArchiveClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def fetch_services(self) -> List[dict]:
        """
        Fetch the comprehensive index of tracked platforms.
        
        Executes a GET request to the /services endpoint.
        This data fuels the frontend autocomplete UI dropdown elements.
        
        Returns
        -------
        List[dict]
            List of service/platform records from the OTA API.
            Each record contains at minimum: id, name, and other platform metadata.
        """
        client = await self._get_client()
        
        try:
            logger.info("Fetching services index from /services endpoint")
            response = await client.get("/services")
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                logger.warning(
                    "Unexpected /services response format. Expected list, got %s",
                    type(data).__name__
                )
                return []
            
            logger.info("Successfully fetched %d service(s)", len(data))
            return data
            
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP status error fetching services: %s — status_code=%s",
                exc,
                exc.response.status_code if exc.response else "unknown"
            )
            return []
        except httpx.RequestError as exc:
            logger.error("Request error fetching services: %s", exc)
            return []
        except Exception as exc:
            logger.error("Unexpected error fetching services: %s", exc, exc_info=True)
            return []

    async def fetch_document_versions(
        self,
        service_id: str,
        document_id: str
    ) -> List[dict]:
        """
        Fetch chronological historical snapshots for a specific document.
        
        Executes a GET request to:
        GET /services/{serviceId}/documents/{documentId}/versions
        
        Parameters
        ----------
        service_id:
            The platform/service identifier (e.g., "amazon", "google").
        document_id:
            The document identifier (e.g., "privacy-policy", "terms-of-service").
            
        Returns
        -------
        List[dict]
            List of version records, each containing:
            - id: version identifier or hash
            - content: the legal text content
            - snapshot_time: ISO timestamp of the version
            - other version metadata
        """
        client = await self._get_client()
        
        try:
            logger.info(
                "Fetching document versions: service_id=%s  document_id=%s",
                service_id,
                document_id
            )
            encoded_document_id = urllib.parse.quote(document_id, safe='')
            endpoint = f"/services/{service_id}/documents/{encoded_document_id}/versions"
            response = await client.get(endpoint)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                logger.warning(
                    "Unexpected versions response format for %s/%s. Expected list, got %s",
                    service_id,
                    document_id,
                    type(data).__name__
                )
                return []
            
            logger.info(
                "Successfully fetched %d version(s) for service=%s document=%s",
                len(data),
                service_id,
                document_id
            )
            return data
            
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP status error fetching document versions: service_id=%s document_id=%s error=%s",
                service_id,
                document_id,
                exc,
                exc_info=True
            )
            return []
        except httpx.RequestError as exc:
            logger.error(
                "Request error fetching document versions: service_id=%s document_id=%s error=%s",
                service_id,
                document_id,
                exc,
                exc_info=True
            )
            return []
        except Exception as exc:
            logger.error(
                "Unexpected error fetching document versions: service_id=%s document_id=%s error=%s",
                service_id,
                document_id,
                exc,
                exc_info=True
            )
            return []

    def create_llama_documents(
        self,
        version_records: List[dict],
        service_id: str,
        document_id: str
    ) -> List[Document]:
        """
        Parse OTA version records into pure LlamaIndex Document objects.
        
        Enforces absolute separation of structural metadata and natural language text
        to prevent vector space pollution.
        
        Parameters
        ----------
        version_records:
            List of version dictionaries from fetch_document_versions().
        service_id:
            The platform/service identifier for metadata.
        document_id:
            The document type identifier for metadata.
            
        Returns
        -------
        List[Document]
            LlamaIndex Document objects with clean text and structured metadata.
        """
        documents: List[Document] = []
        
        for record in version_records:
            # Extract clean legal text content
            # Priority: content field, then clauses text, then raw text
            text_content = self._extract_clean_text(record)
            
            # Build metadata with clean primitives
            metadata = {
                "platform": service_id,
                "document_type": document_id,
                "revision": record.get("id", record.get("version", "unknown")),
                "snapshot_time": record.get("snapshot_time", record.get("timestamp", record.get("created_at", ""))),
            }
            
            # Create Document with clean text and metadata
            doc = Document(
                text=text_content,
                metadata=metadata,
            )
            documents.append(doc)
        
        logger.info(
            "Created %d LlamaIndex Document(s) for service=%s document=%s",
            len(documents),
            service_id,
            document_id
        )
        return documents

    def _extract_clean_text(self, record: dict) -> str:
        """
        Extract pure legal text content from a version record.
        
        Removes any JSON syntax formatting and returns only the raw text.
        
        Parameters
        ----------
        record:
            A single version record dictionary.
            
        Returns
        -------
        str
            Clean legal text content.
        """
        # Try to get the main content field
        content = record.get("content", "")
        
        # If no content, try to extract from clauses
        if not content and "clauses" in record:
            clauses = record.get("clauses", [])
            if isinstance(clauses, list):
                # Extract text from each clause and join
                clause_texts = []
                for clause in clauses:
                    if isinstance(clause, dict):
                        clause_text = clause.get("text", "")
                        if clause_text:
                            clause_texts.append(clause_text)
                    elif isinstance(clause, str):
                        clause_texts.append(clause)
                content = "\n\n".join(clause_texts)
        
        # If still no content, try raw text field
        if not content:
            content = record.get("text", "")
        
        # Ensure we return a string
        return str(content) if content else ""

    async def fetch_all_documents(
        self,
        service_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Execute the full hierarchical ingestion pipeline with dynamic discovery.
        
        1. Fetch services index via fetch_services()
        2. For each service, dynamically inspect available documents schema.
        3. Extract actual keys from the 'documents' field to prevent 404 validation loops.
        4. Apply set intersection if document_ids filter is provided.
        5. Fetch and parse document versions into LlamaIndex Document objects.
        
        Parameters
        ----------
        service_ids:
            Optional list of specific service IDs to fetch.
            If None, fetches all services from the index.
        document_ids:
            Optional list of specific document IDs to filter by.
            
        Returns
        -------
        List[Document]
            All successfully parsed Document objects.
        """
        all_documents: List[Document] = []
        
        # Step 1: Get services index
        services = await self.fetch_services()
        
        if not services:
            logger.warning("No services found — returning empty document list")
            return []
        
        # Filter to specific service IDs if provided
        if service_ids:
            services = [s for s in services if s.get("id") in service_ids]
        
        # Step 2: Iterate through services and fetch document versions dynamically
        for service in services:
            service_id = service.get("id", service.get("name", ""))
            if not service_id:
                continue
                
            documents_schema = service.get("documents", {})
            if not documents_schema:
                logger.warning("Service object for service_id=%s contains an entirely empty 'documents' schema.", service_id)
                continue
                
            discovered_keys = list(documents_schema.keys())
            
            # Use set intersection if an explicit filter override list was supplied
            if document_ids:
                query_targets = set(document_ids).intersection(set(discovered_keys))
            else:
                query_targets = set(discovered_keys)
                
            for document_id in query_targets:
                # Format URL properly: extracted dynamic keys may contain path forward-slashes.
                # In most standard REST routers, we pass this straight to GET /services/{serviceId}/documents/{documentId}/versions.
                version_records = await self.fetch_document_versions(
                    service_id=service_id,
                    document_id=document_id
                )
                
                if version_records:
                    documents = self.create_llama_documents(
                        version_records=version_records,
                        service_id=service_id,
                        document_id=document_id
                    )
                    all_documents.extend(documents)
        
        logger.info("Total documents fetched: %d", len(all_documents))
        return all_documents
