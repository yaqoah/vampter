"""
ingestion.api_client
====================
Async HTTP client targeting the Open Terms Archive GitHub repositories.

Responsibility
--------------
Fetch platform data from OTA version repositories and yield typed
``llama_index.core.Document`` objects enriched with structured metadata.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.parse
from typing import List, Optional

import httpx
from llama_index.core import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Settings for OTA GitHub Configuration
# ---------------------------------------------------------------------------

class OTASettings(BaseModel):
    """
    Configuration for Open Terms Archive data sources.
    
    Queries GitHub repositories containing platform version data.
    """
    ota_repos: List[str] = Field(
        default=[
            "OpenTermsArchive/genai-contrib-versions",
            "OpenTermsArchive/pga-versions",
            "OpenTermsArchive/contrib-versions",
        ],
        description="OTA GitHub repos containing platform version data"
    )
    timeout: float = Field(
        default=60.0,
        description="HTTP request timeout in seconds"
    )


# ---------------------------------------------------------------------------
# OpenTermsArchiveClient
# ---------------------------------------------------------------------------

def _get_github_auth_headers() -> dict:
    """Extract GitHub auth headers from environment."""
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return {
            "User-Agent": "Vampter-App",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    return {"User-Agent": "Vampter-App"}

# Document type mappings for OTA naming conventions
DOCUMENT_TYPE_MAPPINGS = {
    "privacy-policy": ["Privacy Policy.md", "privacy-policy.json", "Privacy-Policy.md"],
    "terms-of-service": ["Terms of Service.md", "terms-of-service.json", "ToS.md"],
    "terms": ["Terms of Service.md", "terms.json", "ToS.md"],
}

# Pre-cached platform directory mappings (populated on first access)
PLATFORM_DIR_CACHE: dict[str, str] = {}


class OpenTermsArchiveClient:
    """
    Client for fetching platform data from OTA GitHub repos.
    
    Extracts platform names from repository directory structure.
    Each top-level directory in an OTA repo represents one platform.
    """

    def __init__(self, settings: Optional[OTASettings] = None):
        self._settings = settings or OTASettings()
        self._dir_cache: dict[str, str] = {}  # platform_id -> dir_name
        self._services_cache: dict[str, dict] = {}  # platform_id -> service info

    async def _build_dir_cache(self) -> None:
        """Populate platform directory cache and services list in one pass."""
        if self._dir_cache:
            return
        
        github_client = httpx.AsyncClient(
            base_url="https://api.github.com",
            timeout=self._settings.timeout,
            headers=_get_github_auth_headers(),
        )
        
        try:
            for repo in self._settings.ota_repos:
                page = 1
                while True:
                    response = await github_client.get(
                        f"/repos/{repo}/contents",
                        params={"per_page": 100, "page": page}
                    )
                    if response.status_code != 200:
                        logger.warning("Failed to fetch from %s: status %s", repo, response.status_code)
                        break
                    
                    contents = response.json()
                    if not isinstance(contents, list):
                        break
                    
                    if not contents:
                        break
                    
                    for item in contents:
                        if item.get("type") == "dir":
                            platform_dir = item.get("name", "")
                            if platform_dir and platform_dir not in [".github", ".git"]:
                                platform_id = platform_dir.lower().replace(" ", "-").replace("_", "-")
                                # Only set if not already present (dedupe across repos)
                                if platform_id not in self._dir_cache:
                                    self._dir_cache[platform_id] = platform_dir
                                    self._services_cache[platform_id] = {
                                        "id": platform_id, 
                                        "name": platform_dir,
                                        "dir_name": platform_dir
                                    }
                    
                    link_header = response.headers.get("Link", "")
                    if 'rel="next"' not in link_header:
                        break
                    page += 1
            
            logger.info("Cached %d platform directory mappings", len(self._dir_cache))
        finally:
            await github_client.aclose()
    
    async def _ensure_dir_cache(self) -> None:
        """Backward compatibility - redirects to _build_dir_cache."""
        await self._build_dir_cache()

    def fetch_services_sync(self) -> List[dict]:
        """
        Synchronously fetch the index of tracked platforms from OTA GitHub repos.
        
        Returns
        -------
        List[dict]
            Platform records: [{"id": "netflix", "name": "Netflix"}, ...]
        """
        platforms = {}
        
        github_client = httpx.Client(
            base_url="https://api.github.com",
            timeout=self._settings.timeout,
            headers=_get_github_auth_headers(),
        )
        
        try:
            for repo in self._settings.ota_repos:
                page = 1
                while True:
                    response = github_client.get(
                        f"/repos/{repo}/contents",
                        params={"per_page": 100, "page": page}
                    )
                    if response.status_code != 200:
                        logger.warning("Failed to fetch from %s: status %s", repo, response.status_code)
                        break
                    
                    contents = response.json()
                    if not isinstance(contents, list):
                        break
                    
                    if not contents:
                        break
                    
                    for item in contents:
                        if item.get("type") == "dir":
                            platform_dir = item.get("name", "")
                            if platform_dir and platform_dir not in [".github", ".git"]:
                                platform_id = platform_dir.lower().replace(" ", "-").replace("_", "-")
                                platform_name = platform_dir
                                platforms[platform_id] = {"id": platform_id, "name": platform_name}
                    
                    link_header = response.headers.get("Link", "")
                    if 'rel="next"' not in link_header:
                        break
                    page += 1
            
            platforms_list = list(platforms.values())
            # Also populate the caches for async methods
            self._dir_cache.update({p["id"]: p.get("dir_name", p["id"]) for p in platforms_list})
            self._services_cache.update({p["id"]: p for p in platforms_list})
            logger.info("Fetched %d unique platforms from OTA GitHub repos (sync)", len(platforms_list))
            return platforms_list
            
        except Exception as exc:
            logger.error("Failed to fetch services from GitHub: %s", exc)
            return []
        finally:
            github_client.close()

    async def fetch_services(self) -> List[dict]:
        """
        Fetch the index of tracked platforms from OTA GitHub repos.
        
        Returns
        -------
        List[dict]
            Platform records: [{"id": "netflix", "name": "Netflix"}, ...]
        """
        # Use cached data if available
        if self._services_cache:
            return list(self._services_cache.values())
        
        platforms = {}
        
        github_client = httpx.AsyncClient(
            base_url="https://api.github.com",
            timeout=self._settings.timeout,
            headers=_get_github_auth_headers(),
        )
        
        try:
            for repo in self._settings.ota_repos:
                # Fetch all pages using pagination
                page = 1
                while True:
                    response = await github_client.get(
                        f"/repos/{repo}/contents",
                        params={"per_page": 100, "page": page}
                    )
                    if response.status_code != 200:
                        logger.warning("Failed to fetch from %s: status %s", repo, response.status_code)
                        break
                    
                    contents = response.json()
                    if not isinstance(contents, list):
                        break
                    
                    if not contents:
                        break
                    
                    for item in contents:
                        if item.get("type") == "dir":
                            platform_dir = item.get("name", "")
                            if platform_dir and platform_dir not in [".github", ".git"]:
                                platform_id = platform_dir.lower().replace(" ", "-").replace("_", "-")
                                platform_name = platform_dir
                                # Store both id and original directory name for lookups
                                platforms[platform_id] = {"id": platform_id, "name": platform_name, "dir_name": platform_dir}
                    
                    # Check if there are more pages
                    link_header = response.headers.get("Link", "")
                    if 'rel="next"' not in link_header:
                        break
                    page += 1
            
            platforms_list = list(platforms.values())
            # Populate both caches
            self._dir_cache.update({p["id"]: p["dir_name"] for p in platforms_list})
            self._services_cache.update({p["id"]: p for p in platforms_list})
            logger.info("Fetched %d unique platforms from OTA GitHub repos", len(platforms_list))
            return platforms_list
            
        except Exception as exc:
            logger.error("Failed to fetch services from GitHub: %s", exc)
            return []
        finally:
            await github_client.aclose()

    async def fetch_document_versions(
        self,
        service_id: str,
        document_id: str,
        dir_name: Optional[str] = None
    ) -> List[dict]:
        """
        Fetch policy versions for a specific platform/document.
        
        For OTA GitHub repos, versions are stored as .md files within platform directories.
        Uses cached directory mappings when available.
        """
        github_client = httpx.AsyncClient(
            base_url="https://api.github.com",
            timeout=self._settings.timeout,
            headers=_get_github_auth_headers(),
        )
        
        try:
            # Use provided dir_name or resolve from cache
            if not dir_name:
                # Use cached directory name if available
                dir_name = self._dir_cache.get(service_id)
                if not dir_name:
                    # Fallback to direct resolution (less efficient)
                    dir_name = await self._resolve_platform_name(github_client, service_id)
                    if dir_name:
                        self._dir_cache[service_id] = dir_name
                
                if not dir_name:
                    logger.warning("Platform '%s' not found in any OTA repo", service_id)
                    return []
            
            # Get possible document filenames for this document type
            doc_filenames = DOCUMENT_TYPE_MAPPINGS.get(document_id, [f"{document_id}.md", f"{document_id}.json"])
            
            # For each OTA repo, look for the document file
            for repo in self._settings.ota_repos:
                for doc_name in doc_filenames:
                    # URL-encode the filename (for spaces in "Privacy Policy.md")
                    encoded_name = urllib.parse.quote(doc_name)
                    path = f"/repos/{repo}/contents/{dir_name}/{encoded_name}"
                    response = await github_client.get(path)
                    if response.status_code == 200:
                        file_info = response.json()
                        download_url = file_info.get("download_url")
                        if download_url:
                            # Fetch raw markdown content
                            async with httpx.AsyncClient(timeout=self._settings.timeout) as content_client:
                                content_response = await content_client.get(download_url)
                                if content_response.status_code == 200:
                                    # Parse markdown into structured format
                                    content = content_response.text
                                    return [{
                                        "id": "latest",
                                        "content": content,
                                        "source": download_url
                                    }]
            
            # If no specific file, try to list platform directory for all .md files
            for repo in self._settings.ota_repos:
                dir_path = f"/repos/{repo}/contents/{dir_name}"
                response = await github_client.get(dir_path)
                if response.status_code == 200:
                    files = response.json()
                    if isinstance(files, list):
                        versions = []
                        for f in files:
                            if f.get("type") == "file" and f.get("name", "").endswith(".md"):
                                download_url = f.get("download_url")
                                if download_url:
                                    async with httpx.AsyncClient(timeout=self._settings.timeout) as content_client:
                                        ver_response = await content_client.get(download_url)
                                        if ver_response.status_code == 200:
                                            versions.append({
                                                "id": f.get("name"),
                                                "content": ver_response.text,
                                                "source": download_url
                                            })
                        return versions
            
            return []
            
        except Exception as exc:
            logger.error("Failed to fetch document versions: %s", exc)
            return []
        finally:
            await github_client.aclose()

    async def _resolve_platform_name(self, client: httpx.AsyncClient, service_id: str) -> Optional[str]:
        """Find the actual case-preserved platform directory name in OTA repos."""
        service_lower = service_id.lower()
        
        for repo in self._settings.ota_repos:
            response = await client.get(f"/repos/{repo}/contents")
            if response.status_code == 200:
                items = response.json()
                if isinstance(items, list):
                    for item in items:
                        if item.get("type") == "dir":
                            dir_name = item.get("name", "")
                            # Match case-insensitive
                            if dir_name.lower().replace(" ", "-").replace("_", "-") == service_lower:
                                return dir_name
                        # Also check if the platform name matches
                        item_name = item.get("name", "")
                        item_id = item_name.lower().replace(" ", "-").replace("_", "-")
                        # Check various name formats
                        if item_id == service_lower:
                            return item_name
        
        return None

    def create_llama_documents(
        self,
        version_records: List[dict],
        service_id: str,
        document_id: str
    ) -> List[Document]:
        """
        Parse OTA version records into pure LlamaIndex Document objects.
        """
        documents: List[Document] = []
        
        for record in version_records:
            text_content = self._extract_clean_text(record)
            
            metadata = {
                "platform": service_id,
                "document_type": document_id,
                "revision": record.get("id", record.get("version", "unknown")),
                "snapshot_time": record.get("snapshot_time", record.get("timestamp", record.get("created_at", ""))),
            }
            
            doc = Document(
                text=text_content,
                metadata=metadata,
            )
            documents.append(doc)
        
        logger.info(
            "Created %d Document(s) for platform=%s document=%s",
            len(documents),
            service_id,
            document_id
        )
        return documents

    def _extract_clean_text(self, record: dict) -> str:
        """Extract pure legal text content from a version record."""
        # Handle raw markdown content (string)
        if "content" in record and isinstance(record.get("content"), str):
            content = record["content"]
            # Check if it's markdown or JSON
            if content.strip().startswith("{"):
                # Try to parse as JSON
                try:
                    import json
                    json_content = json.loads(content)
                    if isinstance(json_content, dict):
                        # Extract from structured JSON
                        if "clauses" in json_content:
                            clauses = json_content.get("clauses", [])
                            if isinstance(clauses, list):
                                clause_texts = []
                                for clause in clauses:
                                    if isinstance(clause, dict):
                                        clause_texts.append(clause.get("text", ""))
                                    elif isinstance(clause, str):
                                        clause_texts.append(clause)
                                return "\n\n".join(clause_texts)
                        return json_content.get("text", json_content.get("content", ""))
                except json.JSONDecodeError:
                    pass
            # Return markdown as-is
            return content
        
        content = record.get("content", "")
        
        if not content and "clauses" in record:
            clauses = record.get("clauses", [])
            if isinstance(clauses, list):
                clause_texts = []
                for clause in clauses:
                    if isinstance(clause, dict):
                        clause_text = clause.get("text", "")
                        if clause_text:
                            clause_texts.append(clause_text)
                    elif isinstance(clause, str):
                        clause_texts.append(clause)
                content = "\n\n".join(clause_texts)
        
        if not content:
            content = record.get("text", "")
        
        return str(content) if content else ""

    async def fetch_all_documents(
        self,
        service_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Execute the full ingestion pipeline with dynamic discovery.
        Uses parallel fetching for improved performance.
        """
        all_documents: List[Document] = []
        
        # Pre-fetch services and populate directory cache in one pass
        await self._build_dir_cache()
        
        # Get services from the cache
        services = list(self._services_cache.values())
        
        if not services:
            logger.warning("No services found — returning empty document list")
            return []
        
        if service_ids:
            services = [s for s in services if s.get("id") in service_ids]
        
        # Fetch documents in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def fetch_platform_docs(service: dict) -> List[Document]:
            async with semaphore:
                service_id = service.get("id", "")
                if not service_id:
                    return []
                
                dir_name = self._dir_cache.get(service_id, service_id)
                
                version_records = await self.fetch_document_versions(
                    service_id=service_id,
                    dir_name=dir_name,
                    document_id="privacy-policy"
                )
                
                if version_records:
                    return self.create_llama_documents(
                        version_records=version_records,
                        service_id=service_id,
                        document_id="privacy-policy"
                    )
                return []
        
        # Execute all fetches concurrently
        results = await asyncio.gather(*[fetch_platform_docs(s) for s in services])
        
        for docs in results:
            all_documents.extend(docs)
        
        logger.info("Total documents fetched: %d", len(all_documents))
        return all_documents

    async def fetch_platform_documents(
        self,
        service_id: str,
        document_id: str = "privacy-policy"
    ) -> List[Document]:
        """
        Fetch documents for a specific platform only.
        
        Parameters
        ----------
        service_id:
            The platform identifier (e.g., "netflix", "spotify")
        document_id:
            The document type to fetch (default: "privacy-policy")
            
        Returns
        -------
        List[Document]
            LlamaIndex Document objects for the specific platform.
        """
        # Resolve platform directory name
        services = await self.fetch_services()
        service_record = next((s for s in services if s.get("id") == service_id), None)
        dir_name = service_record.get("dir_name", service_id) if service_record else service_id
        
        version_records = await self.fetch_document_versions(
            service_id=service_id,
            dir_name=dir_name,
            document_id=document_id
        )
        
        if version_records:
            documents = self.create_llama_documents(
                version_records=version_records,
                service_id=service_id,
                document_id=document_id
            )
            logger.info("Fetched %d documents for platform=%s", len(documents), service_id)
            return documents
        
        logger.info("No documents found for platform=%s", service_id)
        return []

    async def __aenter__(self) -> "OpenTermsArchiveClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass