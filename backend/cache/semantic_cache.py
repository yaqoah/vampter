"""
cache.semantic_cache
====================
Cosine-similarity semantic cache backed by Redis.

Namespace layout
----------------
Every cached entry occupies two Redis keys under a scoped composite namespace:

    vampter_cache:{company_name}:embeddings   → Redis hash
        field = sha256(query_vector bytes)
        value = float32 embedding bytes (little-endian)

    vampter_cache:{company_name}:payloads     → Redis hash
        field = sha256(query_vector bytes)
        value = JSON-serialised AuditReport payload

Cache flow
----------
1. ``lookup(company_name, query_vector)``
   - Retrieve all stored embeddings for the company.
   - Compute cosine similarity between the incoming vector and each stored one.
   - If best similarity >= ``threshold`` (default 0.92), deserialise and return
     the corresponding payload.  Total hot-path latency target: < 5 ms.

2. ``store(company_name, query_vector, payload)``
   - Serialise the embedding to bytes and the payload to JSON.
   - Write both fields to their respective Redis hashes.
   - Apply a TTL of 24 hours (configurable) to both keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np

from cache.connection import get_async_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default cosine-similarity threshold for a cache hit.
DEFAULT_THRESHOLD: float = 0.92

#: TTL for cache entries — 24 hours.
CACHE_TTL_SECONDS: int = 86_400

_EMB_KEY_SUFFIX = "embeddings"
_PAY_KEY_SUFFIX = "payloads"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec_key(company_name: str) -> str:
    return f"vampter_cache:{company_name}:{_EMB_KEY_SUFFIX}"


def _pay_key(company_name: str) -> str:
    return f"vampter_cache:{company_name}:{_PAY_KEY_SUFFIX}"


def _hash_vector(vector: List[float]) -> str:
    """Produce a stable hex key for a float vector."""
    raw = struct.pack(f"{len(vector)}f", *vector)
    return hashlib.sha256(raw).hexdigest()


def _encode_vector(vector: List[float]) -> bytes:
    """Serialise a float list to little-endian bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def _decode_vector(data: bytes) -> np.ndarray:
    """Deserialise little-endian bytes back to a numpy float32 array."""
    n = len(data) // 4
    return np.array(struct.unpack(f"{n}f", data), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


class SemanticCache:
    """
    Asynchronous semantic cache with cosine-similarity matching.

    Parameters
    ----------
    threshold:
        Minimum cosine similarity for a lookup to be considered a hit.
    ttl:
        Cache entry TTL in seconds.  Default is 24 hours.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        ttl: int = CACHE_TTL_SECONDS,
    ) -> None:
        self.threshold = threshold
        self.ttl = ttl

    async def lookup(
        self,
        company_name: str,
        query_vector: List[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Search the cache for a semantically similar previous query.

        Parameters
        ----------
        company_name:
            Scopes the search to a single company's cache partition.
        query_vector:
            Dense embedding of the incoming user query.

        Returns
        -------
        Optional[Dict[str, Any]]
            Deserialised cached payload if a hit is found, else ``None``.
        """
        t0 = time.monotonic()
        client = await get_async_client()

        try:
            emb_key = _vec_key(company_name)
            stored_embeddings: Dict[bytes, bytes] = await client.hgetall(emb_key)

            if not stored_embeddings:
                return None

            query_arr = np.array(query_vector, dtype=np.float32)
            best_field: Optional[bytes] = None
            best_score: float = -1.0

            for field_bytes, emb_bytes in stored_embeddings.items():
                stored_arr = _decode_vector(emb_bytes)
                score = _cosine_similarity(query_arr, stored_arr)
                if score > best_score:
                    best_score = score
                    best_field = field_bytes

            elapsed_ms = (time.monotonic() - t0) * 1000

            if best_score >= self.threshold and best_field is not None:
                pay_key = _pay_key(company_name)
                raw_payload = await client.hget(pay_key, best_field)
                if raw_payload:
                    logger.info(
                        "Cache HIT  company='%s'  similarity=%.4f  latency=%.2f ms",
                        company_name,
                        best_score,
                        elapsed_ms,
                    )
                    return json.loads(raw_payload)

            logger.debug(
                "Cache MISS  company='%s'  best_similarity=%.4f  latency=%.2f ms",
                company_name,
                best_score,
                elapsed_ms,
            )
            return None

        finally:
            await client.aclose()

    async def store(
        self,
        company_name: str,
        query_vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """
        Persist a query embedding and its associated payload to Redis.

        Parameters
        ----------
        company_name:
            Cache partition key.
        query_vector:
            Dense embedding of the user query.
        payload:
            Serialisable dict representing the ``AuditReport``.
        """
        client = await get_async_client()
        try:
            field = _hash_vector(query_vector)
            emb_key = _vec_key(company_name)
            pay_key = _pay_key(company_name)

            await client.hset(emb_key, field, _encode_vector(query_vector))
            await client.hset(pay_key, field, json.dumps(payload))

            # Refresh TTL on both keys after every write.
            await client.expire(emb_key, self.ttl)
            await client.expire(pay_key, self.ttl)

            logger.debug(
                "Cache STORE  company='%s'  field=%s", company_name, field[:12]
            )
        finally:
            await client.aclose()
