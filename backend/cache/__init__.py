"""
cache
=====
Redis-backed semantic cache layer for the Vampter backend.

Public API
----------
- ``get_redis_pool``   : Async Redis connection pool factory.
- ``SemanticCache``    : Cosine-similarity hit/miss cache keyed by company + query vector.
"""

from cache.connection import get_redis_pool
from cache.semantic_cache import SemanticCache

__all__ = ["get_redis_pool", "SemanticCache"]
