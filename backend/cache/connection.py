"""
cache.connection
================
Async Redis connection pool factory.

Design
------
A single ``ConnectionPool`` is constructed once at import time and shared
across all coroutines via ``get_redis_pool()``.  This avoids the cost of
re-establishing TCP handshakes on every request while remaining safe under
``asyncio`` concurrency.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import redis.asyncio as aioredis

from config import settings

logger = logging.getLogger(__name__)

_POOL: aioredis.ConnectionPool | None = None


def get_redis_pool() -> aioredis.ConnectionPool:
    """
    Return the shared async Redis connection pool, creating it on first call.

    The pool is configured with ``max_connections=20`` to cap resource usage
    under concurrent FastAPI request handling.

    Returns
    -------
    redis.asyncio.ConnectionPool
        A ready-to-use async pool.
    """
    global _POOL
    if _POOL is None:
        password = (
            settings.redis_password.get_secret_value()
            if settings.redis_password
            else None
        )
        _POOL = aioredis.ConnectionPool(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=password,
            max_connections=20,
            decode_responses=False,  # We manage binary serialisation ourselves
        )
        logger.info(
            "Redis connection pool created — host=%s  port=%d  db=%d  max_connections=20",
            settings.redis_host,
            settings.redis_port,
            settings.redis_db,
        )
    return _POOL


async def get_async_client() -> aioredis.Redis:
    """
    Return a single async Redis client bound to the shared connection pool.

    Callers are responsible for calling ``await client.aclose()`` when
    finished, or using the client as an async context manager.
    """
    return aioredis.Redis(connection_pool=get_redis_pool())
