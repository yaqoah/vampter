"""
flush_redis.py
==============

Utility script to force-flush the semantic cache by deleting all keys
matching the vampter_cache:* pattern in Redis.
"""

import asyncio
import logging

from cache.connection import get_async_client

logger = logging.getLogger(__name__)

async def flush_semantic_cache() -> None:
    """
    Delete all keys matching the vampter_cache:* pattern in Redis.
    """
    client = await get_async_client()
    try:
        # Find all keys matching the pattern
        keys = await client.keys("vampter_cache:*")

        if not keys:
            logger.info("No semantic cache keys found to flush.")
            return

        # Delete all matching keys
        deleted = await client.delete(*keys)
        logger.info("Flushed semantic cache: deleted %d keys.", deleted)
    finally:
        await client.aclose()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(flush_semantic_cache())