#!/usr/bin/env python3
"""
flush_redis.py
==============
Utility script to completely flush the local Redis database.
"""

import asyncio
import sys
from pathlib import Path

# Adjust python path to import config
sys.path.append(str(Path(__file__).parent.absolute()))

from cache.connection import get_async_client

async def main():
    print("Connecting to Redis and flushing all databases...")
    try:
        client = await get_async_client()
        await client.flushall()
        print("Successfully flushed Redis cache (FLUSHALL completed).")
        await client.aclose()
    except Exception as exc:
        print(f"Error flushing Redis: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
