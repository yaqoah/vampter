"""
Quick script to check database state after clear-platform interruption.

NOTE: This checks localhost. For cloud databases, the backend must be running
and you can check via the /debug/neo4j endpoint.
"""
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_db_state():
    """Check Neo4j and Qdrant state."""
    from neo4j import AsyncGraphDatabase
    from qdrant_client import AsyncQdrantClient
    
    print("=" * 60)
    print("DATABASE STATE CHECK")
    print("=" * 60)
    print(f"Neo4j URI: {settings.neo4j_uri}")
    print(f"Qdrant URL: {settings.qdrant_url or 'localhost'}")
    print()
    
    # Check Neo4j
    try:
        auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
        # Use the URL from settings (could be cloud)
        uri = settings.neo4j_uri or "bolt://localhost:7687"
        async with AsyncGraphDatabase.driver(uri, auth=auth) as driver:
            async with driver.session() as session:
                # Count Chunk nodes
                result = await session.run("MATCH (c:Chunk) RETURN count(c) AS count")
                chunk_count = (await result.single())["count"]
                print(f"Neo4j Chunk nodes: {chunk_count}")
                
                # Count Platform nodes
                result = await session.run("MATCH (p:Platform) RETURN count(p) AS count")
                platform_count = (await result.single())["count"]
                print(f"Neo4j Platform nodes: {platform_count}")
                
                # Get unique platforms from Chunk nodes
                result = await session.run(
                    "MATCH (c:Chunk) WHERE c._properties IS NOT NULL AND c._properties.platform IS NOT NULL "
                    "RETURN DISTINCT c._properties.platform AS platform "
                    "ORDER BY platform LIMIT 20"
                )
                platforms = [rec["platform"] for rec in await result.data()]
                print(f"\nSample platforms (from Chunk nodes):")
                for p in platforms[:10]:
                    print(f"  - {p}")
                if len(platforms) > 10:
                    print(f"  ... and {len(platforms) - 10} more")
                
                # Check for spotify specifically
                result = await session.run(
                    "MATCH (c:Chunk) WHERE c._properties.platform = 'spotify' "
                    "RETURN count(c) AS count"
                )
                spotify_count = (await result.single())["count"]
                print(f"\nSpotify Chunk nodes: {spotify_count}")
                
                # Check for netflix and openai
                for plat in ["netflix", "openai"]:
                    result = await session.run(
                        "MATCH (c:Chunk) WHERE c._properties.platform = $plat "
                        "RETURN count(c) AS count",
                        plat=plat
                    )
                    count = (await result.single())["count"]
                    print(f"{plat.capitalize()} Chunk nodes: {count}")
                    
    except Exception as e:
        print(f"Neo4j error: {e}")
    
    # Check Qdrant
    try:
        if settings.qdrant_url:
            qc = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
            )
        else:
            qc = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
        
        collections = await qc.get_collections()
        if any(c.name == "vampter_docs" for c in collections.collections):
            count_result = await qc.count(collection_name="vampter_docs")
            print(f"\nQdrant vectors: {count_result.count}")
    except Exception as exc:
        logger.warning("Qdrant check failed: %s", exc)
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(check_db_state())