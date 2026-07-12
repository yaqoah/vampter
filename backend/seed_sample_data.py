"""
Seed script to populate Neo4j and Qdrant with sample policy data.
This bypasses GitHub API rate limits by using hardcoded sample data.
"""

import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_index.core import Document, Settings as LlamaSettings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import qdrant_client

from config import settings
from ingestion.stores import QDRANT_COLLECTION_NAME, run_neo4j_schema_migration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample policy data for testing - matching exact Cypher schema
SAMPLE_POLICIES = [
    {
        "platform": "netflix",
        "document": "privacy-policy",
        "revision": "2024-01",
        "clauses": [
            {"text": "Netflix collects personal information including name, email, payment information, and viewing activity.", "category": "Data Collection"},
            {"text": "Information is used to personalize recommendations, process payments, and improve services.", "category": "Data Usage"},
            {"text": "Netflix shares data with third-party partners for content delivery and analytics.", "category": "Third-Party Sharing"},
            {"text": "Users can access, correct, or delete their personal information through account settings.", "category": "User Rights"},
            {"text": "Netflix implements industry-standard security measures to protect user data.", "category": "Security Measures"},
        ]
    },
    {
        "platform": "spotify",
        "document": "privacy-policy",
        "revision": "2024-01",
        "clauses": [
            {"text": "Spotify collects account information including name, email, phone number, and profile information.", "category": "Data Collection"},
            {"text": "Collection of listening history, playlists, and interaction data for service improvement.", "category": "Usage Tracking"},
            {"text": "Spotify uses cookies to improve user experience and advertising targeting.", "category": "Tracking & Cookies"},
            {"text": "Integration with social media platforms and advertising partners for data sharing.", "category": "Third-Party Integration"},
            {"text": "Account data is retained until deletion request is processed, subject to legal requirements.", "category": "Data Retention"},
        ]
    },
    {
        "platform": "openai",
        "document": "privacy-policy",
        "revision": "2024-01",
        "clauses": [
            {"text": "OpenAI collects information from API usage, website interactions, and support contacts.", "category": "Data Collection"},
            {"text": "Content may be used to improve models unless opted out via ChatGPT settings.", "category": "Training Data"},
            {"text": "Limited sharing with service providers and for legal compliance purposes.", "category": "Third-Party Sharing"},
            {"text": "Users can access, correct, or delete their data through privacy portal.", "category": "User Rights"},
            {"text": "Encryption and access controls protect user information in transit and at rest.", "category": "Security"},
        ]
    },
]


def configure_embedding_model():
    """Configure local embedding model."""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        LlamaSettings.embed_model = embed_model
        logger.info("Embedding model configured: BAAI/bge-small-en-v1.5")
    except ImportError:
        raise RuntimeError("Install 'llama-index-embeddings-huggingface' for embeddings")


async def seed_neo4j_directly():
    """Write Document nodes directly to Neo4j with exact properties for Cypher query."""
    from neo4j import AsyncGraphDatabase
    
    uri = settings.neo4j_uri
    auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())
    
    async with AsyncGraphDatabase.driver(uri, auth=auth) as driver:
        async with driver.session() as session:
            for policy in SAMPLE_POLICIES:
                platform_id = policy["platform"]
                
                # Create Platform node for filtering to work
                await session.run(
                    "MERGE (p:Platform {id: $platform_id}) SET p.name = $platform_name",
                    platform_id=platform_id,
                    platform_name=platform_id.capitalize()
                )
                
                for i, clause in enumerate(policy["clauses"], 1):
                    clause_text = clause["text"] if isinstance(clause, dict) else clause
                    clause_category = clause.get("category", f"Clause {i}") if isinstance(clause, dict) else f"Clause {i}"
                    
                    # Create Document nodes with exact properties the Cypher query expects
                    cypher = """
                    MERGE (d:Document {id: $doc_id})
                    SET d.platform = $platform,
                        d.name = $document,
                        d.revision = $revision,
                        d.clause_id = $clause_id,
                        d.text = $text
                    MERGE (p:Platform {id: $platform})
                    MERGE (p)-[:TRACKS_POLICY]->(d)
                    """
                    await session.run(
                        cypher,
                        doc_id=f"{policy['platform']}-privacy-{i}",
                        platform=policy["platform"],
                        document=policy["document"],
                        revision=policy["revision"],
                        clause_id=clause_category,
                        text=clause_text
                    )
                    logger.info(f"Created Document node: {policy['platform']}/{clause_category}")


async def seed_qdrant():
    """Seed Qdrant with document embeddings for vector search."""
    from llama_index.core import VectorStoreIndex
    
    # Create documents for vector indexing
    documents = []
    for policy in SAMPLE_POLICIES:
        # Use full policy text as a single document for vector search
        clause_texts = [clause["text"] if isinstance(clause, dict) else clause for clause in policy["clauses"]]
        doc = Document(
            text="\n".join(clause_texts),
            metadata={
                "platform": policy["platform"],
                "document_type": policy["document"],
                "revision": policy["revision"],
            }
        )
        documents.append(doc)
    
    logger.info("Created %d sample documents for vector indexing", len(documents))
    
    # Initialize Qdrant store
    qdrant_client_obj = qdrant_client.QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_port_grpc,
    )
    vector_store = QdrantVectorStore(client=qdrant_client_obj, collection_name=QDRANT_COLLECTION_NAME)
    
    # Build VectorStoreIndex (not PropertyGraphIndex - simpler, just vectors)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    )


async def seed_databases():
    """Seed Neo4j and Qdrant with sample policy documents."""
    logger.info("=== VAMPTER DATABASE SEED ===")
    
    # Configure embedding model
    configure_embedding_model()
    
    # Run schema migration first
    run_neo4j_schema_migration(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
    )
    
    # Seed Qdrant with vector embeddings
    await seed_qdrant()
    
    # Write Document nodes directly to Neo4j with exact Cypher properties
    await seed_neo4j_directly()
    
    logger.info("=== SEED COMPLETE ===")
    return True


if __name__ == "__main__":
    asyncio.run(seed_databases())