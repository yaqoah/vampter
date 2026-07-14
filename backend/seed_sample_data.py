"""
Seed script to populate Neo4j and Qdrant with sample policy data.
This bypasses GitHub API rate limits by using hardcoded sample data.

NOTE: This script uses PropertyGraphIndex to create the same schema as the
full ingestion pipeline (Chunk nodes with _properties metadata), ensuring
compatibility with the graph_node queries.
"""

import asyncio
import logging
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llama_index.core import Document, Settings as LlamaSettings, PropertyGraphIndex, StorageContext
import qdrant_client

from config import settings
from ingestion.stores import QDRANT_COLLECTION_NAME, init_neo4j_store, init_qdrant_store, run_neo4j_schema_migration
from ingestion.graph_extractor import build_schema_extractor

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
    """Write sample policy data using PropertyGraphIndex for consistent schema."""
    
    # Build vector store and graph store for PropertyGraphIndex
    vector_store = init_qdrant_store(
        host=settings.qdrant_host if not settings.qdrant_url else None,
        port=settings.qdrant_port if not settings.qdrant_url else None,
        grpc_port=settings.qdrant_port_grpc if not settings.qdrant_url else None,
        api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None,
        collection_name=QDRANT_COLLECTION_NAME,
        url=settings.qdrant_url,
    )
    
    graph_store = init_neo4j_store(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
    )
    
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        property_graph_store=graph_store,
    )
    
    # Create documents for PropertyGraphIndex
    documents = []
    for policy in SAMPLE_POLICIES:
        # Each clause becomes a separate document node for better granularity
        for clause in policy["clauses"]:
            clause_text = clause["text"] if isinstance(clause, dict) else clause
            clause_category = clause.get("category", "Clause") if isinstance(clause, dict) else "Clause"
            
            doc = Document(
                text=clause_text,
                metadata={
                    "platform": policy["platform"],
                    "document_type": policy["document"],
                    "revision": policy["revision"],
                    "clause_category": clause_category,
                }
            )
            documents.append(doc)
    
    logger.info("Created %d sample documents for PropertyGraphIndex", len(documents))
    
    # Build the schema extractor for graph triples
    if settings.mistral_api_key:
        from ingestion.pipeline import _build_llm
        llm = _build_llm(settings)
        kg_extractor = build_schema_extractor(llm=llm)
    else:
        # Without LLM, we can't extract graph triples - just use vector store
        logger.warning("No MISTRAL_API_KEY - skipping graph extraction, using vector-only")
        kg_extractor = None
    
    # Build PropertyGraphIndex (same as main ingestion pipeline)
    loop = asyncio.get_event_loop()
    
    def build_index():
        if kg_extractor:
            return PropertyGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                kg_extractors=[kg_extractor],
                show_progress=True,
            )
        else:
            return PropertyGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
            )
    
    index = await loop.run_in_executor(None, build_index)
    logger.info("PropertyGraphIndex built with %d nodes", len(documents))


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
    
    # Use PropertyGraphIndex for both Neo4j and Qdrant (unified approach)
    await seed_neo4j_directly()
    
    logger.info("=== SEED COMPLETE ===")
    return True


if __name__ == "__main__":
    asyncio.run(seed_databases())