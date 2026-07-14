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
        "platform": "chatgpt",
        "document": "privacy-policy",
        "revision": "2024-01",
        "clauses": [
            {"text": "ChatGPT collects user conversations, account information, and usage data to improve the service.", "category": "Data Collection"},
            {"text": "Conversations may be reviewed by AI trainers to enhance model performance.", "category": "Data Usage"},
            {"text": "Data is shared with service providers and for legal compliance when required.", "category": "Third-Party Sharing"},
            {"text": "Users can access and delete their conversation history through account settings.", "category": "User Rights"},
            {"text": "End-to-end encryption is applied to protect sensitive user data.", "category": "Security Measures"},
        ]
    },
    {
        "platform": "grok",
        "document": "privacy-policy",
        "revision": "2024-01",
        "clauses": [
            {"text": "Grok collects interaction data, prompts, and model responses during usage.", "category": "Data Collection"},
            {"text": "User data is used to improve model accuracy and personalize responses.", "category": "Data Usage"},
            {"text": "Information is shared with trusted partners for service provision and analytics.", "category": "Third-Party Sharing"},
            {"text": "Account deletion requests are processed within 30 days of submission.", "category": "User Rights"},
            {"text": "Strong encryption and access controls protect user information.", "category": "Security Measures"},
        ]
    },
    {
        "platform": "tiktok",
        "document": "privacy-policy",
        "revision": "2024-01",
        "clauses": [
            {"text": "TikTok collects device information, browsing history, location data, and content interactions.", "category": "Data Collection"},
            {"text": "Data is used to personalize content feed and deliver targeted advertising.", "category": "Data Usage"},
            {"text": "Information is shared with advertising partners and third-party service providers.", "category": "Third-Party Sharing"},
            {"text": "Users can download their data or request account deletion through settings.", "category": "User Rights"},
            {"text": "Security measures include encryption and regular security audits.", "category": "Security Measures"},
        ]
    },
    {
        "platform": "facebook",
        "document": "privacy-policy",
        "revision": "2024-01",
        "clauses": [
            {"text": "Facebook collects personal information, posts, messages, and interaction data across platforms.", "category": "Data Collection"},
            {"text": "Data is used to personalize ads, content, and improve service functionality.", "category": "Data Usage"},
            {"text": "Information is shared with Instagram, WhatsApp, and third-party advertisers.", "category": "Third-Party Sharing"},
            {"text": "Users can access, download, or delete their data through privacy settings.", "category": "User Rights"},
            {"text": "Encryption and security controls are implemented across all data storage.", "category": "Security Measures"},
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