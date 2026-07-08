"""
ingestion
=========
Vampter asynchronous document ingestion engine.

Public API
----------
- ``run_pipeline``       : Orchestrate the full async ingestion run.
- ``init_qdrant_store``  : Construct a ready Qdrant vector store.
- ``init_neo4j_store``   : Construct a ready Neo4j property-graph store.
"""

from ingestion.pipeline import run_pipeline
from ingestion.stores import init_qdrant_store, init_neo4j_store

__all__ = [
    "run_pipeline",
    "init_qdrant_store",
    "init_neo4j_store",
]
