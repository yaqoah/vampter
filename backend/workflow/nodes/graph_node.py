"""
workflow.nodes.graph_node
=========================
Neo4j multi-hop Cypher walk retrieval node.

Responsibility
--------------
Executes a multi-hop Cypher traversal against the Neo4j graph database,
following the exact schema laid down by the API ingestion pipeline:

    (Platform)-[:TRACKS_POLICY]->(Document)
              -[:HAS_REVISION_VERSION]->(Revision)
              -[:CONTAINS_CLAUSE]->(Clause)

Suitable for queries involving cross-revision contradictions, policy
change histories, or structural analysis that requires navigating the
full document lineage chain.

If Neo4j is unavailable or unpopulated, falls back gracefully to
the vector search path with a warning.
"""

from __future__ import annotations

import logging
from typing import List

from config import settings
from workflow.state import AuditState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher query — full 4-hop traversal aligned to ingestion graph schema
# ---------------------------------------------------------------------------

# Match Chunk nodes (LlamaIndex PropertyGraphStore default label)
_CYPHER_TEMPLATE = """
MATCH (n:Chunk)
WHERE n.text IS NOT NULL
  AND toLower(n.text) CONTAINS toLower($company_name)
RETURN
    n.name        AS platform,
    n.text        AS clause_text
LIMIT 50
"""

# Query for any node with text (fallback)
_CYPHER_ANY_TEMPLATE = """
MATCH (n)
WHERE n.text IS NOT NULL
  AND toLower(n.text) CONTAINS toLower($company_name)
RETURN
    coalesce(n.name, '') AS platform,
    n.text                AS clause_text
LIMIT 50
"""

# Query for Platform-Revision-Clause path (when schema exists)
_CYPHER_PLATFORM_PATH_TEMPLATE = """
MATCH (p:Platform)-[:TRACKS_POLICY]->(d:Document)-[:HAS_REVISION_VERSION]->(r:Revision)-[:CONTAINS_CLAUSE]->(c:Clause)
WHERE toLower(toString(p.name)) CONTAINS toLower($company_name)
RETURN
    p.name + ': ' + c.text AS clause_text
LIMIT 50
"""

# Query by _properties field (LlamaIndex PropertyGraphStore stores metadata here)
_CYPHER_BY_PROPS_TEMPLATE = """
MATCH (n)
WHERE n._properties IS NOT NULL
  AND toLower(toString(apoc.text.join(keys(n._properties), ''))) CONTAINS toLower($company_name)
RETURN
    n.name AS platform,
    n.text AS clause_text
LIMIT 50
"""

# Query for Clause nodes specifically (from SchemaLLMPathExtractor)
_CYPHER_CLAUSE_TEMPLATE = """
MATCH (c:Clause)
WHERE toLower(toString(c.text)) CONTAINS toLower($company_name)
RETURN c.text AS clause_text
LIMIT 50
"""

# Query for Document nodes specifically
_CYPHER_DOC_CLAUSE_TEMPLATE = """
MATCH (d:Document)-[:CONTAINS_CLAUSE]->(c:Clause)
WHERE toLower(toString(d.name)) CONTAINS toLower($company_name)
   OR toLower(toString(d.platform)) CONTAINS toLower($company_name)
RETURN c.text AS clause_text
LIMIT 50
"""

# Full path: Platform -> Document -> Revision -> Clause
_CYPHER_FULL_PATH_TEMPLATE = """
MATCH (p:Platform)-[:TRACKS_POLICY]->(d:Document)-[:HAS_REVISION_VERSION]->(r:Revision)-[:CONTAINS_CLAUSE]->(c:Clause)
WHERE toLower(toString(p.name)) CONTAINS toLower($company_name)
RETURN
    p.name + ': ' + c.text AS clause_text
LIMIT 50
"""

# Document nodes (seed data compatibility)
_CYPHER_BY_DOCUMENT_TEMPLATE = """
MATCH (n:Document)
WHERE toLower(n.platform) CONTAINS toLower($company_name)
RETURN
    n.platform    AS platform,
    n.text        AS clause_text
LIMIT 50
"""

# Query by Platform name when ID doesn't match
_CYPHER_BY_NAME_TEMPLATE = """
MATCH (p:Platform)
WHERE toLower(p.name) CONTAINS toLower($company_name)
MATCH (n)
WHERE toLower(n.text) CONTAINS toLower(p.id)
RETURN
    p.id        AS platform,
    n.text      AS clause_text
LIMIT 50
"""

# Fallback: Any node with text property containing platform name
_CYPHER_FALLBACK_TEMPLATE = """
MATCH (n)
WHERE (n.text IS NOT NULL OR n.name IS NOT NULL)
  AND toLower(toString(n.text)) CONTAINS toLower($company_name)
RETURN
    coalesce(n.platform, n.name)    AS platform,
    n.text                          AS clause_text
LIMIT 50
"""


async def graph_node(state: AuditState) -> AuditState:
    """
    Walk the Neo4j policy graph and return structured text passages.

    Traverses the four-hop chain:
    ``Platform → Document → Revision → Clause``

    If Neo4j is unavailable or unpopulated, falls back to vector search.

    Parameters
    ----------
    state:
        Current ``AuditState`` with ``company_name``.

    Returns
    -------
    AuditState
        Updated state with ``retrieved_passages`` populated.
    """
    company_name = state.get("company_name", "")
    passages: List[str] = []
    
    try:
        from neo4j import AsyncGraphDatabase  # type: ignore[import]

        uri = settings.neo4j_uri
        auth = (settings.neo4j_user, settings.neo4j_password.get_secret_value())

        async with AsyncGraphDatabase.driver(uri, auth=auth) as driver:
            async with driver.session() as session:
                records = []
                
                # Debug: Show what labels exist in the database
                try:
                    debug_result = await session.run(
                        "MATCH (n) RETURN DISTINCT labels(n) AS labels LIMIT 10"
                    )
                    label_records = await debug_result.data()
                    seen_labels = set()
                    for lr in label_records:
                        if lr.get("labels"):
                            seen_labels.update(lr["labels"])
                    logger.info("DEBUG: Neo4j labels found: %s", list(seen_labels))
                except Exception as e:
                    logger.debug("Debug label query failed: %s", e)
                
                # Pattern 1: Match nodes with text containing company name
                try:
                    result = await session.run(_CYPHER_TEMPLATE, company_name=company_name)
                    records = await result.data()
                    if records:
                        logger.info("Graph query matched via text search - found %d records", len(records))
                        logger.info("DEBUG: Sample record keys: %s", list(records[0].keys()) if records else "none")
                except Exception as e:
                    logger.debug("Pattern 1 failed: %s", e)
                
                # Pattern 2: Document nodes with category field (seed data format)
                if not records:
                    try:
                        result = await session.run(
                            _CYPHER_BY_DOCUMENT_TEMPLATE, company_name=company_name
                        )
                        records = await result.data()
                        if records:
                            logger.info("Graph query matched via Document nodes")
                    except Exception as e:
                        logger.debug("Pattern 2 failed: %s", e)
                
                # Pattern 3: Full path traversal (Platform -> Document -> Revision -> Clause)
                if not records:
                    try:
                        result = await session.run(
                            _CYPHER_FULL_PATH_TEMPLATE, company_name=company_name
                        )
                        records = await result.data()
                        if records:
                            logger.info("Graph query matched via full path traversal")
                    except Exception as e:
                        logger.debug("Pattern 3 failed: %s", e)
                
                # Pattern 4: Match by Platform name
                if not records:
                    try:
                        result = await session.run(
                            _CYPHER_BY_NAME_TEMPLATE, company_name=company_name
                        )
                        records = await result.data()
                        if records:
                            logger.info("Graph query matched via Platform name")
                    except Exception as e:
                        logger.debug("Pattern 4 failed: %s", e)
                
                # Pattern 5: Fallback - any node with text
                if not records:
                    try:
                        result = await session.run(
                            _CYPHER_FALLBACK_TEMPLATE, company_name=company_name
                        )
                        records = await result.data()
                        if records:
                            logger.info("Graph query matched via fallback")
                    except Exception as e:
                        logger.debug("Pattern 5 failed: %s", e)

        for rec in records:
            clause_text = rec.get("clause_text", "") or rec.get("text", "")
            if clause_text:
                text_str = str(clause_text).strip()
                if len(text_str) > 50:  # Only keep substantive chunks
                    passages.append(text_str)
                    # Log first passage content for debugging
                    if len(passages) == 1:
                        logger.info("DEBUG: First passage preview: %s...", text_str[:200].replace('\n', ' '))

        logger.info(
            "Graph node retrieved %d clause(s) for company='%s'",
            len(passages),
            company_name,
        )
        
        if passages:
            return {**state, "retrieved_passages": passages}

    except Exception as exc:
        logger.warning(
            "Graph node error: %s — attempting vector fallback.", exc
        )

    # Fallback: Use vector search when Neo4j is unavailable or empty
    logger.info(
        "Graph search returned no results for '%s' — falling back to vector search.",
        company_name,
    )
    
    try:
        from workflow.nodes.vector_node import vector_node
        fallback_state = await vector_node(state)
        return fallback_state
    except Exception as fallback_exc:
        logger.error("Vector fallback also failed: %s", fallback_exc)
        return {**state, "retrieved_passages": []}
