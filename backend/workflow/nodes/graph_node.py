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

_CYPHER_TEMPLATE = """
MATCH (n:Document)
WHERE toLower(n.platform) CONTAINS toLower($company_name)
RETURN
    n.platform    AS platform,
    n.name        AS document,
    n.revision    AS revision,
    n.clause_id   AS clause_id,
    n.text        AS clause_text
ORDER BY n.revision DESC, n.clause_id ASC
LIMIT 50
"""

# Alternative query that matches by platform name when ID doesn't match
_CYPHER_BY_NAME_TEMPLATE = """
MATCH (p:Platform)
WHERE toLower(p.name) CONTAINS toLower($company_name)
MATCH (n:Document)
WHERE n.platform = p.id OR n.platform CONTAINS p.name
RETURN
    n.platform    AS platform,
    n.name        AS document,
    n.revision    AS revision,
    n.clause_id   AS clause_id,
    n.text        AS clause_text
ORDER BY n.revision DESC, n.clause_id ASC
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
                # First try to match by platform ID (direct match)
                result = await session.run(
                    _CYPHER_TEMPLATE, company_name=company_name
                )
                records = await result.data()
                
                # If no results, try matching by platform name
                if not records:
                    result = await session.run(
                        _CYPHER_BY_NAME_TEMPLATE, company_name=company_name
                    )
                    records = await result.data()

        for rec in records:
            platform = rec.get("platform", "")
            document = rec.get("document", "")
            revision = rec.get("revision", "")
            clause_id = rec.get("clause_id", "")
            clause_text = rec.get("clause_text", "")

            passage = (
                f"[{platform}] {document} ({revision}) — {clause_id}: {clause_text}"
            )
            passages.append(passage)

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
