"""
ingestion.graph_extractor
=========================
Legal ontology schema and ``SchemaLLMPathExtractor`` configuration.

Domain ontology
---------------
The extractor maps Open Terms Archive policy document structure onto
a directed property graph using the following typed ontology:

**Entity / Node Types**

+-----------------+------------------------------------------------------+
| Label           | Semantics                                            |
+=================+======================================================+
| ``Platform``    | The OTA software platform or product family          |
|                 | (e.g. "Android Auto", "iOS CarPlay")                 |
+-----------------+------------------------------------------------------+
| ``Document``    | A discrete policy or legal document                  |
|                 | (e.g. "OTA Privacy Policy")                          |
+-----------------+------------------------------------------------------+
| ``Revision``    | A specific version/revision of a document            |
|                 | (e.g. "v4.1.0")                                      |
+-----------------+------------------------------------------------------+
| ``Clause``      | A numbered or titled section / sub-clause within a   |
|                 | policy document (e.g. "§4.2 Data Retention")         |
+-----------------+------------------------------------------------------+

**Relation / Edge Types**

+------------------------+-----------------------------------------------+
| Relation               | Semantics                                     |
+========================+===============================================+
| ``TRACKS_POLICY``      | (Platform) → tracks → (Document)              |
+------------------------+-----------------------------------------------+
| ``HAS_REVISION_VERSION``| (Document) → has version → (Revision)        |
+------------------------+-----------------------------------------------+
| ``CONTAINS_CLAUSE``    | (Revision) → contains → (Clause)              |
+------------------------+-----------------------------------------------+

Design note
-----------
``SchemaLLMPathExtractor`` is the preferred extractor for structured
domains: it instructs the LLM to *only* emit triples that conform to
the declared schema, producing far fewer hallucinated relationships than
the unconstrained ``SimpleLLMPathExtractor``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ontology Definitions
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "Platform",
    "Document", 
    "Revision",
    "Clause",
]

RELATION_TYPES = [
    "TRACKS_POLICY",
    "HAS_REVISION_VERSION",
    "CONTAINS_CLAUSE",
]

# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """
You are a legal document analysis engine specialised in Open Terms Archive
software update policy documents. Your task is to extract structured
knowledge-graph triples from the provided document text.

You MUST ONLY output triples whose subject type, relation type, and object type
appear in the approved schema below. Ignore all other entity relationships.

=== APPROVED SCHEMA ===
Entity types : Platform, Document, Revision, Clause
Relation types: TRACKS_POLICY, HAS_REVISION_VERSION, CONTAINS_CLAUSE

Valid triple patterns:
  (Platform)  -[TRACKS_POLICY]->        (Document)
  (Document)  -[HAS_REVISION_VERSION]-> (Revision)
  (Revision)  -[CONTAINS_CLAUSE]->      (Clause)

=== EXTRACTION RULES ===
1. Extract Platform names exactly as they appear in titles or metadata.
2. Document names should reflect official document titles.
3. Revisions should capture the version of the document.
4. Clauses should correspond to numbered sections (§1, §2.1, "Article 3", etc.)
   or titled sub-sections.
5. Do NOT invent entities not present in the text.
6. Output each triple on a separate line in the format:
   (<subject>) -[<relation>]-> (<object>)
""".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_schema_extractor(
    llm: LLM,
    strict_mode: bool = True,
    max_triplets_per_chunk: int = 10,
) -> SchemaLLMPathExtractor:
    """
    Construct a ``SchemaLLMPathExtractor`` pre-loaded with the Vampter
    legal ontology schema.

    Parameters
    ----------
    llm:
        The language model to use for triple extraction.  Should be a
        capable instruction-following model (Poolside Laguna recommended).
    strict_mode:
        When ``True`` (default), the extractor filters out any triple
        whose entity or relation types are not declared in the schema.
        Set ``False`` during development to inspect the raw LLM output.
    max_triplets_per_chunk:
        Upper bound on the number of triples extracted per text chunk.
        Higher values increase cost and latency; 10 is a practical default
        for clause-level chunks.

    Returns
    -------
    SchemaLLMPathExtractor
        A fully configured extractor ready to be passed to
        ``PropertyGraphIndex``.
    """
    logger.info(
        "Building SchemaLLMPathExtractor — entity_types=%s  relation_types=%s",
        ENTITY_TYPES,
        RELATION_TYPES,
    )

    extractor = SchemaLLMPathExtractor(
        llm=llm,
        extract_prompt=_EXTRACTION_SYSTEM_PROMPT,
        possible_entities=ENTITY_TYPES,
        possible_relations=RELATION_TYPES,
        strict=strict_mode,
        max_triplets_per_chunk=max_triplets_per_chunk,
        num_workers=4,
    )

    logger.info("SchemaLLMPathExtractor constructed successfully.")
    return extractor
