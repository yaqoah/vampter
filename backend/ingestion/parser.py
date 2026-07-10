"""
ingestion.parser
================
Node parsing & text chunking strategy for API JSON payloads.

Pipeline composition
--------------------
Incoming ``Document`` objects pass through a single-stage ``IngestionPipeline``:

1. **SentenceSplitter** — chunker that splits any stringified JSON
   node exceeding *chunk_size* tokens, using a configurable sliding
   *chunk_overlap* window to preserve cross-boundary context.

Embedding model selection
-------------------------
LlamaIndex 0.10.x ships ``resolve_embed_model`` which accepts a URI
scheme to locate a local model:

* ``"local:BAAI/bge-small-en-v1.5"``  →  HuggingFace ONNX / PyTorch
* ``"local:BAAI/bge-base-en-v1.5"``   →  higher accuracy at more RAM

The resolved model is registered on the global ``LlamaSettings``
singleton so every downstream index call automatically inherits it
without repetition.
"""

from __future__ import annotations

import logging
from typing import List

from llama_index.core import Document, Settings as LlamaSettings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: HuggingFace model used for local dense vector encoding.
#: BAAI/bge-small-en-v1.5 is compact (134 MB) and ONNX-optimised for speed.
DEFAULT_EMBED_MODEL_URI: str = "local:BAAI/bge-small-en-v1.5"

#: Chunk size in *tokens* for the sentence-level fallback splitter.
DEFAULT_CHUNK_SIZE: int = 512

#: Sliding overlap window in *tokens* to preserve cross-chunk context.
DEFAULT_CHUNK_OVERLAP: int = 64

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_embedding_model(
    embed_model_uri: str = DEFAULT_EMBED_MODEL_URI,
) -> None:
    """
    Resolve and globally register a local embedding model.

    Attempts the following resolution chain in order:

    1. ``llama_index.embeddings.huggingface.HuggingFaceEmbedding``
       (requires ``llama-index-embeddings-huggingface`` package).
    2. ``llama_index.embeddings.fastembed.FastEmbedEmbedding``
       (requires ``llama-index-embeddings-fastembed`` package).

    The resolved model is registered on ``llama_index.core.Settings``
    so every downstream index call inherits it automatically.

    Parameters
    ----------
    embed_model_uri:
        Accepted as-is for HuggingFace (used as ``model_name`` after
        stripping the ``"local:"`` prefix).

    Raises
    ------
    RuntimeError
        If no embedding model is available. Install one of the required
        packages for real semantic search.
    """
    # Strip the "local:" scheme prefix to get the raw model name.
    model_name = embed_model_uri.removeprefix("local:")

    # ── Attempt 1: HuggingFaceEmbedding ────────────────────────────────────
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import]
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        LlamaSettings.embed_model = embed_model
        logger.info(
            "Embedding model: HuggingFaceEmbedding  model='%s'", model_name
        )
        return
    except ImportError:
        logger.debug(
            "llama-index-embeddings-huggingface not available, trying FastEmbed ..."
        )

    # ── Attempt 2: FastEmbedEmbedding ───────────────────────────────────────
    try:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding  # type: ignore[import]
        embed_model = FastEmbedEmbedding(model_name=model_name)
        LlamaSettings.embed_model = embed_model
        logger.info(
            "Embedding model: FastEmbedEmbedding  model='%s'", model_name
        )
        return
    except ImportError:
        logger.debug(
            "llama-index-embeddings-fastembed not available either."
        )

    # ── No embedding model available ───────────────────────────────────────
    raise RuntimeError(
        "No embedding model available. Install 'llama-index-embeddings-huggingface' "
        "or 'llama-index-embeddings-fastembed' for semantic search."
    )


def build_node_pipeline(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestionPipeline:
    """
    Construct the single-stage node parsing & chunking ``IngestionPipeline``.

    The returned pipeline is stateless and safe to reuse across multiple
    ``pipeline.run()`` calls.

    Parameters
    ----------
    chunk_size:
        Maximum token length per chunk for the ``SentenceSplitter``.
    chunk_overlap:
        Sliding window overlap in tokens between adjacent chunks.

    Returns
    -------
    IngestionPipeline
        A composed pipeline ready to transform ``Document`` lists into
        ``BaseNode`` lists.
    """
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    pipeline = IngestionPipeline(
        transformations=[
            sentence_splitter,
        ]
    )

    logger.info(
        "IngestionPipeline built — chunk_size=%d  chunk_overlap=%d",
        chunk_size,
        chunk_overlap,
    )
    return pipeline


async def parse_documents_to_nodes(
    documents: List[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_model_uri: str = DEFAULT_EMBED_MODEL_URI,
) -> List[BaseNode]:
    """
    High-level async wrapper: configure embeddings then run the node pipeline.

    This function:
    1. Resolves and globally registers the embedding model.
    2. Runs the single-stage ``IngestionPipeline`` on all supplied documents.
    3. Returns the flat list of parsed ``BaseNode`` objects, ready for
       dual-store indexing.

    Parameters
    ----------
    documents:
        ``Document`` objects produced by ``ingestion.api_client``.
    chunk_size:
        Forwarded to ``build_node_pipeline``.
    chunk_overlap:
        Forwarded to ``build_node_pipeline``.
    embed_model_uri:
        Forwarded to ``configure_embedding_model``.

    Returns
    -------
    List[BaseNode]
        Flat sequence of all parsed and enriched nodes.
    """
    configure_embedding_model(embed_model_uri)
    pipeline = build_node_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info("Running IngestionPipeline on %d document(s) …", len(documents))
    nodes: List[BaseNode] = await pipeline.arun(documents=documents)
    logger.info("Produced %d node(s) from %d document(s).", len(nodes), len(documents))
    return nodes