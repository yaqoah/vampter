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
LlamaIndex 0.10.x uses Mistral cloud embeddings for deployment-friendly
operation (no local model downloads, minimal RAM footprint).
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

#: Mistral cloud embedding model name.
#: Uses Mistral AI API for embeddings (no local model download required).
DEFAULT_EMBED_MODEL_NAME: str = "mistral-embed"

#: Default max context tokens before truncation for prompt compression.
DEFAULT_MAX_CONTEXT_TOKENS: int = 8000

#: Chunk size in *tokens* for the sentence-level fallback splitter.
DEFAULT_CHUNK_SIZE: int = 512

#: Sliding overlap window in *tokens* to preserve cross-chunk context.
DEFAULT_CHUNK_OVERLAP: int = 64

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_embedding_model(
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
) -> None:
    """
    Configure and globally register Mistral cloud embedding model.

    Uses the Mistral AI embedding API with the model name.
    The MISTRAL_API_KEY environment variable must be set.

    The resolved model is registered on ``llama_index.core.Settings``
    so every downstream index call inherits it automatically.

    Parameters
    ----------
    embed_model_name:
        The Mistral embedding model to use (default: "mistral-embed").

    Raises
    ------
    RuntimeError
        If the Mistral AI embedding package is not available or API key missing.
    """
    try:
        from llama_index.embeddings.mistralai import MistralAIEmbedding
        embed_model = MistralAIEmbedding(model_name=embed_model_name)
        LlamaSettings.embed_model = embed_model
        logger.info(
            "Embedding model: MistralAIEmbedding  model='%s'", embed_model_name
        )
    except ImportError:
        raise RuntimeError(
            "llama-index-embeddings-mistralai not available. "
            "Install it with: pip install llama-index-embeddings-mistralai"
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
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
) -> List[BaseNode]:
    """
    High-level async wrapper: configure embeddings then run the node pipeline.

    This function:
    1. Configures and globally registers the embedding model.
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
    embed_model_name:
        The Mistral embedding model to use (default: "mistral-embed").

    Returns
    -------
    List[BaseNode]
        Flat sequence of all parsed and enriched nodes.
    """
    configure_embedding_model(embed_model_name)
    pipeline = build_node_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info("Running IngestionPipeline on %d document(s) …", len(documents))
    nodes: List[BaseNode] = await pipeline.arun(documents=documents)
    logger.info("Produced %d node(s) from %d document(s).", len(nodes), len(documents))
    return nodes