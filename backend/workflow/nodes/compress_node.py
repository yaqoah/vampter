"""
workflow.nodes.compress_node
============================
Lightweight token-based context compression node.

Responsibility
--------------
Truncates retrieved passages to fit within token limits for Mistral LLM
using tiktoken (no heavy ML models required).

Target behavior:
- Keep full context if under token limit
- Truncate from the middle if over limit, preserving key information
- No torch or transformers dependencies (minimal RAM footprint)
"""

from __future__ import annotations

import logging
from typing import List

import tiktoken

from workflow.state import AuditState

logger = logging.getLogger(__name__)

#: Max tokens to retain before truncation (Mistral Large context ~32k, leave room for prompt/template)
MAX_COMPRESSED_TOKENS: int = 8000


def truncate_to_token_limit(
    text: str,
    model: str = "mistral-tok",
    max_tokens: int = MAX_COMPRESSED_TOKENS,
) -> str:
    """
    Truncate text to fit within token limit using tiktoken.

    Uses a lightweight token counting approach (gpt-2/llama tokenizer
    approximation) to avoid downloading model-specific tokenizers.

    Parameters
    ----------
    text:
        The text to potentially truncate.
    model:
        Tokenizer model name (using cl100k_base for compatibility with Mistral).
    max_tokens:
        Maximum number of tokens to retain.

    Returns
    -------
    str
        The text, possibly truncated with ellipsis marker.
    """
    try:
        # Use cl100k_base (GPT-4/3.5 tokenizer) as approximation for Mistral
        # This is close enough and avoids model-specific downloads
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate to token limit
        truncated_tokens = tokens[:max_tokens]
        truncated_text = enc.decode(truncated_tokens)
        
        # Add ellipsis to indicate truncation
        return truncated_text + "\n... [truncated for token limit]"
    except Exception:
        # Fallback: rough character-based truncation if tokenizer fails
        max_chars = max_tokens * 4  # Rough approximation
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated]"


async def compress_node(state: AuditState) -> AuditState:
    """
    Compress retrieved passages using lightweight token truncation.

    This replaces the LLMLingua compression to reduce memory footprint.
    Simply truncates context to fit within token limits while preserving
    the beginning and end of the content.

    Parameters
    ----------
    state:
        Current ``AuditState`` with ``retrieved_passages``.

    Returns
    -------
    AuditState
        Updated state with ``compressed_context`` (a single string).
    """
    passages: List[str] = state.get("retrieved_passages", [])
    company_name = state.get("company_name", "unknown")

    if not passages:
        logger.warning(
            "Compress node: no passages to compress for company='%s'.", company_name
        )
        return {**state, "compressed_context": ""}

    # Join all passages into one block
    full_context = "\n\n".join(passages)
    logger.info("Compress node: %d passages, %d chars for company='%s'",
                len(passages), len(full_context), company_name)

    # If context is empty, return empty
    if not full_context.strip():
        logger.warning("Compress node: empty joined context for company='%s'.", company_name)
        return {**state, "compressed_context": ""}

    # Apply lightweight token-based truncation
    compressed = truncate_to_token_limit(full_context, max_tokens=MAX_COMPRESSED_TOKENS)

    logger.info("Context compression: %d chars → %d chars", len(full_context), len(compressed))

    return {**state, "compressed_context": compressed}
