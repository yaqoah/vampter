"""
workflow.nodes.compress_node
============================
LLMLingua context compression node.

Responsibility
--------------
Applies ``llmlingua.PromptCompressor`` to the list of retrieved passages
before they are forwarded to the generation node.

Why compress?
- Legal policy documents are verbose.  Boilerplate phrasing (recitals,
  definitions, signature blocks) adds token overhead without improving
  the quality of the generated audit report.
- Compression reduces LLM latency and cost while preserving the specific
  numeric, structural, and obligation-bearing language that matters.

Target compression ratio: ~0.5 (50 % token reduction).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

from workflow.state import AuditState

logger = logging.getLogger(__name__)

_TARGET_TOKEN_RATIO: float = 0.5


async def compress_node(state: AuditState) -> AuditState:
    """
    Compress retrieved passages using LLMLingua.

    Runs the synchronous ``PromptCompressor.compress_prompt`` in the
    default thread executor to avoid blocking the event loop.

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
        # Still return empty context to indicate no data
        return {**state, "compressed_context": ""}

    # Join all passages into one block for the compressor.
    full_context = "\n\n".join(passages)
    logger.info("Compress node: %d passages, %d chars, %d words for company='%s'", 
                len(passages), len(full_context), len(full_context.split()), company_name)
    
    # DEBUG: Log passage content preview
    if passages:
        for i, p in enumerate(passages[:3]):
            logger.debug("Compress node: passage[%d] = %d chars: %s...", i, len(p), p[:50].replace('\n', ' '))

    # If context is empty after joining, return empty
    if not full_context.strip():
        logger.warning("Compress node: empty joined context for company='%s'.", company_name)
        return {**state, "compressed_context": ""}

    # Start with uncompressed context as the default
    final_compressed = full_context

    # Only attempt compression if we have meaningful text
    if len(full_context.strip()) < 20:
        logger.info("Context too short for compression (%d chars) — using raw context.", len(full_context))
        return {**state, "compressed_context": final_compressed}

    try:
        from llmlingua import PromptCompressor  # type: ignore[import]

        def _compress() -> tuple[str, int, int]:
            compressor = PromptCompressor(
                model_name="lgaalves/gpt2-dolly",   # lightweight local model
                use_llmlingua2=False,
                device_map="cpu",
            )
            result = compressor.compress_prompt(
                full_context,
                rate=_TARGET_TOKEN_RATIO,
                force_tokens=["\n"],
            )
            compressed_text = result.get("compressed_prompt", full_context)
            return compressed_text, result.get("origin_tokens", 0), result.get("compressed_tokens", 0)

        loop = asyncio.get_event_loop()
        compressed, origin_tokens, compressed_tokens = await loop.run_in_executor(None, _compress)

        # Use compressed text only if it's non-empty and actually has tokens
        if compressed and compressed.strip() and (compressed_tokens > 0 or origin_tokens > 0):
            # If compression didn't actually compress but returned valid text, still use it
            if compressed_tokens == 0 and origin_tokens > 0:
                # Compression returned empty but original had content - use original
                final_compressed = full_context
                logger.info(
                    "Compression returned empty (%.0f → %.0f tokens) — using original context for company='%s'",
                    origin_tokens, compressed_tokens, company_name
                )
            else:
                final_compressed = compressed
                logger.info(
                    "LLMLingua compression: %d → %d tokens (%.0f%% retained)  company='%s'",
                    origin_tokens,
                    compressed_tokens,
                    (compressed_tokens / max(origin_tokens, 1)) * 100,
                    company_name,
                )
        else:
            # Compression failed - keep the uncompressed context we already have
            logger.info(
                "LLMLingua compression produced empty result — keeping original context for company='%s'",
                company_name,
            )
            # final_compressed is already set to full_context at line 72

    except ImportError:
        logger.warning(
            "llmlingua not installed — skipping compression, using raw context."
        )
        final_compressed = full_context
    except Exception as exc:
        logger.error(
            "LLMLingua compression failed: %s — using uncompressed context.", exc
        )
        final_compressed = full_context

    return {**state, "compressed_context": final_compressed}
