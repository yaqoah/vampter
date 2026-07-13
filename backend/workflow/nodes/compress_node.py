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
    
    # Attempt compression
    try:
        from llmlingua import PromptCompressor  # type: ignore[import]

        def _compress() -> str:
            compressor = PromptCompressor(
                model_name="lgaalves/gpt2-dolly",
                use_llmlingua2=False,
                device_map="cpu",
            )
            result = compressor.compress_prompt(
                full_context,
                rate=_TARGET_TOKEN_RATIO,
                force_tokens=["\n"],
            )
            
            # Handle different return formats from llmlingua
            if isinstance(result, dict):
                return result.get("compressed_prompt", full_context) or full_context
            elif isinstance(result, str):
                return result
            elif isinstance(result, (list, tuple)):
                # llmlingua sometimes returns tuples like (compressed_prompt, origin_tokens, compressed_tokens)
                # Try to extract first element that looks like text
                for item in result:
                    if isinstance(item, str) and len(item) > 50:
                        return item
                return full_context
            return full_context

        loop = asyncio.get_event_loop()
        compressed = await loop.run_in_executor(None, _compress)

        # Only use compression if it actually helped (reduced tokens)
        if compressed and len(compressed.strip()) > 0 and len(compressed) < len(full_context):
            final_compressed = compressed
            logger.info("LLMLingua compression: %d → %d chars", len(full_context), len(compressed))
        else:
            logger.info("LLMLingua compression skipped — using raw context.")
            
    except Exception as exc:
        logger.warning("LLMLingua compression skipped: %s", exc)

    return {**state, "compressed_context": final_compressed}
