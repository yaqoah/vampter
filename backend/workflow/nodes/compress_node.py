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
        return {**state, "compressed_context": ""}

    # Join all passages into one block for the compressor.
    full_context = "\n\n".join(passages)

    compressed: str = full_context  # safe fallback

    try:
        from llmlingua import PromptCompressor  # type: ignore[import]

        def _compress() -> str:
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
            return result.get("compressed_prompt", full_context)

        loop = asyncio.get_event_loop()
        compressed = await loop.run_in_executor(None, _compress)

        original_tokens = len(full_context.split())
        compressed_tokens = len(compressed.split())
        ratio = compressed_tokens / max(original_tokens, 1)
        logger.info(
            "LLMLingua compression: %d → %d tokens (%.0f%% retained)  company='%s'",
            original_tokens,
            compressed_tokens,
            ratio * 100,
            company_name,
        )

    except ImportError:
        logger.warning(
            "llmlingua not installed — skipping compression, using raw context."
        )
    except Exception as exc:
        logger.error(
            "LLMLingua compression failed: %s — using uncompressed context.", exc
        )

    return {**state, "compressed_context": compressed}
