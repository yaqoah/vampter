"""
workflow.nodes.generate_node
============================
Structured generation node — Gemini Flash native SDK.

Responsibility
--------------
Calls Gemini Flash via ``google-genai`` SDK with native structured output
configuration to generate validated ``AuditReport`` schema directly.

Flow
----
1. Build a structured system + user prompt from ``compressed_context``.
2. Use native Gemini ``response_schema`` parameter with Pydantic model.
3. Parse JSON response and validate into ``AuditReport`` instance.
4. Write the validated ``AuditReport`` instance to ``state["report"]``.

Requirements
------------
At least one LLM provider must be configured (Gemini or Groq).
If no provider is available, an HTTPException is raised.
"""

from __future__ import annotations

import json
import logging
from fastapi import HTTPException, status

from config import settings
from workflow.state import AuditState, AuditReport

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are Vampter — an AI policy document auditor specialised in analysing
Open Terms Archive software update legal documents.

Given the compressed policy context provided, generate a structured audit
report. Your output MUST strictly conform to the AuditReport schema with:
- A numeric vulnerability_score (0.0 = clean, 100.0 = critical risk)
- A threat_level classification
- At least 3 raw_insights bullets citing specific policy sections
- timeline_trends if revision date information is present
- graph_nodes and graph_edges representing the policy document structure

Be precise, factual, and conservative in scoring. Only flag genuine risks.
""".strip()


async def generate_node(state: AuditState) -> AuditState:
    """
    Generate a validated ``AuditReport`` from compressed policy context.

    Uses native Gemini SDK with ``response_schema`` for structured JSON output.
    Falls back to Groq if Gemini key is missing or fails.

    Parameters
    ----------
    state:
        Current ``AuditState`` with ``compressed_context``,
        ``company_name``, and ``query``.

    Returns
    -------
    AuditState
        Updated state with ``report`` set to an ``AuditReport`` instance.
    """
    company_name = state.get("company_name", "Unknown")
    query = state.get("query", "")
    context = state.get("compressed_context", "")

    gemini_api_key = (
        settings.gemini_api_key.get_secret_value()
        if settings.gemini_api_key
        else None
    )

    groq_api_key = (
        settings.groq_api_key.get_secret_value()
        if settings.groq_api_key
        else None
    )

    user_message = (
        f"Company / Platform: {company_name}\n"
        f"User Query: {query}\n\n"
        f"=== POLICY CONTEXT ===\n{context}\n"
        f"=== END OF CONTEXT ===\n\n"
        "Generate the structured AuditReport based solely on the context above."
    )

    # ── Primary engine: Gemini via native SDK ─────────────────────────────────
    if gemini_api_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_api_key)

            # Use native Gemini SDK with Pydantic model as response_schema
            # The SDK accepts the model class directly for structured output
            response = client.models.generate_content(
                model=settings.llm_model or "gemini-1.5-flash",
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=AuditReport,
                ),
            )

            # Parse the JSON response and validate into AuditReport
            if response and response.text:
                json_data = json.loads(response.text)
                report = AuditReport(**json_data)

                logger.info(
                    "Generation complete (Gemini) — company='%s'  score=%.1f  threat=%s",
                    company_name,
                    report.vulnerability_score,
                    report.threat_level,
                )
                return {**state, "report": report}
            else:
                raise ValueError("Empty response from Gemini model")

        except Exception as gemini_exc:
            logger.warning(
                "Gemini generation failed (%s: %s) — pivoting to Groq failover.",
                type(gemini_exc).__name__,
                gemini_exc,
                exc_info=gemini_exc,
            )

    # ── Failover engine: Groq via instructor ──────────────────────────────────
    if groq_api_key:
        try:
            import instructor
            from groq import Groq

            logger.info("Attempting Groq generation for company='%s'...", company_name)
            groq_client = instructor.from_groq(
                Groq(api_key=groq_api_key),
                mode=instructor.Mode.JSON,
            )

            report = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=AuditReport,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            logger.info(
                "Generation complete (Groq) — company='%s'  score=%.1f  threat=%s",
                company_name,
                report.vulnerability_score,
                report.threat_level,
            )
            return {**state, "report": report}

        except Exception as groq_exc:
            logger.error(
                "Groq generation also failed (%s: %s).",
                type(groq_exc).__name__,
                groq_exc,
                exc_info=groq_exc,
            )
    else:
        logger.warning(
            "GROQ_API_KEY not set — no LLM provider available for company='%s'.",
            company_name,
        )

    # ── No LLM provider available ─────────────────────────────────────────────
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="LLM_PROVIDER_MISSING"
    )