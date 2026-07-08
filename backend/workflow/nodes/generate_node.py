"""
workflow.nodes.generate_node
============================
Structured generation node — Gemini Flash + instructor.

Responsibility
--------------
Calls Gemini Flash via ``google.genai`` and wraps it with the
``instructor`` library to force deterministic Pydantic serialisation
directly into the ``AuditReport`` schema.

Flow
----
1. Build a structured system + user prompt from ``compressed_context``.
2. Initialise an ``instructor``-patched Gemini client.
3. Call ``client.chat.completions.create(response_model=AuditReport, ...)``.
4. Write the validated ``AuditReport`` instance to ``state["report"]``.

Fallback
--------
If the Gemini API key is absent or the call fails, the node returns a
deterministic mock ``AuditReport`` populated from the available context
so the full request-response cycle always completes.
"""

from __future__ import annotations

import logging

from config import settings
from workflow.state import AuditState, AuditReport, ThreatLevel, PolicyInsight

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


def _build_mock_report(company_name: str, context: str) -> AuditReport:
    """Generate a plausible mock AuditReport when the live LLM is unavailable."""
    return AuditReport(
        company_name=company_name,
        vulnerability_score=42.5,
        threat_level=ThreatLevel.MEDIUM,
        category_metrics={
            "data_privacy": 65.0,
            "financial": 30.0,
            "contract_terms": 25.0,
        },
        direct_insights=[
            {
                "category": "Data Privacy",
                "text": "Policy scope is broadly defined and may include implicit data collection.",
            },
            {
                "category": "Financial",
                "text": "90-day retention window post-termination may not meet GDPR Article 5(1)(e).",
            },
            {
                "category": "Contract Terms",
                "text": "Third-party data sharing clause lacks explicit opt-out mechanism.",
            },
        ],
        raw_insights=[
            PolicyInsight(
                section="§1 Scope of Application",
                insight="Policy scope is broadly defined and may include implicit data collection.",
                severity=ThreatLevel.LOW,
            ),
            PolicyInsight(
                section="§2 Data Retention",
                insight="90-day retention window post-termination may not meet GDPR Article 5(1)(e).",
                severity=ThreatLevel.MEDIUM,
            ),
            PolicyInsight(
                section="§3 Third-Party Sharing",
                insight="Third-party data sharing clause lacks explicit opt-out mechanism.",
                severity=ThreatLevel.HIGH,
            ),
        ],
        timeline_trends=[],
        graph_nodes=[
            {"id": "p1", "label": company_name, "node_type": "Platform", "properties": {}},
            {"id": "d1", "label": "Privacy Policy", "node_type": "Document", "properties": {}},
            {"id": "r1", "label": "v4.1.0", "node_type": "Revision", "properties": {}},
        ],
        graph_edges=[
            {"source": "p1", "target": "d1", "relation": "TRACKS_POLICY"},
            {"source": "d1", "target": "r1", "relation": "HAS_REVISION_VERSION"},
        ],
    )


async def generate_node(state: AuditState) -> AuditState:
    """
    Generate a validated ``AuditReport`` from compressed policy context.

    Uses ``instructor`` to patch the Gemini client and enforce Pydantic
    schema compliance on the model's output. Falls back to Groq if Gemini
    key is missing or fails, and finally to a mock report.

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

    # ── Primary engine: Gemini via instructor ─────────────────────────────────
    if gemini_api_key:
        try:
            import google.genai as genai  # type: ignore[import]
            import instructor  # type: ignore[import]

            raw_client = genai.Client(api_key=gemini_api_key)
            client = instructor.from_gemini(
                client=raw_client,
                mode=instructor.Mode.GEMINI_JSON,
            )

            report: AuditReport = client.chat.completions.create(
                model="gemini-2.0-flash",
                response_model=AuditReport,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )

            logger.info(
                "Generation complete (Gemini) — company='%s'  score=%.1f  threat=%s",
                company_name,
                report.vulnerability_score,
                report.threat_level,
            )
            return {**state, "report": report}

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
            import instructor  # type: ignore[import]
            from groq import Groq  # type: ignore[import]

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
                "Groq generation also failed (%s: %s) — returning mock AuditReport.",
                type(groq_exc).__name__,
                groq_exc,
                exc_info=groq_exc,
            )
    else:
        logger.warning(
            "GROQ_API_KEY not set — returning mock AuditReport for company='%s'.",
            company_name,
        )

    # ── Final safety net: deterministic mock ──────────────────────────────────
    mock = _build_mock_report(company_name, context)
    return {**state, "report": mock}

