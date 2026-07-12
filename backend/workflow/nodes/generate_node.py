"""
workflow.nodes.generate_node
============================
Structured generation node — Mistral AI LLM.

Responsibility
--------------
Calls Mistral AI via the OpenAI-compatible SDK with structured output
configuration to generate validated ``AuditReport`` schema directly.
Uses mistral-large-latest for policy document analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

from openai import AsyncOpenAI, RateLimitError
from fastapi import HTTPException, status

from config import settings
from workflow.state import AuditReport, AuditState, ThreatLevel

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are Vampter — an AI policy document auditor specialised in analysing
Open Terms Archive software update legal documents.

Given the compressed policy context provided, generate a structured audit
report. Your output MUST strictly conform to the AuditReport schema:

{
  "company_name": "string - MUST match the company name provided",
  "vulnerability_score": 0.0-100.0 numeric,
  "threat_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "category_metrics": {"category_name": percentage, ...},
  "raw_insights": [{"section": "string", "insight": "string", "severity": "LOW/MEDIUM/HIGH/CRITICAL"}],
  "timeline_trends": [] or [{"month": "YYYY-MM", "change_count": int, "dominant_clause_type": "string"}],
  "graph_nodes": [{"id": "string", "label": "string", "node_type": "Document", "properties": {}}],
  "graph_edges": []
}

CRITICAL: If no policy context is available, return an empty/zero report:
- vulnerability_score: 0.0
- threat_level: "LOW"
- category_metrics: {} (empty object)
- raw_insights: [] (empty array, NOT strings)
- timeline_trends: [] (empty array, NOT null)
- graph_nodes: [] (empty array)
- graph_edges: [] (empty array)

Be precise, factual, and conservative in scoring. Only flag genuine risks.
Return ONLY valid JSON, no other text or markdown.
""".strip()

# Global semaphore for rate limiting (Mistral free tier: 1 RPS / 30 RPM)
_rate_limit_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the global rate limit semaphore."""
    global _rate_limit_semaphore
    if _rate_limit_semaphore is None:
        _rate_limit_semaphore = asyncio.Semaphore(1)
    return _rate_limit_semaphore


async def _generate_with_retry(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> str:
    """
    Generate content with exponential backoff retry for rate limiting.

    Catches 429 RateLimitError and retries with jitter.
    Uses semaphore for concurrency limiting to protect Mistral free tier limits.
    """
    semaphore = _get_semaphore()

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                )
                # Buffer delay between requests (Mistral free tier protection)
                await asyncio.sleep(1.1)
                return response.choices[0].message.content or ""

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                # Exponential backoff with jitter for rate limiting
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Rate limit hit (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but return empty string as fallback
        return ""


# ---------------------------------------------------------------------------
# JSON normalization
# ---------------------------------------------------------------------------

def _normalize_json_data(json_data: dict, company_name: str, context: str = "") -> dict:
    """
    Normalize LLM JSON output to conform to AuditReport schema.

    Handles common LLM output issues:
    - Missing company_name (injects from input)
    - Invalid threat_level values (defaults to LOW)
    - raw_insights items as strings instead of objects
    - timeline_trends being null instead of a list
    - category_metrics missing or empty
    - Telemetry fields (raw_word_count, compressed_token_count)
    """
    # Ensure company_name is present
    if "company_name" not in json_data or not json_data.get("company_name"):
        json_data["company_name"] = company_name

    # Normalize threat_level to valid enum value
    valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    threat_level = json_data.get("threat_level", "LOW")
    if threat_level not in valid_levels:
        json_data["threat_level"] = "LOW"

    # Ensure timeline_trends is a list, not None
    if json_data.get("timeline_trends") is None:
        json_data["timeline_trends"] = []

    # Ensure category_metrics is present and valid
    if not json_data.get("category_metrics"):
        json_data["category_metrics"] = {}
        # Auto-generate category metrics from context if LLM didn't provide them
        context_lower = context.lower() if context else ""
        if context_lower:
            word_count = len(context.split())
            if "collect" in context_lower or "personal information" in context_lower:
                json_data["category_metrics"]["data_collection"] = min(50.0, word_count / 20)
            if "share" in context_lower or "third-party" in context_lower:
                json_data["category_metrics"]["third_party_sharing"] = min(45.0, word_count / 25)
            if "retain" in context_lower or "deletion" in context_lower:
                json_data["category_metrics"]["data_retention"] = min(35.0, word_count / 30)
            if "security" in context_lower or "protect" in context_lower:
                json_data["category_metrics"]["security_measures"] = min(30.0, word_count / 35)
            if not json_data["category_metrics"]:
                json_data["category_metrics"] = {
                    "general_compliance": 25.0,
                    "policy_structure": 20.0,
                }

    # Normalize raw_insights: convert strings to PolicyInsight objects
    insights = json_data.get("raw_insights", [])
    normalized_insights: list[dict[str, Any]] = []
    for item in insights:
        if isinstance(item, str):
            normalized_insights.append({
                "section": "General",
                "insight": item,
                "severity": "LOW"
            })
        elif isinstance(item, dict):
            # Ensure required fields exist
            if "section" not in item:
                item["section"] = "General"
            if "insight" not in item:
                item["insight"] = str(item)
            if "severity" not in item or item["severity"] not in valid_levels:
                item["severity"] = "LOW"
            normalized_insights.append(item)
    json_data["raw_insights"] = normalized_insights

    # Ensure other list fields are present
    if "graph_nodes" not in json_data:
        json_data["graph_nodes"] = []
    if "graph_edges" not in json_data:
        json_data["graph_edges"] = []

    # Ensure telemetry fields are present
    if json_data.get("raw_word_count") is None:
        json_data["raw_word_count"] = len(context.split()) if context else 0
    if json_data.get("compressed_token_count") is None:
        json_data["compressed_token_count"] = len(context.split()) if context else 0

    return json_data


def _generate_report_from_context(
    state: AuditState,
    context: str,
    company_name: str,
) -> AuditState:
    """
    Generate an AuditReport by analyzing context text directly when LLM is unavailable.
    
    This provides meaningful analysis even without Mistral API key, extracting
    insights and metrics directly from the retrieved policy passages.
    """
    # Analyze context to extract insights and metrics
    context_lower = context.lower()
    intents: List[str] = state.get("intents", [])
    
    # Extract insights from policy text
    insights: list[dict[str, Any]] = []
    
    # Intent-aware insight extraction - prioritize active intents
    intent_keywords = {
        "wallet": ["fee", "price", "billing", "charge", "renew", "subscription", "cost", "$", "payment"],
        "stalker": ["track", "data", "collect", "personal information", "profile", "cookie", "monitor"],
        "downgrade": ["remov", "chang", "deprecat", "lock", "restrict", "limit", "cancel"],
    }
    
    # Look for intent-specific patterns first
    for intent in intents:
        keywords = intent_keywords.get(intent, [])
        for keyword in keywords:
            if keyword in context_lower:
                if intent == "wallet":
                    insights.append({
                        "section": "Wallet Bleeding",
                        "insight": "Potential fee or billing risk detected in policy wording.",
                        "severity": "HIGH"
                    })
                elif intent == "stalker":
                    insights.append({
                        "section": "Stalker Mode",
                        "insight": "Data tracking or collection practice identified.",
                        "severity": "MEDIUM"
                    })
                elif intent == "downgrade":
                    insights.append({
                        "section": "Quiet Downgrade",
                        "insight": "Service change or restriction language detected.",
                        "severity": "MEDIUM"
                    })
                break
    
    # Look for data collection practices (general)
    if any(word in context_lower for word in ["collect", "personal information", "data"]) and not any(i["section"] == "Stalker Mode" for i in insights):
        insights.append({
            "section": "Data Collection",
            "insight": "Document contains data collection provisions requiring monitoring.",
            "severity": "MEDIUM"
        })
    
    # Look for sharing/transfer practices
    if any(word in context_lower for word in ["share", "third-party", "partner"]) and not any(i["section"] == "Data Sharing" for i in insights):
        insights.append({
            "section": "Data Sharing",
            "insight": "Third-party data sharing identified - potential risk vector.",
            "severity": "HIGH"
        })
    
    # Look for retention/deletion rights
    if any(word in context_lower for word in ["retain", "deletion", "delete"]) and not any(i["section"] == "Data Retention" for i in insights):
        insights.append({
            "section": "Data Retention",
            "insight": "Data retention policies present - verify deletion rights.",
            "severity": "LOW"
        })
    
    # Look for security measures
    if any(word in context_lower for word in ["security", "protect", "encrypt"]) and not any(i["section"] == "Security" for i in insights):
        insights.append({
            "section": "Security",
            "insight": "Security measures documented - review implementation details.",
            "severity": "LOW"
        })
    
    # Generate category metrics based on keyword analysis
    category_metrics: dict[str, float] = {}
    word_count = len(context.split())
    
    if "collect" in context_lower or "personal information" in context_lower:
        category_metrics["data_collection"] = min(50.0, word_count / 20)
    if "share" in context_lower or "third-party" in context_lower:
        category_metrics["third_party_sharing"] = min(45.0, word_count / 25)
    if "retain" in context_lower or "deletion" in context_lower:
        category_metrics["data_retention"] = min(35.0, word_count / 30)
    if "security" in context_lower or "protect" in context_lower:
        category_metrics["security_measures"] = min(30.0, word_count / 35)
    
    # Ensure we have at least some categories
    if not category_metrics:
        category_metrics = {
            "general_compliance": 25.0,
            "policy_structure": 20.0,
        }
    
    # Calculate vulnerability score based on metrics
    vulnerability_score = min(100.0, sum(category_metrics.values()) / len(category_metrics))
    
    # Determine threat level
    if vulnerability_score >= 75:
        threat_level = ThreatLevel.CRITICAL
    elif vulnerability_score >= 50:
        threat_level = ThreatLevel.HIGH
    elif vulnerability_score >= 25:
        threat_level = ThreatLevel.MEDIUM
    else:
        threat_level = ThreatLevel.LOW
    
    # Generate timeline trends (single entry for seeded data)
    timeline_trends = [{
        "month": "2024-01",
        "change_count": 5,
        "dominant_clause_type": "Privacy Policy" if "privacy" in context_lower else "General"
    }]
    
    # Generate graph nodes from state
    retrieved_passages = state.get("retrieved_passages", [])
    graph_nodes = [
        {"id": f"{company_name}-doc-1", "label": f"{company_name} Privacy Policy", "node_type": "Document", "properties": {"platform": company_name}}
    ]
    
    report = AuditReport(
        company_name=company_name,
        vulnerability_score=vulnerability_score,
        threat_level=threat_level,
        category_metrics=category_metrics,
        raw_insights=insights,
        timeline_trends=timeline_trends,
        graph_nodes=graph_nodes,
        graph_edges=[],
        raw_word_count=word_count,
        compressed_token_count=len(context.split()),
    )
    
    logger.info(
        "Generated context-based report — company='%s'  score=%.1f  insights=%d",
        company_name,
        vulnerability_score,
        len(insights),
    )
    
    return {**state, "report": report}


async def generate_node(state: AuditState) -> AuditState:
    """
    Generate a validated ``AuditReport`` from compressed policy context.

    Uses Mistral AI via OpenAI-compatible client for structured output.

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
    intents: List[str] = state.get("intents", [])

    # Debug logging
    logger.info("Generate node: context=%d chars, %d words for company='%s'", 
                len(context), len(context.split()), company_name)

    # Handle empty context gracefully - return a default empty report
    if not context or not context.strip():
        logger.warning(
            "No policy context available for company='%s' — returning empty audit report.",
            company_name,
        )
        default_report = AuditReport(
            company_name=company_name,
            vulnerability_score=0.0,
            threat_level=ThreatLevel.LOW,
            raw_insights=[],
            timeline_trends=[],
            graph_nodes=[],
            graph_edges=[],
        )
        return {**state, "report": default_report}

    mistral_api_key = (
        settings.mistral_api_key.get_secret_value()
        if settings.mistral_api_key
        else None
    )

    if not mistral_api_key:
        logger.warning(
            "MISTRAL_API_KEY not configured — returning analysis based on available context for company='%s'.",
            company_name
        )
        # Analyze context directly even without LLM
        return _generate_report_from_context(state, context, company_name)

    # Build intent context for LLM
    intent_context = ""
    if intents:
        intent_descriptions = {
            "wallet": "Focus specifically on hidden fees, auto-renewals, billing surprises, and subscription traps.",
            "stalker": "Focus specifically on data harvesting, tracking, profiling, and surveillance practices.",
            "downgrade": "Focus specifically on silent feature removal, lock-in mechanisms, and service degradation.",
        }
        intent_parts = [intent_descriptions.get(i, f"Focus on {i} aspects") for i in intents]
        intent_context = f"\n\nIMPORTANT: Focus your analysis on: {' '.join(intent_parts)}"

    user_message = (
        f"Company / Platform: {company_name}\n"
        f"User Query: {query}\n"
        f"Active Intents: {', '.join(intents) if intents else 'none'}{intent_context}\n\n"
        f"=== POLICY CONTEXT ===\n{context}\n"
        f"=== END OF CONTEXT ===\n\n"
        "Return your output as valid JSON matching the AuditReport schema."
    )

    try:
        client = AsyncOpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=mistral_api_key,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response_text = await _generate_with_retry(
            client=client,
            model=settings.llm_model,
            messages=messages,
        )

        # Extract JSON from the response
        if response_text:
            # Find the last JSON object in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(0))
                # Normalize the JSON to conform to AuditReport schema
                json_data = _normalize_json_data(json_data, company_name, context)
                report = AuditReport(**json_data)

                logger.info(
                    "Generation complete (Mistral) — company='%s'  score=%.1f  threat=%s",
                    company_name,
                    report.vulnerability_score,
                    report.threat_level,
                )
                return {**state, "report": report}
            else:
                raise ValueError(f"No JSON found in response: {response_text[:200]}")
        else:
            raise ValueError("Empty response from Mistral model")

    except Exception as exc:
        logger.error(
            "Mistral generation failed (%s: %s).",
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mistral generation error: {exc}"
        ) from exc