"""
workflow.state
==============
Pydantic v2 data models for the LangGraph orchestration engine.

Schema hierarchy
----------------

``AuditReport`` (final serialised output delivered to the client):
    ├── company_name          : str
    ├── vulnerability_score   : float  (0.0 – 100.0)
    ├── threat_level          : Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    ├── raw_insights          : List[PolicyInsight]
    ├── timeline_trends       : List[TimelineTrend]
    ├── graph_nodes           : List[GraphNode]
    └── graph_edges           : List[GraphEdge]

``AuditState`` (internal LangGraph state carrier — TypedDict):
    Carries intermediate data between nodes throughout graph execution.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enum — Threat Level
# ---------------------------------------------------------------------------


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# AuditReport sub-schemas
# ---------------------------------------------------------------------------


class PolicyInsight(BaseModel):
    """A single bullet insight extracted from policy analysis."""

    section: str = Field(description="Policy section reference (e.g. '§3.2 Data Retention')")
    insight: str = Field(description="Human-readable insight bullet point")
    severity: ThreatLevel = Field(default=ThreatLevel.LOW, description="Severity classification")


class TimelineTrend(BaseModel):
    """A month-keyed data point tracking policy change frequency over time."""

    month: str = Field(description="ISO month string e.g. '2024-03'")
    change_count: int = Field(ge=0, description="Number of policy revisions in that month")
    dominant_clause_type: Optional[str] = Field(
        default=None,
        description="Most frequently changed clause category in this month",
    )


class GraphNode(BaseModel):
    """A visual node in the frontend policy graph rendering."""

    id: str = Field(description="Unique node identifier")
    label: str = Field(description="Display label")
    node_type: str = Field(description="Entity type: Platform | Document | Revision | Clause")
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A directed edge in the frontend policy graph rendering."""

    source: str = Field(description="Source node id")
    target: str = Field(description="Target node id")
    relation: str = Field(
        description="Relation type: TRACKS_POLICY | HAS_REVISION_VERSION | CONTAINS_CLAUSE"
    )


# ---------------------------------------------------------------------------
# AuditReport — top-level output schema
# ---------------------------------------------------------------------------


class AuditReport(BaseModel):
    """
    Fully validated audit report — the final output returned by the API.

    This schema is passed to ``instructor`` to force deterministic
    serialisation from the Gemini Flash LLM response.
    """

    company_name: str = Field(description="Target company or platform name")
    vulnerability_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Numeric vulnerability score from 0 (clean) to 100 (critical)",
    )
    threat_level: ThreatLevel = Field(
        description="Overall threat classification derived from vulnerability_score"
    )
    category_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-category risk percentages (keyed by category name)",
    )
    direct_insights: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Ordered policy insight bullets for frontend (alias for raw_insights)",
    )
    raw_insights: List[PolicyInsight] = Field(
        default_factory=list,
        description="Ordered list of policy insight bullets",
    )
    timeline_trends: List[TimelineTrend] = Field(
        default_factory=list,
        description="Month-by-month policy change frequency data",
    )
    graph_nodes: List[GraphNode] = Field(
        default_factory=list,
        description="Visual graph nodes for frontend rendering",
    )
    graph_edges: List[GraphEdge] = Field(
        default_factory=list,
        description="Visual graph edges for frontend rendering",
    )

    # Ingestion telemetry metrics for frontend display
    raw_word_count: Optional[int] = Field(
        default=None,
        description="Raw word count of ingested documents before compression",
    )
    compressed_token_count: Optional[int] = Field(
        default=None,
        description="Token count after LLMLingua compression",
    )

    # Alias fields for frontend compatibility
    greed_trajectory_timeline: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Alias for timeline_trends for Greed Trajectory Engine",
    )
    contradiction_matrix_nodes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Alias for graph_nodes for Contradiction Matrix Web",
    )


# ---------------------------------------------------------------------------
# AuditState — internal LangGraph TypedDict
# ---------------------------------------------------------------------------


class AuditState(TypedDict, total=False):
    """
    Internal state carrier flowing through the LangGraph StateGraph nodes.

    All fields are optional (``total=False``) so individual nodes can
    write only their own outputs without blanking unrelated fields.
    """

    # Input fields (set by the FastAPI router before graph invocation)
    company_name: str
    query: str
    intents: List[str]

    # Router node output
    route: Literal["vector", "graph"]

    # Retrieval node output (set by either vector_node or graph_node)
    retrieved_passages: List[str]

    # Compression node output
    compressed_context: str

    # Generation node output
    report: AuditReport

    # Error tracking
    error: Optional[str]
