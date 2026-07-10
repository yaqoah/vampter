// ─── Backend API response models ──────────────────────────────────────────────
// Mirrors the Pydantic AuditReport from backend/workflow/state.py

export interface VisualInsight {
  /** Policy section reference (e.g. '§3.2 Data Retention') */
  category: string
  /** Human-readable insight bullet */
  text: string
  // Backend raw_insights fields (may arrive instead of category/text)
  section?: string
  insight?: string
  severity?: string
}

export interface AuditResult {
  /** Numeric 0–100 vulnerability score */
  vulnerability_score: number
  /** Overall threat classification: LOW | MEDIUM | HIGH | CRITICAL */
  threat_level: string
  /** Per-category risk percentages (keyed by category name) */
  category_metrics: Record<string, number>
  /** Ordered policy insight bullets — mapped from backend raw_insights */
  direct_insights: VisualInsight[]
  /** Month-by-month policy change frequency for the trajectory chart */
  greed_trajectory_timeline: Array<Record<string, any>>

  // ── Ingestion telemetry metrics ──────────────────────────────────
  /** Raw word count before compression */
  raw_word_count?: number
  /** Tokens after LLMLingua compression */
  compressed_token_count?: number

  // ── Optional passthrough fields from the full AuditReport schema ──────
  company_name?: string
  /** Backend field name alias */
  raw_insights?: Array<{
    section: string
    insight: string
    severity: string
  }>
  timeline_trends?: Array<{
    month: string
    change_count: number
    dominant_clause_type?: string
  }>
  graph_nodes?: Array<Record<string, any>>
  graph_edges?: Array<Record<string, any>>
  contradiction_matrix_nodes?: Array<Record<string, any>>
}
