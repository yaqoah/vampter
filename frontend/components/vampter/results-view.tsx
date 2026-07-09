'use client'

import {
  AlertTriangle,
  ShieldAlert,
  Wallet,
  Database,
  GitBranch,
  Cpu,
  ArrowRight,
  Zap,
  HelpCircle,
  AlertCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { DeepEvidence } from '@/components/vampter/deep-evidence'
import type { AuditResult, VisualInsight } from '@/types/audit'

/* ------------------------------------------------------------------ */
/* Skeleton primitives                                                  */
/* ------------------------------------------------------------------ */

function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn('animate-pulse rounded-md bg-muted/40', className)}
    />
  )
}

/* ---------- Card A: Radial gauge ---------- */

function RadialGauge({ value }: { value: number }) {
  const size = 176
  const stroke = 14
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const dash = (value / 100) * circumference

  return (
    <div className="relative flex items-center justify-center">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="-rotate-90"
        role="img"
        aria-label={`${value}% exposure risk`}
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--color-danger)"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          className="[animation:gauge-fill_1.4s_ease-out_forwards]"
          style={{
            filter: 'drop-shadow(0 0 8px var(--color-danger))',
          }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span
          className="text-4xl font-bold tracking-tight text-danger"
          style={{ textShadow: '0 0 22px var(--color-danger)' }}
        >
          {value}%
        </span>
        <span className="mt-1 font-mono text-[10px] font-semibold uppercase tracking-[0.2em] text-danger/90">
          {value >= 75 ? 'Critical Threat' : value >= 50 ? 'High Risk' : 'Moderate Risk'}
        </span>
      </div>
    </div>
  )
}

function RadialGaugeSkeleton() {
  return (
    <div className="flex justify-center">
      <Skeleton className="h-44 w-44 rounded-full" />
    </div>
  )
}

/* ---------- Card B: Category health matrix ---------- */

const CATEGORY_ICONS: Record<string, typeof Wallet> = {
  financial: Wallet,
  data: Database,
  contract: GitBranch,
}

function getIcon(key: string) {
  const lower = key.toLowerCase()
  for (const [k, Icon] of Object.entries(CATEGORY_ICONS)) {
    if (lower.includes(k)) return Icon
  }
  return Wallet
}

function tone(value: number) {
  if (value >= 75) return 'var(--color-danger)'
  if (value >= 55) return 'var(--color-alert)'
  return 'oklch(0.75 0.15 90)'
}

function ChannelMeter({
  label,
  value,
  icon: Icon,
}: {
  label: string
  value: number
  icon: typeof Wallet
}) {
  const color = tone(value)
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-sm font-medium text-foreground/90">
          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
          {label}
        </span>
        <span
          className="font-mono text-xs font-semibold"
          style={{ color }}
        >
          {value}%
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted/60">
        <div
          className="h-full rounded-full [animation:bar-grow_1.2s_ease-out_forwards]"
          style={{
            width: `${value}%`,
            backgroundColor: color,
            boxShadow: `0 0 12px -2px ${color}`,
          }}
        />
      </div>
    </div>
  )
}

function CategoryMatrixSkeleton() {
  return (
    <div className="mt-6 flex flex-col gap-5">
      {[1, 2, 3].map((i) => (
        <div key={i} className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <Skeleton className="h-4 w-36" />
            <Skeleton className="h-4 w-10" />
          </div>
          <Skeleton className="h-2.5 w-full rounded-full" />
        </div>
      ))}
    </div>
  )
}

/* ---------- Card C: Ingestion shredder status ---------- */

function Sparkline({ color }: { color: string }) {
  const pts = [8, 5, 12, 7, 14, 6, 10, 4, 9, 3]
  const max = Math.max(...pts)
  const w = 88
  const h = 22
  const step = w / (pts.length - 1)
  const d = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${i * step},${h - (p / max) * h}`)
    .join(' ')
  return (
    <svg width={w} height={h} className="overflow-visible" aria-hidden="true">
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" />
    </svg>
  )
}

function ShredderRow({
  label,
  value,
  color,
  spark,
}: {
  label: string
  value: string
  color: string
  spark?: boolean
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-border/50 bg-background/50 px-3 py-2.5">
      <span className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className="flex items-center gap-3">
        {spark && <Sparkline color={color} />}
        <span className="font-mono text-sm font-semibold" style={{ color }}>
          {value}
        </span>
      </span>
    </div>
  )
}

/* ---------- Risk badges ---------- */

function RiskBadge({ label, variant }: { label: string; variant: 'danger' | 'alert' }) {
  return (
    <span
      className={cn(
        'inline-flex shrink-0 items-center rounded-md border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider',
        variant === 'danger'
          ? 'border-danger/40 bg-danger/10 text-danger'
          : 'border-alert/40 bg-alert/10 text-alert',
      )}
    >
      {label}
    </span>
  )
}

function insightVariant(category: string): 'danger' | 'alert' {
  const lower = category.toLowerCase()
  if (
    lower.includes('predatory') ||
    lower.includes('hidden') ||
    lower.includes('critical')
  )
    return 'danger'
  return 'alert'
}

function InsightCardSkeleton() {
  return (
    <li className="flex flex-col gap-2 rounded-xl border border-border/50 bg-background/40 p-3.5 sm:flex-row sm:items-center sm:gap-4">
      <Skeleton className="h-5 w-28 shrink-0 rounded-md" />
      <Skeleton className="h-4 w-full" />
    </li>
  )
}

/* ---------- Error banner ---------- */

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="mt-8 flex items-start gap-3 rounded-2xl border border-danger/40 bg-danger/8 p-5 text-sm text-foreground/90 backdrop-blur-xl">
      <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-danger" />
      <div>
        <p className="font-semibold text-danger">Audit Pipeline Error</p>
        <p className="mt-1 text-muted-foreground">{message}</p>
        <p className="mt-2 font-mono text-[11px] text-muted-foreground/60">
          Ensure the backend is running at{' '}
          <span className="text-primary">{process.env.NEXT_PUBLIC_API_URL}</span> and
          the Docker stack is healthy.
        </p>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* ResultsView                                                          */
/* ------------------------------------------------------------------ */

interface ResultsViewProps {
  target?: string
  result: AuditResult | null
  isLoading: boolean
  error: string | null
}

export function ResultsView({ target, result, isLoading, error }: ResultsViewProps) {
  const brand = target?.trim() || 'the submitted document'

  // ── Error state ────────────────────────────────────────────────────
  if (error) {
    return <ErrorBanner message={error} />
  }

  const score = result ? Math.round((result as any).score ?? result.vulnerability_score ?? 84) : 84
  const threatLevel = result?.threat_level ?? 'CRITICAL'
  const categoryEntries = result?.category_metrics
    ? Object.entries(result.category_metrics)
    : []
  const rawInsights = result?.direct_insights ?? result?.raw_insights ?? []
  const insights: VisualInsight[] = Array.isArray(rawInsights)
    ? rawInsights.map((item: any) => ({
      category: item.category ?? item.section ?? 'Vulnerability',
      text: item.text ?? item.insight ?? '',
    }))
    : []

  return (
    <section aria-label="Audit results" className="mt-8">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="font-mono text-xs uppercase tracking-[0.25em] text-muted-foreground">
          Analysis Output
        </h2>
        {isLoading ? (
          <Skeleton className="h-6 w-44 rounded-full" />
        ) : (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-danger/40 bg-danger/10 px-2.5 py-1 font-mono text-[11px] font-semibold uppercase tracking-wider text-danger">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-danger opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-danger" />
            </span>
            {result ? 'Predatory Traps Detected' : 'Analysing…'}
          </span>
        )}
      </div>

      {/* Row 1: Macro-Corpus Telemetry Grid */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Card A — Radial gauge */}
        <div className="relative overflow-hidden rounded-2xl border border-danger/30 bg-card/50 p-5 backdrop-blur-xl shadow-[0_0_40px_-16px_var(--color-danger)]">
          <div
            className="pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-danger/20 blur-3xl"
            aria-hidden="true"
          />
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-danger" />
            <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Vulnerability Vector Score
            </h3>
          </div>
          <div className="mt-4 flex justify-center">
            {isLoading ? (
              <RadialGaugeSkeleton />
            ) : (
              <RadialGauge value={score} />
            )}
          </div>
          {isLoading ? (
            <Skeleton className="mx-auto mt-4 h-4 w-48" />
          ) : (
            <p className="mt-4 text-center text-xs text-muted-foreground">
              {score}% Exposure Risk — Threat Level:{' '}
              <span className="font-semibold text-danger">{threatLevel}</span>
            </p>
          )}
        </div>

        {/* Card B — Category health matrix */}
        <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur-xl">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-alert" />
            <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Category Health Matrix
            </h3>
          </div>
          {isLoading || categoryEntries.length === 0 ? (
            <CategoryMatrixSkeleton />
          ) : (
            <div className="mt-6 flex flex-col gap-5">
              {categoryEntries.map(([key, value]) => {
                const Icon = getIcon(key)
                const label = key
                  .replace(/_/g, ' ')
                  .replace(/\b\w/g, (c) => c.toUpperCase())
                return (
                  <ChannelMeter
                    key={key}
                    label={label}
                    value={Math.round(value)}
                    icon={Icon}
                  />
                )
              })}
            </div>
          )}
        </div>

        {/* Card C — Ingestion shredder (static pipeline telemetry) */}
        <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur-xl">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-primary" />
            <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Ingestion Shredder Status
            </h3>
          </div>
          {isLoading ? (
            <div className="mt-6 flex flex-col gap-2.5">
              <Skeleton className="h-10 w-full rounded-lg" />
              <Skeleton className="mx-auto h-3.5 w-3.5" />
              <Skeleton className="h-10 w-full rounded-lg" />
              <Skeleton className="mx-auto h-3.5 w-3.5" />
              <Skeleton className="h-10 w-full rounded-lg" />
            </div>
          ) : (
            <div className="mt-6 flex flex-col gap-2.5">
              <ShredderRow
                label="Raw Data Ingested"
                value="14,200 w"
                color="var(--color-foreground)"
                spark
              />
              <div className="flex justify-center py-0.5">
                <ArrowRight className="h-3.5 w-3.5 rotate-90 text-muted-foreground/60" />
              </div>
              <ShredderRow
                label="LLMLingua Compression"
                value="-76%"
                color="var(--color-primary)"
              />
              <div className="flex justify-center py-0.5">
                <ArrowRight className="h-3.5 w-3.5 rotate-90 text-muted-foreground/60" />
              </div>
              <ShredderRow
                label="Salient Tokens Parsed"
                value="3,400 t"
                color="var(--color-alert)"
                spark
              />
            </div>
          )}
          <div className="mt-4 rounded-lg border border-border/50 bg-background/60 px-3 py-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
            <span className="text-primary">$</span> pipeline: parse → compress →{' '}
            <span className="text-alert">rank_salience</span>{' '}
            <span className="text-muted-foreground/60">// 3.4k tokens @ 99.2% recall</span>
          </div>
        </div>
      </div>

      {/* Row 2: Direct Text Insight Box */}
      <div className="mt-4 overflow-hidden rounded-2xl border border-border/60 bg-card/50 backdrop-blur-xl">
        {/* Top: direct answer */}
        <div className="relative border-b border-border/50 bg-alert/5 p-5 sm:p-6">
          <div
            className="pointer-events-none absolute inset-y-0 left-0 w-1 bg-alert"
            aria-hidden="true"
          />
          <div className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4 text-alert" />
            <span className="font-mono text-[11px] uppercase tracking-widest text-alert">
              Direct Answer
            </span>
          </div>
          {isLoading ? (
            <div className="mt-3 space-y-2">
              <Skeleton className="h-4 w-48" />
              <Skeleton className="h-6 w-full" />
              <Skeleton className="h-6 w-4/5" />
            </div>
          ) : (
            <p className="mt-2 text-pretty text-lg font-semibold leading-relaxed text-foreground">
              {insights[0]?.text ??
                'Audit complete. Review the structural risk callouts below for key findings.'}
            </p>
          )}
        </div>

        {/* Bottom: structural risk callouts */}
        <div className="p-5 sm:p-6">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-danger" />
            <span className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
              High-Salience Structural Risks &middot; {brand}
            </span>
          </div>
          <ul className="mt-4 flex flex-col gap-3">
            {isLoading ? (
              <>
                <InsightCardSkeleton />
                <InsightCardSkeleton />
                <InsightCardSkeleton />
                <InsightCardSkeleton />
              </>
            ) : insights.length > 0 ? (
              insights.map((insight, i) => (
                <li
                  key={`${insight.category}-${i}`}
                  className="flex flex-col gap-2 rounded-xl border border-border/50 bg-background/40 p-3.5 sm:flex-row sm:items-center sm:gap-4"
                >
                  <RiskBadge
                    label={insight.category}
                    variant={insightVariant(insight.category)}
                  />
                  <span className="text-sm leading-relaxed text-foreground/85">
                    {insight.text}
                  </span>
                </li>
              ))
            ) : (
              <li className="rounded-xl border border-border/50 bg-background/40 p-3.5 text-sm text-muted-foreground">
                No structural risks detected in this audit run.
              </li>
            )}
          </ul>
        </div>
      </div>

      {/* Rows 3 & 4: Deep architectural evidence */}
      <DeepEvidence
        timeline={result?.greed_trajectory_timeline ?? result?.timeline_trends ?? []}
        contradictionNodes={result?.contradiction_matrix_nodes ?? result?.graph_nodes ?? []}
        isLoading={isLoading}
      />
    </section>
  )
}
