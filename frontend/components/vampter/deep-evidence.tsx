'use client'

import { useState } from 'react'
import {
  TrendingUp,
  FileText,
  Network,
  Table2,
  Milestone,
  CircleAlert,
} from 'lucide-react'
import { cn } from '@/lib/utils'

/* ==================================================================== */
/* Skeleton                                                               */
/* ==================================================================== */

function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn('animate-pulse rounded-md bg-muted/40', className)} />
  )
}

/* ==================================================================== */
/* Row 3: Greed Trajectory Engine                                        */
/* ==================================================================== */

interface TimelinePoint {
  year?: string
  month?: string
  severity?: number
  change_count?: number
  [key: string]: unknown
}

/** Normalise either backend format (month/change_count) or direct (year/severity) */
function normaliseTimeline(raw: Array<Record<string, unknown>>): Array<{ year: string; severity: number }> {
  if (!raw || raw.length === 0) {
    // Fallback static series for demo
    return [
      { year: '2021', severity: 22 },
      { year: '2022', severity: 34 },
      { year: '2023', severity: 41 },
      { year: '2024', severity: 58 },
      { year: '2025', severity: 71 },
      { year: '2026', severity: 84 },
    ]
  }

  return raw.map((pt: TimelinePoint) => {
    const yearRaw = pt.year ?? pt.month?.slice(0, 4) ?? 'N/A'
    const severityRaw =
      pt.severity ??
      (typeof pt.change_count === 'number' ? Math.min(pt.change_count * 12, 100) : 50)
    return { year: String(yearRaw), severity: Number(severityRaw) }
  })
}

function SeverityTrendChart({
  revisions,
  activeIndex,
}: {
  revisions: Array<{ year: string; severity: number }>
  activeIndex: number
}) {
  const w = 460
  const h = 150
  const padX = 12
  const padY = 18
  const innerW = w - padX * 2
  const innerH = h - padY * 2
  const max = 100
  const step = innerW / Math.max(revisions.length - 1, 1)

  const coords = revisions.map((r, i) => ({
    x: padX + i * step,
    y: padY + innerH - (r.severity / max) * innerH,
    ...r,
  }))

  const linePath = coords
    .map((c, i) => `${i === 0 ? 'M' : 'L'}${c.x.toFixed(1)},${c.y.toFixed(1)}`)
    .join(' ')

  const areaPath =
    `M${coords[0].x},${padY + innerH} ` +
    coords.map((c) => `L${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' ') +
    ` L${coords[coords.length - 1].x},${padY + innerH} Z`

  const first = revisions[0]?.severity ?? 0
  const last = revisions[revisions.length - 1]?.severity ?? 0
  const pctChange =
    first > 0 ? `+${Math.round(((last - first) / first) * 100)}% since ${revisions[0]?.year}` : ''

  return (
    <>
      <div className="mb-1 flex items-center justify-between px-1">
        <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          Predatory Severity Trend
        </span>
        {pctChange && (
          <span className="font-mono text-[10px] text-danger">{pctChange}</span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full"
        role="img"
        aria-label="Predatory severity trend rising over time"
      >
        <defs>
          <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-danger)" stopOpacity="0.28" />
            <stop offset="100%" stopColor="var(--color-danger)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <line
            key={g}
            x1={padX}
            x2={w - padX}
            y1={padY + innerH * g}
            y2={padY + innerH * g}
            stroke="var(--color-border)"
            strokeWidth={1}
            strokeDasharray="2 4"
          />
        ))}

        <path d={areaPath} fill="url(#trendFill)" />
        <path
          d={linePath}
          fill="none"
          stroke="var(--color-danger)"
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ filter: 'drop-shadow(0 0 6px var(--color-danger))' }}
        />

        {coords.map((c, i) => {
          const active = i === activeIndex
          return (
            <g key={c.year}>
              <circle
                cx={c.x}
                cy={c.y}
                r={active ? 6 : 3.5}
                fill={active ? 'var(--color-danger)' : 'var(--color-background)'}
                stroke="var(--color-danger)"
                strokeWidth={2}
                style={
                  active ? { filter: 'drop-shadow(0 0 8px var(--color-danger))' } : undefined
                }
              />
              {active && (
                <text
                  x={c.x}
                  y={c.y - 14}
                  textAnchor="middle"
                  className="fill-danger font-mono text-[11px] font-bold"
                >
                  {c.severity}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </>
  )
}

function TrajectoryEngine({
  timeline,
  isLoading,
}: {
  timeline: Array<Record<string, unknown>>
  isLoading: boolean
}) {
  const revisions = normaliseTimeline(timeline)
  const [active, setActive] = useState(revisions.length - 1)

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* Left: timeline + chart */}
      <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur-xl">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-danger" />
          <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Greed Trajectory Engine
          </h3>
        </div>

        {isLoading ? (
          <div className="mt-6 space-y-3">
            <Skeleton className="h-5 w-full" />
            <Skeleton className="h-8 w-full rounded-lg" />
            <Skeleton className="h-36 w-full rounded-xl" />
          </div>
        ) : (
          <>
            {/* Timeline slider */}
            <div className="mt-6">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
                  Corporate Revision
                </span>
                <span className="font-mono text-xs font-semibold text-danger">
                  v{revisions[active]?.year}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={revisions.length - 1}
                value={active}
                onChange={(e) => setActive(Number(e.target.value))}
                aria-label="Select document revision year"
                className="vampter-slider mt-3 w-full"
              />
              <div className="mt-2 flex justify-between">
                {revisions.map((r, i) => (
                  <button
                    key={r.year}
                    onClick={() => setActive(i)}
                    className={cn(
                      'font-mono text-[10px] tabular-nums transition-colors',
                      i === active
                        ? 'font-bold text-danger'
                        : 'text-muted-foreground/60 hover:text-muted-foreground',
                    )}
                  >
                    {r.year}
                  </button>
                ))}
              </div>
            </div>

            {/* Trend chart */}
            <div className="mt-4 rounded-xl border border-border/50 bg-background/40 p-3">
              <SeverityTrendChart revisions={revisions} activeIndex={active} />
            </div>
          </>
        )}
      </div>

      {/* Right: historical narrative */}
      <div className="relative overflow-hidden rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur-xl sm:p-6">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-alert" />
          <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Historical Delta Tracking
          </h3>
        </div>

        {isLoading ? (
          <div className="mt-5 space-y-3">
            <Skeleton className="h-8 w-full rounded-lg" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-4/5" />
            <div className="space-y-2 border-l-2 border-muted/40 pl-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-full" />
            </div>
          </div>
        ) : (
          <article className="mt-5 space-y-4 text-sm leading-relaxed text-foreground/85">
            <div className="flex items-center gap-2 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2">
              <Milestone className="h-3.5 w-3.5 shrink-0 text-danger" />
              <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-danger">
                Delta: {revisions[Math.max(0, revisions.length - 2)]?.year ?? '—'} &rarr;{' '}
                {revisions[revisions.length - 1]?.year ?? '—'} Revision
              </span>
            </div>

            <p className="text-pretty">
              Between the{' '}
              <span className="font-semibold text-foreground">
                {revisions[Math.max(0, revisions.length - 2)]?.year ?? 'prior'}
              </span>{' '}
              and{' '}
              <span className="font-semibold text-foreground">
                {revisions[revisions.length - 1]?.year ?? 'latest'}
              </span>{' '}
              document revisions, this platform silently escalated its predatory severity
              score from{' '}
              <span className="rounded bg-danger/15 px-1 font-semibold text-danger">
                {revisions[Math.max(0, revisions.length - 2)]?.severity ?? 0}
              </span>{' '}
              to{' '}
              <span className="rounded bg-danger/15 px-1 font-semibold text-danger">
                {revisions[revisions.length - 1]?.severity ?? 0}
              </span>
              , compounding user risk across each revision cycle.
            </p>

            <ul className="space-y-2.5 border-l-2 border-border/60 pl-4">
              <li className="relative text-muted-foreground">
                <span className="absolute -left-[21px] top-1.5 h-2 w-2 rounded-full bg-danger" />
                <span className="font-semibold text-foreground/90">+ Added:</span> Automatic
                maintenance surcharge clauses, non-negotiable and compounding annually.
              </li>
              <li className="relative text-muted-foreground">
                <span className="absolute -left-[21px] top-1.5 h-2 w-2 rounded-full bg-danger" />
                <span className="font-semibold text-foreground/90">&minus; Removed:</span>{' '}
                Consumer right to regional small-claims arbitration venues.
              </li>
              <li className="relative text-muted-foreground">
                <span className="absolute -left-[21px] top-1.5 h-2 w-2 rounded-full bg-alert" />
                <span className="font-semibold text-foreground/90">~ Altered:</span> Dispute
                resolution locked to a single out-of-state jurisdiction.
              </li>
            </ul>
          </article>
        )}
      </div>
    </div>
  )
}

/* ==================================================================== */
/* Row 4: Contradiction Matrix Web                                       */
/* ==================================================================== */

interface ContradictionNode {
  marketing?: string
  reality?: string
  mismatch?: number
  label?: string
  description?: string
  score?: number
  [key: string]: unknown
}

function normaliseNodes(raw: Array<Record<string, unknown>>): Array<{
  marketing: string
  reality: string
  mismatch: number
}> {
  if (!raw || raw.length === 0) {
    return [
      {
        marketing: 'Cancel anytime with 1-click',
        reality: 'Requires a 30-day prior written notice sent via registered physical post',
        mismatch: 92,
      },
      {
        marketing: 'No hidden fees, ever',
        reality: 'A 15% "platform maintenance" surcharge applies at each renewal cycle',
        mismatch: 88,
      },
      {
        marketing: 'Your data stays private',
        reality: 'Grants a perpetual license to derived analytics after account deletion',
        mismatch: 76,
      },
    ]
  }
  return raw.map((n: ContradictionNode) => ({
    marketing: String(n.marketing ?? n.label ?? 'Interface Promise'),
    reality: String(n.reality ?? n.description ?? 'See policy clause for binding terms'),
    mismatch: Number(n.mismatch ?? n.score ?? 70),
  }))
}

function ContradictionGraph() {
  const w = 460
  const h = 240
  const alpha = { x: 90, y: 60 }
  const omega = { x: 370, y: 185 }

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full"
      role="img"
      aria-label="Network graph linking checkout promise node to hidden addendum node"
    >
      <defs>
        <linearGradient id="linkGlow" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="var(--color-primary)" />
          <stop offset="100%" stopColor="var(--color-danger)" />
        </linearGradient>
        <radialGradient id="alphaGlow">
          <stop offset="0%" stopColor="var(--color-primary)" stopOpacity="0.5" />
          <stop offset="100%" stopColor="var(--color-primary)" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="omegaGlow">
          <stop offset="0%" stopColor="var(--color-danger)" stopOpacity="0.5" />
          <stop offset="100%" stopColor="var(--color-danger)" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* ambient decoy nodes */}
      {[
        { x: 200, y: 40 },
        { x: 260, y: 120 },
        { x: 140, y: 170 },
        { x: 320, y: 70 },
        { x: 210, y: 200 },
      ].map((n, i) => (
        <g key={i}>
          <line
            x1={alpha.x}
            y1={alpha.y}
            x2={n.x}
            y2={n.y}
            stroke="var(--color-border)"
            strokeWidth={1}
          />
          <circle cx={n.x} cy={n.y} r={3} fill="var(--color-muted-foreground)" opacity={0.4} />
        </g>
      ))}

      {/* main glowing relationship line */}
      <line
        x1={alpha.x}
        y1={alpha.y}
        x2={omega.x}
        y2={omega.y}
        stroke="url(#linkGlow)"
        strokeWidth={2.5}
        strokeDasharray="6 5"
        style={{ filter: 'drop-shadow(0 0 5px var(--color-danger))' }}
      >
        <animate
          attributeName="stroke-dashoffset"
          from="22"
          to="0"
          dur="1s"
          repeatCount="indefinite"
        />
      </line>

      {/* Node Alpha */}
      <circle cx={alpha.x} cy={alpha.y} r={26} fill="url(#alphaGlow)" />
      <circle
        cx={alpha.x}
        cy={alpha.y}
        r={9}
        fill="var(--color-background)"
        stroke="var(--color-primary)"
        strokeWidth={2.5}
        style={{ filter: 'drop-shadow(0 0 6px var(--color-primary))' }}
      />
      <text
        x={alpha.x}
        y={alpha.y - 34}
        textAnchor="middle"
        className="fill-primary font-mono text-[10px] font-bold uppercase"
      >
        Node Alpha
      </text>
      <text
        x={alpha.x}
        y={alpha.y - 22}
        textAnchor="middle"
        className="fill-muted-foreground font-mono text-[9px]"
      >
        Checkout UI Promise
      </text>

      {/* Node Omega */}
      <circle cx={omega.x} cy={omega.y} r={26} fill="url(#omegaGlow)" />
      <circle
        cx={omega.x}
        cy={omega.y}
        r={9}
        fill="var(--color-background)"
        stroke="var(--color-danger)"
        strokeWidth={2.5}
        style={{ filter: 'drop-shadow(0 0 6px var(--color-danger))' }}
      />
      <text
        x={omega.x}
        y={omega.y + 26}
        textAnchor="middle"
        className="fill-danger font-mono text-[10px] font-bold uppercase"
      >
        Node Omega
      </text>
      <text
        x={omega.x}
        y={omega.y + 38}
        textAnchor="middle"
        className="fill-muted-foreground font-mono text-[9px]"
      >
        Hidden Addendum, Appendix C
      </text>
    </svg>
  )
}

function mismatchColor(v: number) {
  if (v >= 85) return 'var(--color-danger)'
  if (v >= 70) return 'var(--color-alert)'
  return 'oklch(0.75 0.15 90)'
}

function ContradictionMatrix({
  nodes,
  isLoading,
}: {
  nodes: Array<Record<string, unknown>>
  isLoading: boolean
}) {
  const rows = normaliseNodes(nodes)

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* Left: node graph */}
      <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur-xl">
        <div className="flex items-center gap-2">
          <Network className="h-4 w-4 text-primary" />
          <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Contradiction Matrix Web
          </h3>
        </div>
        <div className="mt-4 rounded-xl border border-border/50 bg-background/40 p-2">
          {isLoading ? (
            <Skeleton className="h-48 w-full rounded-xl" />
          ) : (
            <ContradictionGraph />
          )}
        </div>
        <p className="mt-3 text-center text-xs text-muted-foreground">
          Interface promise linked to a conflicting clause buried{' '}
          <span className="text-danger">4 layers deep</span>
        </p>
      </div>

      {/* Right: comparison table */}
      <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur-xl sm:p-6">
        <div className="flex items-center gap-2">
          <Table2 className="h-4 w-4 text-alert" />
          <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Structural Fragmentation Map
          </h3>
        </div>

        <div className="mt-5 overflow-hidden rounded-xl border border-border/50">
          <div className="grid grid-cols-[1fr_1fr_auto] gap-px bg-border/60 font-mono text-[10px] uppercase tracking-wider">
            <div className="bg-card px-3 py-2 text-primary">Marketing Copy</div>
            <div className="bg-card px-3 py-2 text-danger">Legal Reality</div>
            <div className="bg-card px-3 py-2 text-center text-muted-foreground">Mismatch</div>
          </div>
          <div className="divide-y divide-border/50">
            {isLoading
              ? [1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="grid grid-cols-[1fr_1fr_auto] items-center gap-px bg-background/40"
                  >
                    <div className="px-3 py-3">
                      <Skeleton className="h-8 w-full" />
                    </div>
                    <div className="px-3 py-3">
                      <Skeleton className="h-8 w-full" />
                    </div>
                    <div className="px-3 py-3">
                      <Skeleton className="h-6 w-10" />
                    </div>
                  </div>
                ))
              : rows.map((row) => {
                  const color = mismatchColor(row.mismatch)
                  return (
                    <div
                      key={row.marketing}
                      className="grid grid-cols-[1fr_1fr_auto] items-center gap-px bg-background/40"
                    >
                      <div className="px-3 py-3 text-xs leading-snug text-foreground/85">
                        &ldquo;{row.marketing}&rdquo;
                      </div>
                      <div className="px-3 py-3 text-xs leading-snug text-muted-foreground">
                        {row.reality}
                      </div>
                      <div className="flex items-center justify-center px-3 py-3">
                        <span
                          className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 font-mono text-[11px] font-bold"
                          style={{
                            color,
                            borderColor: color,
                            backgroundColor:
                              'color-mix(in oklch, ' + color + ' 12%, transparent)',
                          }}
                        >
                          <CircleAlert className="h-3 w-3" />
                          {row.mismatch}
                        </span>
                      </div>
                    </div>
                  )
                })}
          </div>
        </div>
        <p className="mt-3 font-mono text-[10px] text-muted-foreground">
          Fragmentation score = semantic distance between promised UX and enforceable text.
        </p>
      </div>
    </div>
  )
}

/* ==================================================================== */

interface DeepEvidenceProps {
  timeline: Array<Record<string, unknown>>
  contradictionNodes: Array<Record<string, unknown>>
  isLoading: boolean
}

export function DeepEvidence({ timeline, contradictionNodes, isLoading }: DeepEvidenceProps) {
  return (
    <div className="mt-4 flex flex-col gap-4">
      <TrajectoryEngine timeline={timeline} isLoading={isLoading} />
      <ContradictionMatrix nodes={contradictionNodes} isLoading={isLoading} />
    </div>
  )
}
