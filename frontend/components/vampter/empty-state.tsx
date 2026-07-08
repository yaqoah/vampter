import { Radar } from 'lucide-react'

function Bar({ className }: { className?: string }) {
  return <div className={`rounded-md bg-muted/50 ${className ?? ''}`} />
}

export function EmptyState() {
  return (
    <section aria-label="Audit results placeholder" className="mt-8">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="font-mono text-xs uppercase tracking-[0.25em] text-muted-foreground">
          Analysis Output
        </h2>
        <span className="font-mono text-[11px] text-muted-foreground/70">
          Awaiting execution
        </span>
      </div>

      <div className="rounded-3xl border border-dashed border-border/70 bg-card/20 p-5 backdrop-blur-sm sm:p-8">
        <div className="mb-8 flex flex-col items-center justify-center gap-3 text-center">
          <span className="flex h-12 w-12 items-center justify-center rounded-full border border-border/70 bg-card/50 text-muted-foreground">
            <Radar className="h-5 w-5" />
          </span>
          <p className="max-w-md text-pretty text-sm text-muted-foreground">
            Your structured vulnerability report will render here. Run an audit to
            populate risk scores, fee timelines, and clause-level findings.
          </p>
        </div>

        {/* Row 1: score + summary */}
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="rounded-2xl border border-dashed border-border/60 p-5">
            <Bar className="h-3 w-24" />
            <div className="mt-5 flex items-center justify-center">
              <div className="h-28 w-28 rounded-full border-4 border-dashed border-border/60" />
            </div>
            <Bar className="mx-auto mt-5 h-2.5 w-20" />
          </div>

          <div className="rounded-2xl border border-dashed border-border/60 p-5 lg:col-span-2">
            <Bar className="h-3 w-28" />
            <div className="mt-5 space-y-3">
              <Bar className="h-2.5 w-full" />
              <Bar className="h-2.5 w-11/12" />
              <Bar className="h-2.5 w-4/5" />
              <Bar className="h-2.5 w-2/3" />
            </div>
            <div className="mt-6 grid grid-cols-3 gap-3">
              <Bar className="h-14" />
              <Bar className="h-14" />
              <Bar className="h-14" />
            </div>
          </div>
        </div>

        {/* Row 2: chart wireframe */}
        <div className="mt-4 rounded-2xl border border-dashed border-border/60 p-5">
          <Bar className="h-3 w-32" />
          <div className="mt-6 flex h-40 items-end gap-3">
            {[40, 65, 30, 80, 55, 70, 45, 90, 60].map((h, i) => (
              <div
                key={i}
                className="flex-1 rounded-t-md bg-muted/40"
                style={{ height: `${h}%` }}
              />
            ))}
          </div>
        </div>

        {/* Row 3: findings list */}
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className="flex items-center gap-4 rounded-2xl border border-dashed border-border/60 p-4"
            >
              <div className="h-10 w-10 shrink-0 rounded-lg bg-muted/40" />
              <div className="flex-1 space-y-2">
                <Bar className="h-2.5 w-1/2" />
                <Bar className="h-2 w-4/5" />
              </div>
              <Bar className="h-6 w-12" />
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
