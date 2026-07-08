import { SiteHeader } from '@/components/vampter/site-header'
import { AuditConsole } from '@/components/vampter/audit-console'

export default function Page() {
  return (
    <div className="relative min-h-dvh overflow-hidden">
      {/* Deep indigo radial depth */}
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        aria-hidden="true"
        style={{
          background:
            'radial-gradient(60% 45% at 50% 0%, oklch(0.3 0.09 274 / 0.4), transparent 70%)',
        }}
      />
      <div
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-px bg-gradient-to-r from-transparent via-border to-transparent"
        aria-hidden="true"
      />

      <SiteHeader />

      <main className="mx-auto max-w-6xl px-4 pb-24 pt-10 sm:px-6 sm:pt-16">
        <div className="mx-auto max-w-3xl text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card/40 px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-muted-foreground backdrop-blur-md">
            AI Legal &amp; Financial Auditor
          </span>
          <h1 className="mt-5 text-balance text-4xl font-semibold tracking-tight text-foreground sm:text-5xl">
            Expose the fine print before it bites.
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-pretty text-base leading-relaxed text-muted-foreground">
            Vampter runs a forensic audit on any platform&apos;s terms, contracts,
            and billing to surface hidden fees, data risks, and lock-in traps.
          </p>
        </div>

        <AuditConsole />
      </main>
    </div>
  )
}
