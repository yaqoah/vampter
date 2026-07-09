'use client'

import { useState, useEffect } from 'react'
import { Search, Zap, Wallet, ScanEye, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'

const PRESETS = ['Amazon', 'Google', 'Netflix', 'Meta', 'Zoom']

interface PlatformOption {
  id: string
  name: string
}

const RADARS = [
  {
    id: 'wallet',
    label: 'Wallet Bleeding',
    icon: Wallet,
    hint: 'Hidden fees & auto-renewals',
  },
  {
    id: 'stalker',
    label: 'Stalker Mode',
    icon: ScanEye,
    hint: 'Data harvesting & tracking',
  },
  {
    id: 'downgrade',
    label: 'The Quiet Downgrade',
    icon: RefreshCw,
    hint: 'Silent feature removal & lock-in',
  },
] as const

interface CommandInputProps {
  onExecute?: (target: string, concern: string, radars: Record<string, boolean>) => void
}

export function CommandInput({ onExecute }: CommandInputProps) {
  const [target, setTarget] = useState('')
  const [open, setOpen] = useState(false)
  const [concern, setConcern] = useState('')
  const [radars, setRadars] = useState<Record<string, boolean>>({
    wallet: true,
    stalker: false,
    downgrade: false,
  })
  const [platforms, setPlatforms] = useState<PlatformOption[]>([])

  useEffect(() => {
    let active = true
    async function fetchPlatforms() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''
        const res = await fetch(`${apiUrl}/api/v1/platforms`)
        if (res.ok) {
          const data = await res.json()
          if (active) {
            setPlatforms(data || [])
          }
        } else {
          if (active) setPlatforms([])
        }
      } catch (err) {
        console.error('Failed to fetch platforms:', err)
        if (active) setPlatforms([])
      }
    }
    fetchPlatforms()
    return () => {
      active = false
    }
  }, [])

  const filtered = platforms.filter(
    (s) =>
      target &&
      (s.name.toLowerCase().includes(target.toLowerCase()) ||
        s.id.toLowerCase().includes(target.toLowerCase())) &&
      s.name.toLowerCase() !== target.toLowerCase(),
  )

  function toggleRadar(id: string) {
    setRadars((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const handleSelect = (val: string) => {
    setTarget(val)
    setOpen(false)
    onExecute?.(val, concern, radars)
  }

  return (
    <div className="relative">
      {/* Glow behind the card */}
      <div
        className="pointer-events-none absolute -inset-x-6 -top-8 bottom-0 rounded-[2rem] bg-primary/5 blur-3xl"
        aria-hidden="true"
      />

      <div className="relative overflow-hidden rounded-3xl border border-border/70 bg-card/40 shadow-2xl shadow-black/40 backdrop-blur-xl">
        <div className="border-b border-border/50 px-5 py-3 sm:px-6">
          <span className="font-mono text-[11px] uppercase tracking-[0.25em] text-muted-foreground">
            Forensic Command Console
          </span>
        </div>

        <div className="flex flex-col gap-5 p-5 sm:p-6">
          {/* Upper: target autocomplete + presets */}
          <div className="flex flex-col gap-3">
            <label
              htmlFor="target"
              className="font-mono text-xs uppercase tracking-widest text-muted-foreground"
            >
              Target Platform or Brand
            </label>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                id="target"
                value={target}
                onChange={(e) => {
                  setTarget(e.target.value)
                  setOpen(true)
                }}
                onFocus={() => setOpen(true)}
                onBlur={() => setTimeout(() => setOpen(false), 120)}
                placeholder="e.g. Netflix, Adobe, Tabby…"
                autoComplete="off"
                className="h-12 w-full rounded-xl border border-border/70 bg-background/60 pl-10 pr-4 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground/70 focus:border-primary/50 focus:ring-2 focus:ring-ring"
              />
              {open && filtered.length > 0 && (
                <ul className="absolute z-20 mt-2 w-full overflow-hidden rounded-xl border border-border/70 bg-popover/95 py-1 shadow-xl backdrop-blur-xl">
                  {filtered.map((s) => (
                    <li key={s.id}>
                      <button
                        type="button"
                        onMouseDown={() => {
                          handleSelect(s.name)
                        }}
                        className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm text-foreground/90 transition-colors hover:bg-primary/10 hover:text-foreground"
                      >
                        <Search className="h-3.5 w-3.5 text-muted-foreground" />
                        {s.name}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Instant presets */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="mr-1 font-mono text-[11px] uppercase tracking-widest text-muted-foreground/80">
                Presets
              </span>
              {PRESETS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => handleSelect(preset)}
                  className={cn(
                    'group rounded-full border px-3.5 py-1.5 text-xs font-medium transition-all duration-200',
                    target === preset
                      ? 'border-primary/60 bg-primary/15 text-primary shadow-[0_0_18px_-6px_var(--color-primary)]'
                      : 'border-border/70 bg-background/40 text-muted-foreground hover:border-primary/40 hover:text-foreground hover:shadow-[0_0_18px_-8px_var(--color-primary)]',
                  )}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          {/* Middle: concern textarea */}
          <div className="flex flex-col gap-2">
            <label
              htmlFor="concern"
              className="font-mono text-xs uppercase tracking-widest text-muted-foreground"
            >
              Primary Concern
            </label>
            <textarea
              id="concern"
              value={concern}
              onChange={(e) => setConcern(e.target.value)}
              rows={3}
              placeholder="What is your main concern? (e.g., 'Fees Check', 'Can I delete my data?', or 'Am I locked into a 12-month contract?'). Leave blank for a full structural vulnerability audit."
              className="w-full resize-y rounded-xl border border-border/70 bg-background/60 px-4 py-3 text-sm leading-relaxed text-foreground outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-primary/50 focus:ring-2 focus:ring-ring"
            />
          </div>

          {/* Lower: vampire radar toggles */}
          <div className="flex flex-col gap-2">
            <span className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
              Vampire Radar
            </span>
            <div className="grid gap-2 sm:grid-cols-3">
              {RADARS.map((radar) => {
                const active = radars[radar.id]
                const Icon = radar.icon
                return (
                  <button
                    key={radar.id}
                    type="button"
                    onClick={() => toggleRadar(radar.id)}
                    aria-pressed={active}
                    className={cn(
                      'flex items-center gap-3 rounded-xl border px-3.5 py-3 text-left transition-all duration-200',
                      active
                        ? 'border-primary/55 bg-primary/10 shadow-[0_0_22px_-10px_var(--color-primary)]'
                        : 'border-border/70 bg-background/40 hover:border-border',
                    )}
                  >
                    <span
                      className={cn(
                        'flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border transition-colors',
                        active
                          ? 'border-primary/50 bg-primary/15 text-primary'
                          : 'border-border/70 bg-card/60 text-muted-foreground',
                      )}
                    >
                      <Icon className="h-4 w-4" />
                    </span>
                    <span className="min-w-0">
                      <span
                        className={cn(
                          'block truncate text-sm font-medium',
                          active ? 'text-foreground' : 'text-foreground/80',
                        )}
                      >
                        {radar.label}
                      </span>
                      <span className="block truncate text-[11px] text-muted-foreground">
                        {radar.hint}
                      </span>
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Action button */}
          <button
            type="button"
            onClick={() => onExecute?.(target, concern, radars)}
            className="group relative mt-1 flex h-12 w-full items-center justify-center gap-2 overflow-hidden rounded-xl bg-primary font-semibold text-primary-foreground transition-all duration-200 hover:shadow-[0_0_30px_-6px_var(--color-primary)] focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background"
          >
            <span
              className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/25 to-transparent transition-transform duration-700 group-hover:translate-x-full"
              aria-hidden="true"
            />
            <span className="relative">Execute Forensic Audit</span>
            <Zap className="relative h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
