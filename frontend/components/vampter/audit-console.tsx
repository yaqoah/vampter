'use client'

import { useState } from 'react'
import { CommandInput } from '@/components/vampter/command-input'
import { EmptyState } from '@/components/vampter/empty-state'
import { ResultsView } from '@/components/vampter/results-view'
import type { AuditResult } from '@/types/audit'

export function AuditConsole() {
  const [executed, setExecuted] = useState(false)
  const [target, setTarget] = useState('')
  const [concern, setConcern] = useState('')
  const [activeRadars, setActiveRadars] = useState<Record<string, boolean>>({})

  const [result, setResult] = useState<AuditResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleExecuteAudit(
    company: string,
    userConcern: string,
    radars: Record<string, boolean>,
  ) {
    if (!company.trim()) return

    setTarget(company)
    setConcern(userConcern)
    setActiveRadars(radars)
    setExecuted(true)
    setIsLoading(true)
    setError(null)
    setResult(null)

    // Build the active radar toggles array from the boolean map
    const activeIntents = Object.entries(radars)
      .filter(([, on]) => on)
      .map(([key]) => key)

    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/v1/audit`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            company_name: company,
            query: userConcern || 'Summarise the latest policy changes and risk profile.',
            intents: activeIntents,
          }),
        },
      )

      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        throw new Error(
          detail?.detail ?? `Server responded with status ${res.status}`,
        )
      }

      const data: AuditResult = await res.json()
      setResult(data)
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'An unexpected error occurred.'
      setError(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="mx-auto mt-10 max-w-3xl">
      <CommandInput
        onExecute={(value, inputConcern, radars) => {
          handleExecuteAudit(value, inputConcern, radars)
        }}
      />
      {executed ? (
        <div className="[animation:result-in_0.5s_ease-out]">
          <ResultsView
            target={target}
            result={result}
            isLoading={isLoading}
            error={error}
          />
        </div>
      ) : (
        <EmptyState />
      )}
    </div>
  )
}
