function GithubMark() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className="h-4 w-4" aria-hidden="true">
      <path d="M12 .5A11.5 11.5 0 0 0 .5 12a11.5 11.5 0 0 0 7.86 10.92c.575.1.785-.25.785-.556 0-.274-.01-1-.015-1.965-3.196.695-3.87-1.54-3.87-1.54-.523-1.33-1.278-1.685-1.278-1.685-1.044-.714.08-.7.08-.7 1.155.082 1.763 1.187 1.763 1.187 1.026 1.758 2.693 1.25 3.35.955.104-.744.402-1.25.73-1.538-2.552-.29-5.236-1.276-5.236-5.68 0-1.255.448-2.28 1.183-3.084-.119-.29-.513-1.46.112-3.043 0 0 .965-.31 3.163 1.178a11 11 0 0 1 5.76 0c2.196-1.488 3.16-1.178 3.16-1.178.626 1.583.232 2.753.114 3.043.736.804 1.18 1.83 1.18 3.084 0 4.415-2.688 5.386-5.25 5.67.414.356.782 1.06.782 2.137 0 1.543-.014 2.787-.014 3.166 0 .309.208.662.79.55A11.5 11.5 0 0 0 23.5 12 11.5 11.5 0 0 0 12 .5Z" />
    </svg>
  )
}

function StakeLogo() {
  return (
    <span className="relative flex h-9 w-9 items-center justify-center rounded-lg border border-primary/30 bg-primary/10">
      <span
        className="absolute inset-0 rounded-lg bg-primary/25 blur-md"
        aria-hidden="true"
      />
      {/* Wooden stake mark */}
      <svg
        viewBox="0 0 24 24"
        fill="none"
        className="relative h-5 w-5 text-primary"
        aria-hidden="true"
      >
        <path
          d="M12 2.5c1.4 2 2.2 3.5 2.2 5.2 0 1.3-.9 2.1-2.2 2.1s-2.2-.8-2.2-2.1c0-1.7.8-3.2 2.2-5.2Z"
          fill="currentColor"
        />
        <path
          d="M10.4 9.4h3.2l-.7 9.3a.9.9 0 0 1-1.8 0l-.7-9.3Z"
          fill="currentColor"
          opacity="0.85"
        />
        <path d="M12 19v2.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </span>
  )
}

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/60 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-3">
          <StakeLogo />
          <span className="font-mono text-sm font-semibold tracking-[0.25em] text-foreground">
            VAMPTER
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 rounded-full border border-border/70 bg-card/60 px-3 py-1.5 backdrop-blur-md sm:flex">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary/70" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
            </span>
            <span className="font-mono text-xs text-muted-foreground">
              System A Core: <span className="text-foreground">Active</span>
            </span>
          </div>

          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            aria-label="View on GitHub"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-border/70 bg-card/60 text-muted-foreground backdrop-blur-md transition-colors hover:border-primary/40 hover:text-foreground"
          >
            <GithubMark />
          </a>
        </div>
      </div>
    </header>
  )
}
