import type { ReactNode } from 'react'

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-line bg-panel ${className}`}>{children}</div>
}

export function Badge({ children, tone = 'neutral', title }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'good' | 'warn' | 'bad'; title?: string }) {
  const tones = {
    neutral: 'bg-panel-2 text-ink-2 border-line',
    accent: 'bg-accent-soft text-[#8fbaf0] border-[#3987e544]',
    good: 'bg-[#3fb68b1a] text-good border-[#3fb68b44]',
    warn: 'bg-[#e3b3411a] text-warn border-[#e3b34144]',
    bad: 'bg-[#e667671a] text-bad border-[#e6676744]',
  }
  return <span title={title} className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>
}

export function Button({ children, onClick, variant = 'secondary', disabled, type = 'button', title }: {
  children: ReactNode; onClick?: () => void; variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  disabled?: boolean; type?: 'button' | 'submit'; title?: string
}) {
  const v = {
    primary: 'bg-accent text-white hover:bg-[#4d95ea] disabled:bg-[#2a4d78] disabled:text-ink-2',
    secondary: 'border border-line bg-panel-2 text-ink hover:border-[#3a414b] disabled:opacity-50',
    ghost: 'text-ink-2 hover:bg-panel-2 hover:text-ink disabled:opacity-50',
    danger: 'text-bad hover:bg-[#e667671a] disabled:opacity-50',
  }
  return (
    <button type={type} title={title} onClick={onClick} disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed ${v[variant]}`}>
      {children}
    </button>
  )
}

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  )
}

export function Empty({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line px-6 py-16 text-center">
      <div className="mb-3 text-muted">{icon}</div>
      <div className="font-medium">{title}</div>
      {children && <div className="mt-1 max-w-md text-sm text-muted">{children}</div>}
    </div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return <div className="rounded-lg border border-[#e6676744] bg-[#e667671a] px-4 py-3 text-sm text-bad">{message}</div>
}
