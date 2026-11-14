import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { BarChart3, Eye, History, KeyRound, Search, Sparkles } from 'lucide-react'
import { api, getApiKey } from '../lib/api'
import { ApiKeyDialog } from './ApiKeyDialog'

type Stats = { search: { searches_today: number; daily_limit: number } }

const NAV = [
  { to: '/', label: 'Research', icon: Search, end: true },
  { to: '/history', label: 'History', icon: History },
  { to: '/watch', label: 'Topic watch', icon: Eye },
  { to: '/dashboard', label: 'Results', icon: BarChart3 },
]

export function Layout() {
  const [keyOpen, setKeyOpen] = useState(!getApiKey())
  const [stats, setStats] = useState<Stats | null>(null)

  useEffect(() => {
    if (keyOpen) return
    const load = () => api.get<Stats>('/stats').then(setStats).catch(() => setStats(null))
    load()
    const t = setInterval(load, 60000)
    return () => clearInterval(t)
  }, [keyOpen])

  const used = stats?.search.searches_today ?? 0
  const limit = stats?.search.daily_limit ?? 0

  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-line bg-[#0e1014] px-3 py-5 md:flex">
        <div className="mb-8 flex items-center gap-2.5 px-2">
          <div className="rounded-lg bg-accent p-1.5"><Sparkles size={18} className="text-white" /></div>
          <div>
            <div className="text-sm font-semibold leading-tight">Research Agent</div>
            <div className="text-[11px] text-muted">multi-agent · cited · checked</div>
          </div>
        </div>
        <nav className="flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end}
              className={({ isActive }) => `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                isActive ? 'bg-panel-2 text-ink' : 'text-ink-2 hover:bg-panel hover:text-ink'}`}>
              <Icon size={17} /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto space-y-3 px-2">
          {limit > 0 && (
            <div>
              <div className="mb-1.5 flex justify-between text-[11px] text-muted">
                <span>Web searches today</span><span>{used} / {limit}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-panel-2">
                <div className={`h-full rounded-full ${used >= limit ? 'bg-bad' : used > limit * 0.8 ? 'bg-warn' : 'bg-accent'}`}
                  style={{ width: `${Math.min(100, (used / limit) * 100)}%` }} />
              </div>
            </div>
          )}
          <button onClick={() => setKeyOpen(true)} className="flex items-center gap-2 text-xs text-muted hover:text-ink">
            <KeyRound size={14} /> App password
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 overflow-x-auto border-b border-line px-4 py-2 md:hidden">
          {NAV.map(({ to, label, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `whitespace-nowrap rounded-md px-3 py-1.5 text-sm ${isActive ? 'bg-panel-2' : 'text-ink-2'}`}>{label}</NavLink>
          ))}
        </header>
        <main className="scroll-thin flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-5 py-8 md:px-8">
            <Outlet />
          </div>
        </main>
      </div>
      {keyOpen && <ApiKeyDialog required={!getApiKey()} onClose={() => setKeyOpen(false)} />}
    </div>
  )
}
