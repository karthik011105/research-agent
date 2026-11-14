import { useState } from 'react'
import { Check, ChevronDown, ShieldAlert, X } from 'lucide-react'
import type { ProgressEvent } from '../lib/api'
import { STAGES, describe, stageOf, type StageId } from '../lib/format'

/** Live view of a running job: a stepper of the pipeline stages plus the detailed event log. */
export function Timeline({ events, finished }: { events: ProgressEvent[]; finished: boolean }) {
  const [open, setOpen] = useState(true)
  const start = events.find(e => e.t)?.t ?? 0
  const reached = new Set(events.map(stageOf).filter(Boolean) as StageId[])
  const current = [...STAGES].reverse().find(s => reached.has(s.id))?.id
  const failed = events.some(e => e.type === 'error' || e.type === 'blocked')
  const shortcut = events.some(e => e.type === 'memory_hit' || e.type === 'direct_answer')
  const rows = events.map(e => ({ e, text: describe(e) })).filter(r => r.text)
  const last = events[events.length - 1]
  const elapsed = last?.t && start ? Math.round(last.t - start) : 0

  return (
    <div className="rounded-xl border border-line bg-panel">
      <div className="flex items-center gap-1 overflow-x-auto px-5 py-4">
        {STAGES.map((s, i) => {
          const idx = STAGES.findIndex(x => x.id === current)
          const done = finished ? (!failed || i < idx) : i < idx
          const active = !finished && s.id === current
          const skipped = shortcut && !reached.has(s.id) && s.id !== 'done'
          return (
            <div key={s.id} className="flex items-center gap-1">
              <div className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium whitespace-nowrap ${
                active ? 'bg-accent-soft text-[#8fbaf0]' : done && !skipped ? 'text-ink' : 'text-muted'}`}>
                <span className={`flex h-5 w-5 items-center justify-center rounded-full border text-[10px] ${
                  failed && finished && s.id === 'done' ? 'border-bad bg-[#e667671a] text-bad'
                  : done && !skipped ? 'border-good bg-[#3fb68b1a] text-good'
                  : active ? 'border-accent text-accent' : 'border-line'}`}>
                  {failed && finished && s.id === 'done' ? <X size={11} /> : done && !skipped ? <Check size={11} /> : active ? <span className="pulse h-1.5 w-1.5 rounded-full bg-accent" /> : i + 1}
                </span>
                {s.label}
              </div>
              {i < STAGES.length - 1 && <div className={`h-px w-5 ${i < idx || finished ? 'bg-[#3a414b]' : 'bg-line'}`} />}
            </div>
          )
        })}
        <div className="ml-auto pl-4 text-xs text-muted tabular-nums">{elapsed > 0 && `${elapsed}s`}</div>
      </div>
      {rows.length > 0 && (
        <div className="border-t border-line">
          <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-5 py-2 text-xs text-muted hover:text-ink">
            <ChevronDown size={14} className={`transition-transform ${open ? '' : '-rotate-90'}`} />
            {open ? 'Hide steps' : `Show ${rows.length} steps`}
          </button>
          {open && (
            <ol className="scroll-thin max-h-64 space-y-1 overflow-y-auto px-5 pb-4">
              {rows.map(({ e, text }, i) => (
                <li key={i} className={`flex gap-3 text-[13px] ${e.type === 'defense' ? 'text-warn' : e.type === 'error' || e.type === 'blocked' ? 'text-bad' : 'text-ink-2'}`}>
                  <span className="w-10 shrink-0 text-right text-muted tabular-nums">
                    {e.t && start ? `${Math.floor((e.t - start) / 60)}:${String(Math.round(e.t - start) % 60).padStart(2, '0')}` : ''}
                  </span>
                  {e.type === 'defense' && <ShieldAlert size={14} className="mt-0.5 shrink-0" />}
                  <span className="min-w-0">{text}</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  )
}
