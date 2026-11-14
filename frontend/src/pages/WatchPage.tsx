import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronDown, Eye, Loader2, Minus, Plus, Trash2 } from 'lucide-react'
import { api, getSessionId, type Monitor, type MonitorRun } from '../lib/api'
import { timeAgo } from '../lib/format'
import { Badge, Button, Card, Empty, ErrorNote, PageHeader } from '../components/ui'

const INTERVALS = [6, 12, 24, 48, 168]
const intervalLabel = (h: number) => (h % 24 === 0 ? (h === 24 ? 'Every day' : h === 168 ? 'Every week' : `Every ${h / 24} days`) : `Every ${h} hours`)

export function WatchPage() {
  const [monitors, setMonitors] = useState<Monitor[] | null>(null)
  const [max, setMax] = useState(5)
  const [topic, setTopic] = useState('')
  const [hours, setHours] = useState(24)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  const load = () => api.get<{ monitors: Monitor[]; max: number }>('/monitors')
    .then(r => { setMonitors(r.monitors); setMax(r.max) }).catch(e => setError(e.message))
  useEffect(() => { load() }, [])

  async function add() {
    if (!topic.trim()) return
    setSaving(true); setError('')
    try {
      await api.post('/monitors', { topic: topic.trim(), interval_hours: hours, session_id: getSessionId() })
      setTopic(''); await load()
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not add') } finally { setSaving(false) }
  }

  async function remove(id: string) {
    await api.del(`/monitors/${id}`).catch(e => setError(e.message))
    load()
  }

  const full = (monitors?.length ?? 0) >= max

  return (
    <div>
      <PageHeader title="Topic watch" subtitle="Topics re-researched on a schedule. Each run shows what changed since the previous one." />
      <Card className="mb-6 p-4">
        <form onSubmit={e => { e.preventDefault(); add() }} className="flex flex-col gap-3 md:flex-row">
          <input value={topic} onChange={e => setTopic(e.target.value)} placeholder="e.g. EU AI Act enforcement news" disabled={full}
            className="flex-1 rounded-lg border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent disabled:opacity-50" />
          <select value={hours} onChange={e => setHours(Number(e.target.value))} disabled={full}
            className="rounded-lg border border-line bg-bg px-3 py-2 text-sm outline-none focus:border-accent">
            {INTERVALS.map(h => <option key={h} value={h}>{intervalLabel(h)}</option>)}
          </select>
          <Button type="submit" variant="primary" disabled={!topic.trim() || saving || full}>
            {saving ? <Loader2 size={15} className="animate-spin" /> : <Plus size={15} />} Watch
          </Button>
        </form>
        <div className="mt-2 text-xs text-muted">
          {full ? `You are watching the maximum of ${max} topics.` : `Up to ${max} topics. The first run starts within 10 minutes; each run uses about 9 web searches.`}
        </div>
      </Card>
      {error && <div className="mb-4"><ErrorNote message={error} /></div>}
      {!monitors && !error && <div className="flex justify-center py-16 text-muted"><Loader2 className="animate-spin" /></div>}
      {monitors?.length === 0 && <Empty icon={<Eye size={28} />} title="No watched topics">Add a topic above, or press "Watch topic" on any report.</Empty>}
      <div className="grid gap-3">{monitors?.map(m => <MonitorCard key={m.id} m={m} onRemove={() => remove(m.id)} />)}</div>
    </div>
  )
}

function MonitorCard({ m, onRemove }: { m: Monitor; onRemove: () => void }) {
  const [open, setOpen] = useState(false)
  const [runs, setRuns] = useState<MonitorRun[] | null>(null)
  const latest = m.latest

  useEffect(() => {
    if (open && !runs) api.get<{ runs: MonitorRun[] }>(`/monitors/${m.id}/runs`).then(r => setRuns(r.runs)).catch(() => setRuns([]))
  }, [open, runs, m.id])

  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3 p-5">
        <button onClick={() => setOpen(!open)} className="text-muted hover:text-ink"><ChevronDown size={18} className={`transition-transform ${open ? '' : '-rotate-90'}`} /></button>
        <div className="min-w-0 flex-1">
          <div className="font-medium">{m.topic}</div>
          <div className="mt-1 flex flex-wrap gap-x-3 text-xs text-muted">
            <span>{intervalLabel(m.interval_hours)}</span>
            <span>{m.last_run_at ? `Last run ${timeAgo(m.last_run_at)}` : 'First run within 10 minutes'}</span>
          </div>
        </div>
        {latest && (latest.changes.first_run
          ? <Badge>First report</Badge>
          : <div className="flex gap-1.5"><Badge tone="good">+{latest.changes.new.length} new</Badge><Badge tone="bad">-{latest.changes.dropped.length} gone</Badge></div>)}
        <Button variant="danger" onClick={onRemove} title="Stop watching"><Trash2 size={15} /></Button>
      </div>
      {open && (
        <div className="border-t border-line px-5 py-4">
          {!runs && <Loader2 size={16} className="animate-spin text-muted" />}
          {runs?.length === 0 && <div className="text-sm text-muted">No runs yet.</div>}
          <div className="space-y-5">
            {runs?.map(r => (
              <div key={r.job_id}>
                <div className="mb-2 flex items-center gap-3 text-xs text-muted">
                  <span>{new Date(r.at + 'Z').toLocaleString()}</span>
                  <Link to={`/history/${r.job_id}`} className="text-accent hover:underline">Open report</Link>
                </div>
                {r.changes.first_run ? <div className="text-sm text-ink-2">First report - {r.changes.unchanged} claims recorded as the baseline.</div> : (
                  <ul className="space-y-1.5 text-sm">
                    {r.changes.new.map((c, i) => <li key={'n' + i} className="flex gap-2 text-ink-2"><Plus size={14} className="mt-1 shrink-0 text-good" />{c}</li>)}
                    {r.changes.dropped.map((c, i) => <li key={'d' + i} className="flex gap-2 text-muted line-through decoration-[#e6676780]"><Minus size={14} className="mt-1 shrink-0 text-bad" />{c}</li>)}
                    {r.changes.new.length + r.changes.dropped.length === 0 && <li className="text-muted">No changes - {r.changes.unchanged} claims unchanged.</li>}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}
