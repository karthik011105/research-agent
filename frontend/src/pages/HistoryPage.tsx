import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, History, Loader2, Search, Trash2 } from 'lucide-react'
import { api, downloadFile, type HistoryItem, type HistoryReport } from '../lib/api'
import { timeAgo } from '../lib/format'
import { OriginBadge, ReportView, ReviewBadge, TrustBadge } from '../components/ReportView'
import { Badge, Button, Card, Empty, ErrorNote, PageHeader } from '../components/ui'

/** Asks first; deletes the report from the history and from the cache and memory, so it is never reused. */
async function deleteReport(jobId: string, topic: string): Promise<boolean> {
  if (!window.confirm(`Delete this search?\n\n"${topic}"\n\nIt is removed from the history and will not be reused from memory.`)) return false
  await api.del(`/reports/${jobId}`)
  return true
}

export function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  async function remove(r: HistoryItem) {
    try {
      if (await deleteReport(r.job_id, r.topic)) setItems(list => list && list.filter(x => x.job_id !== r.job_id))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    const t = setTimeout(() => {
      api.get<{ reports: HistoryItem[] }>(`/reports?limit=100&q=${encodeURIComponent(query)}`)
        .then(r => { setItems(r.reports); setError('') })
        .catch(e => setError(e.message))
    }, query ? 250 : 0)
    return () => clearTimeout(t)
  }, [query])

  return (
    <div>
      <PageHeader title="History" subtitle="Every report you have asked for, newest first."
        right={
          <div className="relative w-full max-w-xs">
            <Search size={15} className="absolute top-1/2 left-3 -translate-y-1/2 text-muted" />
            <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search topics"
              className="w-full rounded-lg border border-line bg-panel py-2 pr-3 pl-9 text-sm outline-none focus:border-accent" />
          </div>
        } />
      {error && <ErrorNote message={error} />}
      {!items && !error && <div className="flex justify-center py-16 text-muted"><Loader2 className="animate-spin" /></div>}
      {items && items.length === 0 && (
        <Empty icon={<History size={28} />} title={query ? 'No matching reports' : 'No reports yet'}>
          {query ? 'Try another word.' : <>Reports appear here once a research job finishes. <Link className="text-accent" to="/">Ask a question</Link>.</>}
        </Empty>
      )}
      <div className="grid gap-3">
        {items?.map(r => (
          <div key={r.job_id} role="button" tabIndex={0} onClick={() => navigate(`/history/${r.job_id}`)}
            onKeyDown={e => { if (e.key === 'Enter') navigate(`/history/${r.job_id}`) }} className="cursor-pointer text-left">
            <Card className="p-5 transition-colors hover:border-[#3a414b] hover:bg-panel-2">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-ink">{r.topic}</div>
                  <p className="mt-1.5 line-clamp-2 text-sm text-muted">{r.preview}</p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-2">
                  <span className="text-xs text-muted">{timeAgo(r.created_at)}</span>
                  <div className="flex gap-1.5">
                    <OriginBadge origin={r.origin} liveSources={r.live_sources} count={r.sources_count} />
                    {r.trust?.score != null && <TrustBadge trust={r.trust} />}
                    {r.notes_count > 0 && <Badge tone="warn">{r.notes_count} note{r.notes_count === 1 ? '' : 's'}</Badge>}
                    <ReviewBadge status={r.review_status} />
                    <button title="Delete this search" onClick={e => { e.stopPropagation(); remove(r) }}
                      className="rounded-md px-1.5 text-muted hover:bg-[#e667671a] hover:text-bad"><Trash2 size={14} /></button>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        ))}
      </div>
    </div>
  )
}

export function HistoryDetailPage() {
  const { jobId = '' } = useParams()
  const [report, setReport] = useState<HistoryReport | null>(null)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    api.get<HistoryReport>(`/reports/${jobId}`).then(setReport).catch(e => setError(e.message))
  }, [jobId])

  async function remove() {
    if (!report) return
    try {
      if (await deleteReport(jobId, report.topic)) navigate('/history')
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div>
      <Link to="/history" className="mb-5 inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink"><ArrowLeft size={15} /> History</Link>
      {error && <ErrorNote message={error} />}
      {!report && !error && <div className="flex justify-center py-16 text-muted"><Loader2 className="animate-spin" /></div>}
      {report && (
        <>
          <div className="mb-4 text-xs text-muted">{new Date(report.created_at + 'Z').toLocaleString()}</div>
          <ReportView topic={report.topic} report={report.report} sources={report.sources} subQuestions={report.sub_questions}
            origin={report.origin} liveSources={report.live_sources} trust={report.trust} reviewStatus={report.review_status}
            onPdf={() => downloadFile(`/reports/${jobId}/pdf`, `report-${jobId.slice(0, 8)}.pdf`)}
            extraActions={<Button variant="danger" onClick={remove} title="Delete this search"><Trash2 size={15} /> Delete</Button>} />
        </>
      )}
    </div>
  )
}
