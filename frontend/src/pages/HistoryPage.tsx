import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, History, Loader2, Search } from 'lucide-react'
import { api, downloadFile, type HistoryItem, type HistoryReport } from '../lib/api'
import { timeAgo } from '../lib/format'
import { OriginBadge, ReportView } from '../components/ReportView'
import { Badge, Card, Empty, ErrorNote, PageHeader } from '../components/ui'

export function HistoryPage() {
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [query, setQuery] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

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
          <button key={r.job_id} onClick={() => navigate(`/history/${r.job_id}`)} className="text-left">
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
                    {r.notes_count > 0 && <Badge tone="warn">{r.notes_count} note{r.notes_count === 1 ? '' : 's'}</Badge>}
                  </div>
                </div>
              </div>
            </Card>
          </button>
        ))}
      </div>
    </div>
  )
}

export function HistoryDetailPage() {
  const { jobId = '' } = useParams()
  const [report, setReport] = useState<HistoryReport | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.get<HistoryReport>(`/reports/${jobId}`).then(setReport).catch(e => setError(e.message))
  }, [jobId])

  return (
    <div>
      <Link to="/history" className="mb-5 inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink"><ArrowLeft size={15} /> History</Link>
      {error && <ErrorNote message={error} />}
      {!report && !error && <div className="flex justify-center py-16 text-muted"><Loader2 className="animate-spin" /></div>}
      {report && (
        <>
          <div className="mb-4 text-xs text-muted">{new Date(report.created_at + 'Z').toLocaleString()}</div>
          <ReportView topic={report.topic} report={report.report} sources={report.sources} subQuestions={report.sub_questions}
            origin={report.origin} liveSources={report.live_sources}
            onPdf={() => downloadFile(`/reports/${jobId}/pdf`, `report-${jobId.slice(0, 8)}.pdf`)} />
        </>
      )}
    </div>
  )
}
