import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Check, Loader2, UserCheck, X } from 'lucide-react'
import { api, downloadFile, type HistoryItem, type HistoryReport } from '../lib/api'
import { timeAgo } from '../lib/format'
import { ReportView, ReviewBadge, TrustBadge } from '../components/ReportView'
import { Badge, Button, Card, Empty, ErrorNote, PageHeader } from '../components/ui'

/** Human-in-the-loop queue: new reports the system doubts (trust below High, fact-check notes, or not scored) wait here.
 *  Nothing on this list is reused for other users until a person approves it. */
export function ReviewPage() {
  const [tab, setTab] = useState<'pending' | 'decided'>('pending')
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    setItems(null)
    api.get<{ reports: HistoryItem[] }>(`/reviews?status=${tab}`)
      .then(r => { setItems(r.reports); setError('') })
      .catch(e => setError(e.message))
  }, [tab])

  return (
    <div>
      <PageHeader title="Human review"
        subtitle="Reports the system is not sure about wait here. Only reports a person approves are reused for other users."
        right={
          <div className="flex rounded-lg border border-line p-0.5 text-sm">
            {(['pending', 'decided'] as const).map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`rounded-md px-3 py-1.5 ${tab === t ? 'bg-panel-2 text-ink' : 'text-muted hover:text-ink'}`}>
                {t === 'pending' ? 'Waiting' : 'Decided'}
              </button>
            ))}
          </div>
        } />
      {error && <ErrorNote message={error} />}
      {!items && !error && <div className="flex justify-center py-16 text-muted"><Loader2 className="animate-spin" /></div>}
      {items && items.length === 0 && (
        <Empty icon={<UserCheck size={28} />} title={tab === 'pending' ? 'Nothing to review' : 'No decisions yet'}>
          {tab === 'pending'
            ? 'New reports with a trust score below 90%, fact-check notes or no score appear here.'
            : 'Approved and rejected reports appear here.'}
        </Empty>
      )}
      <div className="grid gap-3">
        {items?.map(r => (
          <button key={r.job_id} onClick={() => navigate(`/review/${r.job_id}`)} className="text-left">
            <Card className="p-5 transition-colors hover:border-[#3a414b] hover:bg-panel-2">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="font-medium text-ink">{r.topic}</div>
                  <p className="mt-1.5 line-clamp-2 text-sm text-muted">{r.review_note || r.preview}</p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-2">
                  <span className="text-xs text-muted">{timeAgo(r.reviewed_at || r.created_at)}</span>
                  <div className="flex gap-1.5">
                    <TrustBadge trust={r.trust} />
                    {r.notes_count > 0 && <Badge tone="warn">{r.notes_count} note{r.notes_count === 1 ? '' : 's'}</Badge>}
                    <ReviewBadge status={r.review_status} />
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

export function ReviewDetailPage() {
  const { jobId = '' } = useParams()
  const [report, setReport] = useState<HistoryReport | null>(null)
  const [note, setNote] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.get<HistoryReport>(`/reports/${jobId}`).then(setReport).catch(e => setError(e.message))
  }, [jobId])

  async function decide(decision: 'approve' | 'reject') {
    setSaving(true); setError('')
    try {
      const r = await api.post<{ review_status: HistoryReport['review_status']; reviewed_at: string }>(
        `/reviews/${jobId}`, { decision, note })
      setReport(rep => rep && { ...rep, review_status: r.review_status, review_note: note, reviewed_at: r.reviewed_at })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <Link to="/review" className="mb-5 inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink"><ArrowLeft size={15} /> Human review</Link>
      {error && <div className="mb-4"><ErrorNote message={error} /></div>}
      {!report && !error && <div className="flex justify-center py-16 text-muted"><Loader2 className="animate-spin" /></div>}
      {report && (
        <>
          <Card className="mb-5 p-5">
            {report.review_status === 'pending' ? (
              <>
                <div className="font-medium">Your decision</div>
                <p className="mt-1 text-sm text-muted">
                  Check the flagged claims (Fact-check notes) against the sources on the right. <b>Approve</b> lets this report be
                  reused for other users from cache and memory; <b>Reject</b> keeps it out of shared memory for good.
                </p>
                <textarea value={note} onChange={e => setNote(e.target.value)} maxLength={500} rows={2}
                  placeholder="Optional note, e.g. which claim was wrong"
                  className="mt-3 w-full rounded-lg border border-line bg-panel-2 px-3 py-2 text-sm outline-none focus:border-accent" />
                <div className="mt-3 flex gap-2">
                  <Button variant="primary" onClick={() => decide('approve')} disabled={saving}><Check size={15} /> Approve</Button>
                  <Button variant="danger" onClick={() => decide('reject')} disabled={saving}><X size={15} /> Reject</Button>
                </div>
              </>
            ) : (
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <ReviewBadge status={report.review_status} />
                {report.reviewed_at && <span className="text-muted">{new Date(report.reviewed_at + 'Z').toLocaleString()}</span>}
                {report.review_note && <span className="text-ink-2">“{report.review_note}”</span>}
                {report.review_status === 'not_needed' && <span className="text-muted">This report did not need a human review.</span>}
              </div>
            )}
          </Card>
          <ReportView topic={report.topic} report={report.report} sources={report.sources} subQuestions={report.sub_questions}
            origin={report.origin} liveSources={report.live_sources} trust={report.trust} reviewStatus={report.review_status}
            onPdf={() => downloadFile(`/reports/${jobId}/pdf`, `report-${jobId.slice(0, 8)}.pdf`)} />
        </>
      )}
    </div>
  )
}
