import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, Eye, Loader2, RotateCcw, ThumbsDown, ThumbsUp } from 'lucide-react'
import { api, downloadFile, getSessionId, setSessionId, streamProgress, type HistoryReport, type JobResult, type ProgressEvent } from '../lib/api'
import { Timeline } from '../components/Timeline'
import { ReportView } from '../components/ReportView'
import { Button, ErrorNote } from '../components/ui'

const EXAMPLES = [
  'What are the latest AI news this week?',
  'Techniques for reducing hallucinations in large language models',
  'Global electric vehicle sales trends in 2026',
  'Who won the 2026 Nobel Prize in Mathematics and for what work?',
]
const LAST_JOB = 'researchLastJob'

export function ResearchPage() {
  const [topic, setTopic] = useState('')
  const [fresh, setFresh] = useState(true)
  const [jobId, setJobId] = useState<string | null>(sessionStorage.getItem(LAST_JOB))
  const [events, setEvents] = useState<ProgressEvent[]>([])
  const [result, setResult] = useState<JobResult | null>(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [rated, setRated] = useState<number | null>(null)
  const [watch, setWatch] = useState<'idle' | 'saving' | 'saved' | string>('idle')
  const abort = useRef<AbortController | null>(null)
  const submittedHere = useRef(false)
  const navigate = useNavigate()
  const running = !!jobId && !result && !error

  // follow the job: live events by SSE, the result by polling
  useEffect(() => {
    if (!jobId) return
    let stopped = false
    let seen = 0
    const restored = !submittedHere.current
    abort.current?.abort()
    abort.current = new AbortController()
    streamProgress(jobId, e => { seen++; setEvents(prev => [...prev, e]) }, abort.current.signal).catch(() => {})
    const started = Date.now()
    const poll = async () => {
      while (!stopped) {
        try {
          const r = await api.get<JobResult>(`/result/${jobId}`)
          if (r.status === 'pending' && restored && seen === 0 && Date.now() - started > 5000) {
            // a job from an earlier visit: live results expire after an hour, the saved copy is in the history
            try {
              const h = await api.get<HistoryReport>(`/reports/${jobId}`)
              setResult({ status: 'done', topic: h.topic, report: h.report, sources: h.sources, sub_questions: h.sub_questions,
                          origin: h.origin as JobResult['origin'], live_sources: h.live_sources, trust: h.trust ?? undefined })
            } catch {
              sessionStorage.removeItem(LAST_JOB); setJobId(null)
            }
            return
          }
          if (r.status !== 'pending') {
            if (r.status === 'done') setResult(r)
            else setError(r.status === 'blocked' ? `Blocked by the safety filter: ${r.error}` : `The research failed: ${r.error}`)
            return
          }
        } catch (e) {
          if (e instanceof Error && 'status' in e && (e as { status: number }).status === 401) { setError(e.message); return }
        }
        await new Promise(res => setTimeout(res, 2500))
      }
    }
    poll()
    return () => { stopped = true; abort.current?.abort() }
  }, [jobId])

  async function start(question = topic) {
    const q = question.trim()
    if (!q || submitting) return
    setTopic(q); setSubmitting(true); setError(''); setResult(null); setEvents([]); setRated(null); setWatch('idle')
    try {
      const r = await api.post<{ job_id: string; session_id: string }>('/research', {
        topic: q, fresh, session_id: getSessionId(), output_format: 'text',
      })
      setSessionId(r.session_id)
      submittedHere.current = true
      sessionStorage.setItem(LAST_JOB, r.job_id)
      setJobId(r.job_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start the research')
    } finally {
      setSubmitting(false)
    }
  }

  function reset() {
    abort.current?.abort()
    sessionStorage.removeItem(LAST_JOB)
    setJobId(null); setResult(null); setEvents([]); setError(''); setTopic('')
  }

  async function rate(rating: number) {
    if (!jobId) return
    setRated(rating)
    await api.post(`/feedback/${jobId}`, { rating }).catch(() => {})
  }

  async function watchTopic() {
    if (!result?.topic) return
    setWatch('saving')
    try {
      await api.post('/monitors', { topic: result.topic, interval_hours: 24, session_id: getSessionId() })
      setWatch('saved')
    } catch (e) {
      setWatch(e instanceof Error ? e.message : 'Could not save')
    }
  }

  const idle = !jobId

  return (
    <div className="space-y-6">
      {idle ? (
        <div className="mx-auto max-w-3xl pt-10 text-center">
          <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">What do you want to research?</h1>
          <p className="mx-auto mt-3 max-w-xl text-ink-2">
            Agents plan the question, search the web, news, papers and forums in parallel, write a cited report and
            fact-check it against the sources.
          </p>
        </div>
      ) : null}

      <form onSubmit={e => { e.preventDefault(); start() }}
        className={`${idle ? 'mx-auto max-w-3xl' : ''} rounded-2xl border border-line bg-panel p-2 shadow-lg shadow-black/20 focus-within:border-[#3a5f8f]`}>
        <textarea value={topic} onChange={e => setTopic(e.target.value)} rows={idle ? 3 : 1}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); start() } }}
          placeholder="Ask a research question..." disabled={running}
          className="w-full resize-none bg-transparent px-3 py-2.5 text-[15px] outline-none placeholder:text-muted disabled:opacity-60" />
        <div className="flex items-center justify-between gap-3 px-2 pb-1">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-2" title="Skip cached and remembered reports and always search the live web">
            <span className={`relative h-5 w-9 rounded-full transition-colors ${fresh ? 'bg-accent' : 'bg-[#2a2f37]'}`}>
              <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${fresh ? 'left-4.5' : 'left-0.5'}`} />
            </span>
            <input type="checkbox" className="hidden" checked={fresh} onChange={e => setFresh(e.target.checked)} />
            Fresh search
          </label>
          <div className="flex gap-2">
            {!idle && <Button variant="ghost" onClick={reset} disabled={running}><RotateCcw size={15} /> New</Button>}
            <Button type="submit" variant="primary" disabled={!topic.trim() || submitting || running}>
              {submitting || running ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
              {running ? 'Researching' : 'Research'}
            </Button>
          </div>
        </div>
      </form>

      {idle && (
        <div className="mx-auto flex max-w-3xl flex-wrap justify-center gap-2">
          {EXAMPLES.map(q => (
            <button key={q} onClick={() => start(q)}
              className="rounded-full border border-line bg-panel px-3.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:border-[#3a414b] hover:text-ink">
              {q}
            </button>
          ))}
        </div>
      )}

      {error && <ErrorNote message={error} />}
      {jobId && events.length > 0 && <Timeline events={events} finished={!running} />}
      {running && events.length === 0 && (
        <div className="flex items-center gap-3 rounded-xl border border-line bg-panel px-5 py-4 text-sm text-ink-2">
          <Loader2 size={16} className="animate-spin text-accent" /> Waiting for a worker...
        </div>
      )}

      {result?.report && (
        <ReportView
          topic={result.topic || topic}
          report={result.report}
          sources={result.sources || []}
          subQuestions={result.sub_questions}
          origin={result.origin}
          liveSources={result.live_sources}
          trust={result.trust}
          onPdf={() => downloadFile(`/result/${jobId}/pdf`, `report-${jobId?.slice(0, 8)}.pdf`)}
          extraActions={<>
            {result.origin === 'pipeline' && (
              <div className="flex rounded-lg border border-line">
                <button title="Good report" onClick={() => rate(1)} className={`px-2.5 py-2 hover:text-good ${rated === 1 ? 'text-good' : 'text-muted'}`}><ThumbsUp size={15} /></button>
                <button title="Bad report" onClick={() => rate(-1)} className={`border-l border-line px-2.5 py-2 hover:text-bad ${rated === -1 ? 'text-bad' : 'text-muted'}`}><ThumbsDown size={15} /></button>
              </div>
            )}
            {result.origin !== 'assistant' && !result.monitor_id && (
              watch === 'saved'
                ? <Button onClick={() => navigate('/watch')}><Eye size={15} /> Watching</Button>
                : <Button onClick={watchTopic} disabled={watch === 'saving'} title={watch !== 'idle' && watch !== 'saving' ? watch : 'Re-research this topic every day and show what changed'}>
                    <Eye size={15} /> {watch === 'saving' ? 'Saving...' : watch === 'idle' ? 'Watch topic' : 'Could not watch'}
                  </Button>
            )}
          </>}
        />
      )}
    </div>
  )
}
