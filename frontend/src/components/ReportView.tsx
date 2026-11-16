import { useMemo, useRef, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { BookOpen, Check, Copy, Database, ExternalLink, FileDown, Globe, Info, ListTree, ShieldCheck, UserCheck, UserX } from 'lucide-react'
import type { ReviewStatus, Source, Trust } from '../lib/api'
import { domain, kindLabel, prepareReport } from '../lib/format'
import { Badge, Button, Card } from './ui'

type Props = {
  topic: string
  report: string
  sources: Source[]
  subQuestions?: string[]
  origin?: string
  liveSources?: boolean
  trust?: Trust | null
  reviewStatus?: ReviewStatus
  onPdf?: () => void
  extraActions?: ReactNode
}

export function OriginBadge({ origin, liveSources, count }: { origin?: string; liveSources?: boolean; count: number }) {
  if (origin === 'cache') return <Badge tone="accent"><Database size={12} /> From cache</Badge>
  if (origin === 'memory') return <Badge tone="accent"><Database size={12} /> From memory</Badge>
  if (origin === 'assistant') return <Badge>Instant reply</Badge>
  if (!liveSources) return <Badge tone="warn"><Info size={12} /> Model knowledge only</Badge>
  return <Badge tone="good"><Globe size={12} /> Live web · {count} sources</Badge>
}

/** The report's trust score: the share of its factual claims that the two critics confirmed against the sources. */
export function TrustBadge({ trust }: { trust?: Trust | null }) {
  if (!trust) return null
  if (trust.score == null) return <Badge title={`Not scored: ${trust.reason || ''}`}><ShieldCheck size={12} /> Trust: not scored</Badge>
  const tone = trust.label === 'High' ? 'good' : trust.label === 'Medium' ? 'warn' : 'bad'
  return <Badge tone={tone} title={`${trust.claims_verified} of ${trust.claims_checked} facts verified against their sources by two critics (interpretations are counted separately)`}>
    <ShieldCheck size={12} /> Trust {trust.score}% · {trust.label}</Badge>
}

/** Human-in-the-loop status of a report: held for a reviewer, approved or rejected by one. */
export function ReviewBadge({ status }: { status?: ReviewStatus | null }) {
  if (status === 'pending') return <Badge tone="warn" title="A person must approve this report before it is reused for other users"><UserCheck size={12} /> Needs human review</Badge>
  if (status === 'approved') return <Badge tone="good" title="A reviewer approved this report"><UserCheck size={12} /> Human-approved</Badge>
  if (status === 'rejected') return <Badge tone="bad" title="A reviewer rejected this report: it is never reused"><UserX size={12} /> Rejected by reviewer</Badge>
  return null
}

function trustLine(trust?: Trust | null): string | null {
  if (!trust) return null
  if (trust.score == null) return `Trust score not given: ${trust.reason}.`
  const flagged = trust.claims_flagged ? ` (${trust.claims_flagged} flagged)` : ''
  const reasoning = trust.interpretations_checked
    ? ` · ${trust.interpretations_flagged} of ${trust.interpretations_checked} interpretation sentences flagged` : ''
  return `${trust.claims_verified} of ${trust.claims_checked} facts verified against their sources by two AI critics${flagged}${reasoning}` +
    ` · ${(trust.drafts ?? 1) > 1 ? `rewritten ${trust.drafts === 2 ? 'once' : 'twice'} with the critics’ fixes` : 'passed on the first draft'}`
}

export function ReportView({ topic, report, sources, subQuestions = [], origin, liveSources, trust, reviewStatus, onPdf, extraActions }: Props) {
  const { body, notes } = useMemo(() => prepareReport(report), [report])
  const [copied, setCopied] = useState(false)
  const [active, setActive] = useState<number | null>(null)
  const panel = useRef<HTMLDivElement>(null)
  const cited = useMemo(() => new Set([...report.split(/\n#+\s*Sources\s*\n/)[0].matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)]
    .flatMap(m => m[1].split(/\s*,\s*/).map(Number))), [report])

  function focusSource(id: number) {
    setActive(id)
    const el = panel.current?.querySelector(`[data-src="${id}"]`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    el?.classList.remove('flash'); void (el as HTMLElement | undefined)?.offsetWidth; el?.classList.add('flash')
  }

  const components = {
    a: ({ href, children }: { href?: string; children?: ReactNode }) => {
      const m = href?.match(/^#src-(\d+)$/)
      if (m) {
        const id = Number(m[1])
        const src = sources.find(s => s.id === id)
        return <a className="cite" title={src ? `${src.title} - ${domain(src.url)}` : `Source ${id}`}
          onClick={e => { e.preventDefault(); focusSource(id) }} href={href}>{id}</a>
      }
      return <a href={href} target="_blank" rel="noreferrer">{children}</a>
    },
  }

  async function copy() {
    await navigator.clipboard.writeText(report)
    setCopied(true); setTimeout(() => setCopied(false), 1500)
  }

  const shown = sources.filter(s => cited.has(s.id))
  const unused = sources.filter(s => !cited.has(s.id))

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
      <Card className="min-w-0 p-6 md:p-8">
        <div className="mb-6 flex flex-wrap items-start justify-between gap-3 border-b border-line pb-5">
          <div className="min-w-0">
            <h2 className="text-xl font-semibold leading-snug">{topic}</h2>
            <div className="mt-2 flex flex-wrap gap-2"><OriginBadge origin={origin} liveSources={liveSources} count={sources.length} /><TrustBadge trust={trust} /><ReviewBadge status={reviewStatus} /></div>
            {trustLine(trust) && <div className="mt-2 text-xs text-muted">{trustLine(trust)}</div>}
          </div>
          <div className="flex flex-wrap gap-2">
            {extraActions}
            <Button onClick={copy} title="Copy the report text">{copied ? <Check size={15} /> : <Copy size={15} />} {copied ? 'Copied' : 'Copy'}</Button>
            {onPdf && <Button onClick={onPdf} title="Download as PDF"><FileDown size={15} /> PDF</Button>}
          </div>
        </div>
        <article className="report">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{body}</ReactMarkdown>
          {notes && (
            <div className="notes">
              <h2>Fact-check notes</h2>
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{notes}</ReactMarkdown>
            </div>
          )}
        </article>
      </Card>

      <aside className="space-y-4 lg:sticky lg:top-6 lg:self-start">
        {sources.length > 0 && (
          <Card>
            <div className="flex items-center gap-2 border-b border-line px-4 py-3 text-sm font-medium">
              <BookOpen size={15} className="text-muted" /> Sources <span className="text-muted">({shown.length} cited)</span>
            </div>
            <div ref={panel} className="scroll-thin max-h-[60vh] space-y-1 overflow-y-auto p-2">
              {[...shown, ...unused].map(s => (
                <a key={s.id} data-src={s.id} href={s.url} target="_blank" rel="noreferrer"
                  className={`group block rounded-lg px-3 py-2.5 transition-colors hover:bg-panel-2 ${active === s.id ? 'bg-panel-2' : ''} ${cited.has(s.id) ? '' : 'opacity-55'}`}>
                  <div className="flex items-start gap-2.5">
                    <span className="mt-0.5 flex h-5 min-w-5 items-center justify-center rounded bg-accent-soft px-1 text-[10px] font-semibold text-[#8fbaf0]">{s.id}</span>
                    <div className="min-w-0 flex-1">
                      <div className="line-clamp-2 text-[13px] leading-snug text-ink group-hover:text-white">{s.title || domain(s.url)}</div>
                      <div className="mt-1 flex flex-wrap items-center gap-x-2 text-[11px] text-muted">
                        <span>{domain(s.url)}</span>
                        {s.kind && <span>· {kindLabel[s.kind] || s.kind}</span>}
                        {s.published_date && <span>· {s.published_date.slice(0, 10)}</span>}
                        {!cited.has(s.id) && <span>· not cited</span>}
                      </div>
                    </div>
                    <ExternalLink size={13} className="mt-0.5 shrink-0 text-muted opacity-0 group-hover:opacity-100" />
                  </div>
                </a>
              ))}
            </div>
          </Card>
        )}
        {subQuestions.length > 0 && (
          <Card className="p-4">
            <div className="mb-2 flex items-center gap-2 text-sm font-medium"><ListTree size={15} className="text-muted" /> Research plan</div>
            <ol className="space-y-2 text-[13px] text-ink-2">
              {subQuestions.map((q, i) => <li key={i} className="flex gap-2"><span className="text-muted">{i + 1}.</span><span>{q}</span></li>)}
            </ol>
          </Card>
        )}
      </aside>
    </div>
  )
}
