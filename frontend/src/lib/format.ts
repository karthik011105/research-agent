import type { ProgressEvent } from './api'

export function timeAgo(iso: string): string {
  const then = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z').getTime()
  const s = Math.max(0, (Date.now() - then) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  if (s < 86400 * 7) return `${Math.floor(s / 86400)} d ago`
  return new Date(then).toLocaleDateString()
}

export function domain(url: string): string {
  try { return new URL(url).hostname.replace(/^www\./, '') } catch { return url }
}

export const kindLabel: Record<string, string> = { web: 'Web', news: 'News', paper: 'Paper', community: 'Community' }
const kinds = (v: unknown) => ((v as string[]) || []).map(k => kindLabel[k] || k).join(', ')
const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

// ── pipeline stages shown as a stepper ──────────────────────────────────────

export const STAGES = [
  { id: 'plan', label: 'Plan' },
  { id: 'search', label: 'Search' },
  { id: 'read', label: 'Read' },
  { id: 'write', label: 'Write' },
  { id: 'check', label: 'Fact-check' },
  { id: 'done', label: 'Done' },
] as const
export type StageId = (typeof STAGES)[number]['id']

const STAGE_OF: Record<string, StageId> = {
  started: 'plan', planning: 'plan', planned: 'plan', direct_answer: 'done', memory_hit: 'done',
  searching: 'search', searched: 'search', sources: 'search', gap: 'search',
  extracting: 'read', extracted: 'read',
  writing: 'write',
  checking: 'check', checked: 'check', debating: 'check', safety_check: 'check',
  done: 'done', error: 'done', blocked: 'done',
}
export const stageOf = (e: ProgressEvent): StageId | undefined => STAGE_OF[e.type]

/** One human sentence per progress event, or null for events that aren't shown. */
export function describe(e: ProgressEvent): string | null {
  switch (e.type) {
    case 'started': return 'Started'
    case 'memory_hit': return `Found a recent answer in ${e.where}`
    case 'planning': return 'Planning the research'
    case 'direct_answer': return 'Not a research question - answering directly'
    case 'planned': return `Split into ${plural((e.sub_questions as unknown[]).length, 'sub-question')}`
    case 'searching': return `Searching ${kinds(e.kinds)}: ${e.question}`
    case 'searched': return `Sub-question ${e.subq}: ${plural(Number(e.results), 'result')}` + (e.failed ? ` (${plural(Number(e.failed), 'search')} failed)` : '')
    case 'sources': return `${plural(Number(e.total), 'source')} selected` + (e.read_in_full ? `, ${e.read_in_full} read in full` : '') + (e.too_old ? `, ${plural(Number(e.too_old), 'older page')} skipped` : '')
    case 'extracting': return `Reading ${plural(Number(e.sources), 'source')} for sub-question ${e.subq}`
    case 'extracted': return `Sub-question ${e.subq}: ${plural(Number(e.facts), 'fact')} extracted`
    case 'gap': return `Something is missing - searching again`
    case 'writing': return e.revising ? `Rewriting with the fact-checker's fixes (draft ${e.draft})` : `Writing draft ${e.draft}`
    case 'checking': return e.mode === 'single' ? 'Fact-checking the draft against its sources' : 'Two critics are checking every claim'
    case 'debating': return `Critics disagree - debate round ${e.round}`
    case 'checked': return e.passed ? 'Fact-check passed' : `Fact-check: ${plural(Number(e.needs_work), 'claim')} to fix`
    case 'safety_check': return 'Safety check'
    case 'defense': return ({ blocked_domains: `Blocked ${plural(Number(e.count), 'low-quality source')}`,
                              injection_rule: `Removed a suspected prompt injection from ${plural(Number(e.count), 'page')}`,
                              output_guardrail: 'Safety filter flagged the draft - rewriting it' } as Record<string, string>)[e.what as string] || 'Defense triggered'
    case 'done': return 'Report ready'
    case 'blocked': return `Blocked: ${e.error || ''}`
    case 'error': return `Error: ${e.error || ''}`
    default: return null
  }
}

// ── report text → display markdown ──────────────────────────────────────────

export const NOTES_HEADING = '## Fact-check notes'

/** Splits the report into its body and the "Fact-check notes" section, and drops the Sources list (shown in the
 *  side panel). Bold-only lines become headings, and [3] / [1, 4] citations become links the renderer turns into
 *  clickable chips. */
export function prepareReport(report: string): { body: string; notes: string } {
  let text = report.replace(/\r\n/g, '\n')
  let notes = ''
  const n = text.indexOf(NOTES_HEADING)
  if (n >= 0) { notes = text.slice(n + NOTES_HEADING.length).trim(); text = text.slice(0, n) }
  text = text.split(/\n#+\s*Sources\s*\n/)[0]
  const fix = (s: string) => s
    .replace(/^\s*\*\*([^*\n]{2,80})\*\*\s*$/gm, '## $1')
    .replace(/^---\s*$/gm, '')
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g, (_, ids: string) => ids.split(/\s*,\s*/).map(id => `[${id}](#src-${id})`).join(''))
  return { body: fix(text).trim(), notes: notes ? fix(notes) : '' }
}
