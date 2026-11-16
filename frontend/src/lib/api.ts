// Thin client for the FastAPI backend. The app password (API_KEY) is kept in the browser's localStorage.

const KEY_STORAGE = 'researchApiKey'
const SESSION_STORAGE = 'researchSessionId'

export const getApiKey = () => localStorage.getItem(KEY_STORAGE) || ''
export const setApiKey = (key: string) => localStorage.setItem(KEY_STORAGE, key.trim())
export const getSessionId = () => localStorage.getItem(SESSION_STORAGE) || ''
export const setSessionId = (id: string) => localStorage.setItem(SESSION_STORAGE, id)

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) { super(message); this.status = status }
}

function headers(json = false): HeadersInit {
  const h: Record<string, string> = { 'X-API-Key': getApiKey() }
  if (json) h['Content-Type'] = 'application/json'
  return h
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const resp = await fetch(path, { method, headers: headers(body !== undefined), body: body === undefined ? undefined : JSON.stringify(body) })
  if (!resp.ok) {
    let detail = resp.statusText
    try { detail = (await resp.json()).detail || detail } catch { /* not JSON */ }
    if (resp.status === 401) detail = 'Wrong or missing API key. Use the app password (API_KEY), not a Groq or Tavily key.'
    throw new ApiError(resp.status, typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return resp.json() as Promise<T>
}

export const api = {
  get: <T,>(path: string) => request<T>('GET', path),
  post: <T,>(path: string, body: unknown) => request<T>('POST', path, body),
  del: <T,>(path: string) => request<T>('DELETE', path),
}

export async function downloadFile(path: string, filename: string) {
  const resp = await fetch(path, { headers: headers() })
  if (!resp.ok) throw new ApiError(resp.status, 'Download failed')
  const url = URL.createObjectURL(await resp.blob())
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

// ── shapes returned by the API ──────────────────────────────────────────────

export type Trust = {
  score: number | null; label: string; reason?: string
  claims_checked?: number; claims_verified?: number; claims_flagged?: number; critics_agreed?: number
  interpretations_checked?: number; interpretations_flagged?: number
  sources?: number; drafts?: number
}

export type ReviewStatus = 'not_needed' | 'pending' | 'approved' | 'rejected'

export type Source = { id: number; title: string; url: string; published_date?: string; provider?: string; kind?: string }

export type JobResult = {
  status: 'pending' | 'done' | 'error' | 'blocked'
  error?: string
  topic?: string
  report?: string
  origin?: 'pipeline' | 'cache' | 'memory' | 'assistant'
  live_sources?: boolean
  sources_count?: number
  sub_questions?: string[]
  sources?: Source[]
  review_notes?: string[]
  trust?: Trust
  review?: { status: ReviewStatus }
  monitor_id?: string
}

export type HistoryItem = {
  job_id: string; topic: string; origin: string; live_sources: boolean; sources_count: number
  notes_count: number; created_at: string; preview: string; trust?: Trust | null
  review_status?: ReviewStatus; review_note?: string | null; reviewed_at?: string | null
}
export type HistoryReport = HistoryItem & { report: string; sources: Source[]; sub_questions: string[] }

export type Changes = { first_run: boolean; new: string[]; dropped: string[]; unchanged: number }
export type MonitorRun = { job_id: string; changes: Changes; at: string }
export type Monitor = {
  id: string; topic: string; interval_hours: number; active: boolean; created_at: string
  last_run_at: string | null; last_job_id: string | null; latest: MonitorRun | null
}

// ── live progress (Server-Sent Events read with fetch, since EventSource can't send the API key) ──

export type ProgressEvent = { type: string; t?: number; [k: string]: unknown }

export async function streamProgress(jobId: string, onEvent: (e: ProgressEvent) => void, signal: AbortSignal) {
  const resp = await fetch(`/stream/${jobId}`, { headers: headers(), signal })
  if (!resp.ok || !resp.body) return
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let cut
    while ((cut = buffer.indexOf('\n\n')) >= 0) {
      const chunk = buffer.slice(0, cut)
      buffer = buffer.slice(cut + 2)
      const data = chunk.split('\n').filter(l => l.startsWith('data: ')).map(l => l.slice(6)).join('\n')
      if (data) onEvent(JSON.parse(data))
    }
  }
}
