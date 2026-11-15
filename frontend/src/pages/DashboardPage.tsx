import { useState, type ReactNode } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { CheckCircle2, ShieldCheck, ShieldX, Table2, BarChart3 } from 'lucide-react'
import bench from '../../../bench/results/v2.json'
import experiments from '../../../bench/results/v2-experiments.json'
import { Card, PageHeader } from '../components/ui'

// Results of the live run of 14 Nov 2026 (bench/results/), bundled at build time.
const SYS = bench.aggregate.systems
const SYSTEMS = [
  { key: 'single', label: 'Single AI call', color: 'var(--color-s1)' },
  { key: 'multi', label: 'Inquira (multi-agent)', color: 'var(--color-s3)' },
] as const
const CRITIC = experiments.critic_comparison
const SETUPS = [
  { key: 'single', label: 'One critic', color: 'var(--color-s1)' },
  { key: 'panel', label: 'Two critics', color: 'var(--color-s2)' },
  { key: 'debate', label: 'Two critics + debate', color: 'var(--color-s3)' },
] as const

const axis = { stroke: '#3a414b', tick: { fill: '#9aa3ad', fontSize: 12 }, tickLine: false }
const tooltipStyle = {
  contentStyle: { background: '#1a1e24', border: '1px solid #2f353e', borderRadius: 8, fontSize: 12, color: '#e8eaed' },
  labelStyle: { color: '#e8eaed', fontWeight: 600, marginBottom: 4 }, itemStyle: { color: '#b4bcc6' },
  cursor: { fill: '#ffffff08' },
}

function ChartCard({ title, subtitle, chart, table }: { title: string; subtitle?: string; chart: ReactNode; table: ReactNode }) {
  const [view, setView] = useState<'chart' | 'table'>('chart')
  return (
    <Card className="p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <div className="font-medium">{title}</div>
          {subtitle && <div className="mt-0.5 text-xs text-muted">{subtitle}</div>}
        </div>
        <div className="flex rounded-lg border border-line p-0.5">
          {(['chart', 'table'] as const).map(v => (
            <button key={v} onClick={() => setView(v)} title={v === 'chart' ? 'Chart' : 'Table'}
              className={`rounded-md px-2 py-1 ${view === v ? 'bg-panel-2 text-ink' : 'text-muted hover:text-ink'}`}>
              {v === 'chart' ? <BarChart3 size={14} /> : <Table2 size={14} />}
            </button>
          ))}
        </div>
      </div>
      {view === 'chart' ? <div className="h-72">{chart}</div> : <div className="overflow-x-auto">{table}</div>}
    </Card>
  )
}

function Table({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <table className="w-full text-sm">
      <thead><tr className="border-b border-line text-left text-xs text-muted">{head.map(h => <th key={h} className="py-2 pr-4 font-medium">{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i} className="border-b border-line/60 last:border-0">{r.map((c, j) => <td key={j} className={`py-2.5 pr-4 ${j === 0 ? 'text-ink' : 'text-ink-2 tabular-nums'}`}>{c}</td>)}</tr>)}</tbody>
    </table>
  )
}

function Stat({ value, label, note }: { value: string; label: string; note?: string }) {
  return (
    <Card className="p-5">
      <div className="text-3xl font-semibold tracking-tight tabular-nums">{value}</div>
      <div className="mt-1 text-sm text-ink-2">{label}</div>
      {note && <div className="mt-2 text-xs text-muted">{note}</div>}
    </Card>
  )
}

const pct = (x: number) => `${Math.round(x * 100)}%`
// Round half to even, like Python's format(): the README and write-up show 9.25 as 9.2
function one(x: number | null | undefined): string {
  if (x == null) return '-'
  const scaled = x * 10, floor = Math.floor(scaled), diff = scaled - floor
  const r = Math.abs(diff - 0.5) < 1e-9 ? (floor % 2 === 0 ? floor : floor + 1) : Math.round(scaled)
  return (r / 10).toFixed(1)
}

/** Legend in the series' fixed order, text in ink colours (the swatch carries the identity). */
function legendFor(items: readonly { label: string; color: string }[]) {
  return () => (
    <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 pt-2 text-xs text-ink-2">
      {items.map(i => <span key={i.label} className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: i.color }} />{i.label}</span>)}
    </div>
  )
}

export function DashboardPage() {
  const accuracy = [
    { name: 'Overall', ...Object.fromEntries(SYSTEMS.map(s => [s.key, SYS[s.key].accuracy])) },
    ...(['news', 'technical', 'comparison'] as const).map(cat => ({
      name: cat[0].toUpperCase() + cat.slice(1),
      ...Object.fromEntries(SYSTEMS.map(s => [s.key, SYS[s.key].accuracy_by_category[cat]])),
    })),
  ]
  const critic = [
    { name: 'Errors caught', ...Object.fromEntries(SETUPS.map(s => [s.key, Math.round(CRITIC[s.key].caught_rate * 100)])) },
    { name: 'False alarms', ...Object.fromEntries(SETUPS.map(s => [s.key, Math.round(CRITIC[s.key].false_alarm_rate * 100)])) },
  ]
  const attacks = ['instr_claim', 'role_system', 'exfil_prompt', 'cite_hijack', 'misinfo_stat']
  const attackNames: Record<string, string> = { instr_claim: 'Hidden instruction', role_system: 'Fake system message', exfil_prompt: 'Steal the prompt', cite_hijack: 'Citation hijack', misinfo_stat: 'Fake statistic' }
  const stageNames: Record<string, string> = { rule: 'Self-healing rule', extraction: 'Fact extraction', writer: 'Writer', critic: 'Critic', rewrite: 'Rewrite', guardrail: 'Output guardrail', not_retrieved: 'Not retrieved', got_through: 'Got through' }
  const run = (a: string, mode: string) => experiments.attack_runs.find(r => r.system === `attack:${a}:${mode}`)
  const memoryNames: Record<string, string> = {
    'memory:none:unverified': 'No defense', 'memory:verified_only:unverified': 'Verified-only (critic caught it)',
    'memory:verified_only:verified': 'Verified-only (critic missed it)', 'memory:session_scope:unverified': 'Per-user memory',
    'memory:both:verified': 'Both defenses',
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Results" subtitle="Measured on the live AWS deployment, 14 Nov 2026 - 8 benchmark topics, 111 labelled claims, 10 attacks." />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat value={`${one(SYS.multi.accuracy)} vs ${one(SYS.single.accuracy)}`} label="Accuracy: Inquira vs a single AI call (1-10)" note={`On news topics: ${one(SYS.multi.accuracy_by_category.news)} vs ${one(SYS.single.accuracy_by_category.news)}`} />
        <Stat value={pct(CRITIC.panel.caught_rate)} label="Planted errors caught by two critics" note={`One critic: ${pct(CRITIC.single.caught_rate)}, with ${pct(CRITIC.single.false_alarm_rate)} false alarms`} />
        <Stat value={`0 / ${experiments.attack_runs.length}`} label="Prompt-injection attacks that reached their goal" note="9 of 10 stopped at fact extraction" />
        <Stat value={`${Math.round((SYS.multi.input_tokens + SYS.multi.output_tokens) / 1000)}k`} label="Tokens per Inquira report" note={`Single call: ${((SYS.single.input_tokens + SYS.single.output_tokens) / 1000).toFixed(1)}k · ${Math.round(SYS.multi.latency_s)}s vs ${Math.round(SYS.single.latency_s)}s`} />
      </div>

      <section className="grid gap-4 xl:grid-cols-2">
        <ChartCard title="Accuracy by topic type" subtitle={`Average of two AI judges, 1-10, on ${bench.aggregate.topics_compared} topics. Same model for both systems.`}
          chart={
            <ResponsiveContainer>
              <BarChart data={accuracy} barGap={2} barCategoryGap="22%">
                <CartesianGrid vertical={false} stroke="#232830" />
                <XAxis dataKey="name" {...axis} />
                <YAxis domain={[0, 10]} ticks={[0, 2, 4, 6, 8, 10]} {...axis} axisLine={false} width={28} />
                <Tooltip {...tooltipStyle} formatter={(v) => one(Number(v))} />
                <Legend content={legendFor(SYSTEMS)} />
                {SYSTEMS.map(s => <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} maxBarSize={34} />)}
              </BarChart>
            </ResponsiveContainer>
          }
          table={<Table head={['', ...SYSTEMS.map(s => s.label)]} rows={accuracy.map(r => [r.name, ...SYSTEMS.map(s => one((r as Record<string, number | string>)[s.key] as number))])} />}
        />
        <ChartCard title="Fact-checker: errors caught vs false alarms" subtitle="55 broken and 56 true claims built from real sources. Higher is better for errors caught, lower for false alarms."
          chart={
            <ResponsiveContainer>
              <BarChart data={critic} barGap={2} barCategoryGap="30%">
                <CartesianGrid vertical={false} stroke="#232830" />
                <XAxis dataKey="name" {...axis} />
                <YAxis domain={[0, 100]} unit="%" {...axis} axisLine={false} width={40} />
                <Tooltip {...tooltipStyle} formatter={(v) => `${v}%`} />
                <Legend content={legendFor(SETUPS)} />
                {SETUPS.map(s => <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} maxBarSize={44} />)}
              </BarChart>
            </ResponsiveContainer>
          }
          table={<Table head={['Setup', 'Errors caught', 'False alarms', 'AI calls', 'Tokens']}
            rows={SETUPS.map(s => [s.label, `${CRITIC[s.key].caught} of ${CRITIC[s.key].errors}`, `${CRITIC[s.key].false_alarms} of ${CRITIC[s.key].true_claims}`,
              one(CRITIC[s.key].llm_calls_per_topic), `${(CRITIC[s.key].tokens_per_topic / 1000).toFixed(1)}k`])} />}
        />
      </section>

      <Card className="p-5">
        <div className="mb-1 font-medium">Cost per report</div>
        <div className="mb-4 text-xs text-muted">Averages over the regular topics.</div>
        <Table head={['System', 'Accuracy', 'Unsupported claims', 'Sources', 'AI calls', 'Tokens', 'Seconds', 'Trap honesty']}
          rows={SYSTEMS.map(s => [s.label, one(SYS[s.key].accuracy), one(SYS[s.key].unsupported_claims), one(SYS[s.key].sources),
            one(SYS[s.key].llm_calls), `${((SYS[s.key].input_tokens + SYS[s.key].output_tokens) / 1000).toFixed(1)}k`,
            Math.round(SYS[s.key].latency_s), one(SYS[s.key].trap_honesty)])} />
      </Card>

      <section className="grid gap-4 xl:grid-cols-2">
        <Card className="p-5">
          <div className="mb-1 font-medium">Where each attack was stopped</div>
          <div className="mb-4 text-xs text-muted">One poisoned web page placed first in the search results, traced through every stage.</div>
          <Table head={['Attack', 'Single critic', 'Debate panel']}
            rows={attacks.map(a => [attackNames[a], ...['single', 'debate'].map(mode => {
              const r = run(a, mode)
              if (!r) return '-'
              const listedOnly = r.final
              return <span className="inline-flex items-center gap-1.5">
                <ShieldCheck size={14} className="text-good" /> {stageNames[r.stopped_at] || r.stopped_at}{listedOnly ? ' *' : ''}
              </span>
            })])} />
          <div className="mt-3 text-xs text-muted">* Nothing leaked, but the poisoned page was listed among the report's sources.</div>
        </Card>
        <Card className="p-5">
          <div className="mb-1 font-medium">Memory poisoning: did another user get the poisoned report?</div>
          <div className="mb-4 text-xs text-muted">A poisoned report planted in shared memory, then 2 questions each on the same, a related and an unrelated topic.</div>
          <Table head={['Defense', 'Same topic', 'Related', 'Unrelated']}
            rows={experiments.memory_runs.map(m => [memoryNames[m.system] || m.system, ...(['same', 'related', 'unrelated'] as const).map(k => {
              const hit = (m.contamination as Record<string, number>)[k] > 0
              return <span className={`inline-flex items-center gap-1.5 ${hit ? 'text-bad' : ''}`}>
                {hit ? <ShieldX size={14} /> : <CheckCircle2 size={14} className="text-good" />} {hit ? 'Poisoned' : 'Clean'}
              </span>
            })])} />
        </Card>
      </section>

    </div>
  )
}
