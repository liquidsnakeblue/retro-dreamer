import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { TrainingStatus } from '../hooks/useTrainingSocket'
import {
  TrainingProposalCard,
  isTrainingStartProposal,
  type ProposalAction,
  type TrainingStartProposal,
} from './TrainingProposalCard'

const API = '/api/copilot'

type TextEv = {
  seq: number
  ts: number
  kind: 'user' | 'assistant' | 'tool' | 'meta' | 'raw' | 'grounding-warning'
  text: string
  detail?: string // full auxiliary detail (for example tool input or warning evidence)
}

type ProposalEv = {
  seq: number
  ts: number
  kind: 'proposal'
  proposal: unknown
}

type Ev = TextEv | ProposalEv

interface CopilotPanelProps {
  selectedGame: string
  status: TrainingStatus | null
  activeTab: string
  onTrainingRefresh: () => Promise<void>
  onOpenMetrics: () => void
}

type ConfirmResponse = {
  status: string
  plan_id: string
  execution: unknown
  studio_state?: {
    revision?: string
    generated_at?: string
  }
  intent?: {
    type: string
    tab: string
  }
}

const STATE_START = '<STUDIO_STATE>\n'
const STATE_END = '\n</STUDIO_STATE>\n'

export function visibleUserText(text: string): string {
  if (!text.startsWith(STATE_START)) return text
  const end = text.indexOf(STATE_END, STATE_START.length)
  return end === -1 ? '' : text.slice(end + STATE_END.length)
}

/** Chat panel over the resident copilot — a headless claude-local session
 * (Qwen 3.6 27B on the 2x3090 box) driving the studio's HTTP tools.
 * Reasoning model: minutes-long thinking is normal; the meta events keep
 * the human oriented while it works. */
export function CopilotPanel({
  selectedGame,
  status,
  activeTab,
  onTrainingRefresh,
  onOpenMetrics,
}: CopilotPanelProps) {
  const [events, setEvents] = useState<Ev[]>([])
  const [running, setRunning] = useState(false)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [contextReceipt, setContextReceipt] = useState<{ revision: string; observedAt: string } | null>(null)
  const [approvalReady, setApprovalReady] = useState(false)
  const [approvalError, setApprovalError] = useState<string | null>(null)
  const [proposalActions, setProposalActions] = useState<Record<string, ProposalAction>>({})
  const lastSeq = useRef(0)
  const bottomRef = useRef<HTMLDivElement>(null)

  const initializeApprovalSession = useCallback(async (signal?: AbortSignal) => {
    setApprovalError(null)
    try {
      const res = await fetch('/api/training/approval-session', {
        method: 'POST',
        credentials: 'same-origin',
        signal,
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      // The server returns status only. The approval capability remains in an
      // HttpOnly cookie, outside JavaScript and outside the copilot process.
      setApprovalReady(true)
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        setApprovalReady(false)
        setApprovalError(error instanceof Error ? error.message : 'session initialization failed')
      }
      throw error
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void initializeApprovalSession(controller.signal).catch(() => {})
    return () => controller.abort()
  }, [initializeApprovalSession])

  useEffect(() => {
    const t = setInterval(async () => {
      try {
        const d = await fetch(`${API}/events?since=${lastSeq.current}`).then((r) => r.json())
        setRunning(d.running)
        if (d.last_seq < lastSeq.current) {
          // Copilot (or server) restarted: seq space rewound. Reset so the
          // next poll refetches the new session from 0 — otherwise the panel
          // silently ignores everything until seq outgrows the old session.
          lastSeq.current = 0
          setEvents([])
          setProposalActions({})
          return
        }
        if (d.events?.length) {
          lastSeq.current = d.last_seq
          setEvents((prev) => [...prev, ...d.events].slice(-400))
        }
      } catch {
        /* server restart etc. */
      }
    }, 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  useEffect(() => {
    if (!selectedGame) return
    fetch(`/api/studio/state?focus_game_id=${encodeURIComponent(selectedGame)}`)
      .then((res) => res.ok ? res.json() : Promise.reject())
      .then((state) => setContextReceipt({ revision: state.revision, observedAt: state.generated_at }))
      .catch(() => {})
  }, [selectedGame, status?.state, status?.game_id])

  async function start() {
    setBusy(true)
    try {
      await fetch(`${API}/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
    } finally {
      setBusy(false)
    }
  }

  async function stop() {
    await fetch(`${API}/stop`, { method: 'POST' })
  }

  async function send() {
    const text = input.trim()
    if (!text) return
    setInput('')
    const res = await fetch(`${API}/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, focus_game_id: selectedGame, active_tab: activeTab }),
    })
    if (res.ok) {
      const receipt = await res.json()
      setContextReceipt({ revision: receipt.studio_revision, observedAt: receipt.observed_at })
    }
  }

  function proposalAction(id: string): ProposalAction {
    return proposalActions[id] || { status: 'pending' }
  }

  function updateProposalAction(id: string, action: ProposalAction) {
    setProposalActions((current) => ({ ...current, [id]: action }))
  }

  async function brokerAction(proposal: TrainingStartProposal, action: 'confirm' | 'cancel') {
    updateProposalAction(proposal.id, { status: action === 'confirm' ? 'confirming' : 'cancelling' })
    try {
      const request = () => fetch(
        `/api/training/plans/${encodeURIComponent(proposal.id)}/${action}`,
        { method: 'POST', credentials: 'same-origin' },
      )
      let res = await request()
      if (res.status === 403) {
        await initializeApprovalSession()
        res = await request()
      }
      const response = await readJson(res)
      if (!res.ok) throw new Error(apiError(response, res.status))

      if (action === 'cancel') {
        updateProposalAction(proposal.id, { status: 'cancelled' })
        return
      }

      const confirmed = response as ConfirmResponse
      if (confirmed.status !== 'confirmed' || confirmed.plan_id !== proposal.id) {
        throw new Error('Broker returned an invalid confirmation receipt')
      }
      updateProposalAction(proposal.id, { status: 'confirmed' })
      if (confirmed.studio_state?.revision && confirmed.studio_state.generated_at) {
        setContextReceipt({
          revision: confirmed.studio_state.revision,
          observedAt: confirmed.studio_state.generated_at,
        })
      }
      if (confirmed.intent?.type === 'open_tab' && confirmed.intent.tab === 'metrics') {
        onOpenMetrics()
      }
      try {
        await onTrainingRefresh()
      } catch {
        // The broker receipt is authoritative. Polling will reconcile ambient
        // training state without downgrading a successfully confirmed card.
      }
    } catch (error) {
      updateProposalAction(proposal.id, {
        status: 'error',
        error: error instanceof Error ? error.message : 'Broker request failed',
      })
    }
  }

  return (
    <div className="absolute inset-0 flex flex-col bg-retro-card">
      <div className="px-4 py-3 border-b border-retro-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-retro-text">Copilot</h2>
          <span className={`flex items-center gap-1.5 text-[10px] font-semibold ${running ? 'text-retro-success' : 'text-retro-text-dim'}`}>
            <span className={`w-2 h-2 rounded-full ${running ? 'bg-retro-success animate-pulse' : 'bg-retro-border'}`} />
            {running ? 'Qwen 3.6 27B — local' : 'offline'}
          </span>
          {contextReceipt && (
            <span className="text-[10px] text-retro-text-dim font-mono" title={contextReceipt.observedAt}>
              context r{contextReceipt.revision.slice(0, 8)} · observed {new Date(contextReceipt.observedAt).toLocaleTimeString()}
            </span>
          )}
        </div>
        <div className="flex gap-2">
          {!running ? (
            <button onClick={start} disabled={busy}
              className="px-3 py-1.5 text-xs font-semibold rounded bg-retro-success text-white disabled:opacity-50">
              {busy ? 'Starting…' : 'Start Copilot'}
            </button>
          ) : (
            <button onClick={stop}
              className="px-3 py-1.5 text-xs rounded bg-retro-surface border border-retro-border text-retro-text">
              ◼ Stop
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2">
        {events.length === 0 && (
          <p className="text-xs text-retro-text-dim text-center mt-8">
            Start the copilot, then ask it to onboard a game, audit a reward, or diagnose the run.
            <br />It drives the same studio tools you see in the UI. Thinking can take minutes — it's a reasoning model.
          </p>
        )}
        {events.map((e) => (
          <EventRow
            key={e.seq}
            e={e}
            approvalReady={approvalReady}
            approvalError={approvalError}
            proposalAction={proposalAction}
            onConfirm={(proposal) => brokerAction(proposal, 'confirm')}
            onCancel={(proposal) => brokerAction(proposal, 'cancel')}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="p-3 border-t border-retro-border flex gap-2 shrink-0">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          disabled={!running}
          rows={2}
          placeholder={running ? 'Message the copilot… (Enter to send)' : 'Start the copilot first'}
          className="flex-1 bg-retro-surface border border-retro-border rounded px-3 py-2 text-xs text-retro-text resize-none disabled:opacity-40"
        />
        <button onClick={send} disabled={!running || !input.trim()}
          className="px-4 rounded bg-retro-accent text-black text-xs font-semibold disabled:opacity-40">
          Send
        </button>
      </div>
    </div>
  )
}

interface EventRowProps {
  e: Ev
  approvalReady: boolean
  approvalError: string | null
  proposalAction: (id: string) => ProposalAction
  onConfirm: (proposal: TrainingStartProposal) => void
  onCancel: (proposal: TrainingStartProposal) => void
}

function EventRow({
  e,
  approvalReady,
  approvalError,
  proposalAction,
  onConfirm,
  onCancel,
}: EventRowProps) {
  if (e.kind === 'proposal') {
    if (!isTrainingStartProposal(e.proposal)) {
      return (
        <div className="px-3 py-2 rounded border border-retro-danger/50 bg-red-950/20 text-[10px] text-retro-danger">
          Rejected a malformed training proposal event. No action is available.
        </div>
      )
    }
    const proposal = e.proposal
    return (
      <TrainingProposalCard
        proposal={proposal}
        action={proposalAction(proposal.id)}
        approvalReady={approvalReady}
        approvalError={approvalError}
        onConfirm={() => onConfirm(proposal)}
        onCancel={() => onCancel(proposal)}
      />
    )
  }
  if (e.kind === 'assistant') {
    return (
      <div className="px-3 py-2 rounded text-xs bg-retro-surface/60 border border-retro-border/50 copilot-md">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{e.text}</ReactMarkdown>
      </div>
    )
  }
  if (e.kind === 'grounding-warning') {
    const warning = e.text.replace(/^Grounding(?: vocabulary)? telemetry:\s*/i, '')
    return (
      <div
        className="px-3 py-1 text-[10px] text-retro-text-dim"
        title={e.detail || e.text}
      >
        <span className="font-semibold text-retro-warning">[⚠ grounding telemetry]</span>{' '}
        {warning}
      </div>
    )
  }
  if (e.kind === 'tool') return <ToolRow e={e} />
  const text = e.kind === 'user' ? visibleUserText(e.text) : e.text
  if (e.kind === 'user' && !text) return null
  const style: Record<string, string> = {
    user: 'text-retro-text bg-retro-surface border-l-2 border-retro-accent whitespace-pre-wrap',
    meta: 'text-retro-text-dim italic text-[10px]',
    raw: 'text-retro-text-dim font-mono text-[10px] whitespace-pre-wrap',
  }
  return <div className={`px-3 py-2 rounded text-xs ${style[e.kind]}`}>{text}</div>
}

/** One-line tool call, expandable when we have (or can recover) the full input.
 * Handles both event formats: the newer backend sends {text: "Bash — desc",
 * detail: "<full command>"}; the older one sent "Bash {json truncated @200}". */
function ToolRow({ e }: { e: TextEv }) {
  const { summary, detail } = parseToolEvent(e)
  if (!detail) {
    return <div className="px-3 py-1.5 rounded text-[10px] font-mono text-retro-text-dim">⚙ {summary}</div>
  }
  return (
    <details className="px-3 py-1.5 rounded text-[10px] text-retro-text-dim group">
      <summary className="cursor-pointer font-mono list-none select-none hover:text-retro-text">
        <span className="inline-block w-3 transition-transform group-open:rotate-90">▸</span>
        ⚙ {summary}
      </summary>
      <pre className="mt-1.5 ml-4 p-2 rounded bg-retro-surface font-mono text-[10px] whitespace-pre-wrap break-all overflow-x-auto max-h-48 overflow-y-auto">
        {detail}
      </pre>
    </details>
  )
}

function parseToolEvent(e: TextEv): { summary: string; detail?: string } {
  if (e.detail !== undefined) return { summary: e.text, detail: e.detail || undefined }

  // Legacy format: "Name {json possibly cut off at 200 chars}"
  const sp = e.text.indexOf(' ')
  if (sp === -1) return { summary: e.text }
  const name = e.text.slice(0, sp)
  const rest = e.text.slice(sp + 1)
  try {
    const inp = JSON.parse(rest)
    const label =
      inp.description || inp.file_path || inp.pattern ||
      (typeof inp.command === 'string' ? inp.command.split('\n')[0].slice(0, 90) : '') ||
      JSON.stringify(inp).slice(0, 90)
    const detail = typeof inp.command === 'string' && inp.command.length > label.length
      ? inp.command
      : JSON.stringify(inp, null, 2)
    return { summary: `${name} — ${label}`, detail: detail !== label ? detail : undefined }
  } catch {
    // Truncated JSON — fish out the most descriptive field we can
    const m = rest.match(/"(?:description|command|file_path|pattern)"\s*:\s*"((?:[^"\\]|\\.)*)/)
    const label = (m ? m[1] : rest).split('\\n')[0].slice(0, 90)
    return { summary: `${name} — ${label}`, detail: rest.length > label.length + 30 ? rest : undefined }
  }
}

async function readJson(res: Response): Promise<unknown> {
  try {
    return await res.json()
  } catch {
    return null
  }
}

function apiError(value: unknown, status: number): string {
  if (typeof value === 'object' && value !== null && 'detail' in value) {
    const detail = (value as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
  }
  return `Broker request failed (HTTP ${status})`
}
