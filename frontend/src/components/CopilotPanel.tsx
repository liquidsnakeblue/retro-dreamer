import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API = '/api/copilot'

type Ev = {
  seq: number
  ts: number
  kind: 'user' | 'assistant' | 'tool' | 'meta' | 'raw'
  text: string
  detail?: string // newer backend: full tool input (e.g. complete bash command)
}

/** Chat panel over the resident copilot — a headless claude-local session
 * (Qwen 3.6 27B on the 2x3090 box) driving the studio's HTTP tools.
 * Reasoning model: minutes-long thinking is normal; the meta events keep
 * the human oriented while it works. */
export function CopilotPanel() {
  const [events, setEvents] = useState<Ev[]>([])
  const [running, setRunning] = useState(false)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const lastSeq = useRef(0)
  const bottomRef = useRef<HTMLDivElement>(null)

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
    await fetch(`${API}/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
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
        {events.map((e) => <EventRow key={e.seq} e={e} />)}
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

function EventRow({ e }: { e: Ev }) {
  if (e.kind === 'assistant') {
    return (
      <div className="px-3 py-2 rounded text-xs bg-retro-surface/60 border border-retro-border/50 copilot-md">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{e.text}</ReactMarkdown>
      </div>
    )
  }
  if (e.kind === 'tool') return <ToolRow e={e} />
  const style: Record<string, string> = {
    user: 'text-retro-text bg-retro-surface border-l-2 border-retro-accent whitespace-pre-wrap',
    meta: 'text-retro-text-dim italic text-[10px]',
    raw: 'text-retro-text-dim font-mono text-[10px] whitespace-pre-wrap',
  }
  return <div className={`px-3 py-2 rounded text-xs ${style[e.kind]}`}>{e.text}</div>
}

/** One-line tool call, expandable when we have (or can recover) the full input.
 * Handles both event formats: the newer backend sends {text: "Bash — desc",
 * detail: "<full command>"}; the older one sent "Bash {json truncated @200}". */
function ToolRow({ e }: { e: Ev }) {
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

function parseToolEvent(e: Ev): { summary: string; detail?: string } {
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
