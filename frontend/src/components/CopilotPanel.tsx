import { useEffect, useRef, useState } from 'react'

const API = '/api/copilot'

type Ev = { seq: number; ts: number; kind: 'user' | 'assistant' | 'tool' | 'meta' | 'raw'; text: string }

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

  const kindStyle: Record<Ev['kind'], string> = {
    user: 'text-retro-text bg-retro-surface border-l-2 border-retro-accent',
    assistant: 'text-retro-text bg-retro-card',
    tool: 'text-retro-text-dim font-mono text-[10px]',
    meta: 'text-retro-text-dim italic text-[10px]',
    raw: 'text-retro-text-dim font-mono text-[10px]',
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
        {events.map((e) => (
          <div key={e.seq} className={`px-3 py-2 rounded text-xs whitespace-pre-wrap ${kindStyle[e.kind]}`}>
            {e.kind === 'tool' ? `⚙ ${e.text}` : e.text}
          </div>
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
