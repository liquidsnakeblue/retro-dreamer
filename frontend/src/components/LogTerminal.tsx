import { useState, useEffect, useRef, useCallback } from 'react'

const API = '/api'
const POLL_INTERVAL = 2000

export function LogTerminal() {
  const [lines, setLines] = useState<string[]>([])
  const [collapsed, setCollapsed] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API}/training/logs?n=80`)
      if (res.ok) {
        const data = await res.json()
        setLines(data.lines)
      }
    } catch {}
  }, [])

  useEffect(() => {
    poll()
    const timer = setInterval(poll, POLL_INTERVAL)
    return () => clearInterval(timer)
  }, [poll])

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines, autoScroll])

  function colorize(line: string) {
    if (line.includes('ERROR') || line.includes('Error')) return 'text-retro-danger'
    if (line.includes('WARNING') || line.includes('Warning') || line.includes('UserWarning')) return 'text-yellow-400'
    if (line.includes('policy_step=')) return 'text-retro-accent'
    if (line.includes('Saving checkpoint')) return 'text-retro-success'
    if (line.includes('reward_env')) return 'text-emerald-300'
    return 'text-retro-text-dim'
  }

  return (
    <div
      className="bg-retro-card rounded-lg border border-retro-border overflow-hidden flex flex-col"
      style={{ height: collapsed ? 'auto' : '200px' }}
    >
      <div className="px-4 py-2 border-b border-retro-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-retro-text">Live Output</h2>
          <span className="text-[10px] text-retro-text-dim tabular-nums">{lines.length} lines</span>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={e => setAutoScroll(e.target.checked)}
              className="accent-retro-accent scale-75"
            />
            <span className="text-[10px] text-retro-text-dim">Auto-scroll</span>
          </label>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="text-[10px] text-retro-text-dim hover:text-retro-text"
          >
            {collapsed ? 'Expand' : 'Collapse'}
          </button>
        </div>
      </div>
      {!collapsed && (
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto overflow-x-hidden p-2 font-mono text-[11px] leading-[1.4] bg-black/30"
          onWheel={() => setAutoScroll(false)}
        >
          {lines.length === 0 ? (
            <span className="text-retro-text-dim">Waiting for output...</span>
          ) : (
            lines.map((line, i) => (
              <div key={i} className={`${colorize(line)} whitespace-pre-wrap break-all`}>
                {line}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
