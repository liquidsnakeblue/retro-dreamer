import { useState, useMemo, useEffect } from 'react'
import { useGameList } from '../hooks/useGameConfig'

type Workspace = {
  game_id: string
  lineages: { name: string; status: string; running: boolean; head_step: number | null }[]
}

interface GameSelectorProps {
  selectedGame: string
  onSelect: (gameId: string) => void
}

function NewGameModal({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [gameId, setGameId] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [system, setSystem] = useState('Snes')
  const [rom, setRom] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  async function submit() {
    if (!gameId || !rom) { setMsg('game id and ROM file are required'); return }
    setBusy(true)
    setMsg('importing…')
    try {
      const fd = new FormData()
      fd.append('rom', rom)
      const q = new URLSearchParams({
        game_id: gameId, display_name: displayName || gameId, system,
      })
      const res = await fetch(`/api/games/import?${q}`, { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || res.status)
      setMsg('✔ imported')
      onCreated(gameId)
      setTimeout(onClose, 600)
    } catch (e) {
      setMsg(`✘ ${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center" onClick={onClose}>
      <div className="bg-retro-card border border-retro-border rounded-lg p-5 w-96 space-y-3" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-sm font-semibold text-retro-text">New Game — import a ROM</h3>
        <input placeholder="game id (e.g. SuperMarioBros-Nes)" value={gameId}
          onChange={(e) => setGameId(e.target.value.replace(/[^A-Za-z0-9_-]/g, ''))}
          className="w-full bg-retro-surface border border-retro-border rounded px-2.5 py-1.5 text-xs text-retro-text" />
        <input placeholder="display name" value={displayName} onChange={(e) => setDisplayName(e.target.value)}
          className="w-full bg-retro-surface border border-retro-border rounded px-2.5 py-1.5 text-xs text-retro-text" />
        <select value={system} onChange={(e) => setSystem(e.target.value)}
          className="w-full bg-retro-surface border border-retro-border rounded px-2 py-1.5 text-xs text-retro-text">
          {['Snes', 'Nes', 'Genesis', 'GameBoy', 'GbAdvance', 'PCEngine', 'Sms'].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <input type="file" accept=".sfc,.smc,.nes,.md,.gen,.bin,.gb,.gbc,.gba,.pce,.sms"
          onChange={(e) => setRom(e.target.files?.[0] ?? null)}
          className="w-full text-xs text-retro-text-dim" />
        <p className="text-[10px] text-retro-text-dim">
          Your ROM stays on this machine. After import: define RAM variables, build the
          reward (🧪 Validate), capture a save state, train.
        </p>
        <div className="flex gap-2 items-center">
          <button onClick={submit} disabled={busy}
            className="px-3 py-1.5 text-xs font-semibold rounded bg-retro-success text-white disabled:opacity-50">
            {busy ? 'Importing…' : 'Import'}
          </button>
          <button onClick={onClose} className="px-3 py-1.5 text-xs rounded bg-retro-surface border border-retro-border text-retro-text">
            Cancel
          </button>
          {msg && <span className="text-[10px] text-retro-text-dim">{msg}</span>}
        </div>
      </div>
    </div>
  )
}

export function GameSelector({ selectedGame, onSelect }: GameSelectorProps) {
  const { games, loading } = useGameList()
  const [search, setSearch] = useState('')
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [showNewGame, setShowNewGame] = useState(false)

  useEffect(() => {
    const load = () =>
      fetch('/api/workspaces')
        .then((r) => r.json())
        .then((d) => setWorkspaces(d.workspaces || []))
        .catch(() => {})
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  const current = games.find((g: any) => (g.game_id || g.id) === selectedGame)
  const ws = workspaces.find((w) => w.game_id === selectedGame)
  const mainLineage = ws?.lineages.find((l) => l.status === 'active') ?? ws?.lineages[0]

  // Split into custom and built-in, filter by search
  const { customGames, builtinGames } = useMemo(() => {
    const q = search.toLowerCase()
    const custom: any[] = []
    const builtin: any[] = []
    for (const g of games) {
      const gid = g.game_id || g.id || ''
      const name = g.display_name || gid
      if (q && !name.toLowerCase().includes(q) && !gid.toLowerCase().includes(q)) continue
      if (g.source === 'builtin') builtin.push(g)
      else custom.push(g)
    }
    return { customGames: custom, builtinGames: builtin }
  }, [games, search])

  const totalShown = customGames.length + builtinGames.length

  return (
    <div className="bg-retro-card rounded-lg border border-retro-border overflow-hidden">
      <div className="px-4 py-3 border-b border-retro-border flex items-center justify-between">
        <h2 className="text-sm font-semibold text-retro-text">Game</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowNewGame(true)}
            className="text-[10px] px-2 py-1 rounded bg-retro-accent/90 text-black font-semibold hover:brightness-110"
          >
            + New Game
          </button>
          <span className="text-[10px] text-retro-text-dim tabular-nums">{games.length} games</span>
        </div>
      </div>
      {showNewGame && (
        <NewGameModal
          onClose={() => setShowNewGame(false)}
          onCreated={(id) => onSelect(id)}
        />
      )}
      <div className="p-4 space-y-2">
        <input
          type="text"
          placeholder="Search games..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full bg-retro-surface border border-retro-border rounded px-2.5 py-1.5 text-xs text-retro-text placeholder:text-retro-text-dim/50 focus:outline-none focus:border-retro-accent"
        />

        <select
          value={selectedGame}
          onChange={e => onSelect(e.target.value)}
          disabled={loading}
          size={Math.min(12, Math.max(4, totalShown + 2))}
          className="w-full bg-retro-surface border border-retro-border rounded px-1 py-1 text-xs text-retro-text disabled:opacity-40 focus:outline-none focus:border-retro-accent"
        >
          {loading && <option value="">Loading games...</option>}

          {customGames.length > 0 && (
            <optgroup label="Custom Games">
              {customGames.map((g: any) => {
                const gid = g.game_id || g.id
                return <option key={gid} value={gid}>{g.display_name || gid}</option>
              })}
            </optgroup>
          )}

          {builtinGames.length > 0 && (
            <optgroup label={`Built-in (${builtinGames.length})`}>
              {builtinGames.map((g: any) => {
                const gid = g.game_id || g.id
                return <option key={gid} value={gid}>{g.display_name || gid}</option>
              })}
            </optgroup>
          )}

          {!loading && totalShown === 0 && search && (
            <option value="" disabled>No matches for "{search}"</option>
          )}
        </select>

        {current && (
          <div className="flex items-center gap-2">
            <p className="text-[10px] text-retro-text-dim font-mono">
              {current.system && <><span className="text-retro-accent">{current.system}</span> · </>}
              {current.source === 'builtin' ? (
                <span className="text-retro-text-dim">built-in</span>
              ) : (
                <span className="text-retro-success">custom</span>
              )}
              {' · '}
              {mainLineage?.running ? (
                <span className="text-retro-success">● training @ {mainLineage.head_step?.toLocaleString()}</span>
              ) : mainLineage?.head_step ? (
                <span className="text-retro-text">brain @ step {mainLineage.head_step.toLocaleString()}</span>
              ) : (
                <span className="text-retro-text-dim">never trained</span>
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
