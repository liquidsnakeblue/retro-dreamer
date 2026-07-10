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

export function GameSelector({ selectedGame, onSelect }: GameSelectorProps) {
  const { games, loading } = useGameList()
  const [search, setSearch] = useState('')
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])

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
        <span className="text-[10px] text-retro-text-dim tabular-nums">{games.length} games</span>
      </div>
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
