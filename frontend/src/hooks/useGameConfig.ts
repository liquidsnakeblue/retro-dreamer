import { useState, useEffect } from 'react'

const API = '/api'

export interface GameInfo {
  game_id: string
  id?: string
  display_name: string
  system: string
  source: 'custom' | 'builtin'
  has_custom_config?: boolean
  rom_path?: string
  rom_ready?: boolean
}

export function useGameList() {
  const [games, setGames] = useState<GameInfo[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function fetchGames() {
      try {
        const res = await fetch(`${API}/games`)
        if (res.ok && !cancelled) {
          const data = await res.json()
          setGames(data)
        }
      } catch {
        // ignore fetch errors
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchGames()
    return () => { cancelled = true }
  }, [])

  return { games, loading }
}

export function useGameStates(gameId: string) {
  const [states, setStates] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!gameId) {
      setStates([])
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)

    async function fetchStates() {
      try {
        const res = await fetch(`${API}/games/${encodeURIComponent(gameId)}/states`)
        if (res.ok && !cancelled) {
          const data = await res.json()
          setStates(data.states || data)
        }
      } catch {
        // ignore fetch errors
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchStates()
    return () => { cancelled = true }
  }, [gameId])

  return { states, loading }
}
