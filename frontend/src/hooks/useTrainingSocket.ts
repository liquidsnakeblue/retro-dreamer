import { useState, useEffect, useCallback } from 'react'

export interface TrainingStatus {
  state: string
  current_step: number
  current_episode: number
  elapsed_time: number
  steps_per_second: number
  avg_return: number
  avg_length: number
  max_return: number
  gpu_memory_used: number
  gpu_memory_total: number
  error_message: string
  game_id: string
  initial_state: string
}

export interface VideoInfo {
  id: string
  source: 'train' | 'eval'
  filename: string
  path: string
  size_mb: number
  modified: number
  step: number
  duration: number
}

const API = '/api'
const POLL_INTERVAL = 5000

export function useTrainingPolling() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [videos, setVideos] = useState<VideoInfo[]>([])
  const [connected, setConnected] = useState(false)

  const poll = useCallback(async () => {
    try {
      const res = await fetch(`${API}/training/status`)
      if (res.ok) {
        const data = await res.json()
        setStatus(data)
        setConnected(true)
      }
    } catch {
      setConnected(false)
    }

    try {
      const res = await fetch(`${API}/videos`)
      if (res.ok) {
        const data = await res.json()
        setVideos(data)
      }
    } catch {}
  }, [])

  useEffect(() => {
    poll()
    const timer = setInterval(poll, POLL_INTERVAL)
    return () => clearInterval(timer)
  }, [poll])

  return { connected, status, videos, refresh: poll }
}
