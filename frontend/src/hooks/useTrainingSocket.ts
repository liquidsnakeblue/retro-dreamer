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

export interface StorageUsage {
  sampled_at: number
  filesystem: {
    total_bytes: number | null
    free_bytes: number | null
    free_percent: number | null
  }
  active_run_bytes: number | null
  active_run_sampled_at: number | null
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
const STORAGE_POLL_INTERVAL = 30000

export function useTrainingPolling() {
  const [status, setStatus] = useState<TrainingStatus | null>(null)
  const [videos, setVideos] = useState<VideoInfo[]>([])
  const [storage, setStorage] = useState<StorageUsage | null>(null)
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

  const pollStorage = useCallback(async (signal?: AbortSignal) => {
    try {
      const res = await fetch(`${API}/storage/usage`, { signal })
      if (!res.ok) {
        setStorage(null)
        return
      }
      const data: StorageUsage = await res.json()
      setStorage(current => (
        current === null || data.sampled_at >= current.sampled_at ? data : current
      ))
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        setStorage(null)
      }
    }
  }, [])

  useEffect(() => {
    poll()
    const timer = setInterval(poll, POLL_INTERVAL)
    return () => clearInterval(timer)
  }, [poll])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let controller: AbortController | undefined
    const run = async () => {
      controller = new AbortController()
      await pollStorage(controller.signal)
      if (!cancelled) timer = setTimeout(run, STORAGE_POLL_INTERVAL)
    }
    run()
    return () => {
      cancelled = true
      controller?.abort()
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [pollStorage])

  return { connected, status, storage, videos, refresh: poll }
}
