import { useRef, useState, useEffect } from 'react'
import Hls from 'hls.js'

const LIVE_BASE = `http://${window.location.hostname}:8092`

/** Live Play: drop the newest checkpoint into the game and watch it play —
 * real 60fps video (5x crisp upscale baked into the stream) with real
 * emulator sound, delivered as HLS so playback is smooth. */
export function LivePlay() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const hlsRef = useRef<Hls | null>(null)
  const [playing, setPlaying] = useState(false)
  const [starting, setStarting] = useState(false)
  const [volume, setVolume] = useState(0.7)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (videoRef.current) videoRef.current.volume = volume
  }, [volume])

  const teardown = () => {
    hlsRef.current?.destroy()
    hlsRef.current = null
    const v = videoRef.current
    if (v) {
      v.pause()
      v.removeAttribute('src')
      v.load()
    }
    fetch(`${LIVE_BASE}/stop`).catch(() => {})
    setPlaying(false)
  }

  const start = async () => {
    const v = videoRef.current
    if (!v) return
    setError(null)
    setStarting(true)
    try {
      await fetch(`${LIVE_BASE}/start`)
      // wait for the playlist (session start includes checkpoint loading)
      const t0 = Date.now()
      while (true) {
        const r = await fetch(`${LIVE_BASE}/status`).then((r) => r.json()).catch(() => null)
        if (r?.playlist_ready) break
        if (!r?.running && Date.now() - t0 > 20000) throw new Error('session died')
        if (Date.now() - t0 > 150000) throw new Error('timeout')
        await new Promise((res) => setTimeout(res, 500))
      }
      if (!Hls.isSupported()) throw new Error('HLS unsupported in this browser')
      // ride ~6s behind the live edge with no rate-chasing: hugging the edge
      // means any producer clock wobble starves the buffer (stutter ~20s in)
      const hls = new Hls({ liveSyncDurationCount: 6, maxLiveSyncPlaybackRate: 1.0 })
      hlsRef.current = hls
      hls.loadSource(`${LIVE_BASE}/live/live.m3u8`)
      hls.attachMedia(v)
      await new Promise<void>((resolve, reject) => {
        hls.on(Hls.Events.MANIFEST_PARSED, () => resolve())
        hls.on(Hls.Events.ERROR, (_e, data) => {
          if (data.fatal) reject(new Error(data.type))
        })
        setTimeout(() => reject(new Error('manifest timeout')), 30000)
      })
      v.volume = volume
      await v.play()
      setPlaying(true)
    } catch (e) {
      setError(`Stream failed to start (${(e as Error).message}) — is the live server on :8092?`)
      teardown()
    } finally {
      setStarting(false)
    }
  }

  useEffect(() => () => teardown(), [])

  return (
    <div className="absolute inset-0 flex flex-col bg-retro-card">
      <div className="px-4 py-3 border-b border-retro-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-retro-text">Live Play</h2>
          {playing && (
            <span className="flex items-center gap-1.5 text-[10px] text-red-400 font-semibold">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              LIVE — newest checkpoint, ~4s behind real time
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-retro-text-dim">🔊</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={volume}
              onChange={(e) => setVolume(Number(e.target.value))}
              className="w-24 accent-retro-accent"
            />
          </div>
          {!playing ? (
            <button
              onClick={start}
              disabled={starting}
              className="px-4 py-1.5 text-xs font-semibold rounded bg-retro-accent text-black hover:brightness-110 disabled:opacity-50"
            >
              {starting ? 'Starting…' : '▶ Play'}
            </button>
          ) : (
            <button
              onClick={teardown}
              className="px-4 py-1.5 text-xs font-semibold rounded bg-retro-surface border border-retro-border text-retro-text hover:bg-retro-surface/60"
            >
              ◼ Stop
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 relative bg-black">
        <video
          ref={videoRef}
          playsInline
          className="absolute inset-0 w-full h-full object-contain"
          onEnded={() => {
            teardown()
            setError('Session ended — press Play to start a new one')
          }}
        />
        {!playing && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-retro-text-dim text-xs bg-black/60">
            <p className="text-2xl mb-2 opacity-30">🎮</p>
            {starting ? (
              <p>Starting session — loading the newest checkpoint…</p>
            ) : (
              <p>Press Play to drop the newest trained AI into the game</p>
            )}
            {error && <p className="text-red-400 mt-2 max-w-md">{error}</p>}
          </div>
        )}
      </div>
    </div>
  )
}
