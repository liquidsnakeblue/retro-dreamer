import { useRef, useState, useEffect } from 'react'
import Hls from 'hls.js'

const LIVE_BASE = `http://${window.location.hostname}:8092`

// Save states verified frame-by-frame 2026-07-09 (F-Zero specific for now;
// generalize into games/<id> metadata when game #2 lands)
const TRACKS = [
  // Real races first — watching = GP mode (training stays practice-only)
  { state: 'gp_knight_beginner', label: '🏁 GP Race 1 — Mute City I' },
  { state: 'gp_knight_r2_bigblue', label: '🏁 GP Race 2 — Big Blue' },
  { state: 'gp_knight_r3_sandocean', label: '🏁 GP Race 3 — Sand Ocean' },
  { state: 'gp_knight_r4_deathwind', label: '🏁 GP Race 4 — Death Wind I' },
  { state: 'gp_knight_r5_silence', label: '🏁 GP Race 5 — Silence' },
  { state: 'go', label: 'Practice — Mute City I' },
  { state: 'BBP1', label: 'Practice — Big Blue' },
  { state: 'SOP1', label: 'Practice — Sand Ocean' },
  { state: 'DWP1', label: 'Practice — Death Wind I' },
  { state: 'SP1', label: 'Practice — Silence' },
  { state: 'PLP1', label: 'Practice — Port Town I' },
  { state: 'WLP1', label: 'Practice — White Land I' },
]

type Mode = 'idle' | 'recording' | 'replay' | 'starting-live' | 'live'

/** Watch the newest checkpoint play. Primary flow: record an episode to a
 * file (flawless, seekable playback), per Schuyler's bk2-style workflow.
 * Live streaming (HLS) kept as a secondary option. */
export function LivePlay() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const hlsRef = useRef<Hls | null>(null)
  const [mode, setMode] = useState<Mode>('idle')
  const [percent, setPercent] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [length, setLength] = useState('60')
  const [track, setTrack] = useState('gp_knight_beginner')
  const [volume, setVolume] = useState(0.7)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (videoRef.current) videoRef.current.volume = volume
  }, [volume])

  const reset = () => {
    hlsRef.current?.destroy()
    hlsRef.current = null
    const v = videoRef.current
    if (v) {
      v.pause()
      v.removeAttribute('src')
      v.load()
    }
    fetch(`${LIVE_BASE}/stop`).catch(() => {})
    setMode('idle')
  }

  useEffect(() => () => reset(), [])

  // ---------- Record & Watch (primary) ----------
  const record = async () => {
    setError(null)
    setMode('recording')
    setPercent(0)
    setElapsed(0)
    const ticker = setInterval(() => setElapsed((e) => e + 1), 1000)
    try {
      await fetch(`${LIVE_BASE}/record?seconds=${length}&state=${encodeURIComponent(track)}`)
      const t0 = Date.now()
      while (true) {
        await new Promise((r) => setTimeout(r, 1000))
        const s = await fetch(`${LIVE_BASE}/record_status`).then((r) => r.json()).catch(() => null)
        if (s?.error) throw new Error(s.error)
        if (s?.percent != null) setPercent(s.percent)
        if (s?.done) break
        if (Date.now() - t0 > 8 * 60000) throw new Error('timeout')
      }
      const v = videoRef.current!
      v.src = `${LIVE_BASE}/rec/recording.mp4?t=${Date.now()}`
      v.controls = true
      v.loop = true
      v.volume = volume
      await v.play()
      setMode('replay')
    } catch (e) {
      setError(`Recording failed: ${(e as Error).message}`)
      setMode('idle')
    } finally {
      clearInterval(ticker)
    }
  }

  // ---------- Live (secondary) ----------
  const startLive = async () => {
    const v = videoRef.current
    if (!v) return
    setError(null)
    setMode('starting-live')
    setElapsed(0)
    const ticker = setInterval(() => setElapsed((e) => e + 1), 1000)
    try {
      await fetch(`${LIVE_BASE}/start?state=${encodeURIComponent(track)}`)
      const t0 = Date.now()
      while (true) {
        const r = await fetch(`${LIVE_BASE}/status`).then((r) => r.json()).catch(() => null)
        if (r?.playlist_ready) break
        if (Date.now() - t0 > 150000) throw new Error('timeout')
        await new Promise((res) => setTimeout(res, 500))
      }
      if (!Hls.isSupported()) throw new Error('HLS unsupported')
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
      v.controls = false
      v.loop = false
      v.volume = volume
      await v.play()
      setMode('live')
    } catch (e) {
      setError(`Live stream failed: ${(e as Error).message}`)
      reset()
    } finally {
      clearInterval(ticker)
    }
  }

  const busy = mode === 'recording' || mode === 'starting-live'
  const showingVideo = mode === 'replay' || mode === 'live'

  return (
    <div className="absolute inset-0 flex flex-col bg-retro-card">
      <div className="px-4 py-3 border-b border-retro-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-retro-text">Watch</h2>
          {mode === 'live' && (
            <span className="flex items-center gap-1.5 text-[10px] text-red-400 font-semibold">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              LIVE — {TRACKS.find((t) => t.state === track)?.label ?? track}, ~6s behind real time
            </span>
          )}
          {mode === 'replay' && (
            <span className="text-[10px] text-retro-success font-semibold">
              RECORDED — {TRACKS.find((t) => t.state === track)?.label ?? track}, newest checkpoint, loops
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-retro-text-dim">🔊</span>
            <input
              type="range" min={0} max={1} step={0.05} value={volume}
              onChange={(e) => setVolume(Number(e.target.value))}
              className="w-24 accent-retro-accent"
            />
          </div>
          <select
            value={track}
            onChange={(e) => setTrack(e.target.value)}
            disabled={busy}
            className="bg-retro-surface border border-retro-border rounded text-xs px-2 py-1.5 text-retro-text"
          >
            {TRACKS.map((t) => (
              <option key={t.state} value={t.state}>{t.label}</option>
            ))}
          </select>
          <select
            value={length}
            onChange={(e) => setLength(e.target.value)}
            disabled={busy}
            className="bg-retro-surface border border-retro-border rounded text-xs px-2 py-1.5 text-retro-text"
          >
            <option value="30">30 sec</option>
            <option value="60">1 min</option>
            <option value="120">2 min</option>
            <option value="180">3 min</option>
            <option value="full">Full episode</option>
          </select>
          {!showingVideo && !busy && (
            <>
              <button
                onClick={record}
                className="px-4 py-1.5 text-xs font-semibold rounded bg-retro-accent text-black hover:brightness-110"
              >
                🎥 Record & Watch
              </button>
              <button
                onClick={startLive}
                className="px-3 py-1.5 text-xs font-semibold rounded bg-retro-surface border border-retro-border text-retro-text hover:bg-retro-surface/60"
              >
                Live
              </button>
            </>
          )}
          {(showingVideo || busy) && (
            <button
              onClick={reset}
              className="px-4 py-1.5 text-xs font-semibold rounded bg-retro-surface border border-retro-border text-retro-text hover:bg-retro-surface/60"
            >
              ◼ Stop
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 relative bg-black">
        <video ref={videoRef} playsInline className="absolute inset-0 w-full h-full object-contain" />
        {!showingVideo && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-retro-text-dim text-xs bg-black/60">
            <p className="text-2xl mb-2 opacity-30">🎮</p>
            {mode === 'recording' && (
              <div className="w-64">
                <p className="mb-2">Recording the newest checkpoint… {percent}% ({elapsed}s)</p>
                <div className="h-1.5 bg-retro-surface rounded-full overflow-hidden">
                  <div className="h-full bg-retro-accent transition-all" style={{ width: `${percent}%` }} />
                </div>
                <p className="mt-2 opacity-60">model load ~30s, then generates ~1.2x realtime</p>
              </div>
            )}
            {mode === 'starting-live' && <p>Starting live session… {elapsed}s (cold start 60-90s)</p>}
            {mode === 'idle' && <p>Record the newest trained AI playing, then watch it — or go Live</p>}
            {error && <p className="text-red-400 mt-2 max-w-md">{error}</p>}
          </div>
        )}
      </div>
    </div>
  )
}
