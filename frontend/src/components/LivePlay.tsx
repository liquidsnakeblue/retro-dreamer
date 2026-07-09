import { useRef, useState, useEffect } from 'react'

const LIVE_BASE = `http://${window.location.hostname}:8092`

/** Live Play: drop the newest checkpoint into the game and watch it play —
 * real 60fps video with real emulator sound, streamed as fMP4. */
export function LivePlay() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [starting, setStarting] = useState(false)
  const [volume, setVolume] = useState(0.7)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (videoRef.current) videoRef.current.volume = volume
  }, [volume])

  const start = async () => {
    const v = videoRef.current
    if (!v) return
    setError(null)
    setStarting(true)
    // cache-buster so every Play gets a fresh session (newest checkpoint)
    v.src = `${LIVE_BASE}/stream.mp4?t=${Date.now()}`
    v.volume = volume
    try {
      // wait for a ~2.5s buffer cushion before starting playback so small
      // producer hiccups never stall the video (session start includes
      // checkpoint loading, so allow a couple of minutes)
      await new Promise<void>((resolve, reject) => {
        const t0 = Date.now()
        const tick = () => {
          if (v.buffered.length > 0 && v.buffered.end(v.buffered.length - 1) >= 2.5) return resolve()
          if (Date.now() - t0 > 150000) return reject(new Error('timeout'))
          if (v.error) return reject(new Error('stream error'))
          setTimeout(tick, 250)
        }
        tick()
      })
      await v.play()
      setPlaying(true)
    } catch (e) {
      setError('Stream failed to start — is the live server running on :8092?')
      v.removeAttribute('src')
      v.load()
    } finally {
      setStarting(false)
    }
  }

  const stop = () => {
    const v = videoRef.current
    if (!v) return
    v.pause()
    v.removeAttribute('src')
    v.load()
    setPlaying(false)
  }

  useEffect(() => () => stop(), [])

  return (
    <div className="h-full flex flex-col bg-retro-card">
      <div className="px-4 py-3 border-b border-retro-border flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-retro-text">Live Play</h2>
          {playing && (
            <span className="flex items-center gap-1.5 text-[10px] text-red-400 font-semibold">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              LIVE — newest checkpoint, ~2s behind real time
            </span>
          )}
        </div>
        <div className="flex items-center gap-4">
          {/* Volume */}
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
              onClick={stop}
              className="px-4 py-1.5 text-xs font-semibold rounded bg-retro-surface border border-retro-border text-retro-text hover:bg-retro-surface/60"
            >
              ◼ Stop
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 flex items-center justify-center bg-black p-2 relative overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="h-full w-full object-contain"
          onError={() => playing && setError('Stream ended')}
        />
        {!playing && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-retro-text-dim text-xs bg-black/60">
            <p className="text-2xl mb-2 opacity-30">🎮</p>
            {starting ? (
              <p>Starting session — loading the newest checkpoint…</p>
            ) : (
              <p>Press Play to drop the newest trained AI into the game</p>
            )}
            {error && <p className="text-red-400 mt-2">{error}</p>}
          </div>
        )}
      </div>
    </div>
  )
}
