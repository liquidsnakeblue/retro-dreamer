import { useState } from 'react'
import type { VideoInfo } from '../hooks/useTrainingSocket'

interface EpisodePlayerProps {
  videos: VideoInfo[]
}

function formatStep(step: number): string {
  if (step >= 1e6) return (step / 1e6).toFixed(1) + 'M'
  if (step >= 1e3) return (step / 1e3).toFixed(1) + 'K'
  return String(step)
}

function formatDuration(seconds: number): string {
  if (seconds <= 0) return '--'
  if (seconds < 1) return Math.round(seconds * 1000) + 'ms'
  if (seconds < 60) return seconds.toFixed(1) + 's'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function timeAgo(ts: number): string {
  const diff = Math.floor(Date.now() / 1000 - ts)
  if (diff < 60) return 'just now'
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago'
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago'
  return Math.floor(diff / 86400) + 'd ago'
}

/** Map duration to a color — longer survival = greener */
function durationColor(seconds: number): string {
  if (seconds <= 0) return 'text-retro-text-dim'
  if (seconds < 0.1) return 'text-retro-danger'
  if (seconds < 0.3) return 'text-orange-400'
  if (seconds < 0.5) return 'text-yellow-400'
  if (seconds < 1.0) return 'text-emerald-400'
  return 'text-retro-success'
}

/** Simple bar showing relative duration within the set */
function DurationBar({ duration, maxDuration }: { duration: number; maxDuration: number }) {
  const pct = maxDuration > 0 ? Math.min(100, (duration / maxDuration) * 100) : 0
  return (
    <div className="h-1 w-full bg-retro-surface rounded-full overflow-hidden mt-1.5">
      <div
        className="h-full rounded-full transition-all"
        style={{
          width: `${pct}%`,
          background: pct > 66 ? '#10b981' : pct > 33 ? '#f59e0b' : '#ef4444',
        }}
      />
    </div>
  )
}

export function EpisodePlayer({ videos }: EpisodePlayerProps) {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null)

  const maxDuration = videos.reduce((m, v) => Math.max(m, v.duration || 0), 0)

  return (
    <div className="bg-retro-card rounded-lg border border-retro-border overflow-hidden flex-1 flex flex-col">
      <div className="px-4 py-3 border-b border-retro-border flex items-center justify-between shrink-0">
        <h2 className="text-sm font-semibold text-retro-text">Episode Replays</h2>
        <span className="text-[10px] text-retro-text-dim tabular-nums">{videos.length} clips</span>
      </div>

      {/* Video player */}
      {selectedVideo && (
        <div className="border-b border-retro-border bg-black/40 shrink-0">
          <video
            src={`/api/videos/${selectedVideo}`}
            controls
            autoPlay
            loop
            className="w-full rounded"
            style={{ imageRendering: 'pixelated' }}
          />
          <div className="px-3 py-2 flex items-center justify-between">
            <span className="text-[10px] text-retro-text-dim font-mono truncate">{selectedVideo}</span>
            <button
              onClick={() => setSelectedVideo(null)}
              className="text-[10px] text-retro-text-dim hover:text-retro-text ml-2 shrink-0"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Video list */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {videos.length === 0 ? (
          <div className="text-center py-10 text-retro-text-dim text-xs px-4">
            <p className="text-lg mb-1 opacity-20">No videos yet</p>
            <p className="text-[10px]">Videos are captured every 200 env steps during training</p>
          </div>
        ) : (
          <div className="divide-y divide-retro-border/50">
            {videos.map((video) => {
              const isSelected = selectedVideo === video.filename
              return (
                <button
                  key={video.filename}
                  onClick={() => setSelectedVideo(video.filename)}
                  className={`w-full text-left px-3 py-2.5 transition-colors ${
                    isSelected
                      ? 'bg-retro-accent/10'
                      : 'hover:bg-retro-surface/60'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    {/* Step badge */}
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="text-[10px] font-mono bg-retro-surface px-1.5 py-0.5 rounded text-retro-accent shrink-0">
                        {formatStep(video.step)}
                      </span>
                      <span className={`text-xs font-semibold tabular-nums ${durationColor(video.duration)}`}>
                        {formatDuration(video.duration)}
                      </span>
                    </div>
                    {/* Timestamp */}
                    <span className="text-[10px] text-retro-text-dim shrink-0 tabular-nums">
                      {timeAgo(video.modified)}
                    </span>
                  </div>
                  <DurationBar duration={video.duration} maxDuration={maxDuration} />
                </button>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
