import type { StorageUsage, TrainingStatus } from '../hooks/useTrainingSocket'

interface StatusBarProps {
  status: TrainingStatus | null
  storage: StorageUsage | null
}

function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function formatNumber(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return n.toFixed(0)
}

function formatBytes(bytes: number | null): string {
  if (bytes === null) return '—'
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`
}

export function StatusBar({ status, storage }: StatusBarProps) {
  const state = status?.state || 'idle'
  const stateColors: Record<string, string> = {
    idle: 'text-retro-text-dim',
    training: 'text-retro-success',
    stopping: 'text-retro-warning',
    error: 'text-retro-danger',
  }

  return (
    <div className="bg-retro-card border-b border-retro-border px-6 py-2 flex items-center gap-6 text-xs shrink-0 overflow-x-auto">
      <div className="flex items-center gap-2">
        <div className={`w-1.5 h-1.5 rounded-full ${
          state === 'training' ? 'bg-retro-success animate-pulse' :
          state === 'error' ? 'bg-retro-danger' : 'bg-retro-text-dim'
        }`} />
        <span className={`font-semibold uppercase tracking-wider ${stateColors[state] || 'text-retro-text-dim'}`}>
          {state}
        </span>
      </div>

      <Stat label="Step" value={formatNumber(status?.current_step || 0)} />
      <Stat label="Episode" value={formatNumber(status?.current_episode || 0)} />
      <Stat label="SPS" value={(status?.steps_per_second || 0).toFixed(1)} />
      <Stat label="Elapsed" value={formatTime(status?.elapsed_time || 0)} />
      <Stat label="GPU" value={`${(status?.gpu_memory_used || 0).toFixed(1)} GB`} />
      <DiskGauge storage={storage} />

      <div className="w-px h-4 bg-retro-border" />
      <Stat label="Avg Return" value={(status?.avg_return || 0).toFixed(1)} highlight />
      <Stat label="Max Return" value={(status?.max_return || 0).toFixed(1)} />

      {status?.error_message && (
        <>
          <div className="w-px h-4 bg-retro-border" />
          <span className="text-retro-danger truncate max-w-xs" title={status.error_message}>
            {status.error_message.split('\n')[0].slice(0, 80)}
          </span>
        </>
      )}
    </div>
  )
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center gap-1.5 shrink-0">
      <span className="text-retro-text-dim">{label}</span>
      <span className={`font-semibold tabular-nums ${highlight ? 'text-retro-speed-glow' : 'text-retro-text'}`}>
        {value}
      </span>
    </div>
  )
}

function DiskGauge({ storage }: { storage: StorageUsage | null }) {
  const rawPercent = storage?.filesystem.free_percent
  const percent = typeof rawPercent === 'number'
    ? Math.max(0, Math.min(100, rawPercent))
    : null
  if (percent === null) {
    return (
      <div className="flex items-center gap-1.5 shrink-0" aria-label="Disk usage unavailable">
        <span className="text-retro-text-dim">Disk</span>
        <div className="w-14 h-1.5 rounded-full bg-retro-border" />
        <span className="font-semibold text-retro-text-dim">—</span>
        <span className="text-retro-text-dim">Run —</span>
      </div>
    )
  }
  const color = percent <= 5
    ? 'bg-retro-danger'
    : percent <= 10
      ? 'bg-retro-warning'
      : 'bg-retro-success'
  const textColor = percent <= 5
    ? 'text-retro-danger'
    : percent <= 10
      ? 'text-retro-warning'
      : 'text-retro-success'
  const percentLabel = `${percent.toFixed(1)}% free`

  return (
    <div
      className="flex items-center gap-1.5 shrink-0"
      role="meter"
      aria-label="Disk free space"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={percent}
      aria-valuetext={`${percentLabel}; active run ${formatBytes(storage?.active_run_bytes ?? null)}`}
      title="Active run size excludes the lineage replay buffer"
    >
      <span className="text-retro-text-dim">Disk</span>
      <div className="w-14 h-1.5 rounded-full bg-retro-border overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${percent}%` }} />
      </div>
      <span className={`font-semibold tabular-nums ${textColor}`}>{percentLabel}</span>
      <span className="text-retro-text-dim">Run {formatBytes(storage?.active_run_bytes ?? null)}</span>
    </div>
  )
}
