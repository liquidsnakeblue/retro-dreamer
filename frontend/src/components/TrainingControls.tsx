import { useState, useEffect } from 'react'
import type { TrainingStatus } from '../hooks/useTrainingSocket'
import { useGameStates } from '../hooks/useGameConfig'

interface TrainingControlsProps {
  status: TrainingStatus | null
  selectedGame: string
}

const API = '/api'

export function TrainingControls({ status, selectedGame }: TrainingControlsProps) {
  const [modelSize, setModelSize] = useState('large')
  const [advisor, setAdvisor] = useState<{ gpu: string; vram_gb: number; recommended: string } | null>(null)

  useEffect(() => {
    fetch(`${API}/advisor/model_size`).then((r) => r.json()).then(setAdvisor).catch(() => {})
  }, [])
  const [batchSize, setBatchSize] = useState(16)
  // SheepRL units: gradient updates per policy step. Paper "train ratio"
  // = this x1024 (batch 16 x seq 64 replayed frames per update).
  const [replayRatio, setReplayRatio] = useState('0.125')
  const [numEnvs, setNumEnvs] = useState(6)
  const [freshStart, setFreshStart] = useState(false)
  const [initialState, setInitialState] = useState('')
  const [loading, setLoading] = useState(false)

  const { states } = useGameStates(selectedGame)

  const state = status?.state || 'idle'
  const isRunning = state === 'training'

  async function apiCall(endpoint: string, method = 'POST', body?: object) {
    setLoading(true)
    try {
      const res = await fetch(`${API}${endpoint}`, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
      })
      return await res.json()
    } catch (e) {
      console.error('API call failed:', e)
    } finally {
      setLoading(false)
    }
  }

  async function handleStart(endpoint: '/training/start' | '/training/switch' = '/training/start') {
    await apiCall(endpoint, 'POST', {
      model_size: modelSize,
      batch_size: batchSize,
      replay_ratio: parseFloat(replayRatio),
      num_envs: numEnvs,
      fresh_start: freshStart,
      game_id: selectedGame,
      initial_state: initialState || undefined,
    })
  }

  // Training a different game than selected → offer an atomic switch
  // (graceful suspend of the running game, then start/resume this one)
  const runningOtherGame = isRunning && status?.game_id && status.game_id !== selectedGame

  const stateOptions: { value: string; label: string }[] = [
    { value: '', label: 'Default' },
    ...states.map(s => ({ value: s, label: s })),
  ]

  return (
    <div className="bg-retro-card rounded-lg border border-retro-border overflow-hidden">
      <div className="px-4 py-3 border-b border-retro-border">
        <h2 className="text-sm font-semibold text-retro-text">Training Controls</h2>
      </div>

      <div className="p-4 space-y-4">
        <div className="flex gap-2">
          {!isRunning ? (
            <button
              onClick={() => handleStart('/training/start')}
              disabled={loading || !selectedGame}
              className="flex-1 bg-retro-success hover:bg-emerald-600 text-white px-3 py-2 rounded text-sm font-semibold transition-colors disabled:opacity-50"
            >
              {loading ? 'Starting...' : 'Start Training'}
            </button>
          ) : runningOtherGame ? (
            <>
              <button
                onClick={() => handleStart('/training/switch')}
                disabled={loading}
                className="flex-1 bg-retro-accent hover:brightness-110 text-black px-3 py-2 rounded text-sm font-semibold transition-colors disabled:opacity-50"
                title={`Gracefully suspend ${status?.game_id}, then train ${selectedGame}`}
              >
                {loading ? 'Switching…' : `⇄ Switch to ${selectedGame}`}
              </button>
              <button
                onClick={() => apiCall('/training/stop')}
                className="bg-retro-danger hover:bg-red-600 text-white px-3 py-2 rounded text-sm font-semibold transition-colors"
              >
                Stop
              </button>
            </>
          ) : (
            <button
              onClick={() => apiCall('/training/stop')}
              className="flex-1 bg-retro-danger hover:bg-red-600 text-white px-3 py-2 rounded text-sm font-semibold transition-colors"
            >
              Stop
            </button>
          )}
        </div>

        <div className="space-y-3">
          <ControlSelect
            label="Model Size"
            value={modelSize}
            onChange={setModelSize}
            options={[
              { value: 'debug', label: 'Debug (XS ~3M)' },
              { value: 'small', label: 'Small (~18M)' },
              { value: 'medium', label: 'Medium (~37M)' },
              { value: 'large', label: 'Large (~77M)' },
              { value: 'xl', label: 'XL (~200M)' },
            ]}
            disabled={isRunning}
          />
          {advisor && (
            <p className="text-[10px] text-retro-text-dim -mt-1">
              {advisor.gpu} ({advisor.vram_gb}GB) — recommended: <span className={modelSize === advisor.recommended ? 'text-retro-success font-semibold' : 'text-retro-accent font-semibold'}>{advisor.recommended.toUpperCase()}</span>
            </p>
          )}

          <ControlSelect
            label="Initial State"
            value={initialState}
            onChange={setInitialState}
            options={stateOptions}
            disabled={isRunning}
          />

          <ControlSlider
            label="Batch Size"
            value={batchSize}
            onChange={setBatchSize}
            min={4}
            max={64}
            step={4}
            disabled={isRunning}
          />

          <ControlSelect
            label="Replay Ratio (grad updates / step)"
            value={replayRatio}
            onChange={setReplayRatio}
            options={[
              { value: '0.03125', label: '0.03 — paper 32 (Atari/Minecraft runs)' },
              { value: '0.0625', label: '0.06 — paper 64 (ProcGen)' },
              { value: '0.125', label: '0.125 — paper 128 (Atari100k)' },
              { value: '0.25', label: '0.25 — paper 256' },
              { value: '0.5', label: '0.5 — paper 512 (control suites)' },
              { value: '1', label: '1.0 — paper 1024 (max sample-efficiency)' },
            ]}
            disabled={isRunning}
          />

          <ControlSlider
            label="Num Environments"
            value={numEnvs}
            onChange={setNumEnvs}
            min={1}
            max={16}
            step={1}
            disabled={isRunning}
          />

          <label className={`flex items-center gap-2 cursor-pointer ${isRunning ? 'opacity-40 pointer-events-none' : ''}`}>
            <input
              type="checkbox"
              checked={freshStart}
              onChange={e => setFreshStart(e.target.checked)}
              disabled={isRunning}
              className="accent-retro-accent"
            />
            <span className="text-xs text-retro-text">Start new model</span>
          </label>
        </div>

        <div className="text-[10px] text-retro-text-dim space-y-1">
          <p>Powered by SheepRL DreamerV3</p>
          <p>Training runs as subprocess with full checkpoint resume</p>
        </div>

        {status?.state === 'error' && (
          <div className="bg-red-900/30 border border-retro-danger rounded p-2 text-xs text-retro-danger">
            {status.error_message?.split('\n')[0] || 'Training error occurred'}
          </div>
        )}
      </div>
    </div>
  )
}

function ControlSelect({ label, value, onChange, options, disabled }: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
  disabled?: boolean
}) {
  return (
    <div>
      <label className="block text-[10px] text-retro-text-dim uppercase tracking-wider mb-1">{label}</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-retro-surface border border-retro-border rounded px-2.5 py-1.5 text-xs text-retro-text disabled:opacity-40 focus:outline-none focus:border-retro-accent"
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}

function ControlSlider({ label, value, onChange, min, max, step, disabled }: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  disabled?: boolean
}) {
  return (
    <div>
      <div className="flex justify-between mb-1">
        <label className="text-[10px] text-retro-text-dim uppercase tracking-wider">{label}</label>
        <span className="text-[10px] text-retro-text tabular-nums">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        disabled={disabled}
        className="w-full h-1 bg-retro-surface rounded-lg appearance-none cursor-pointer accent-retro-accent disabled:opacity-40"
      />
    </div>
  )
}
