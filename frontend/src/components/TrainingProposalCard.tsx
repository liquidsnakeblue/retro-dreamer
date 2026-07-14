export interface TrainingStartProposal {
  type: 'training_start_proposal'
  id: string
  generation: number
  studio_revision: string
  created_at: string
  expires_at: string
  game: {
    id: string
    display_name: string
  }
  mode: 'new' | 'resume' | 'switch'
  launch: {
    strategy: 'new' | 'fresh' | 'resume'
    initial_state: string
    fresh_start: boolean
  }
  superseded_plan_ids: string[]
  head: {
    snapshot_id: string | number
    step: number
    lineage?: string
  } | null
  model: {
    size: string
  }
  states: Array<{
    file: string
    label: string
    description?: string
  }>
  replay_ratio: number
  num_envs: number
  batch_size: number
  batch_length: number
  consequences: string[]
  warnings: string[]
  exact_request: {
    route: string
    body: Record<string, unknown>
  }
}

export type ProposalAction = {
  status: 'pending' | 'confirming' | 'cancelling' | 'confirmed' | 'cancelled' | 'superseded' | 'stale' | 'expired' | 'error'
  error?: string
}

interface TrainingProposalCardProps {
  proposal: TrainingStartProposal
  action: ProposalAction
  approvalReady: boolean
  approvalError: string | null
  onConfirm: () => void
  onCancel: () => void
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Runtime validation keeps proposal actions tied to server data, never prose. */
export function isTrainingStartProposal(value: unknown): value is TrainingStartProposal {
  if (!isObject(value) || value.type !== 'training_start_proposal') return false
  if (!isObject(value.game) || typeof value.game.id !== 'string' || typeof value.game.display_name !== 'string') return false
  if (!isObject(value.model) || typeof value.model.size !== 'string') return false
  if (
    !isObject(value.launch) ||
    !['new', 'fresh', 'resume'].includes(String(value.launch.strategy)) ||
    typeof value.launch.initial_state !== 'string' ||
    typeof value.launch.fresh_start !== 'boolean'
  ) return false
  if (!isObject(value.exact_request) || typeof value.exact_request.route !== 'string' || !isObject(value.exact_request.body)) return false
  if (!['new', 'resume', 'switch'].includes(String(value.mode))) return false
  if (!Array.isArray(value.states) || !value.states.every((state) => (
    isObject(state) &&
    typeof state.file === 'string' &&
    typeof state.label === 'string' &&
    (state.description === undefined || typeof state.description === 'string')
  ))) return false
  if (!Array.isArray(value.consequences) || !value.consequences.every((item) => typeof item === 'string')) return false
  if (!Array.isArray(value.warnings) || !value.warnings.every((item) => typeof item === 'string')) return false
  if (!Array.isArray(value.superseded_plan_ids) || !value.superseded_plan_ids.every((id) => typeof id === 'string')) return false
  return (
    typeof value.id === 'string' &&
    typeof value.generation === 'number' && Number.isInteger(value.generation) && value.generation > 0 &&
    typeof value.studio_revision === 'string' &&
    typeof value.created_at === 'string' &&
    typeof value.expires_at === 'string' &&
    typeof value.replay_ratio === 'number' &&
    typeof value.num_envs === 'number' &&
    typeof value.batch_size === 'number' &&
    typeof value.batch_length === 'number' &&
    value.states.length > 0 &&
    value.launch.initial_state === value.states.map((state) => state.file).join('+') &&
    value.exact_request.body.initial_state === value.launch.initial_state &&
    value.exact_request.body.fresh_start === value.launch.fresh_start &&
    (value.launch.strategy === 'fresh') === value.launch.fresh_start &&
    (value.launch.strategy !== 'resume' || value.head !== null) &&
    (value.launch.strategy !== 'new' || value.head === null) &&
    (value.mode === 'switch' || (value.mode === 'resume') === (value.launch.strategy === 'resume')) &&
    (value.mode === 'switch'
      ? value.exact_request.route === '/api/training/switch'
      : value.exact_request.route === '/api/training/start') &&
    (value.head === null || (
      isObject(value.head) &&
      (typeof value.head.snapshot_id === 'string' || typeof value.head.snapshot_id === 'number') &&
      typeof value.head.step === 'number' &&
      (value.head.lineage === undefined || typeof value.head.lineage === 'string')
    ))
  )
}

export function TrainingProposalCard({
  proposal,
  action,
  approvalReady,
  approvalError,
  onConfirm,
  onCancel,
}: TrainingProposalCardProps) {
  const expiry = new Date(proposal.expires_at)
  const validExpiry = Number.isFinite(expiry.getTime())
  const created = new Date(proposal.created_at)
  const validCreated = Number.isFinite(created.getTime())
  const expired = !validExpiry || expiry.getTime() <= Date.now()
  const acting = action.status === 'confirming' || action.status === 'cancelling'
  const settled = ['confirmed', 'cancelled', 'superseded', 'stale', 'expired'].includes(action.status)
  const disabled = !approvalReady || acting || settled || expired
  const transition = proposal.mode === 'switch' ? 'Switch game' : 'Start game'
  const strategy = proposal.launch.strategy === 'resume'
    ? 'Resume checkpoint'
    : proposal.launch.strategy === 'fresh'
      ? 'Fresh start'
      : 'New brain'
  const resolvedStates = proposal.states.map((state) => (
    state.label === state.file ? state.file : `${state.label} (${state.file})`
  )).join(', ')
  const head = proposal.launch.strategy === 'resume' && proposal.head
    ? `${proposal.head.lineage ? `${proposal.head.lineage} · ` : ''}step ${proposal.head.step.toLocaleString()} (${proposal.head.snapshot_id})`
    : proposal.head
      ? 'Existing head will not be used'
      : 'None — starting a new brain'

  return (
    <section className="rounded-lg border border-retro-accent/60 bg-retro-surface/80 overflow-hidden shadow-lg shadow-black/20">
      <div className="px-4 py-3 border-b border-retro-border bg-retro-accent/10 flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] uppercase tracking-widest font-semibold text-retro-accent-bright">Training proposal</p>
          <h3 className="mt-1 text-sm font-semibold text-retro-text">{proposal.game.display_name}</h3>
          <p className="text-[10px] font-mono text-retro-text-dim">{proposal.game.id} · plan {proposal.id}</p>
        </div>
        <div className="flex flex-wrap justify-end gap-1.5 shrink-0">
          <span className="rounded border border-retro-border bg-retro-card px-2 py-1 text-[10px] font-semibold text-retro-text">
            {transition}
          </span>
          <span className={`rounded border px-2 py-1 text-[10px] font-semibold ${proposal.launch.strategy === 'fresh'
            ? 'border-retro-danger/60 bg-red-950/30 text-retro-danger'
            : 'border-retro-accent/60 bg-retro-accent/10 text-retro-accent-bright'}`}>
            {strategy}
          </span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <div className="rounded border-2 border-retro-accent/70 bg-retro-accent/10 p-3">
          <p className="text-[10px] uppercase tracking-widest font-semibold text-retro-accent-bright">Approval target — launch state</p>
          <p className="mt-1 font-mono text-base font-bold text-retro-text">{proposal.launch.initial_state}</p>
          <p className="mt-1 text-xs text-retro-text">{resolvedStates}</p>
          <p className="mt-2 text-[10px] font-semibold text-retro-text-dim">{strategy} · created {validCreated ? created.toLocaleString() : proposal.created_at}</p>
        </div>

        {proposal.superseded_plan_ids.length > 0 && (
          <p className="rounded border border-retro-border bg-retro-card/70 px-3 py-2 text-[10px] text-retro-text-dim">
            This proposal replaced {proposal.superseded_plan_ids.length} older pending proposal{proposal.superseded_plan_ids.length === 1 ? '' : 's'}.
          </p>
        )}

        <dl className="grid grid-cols-2 xl:grid-cols-4 gap-x-4 gap-y-3">
          <ProposalField label={proposal.launch.strategy === 'resume' ? 'Resume from' : 'Checkpoint'} value={head} wide />
          <ProposalField label="Model" value={proposal.model.size.toUpperCase()} />
          <ProposalField label="Resolved state(s)" value={resolvedStates} />
          <ProposalField label="Replay ratio" value={String(proposal.replay_ratio)} />
          <ProposalField label="Environments" value={String(proposal.num_envs)} />
          <ProposalField label="Batch" value={`${proposal.batch_size} × ${proposal.batch_length}`} />
          <ProposalField label="Studio revision" value={proposal.studio_revision.slice(0, 12)} />
          <ProposalField label="Created" value={validCreated ? created.toLocaleString() : proposal.created_at} />
          <ProposalField label="Expires" value={validExpiry ? expiry.toLocaleString() : proposal.expires_at} />
        </dl>

        {proposal.states.some((state) => state.description) && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-retro-text-dim mb-1.5">State details</p>
            <ul className="space-y-1 text-xs text-retro-text">
              {proposal.states.filter((state) => state.description).map((state) => (
                <li key={state.file}><span className="font-semibold">{state.label}:</span> {state.description}</li>
              ))}
            </ul>
          </div>
        )}

        <ProposalList title="Consequences" items={proposal.consequences} tone="normal" />
        {proposal.warnings.length > 0 && <ProposalList title="Warnings" items={proposal.warnings} tone="warning" />}

        <details className="rounded border border-retro-border bg-retro-card/70 group">
          <summary className="cursor-pointer select-none px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-retro-text-dim hover:text-retro-text">
            <span className="inline-block w-3 transition-transform group-open:rotate-90">▸</span>
            Exact request JSON
          </summary>
          <div className="border-t border-retro-border px-3 py-2">
            <p className="mb-2 font-mono text-[10px] text-retro-accent">POST {proposal.exact_request.route}</p>
            <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-all font-mono text-[10px] leading-relaxed text-retro-text">
              {JSON.stringify(proposal.exact_request.body, null, 2)}
            </pre>
          </div>
        </details>

        <div className="flex items-center justify-between gap-3 border-t border-retro-border pt-3">
          <div className="min-w-0 text-[10px]" aria-live="polite">
            {approvalError && <span className="text-retro-danger">Approval controls unavailable: {approvalError}</span>}
            {!approvalError && !approvalReady && <span className="text-retro-text-dim">Preparing browser approval controls…</span>}
            {expired && <span className="text-retro-danger">This proposal has expired. Ask for a fresh plan.</span>}
            {action.status === 'confirmed' && <span className="text-retro-success font-semibold">Confirmed and submitted.</span>}
            {action.status === 'cancelled' && <span className="text-retro-text-dim font-semibold">Cancelled. Nothing changed.</span>}
            {action.status === 'superseded' && <span className="text-retro-warning font-semibold">Superseded by a newer proposal. Nothing changed.</span>}
            {action.status === 'stale' && <span className="text-retro-warning font-semibold">Stale studio context. Ask for a fresh proposal.</span>}
            {action.status === 'expired' && <span className="text-retro-warning font-semibold">This proposal expired. Ask for a fresh proposal.</span>}
            {action.status === 'error' && <span className="text-retro-danger">{action.error || 'The broker rejected this action.'}</span>}
          </div>
          <div className="flex gap-2 shrink-0">
            <button
              type="button"
              onClick={onCancel}
              disabled={disabled}
              className="px-3 py-1.5 rounded border border-retro-border bg-retro-card text-xs text-retro-text hover:border-retro-text-dim disabled:opacity-40"
            >
              {action.status === 'cancelling' ? 'Cancelling…' : 'Cancel'}
            </button>
            <button
              type="button"
              onClick={onConfirm}
              disabled={disabled}
              className="px-3 py-1.5 rounded bg-retro-success text-white text-xs font-semibold hover:bg-emerald-600 disabled:opacity-40"
            >
              {action.status === 'confirming' ? 'Confirming…' : 'Confirm'}
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

function ProposalField({ label, value, wide = false }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={wide ? 'col-span-2' : ''}>
      <dt className="text-[10px] uppercase tracking-wider text-retro-text-dim">{label}</dt>
      <dd className="mt-0.5 text-xs text-retro-text break-words">{value}</dd>
    </div>
  )
}

function ProposalList({ title, items, tone }: { title: string; items: string[]; tone: 'normal' | 'warning' }) {
  if (items.length === 0) return null
  return (
    <div className={tone === 'warning' ? 'rounded border border-amber-500/40 bg-amber-950/20 p-3' : ''}>
      <p className={`text-[10px] uppercase tracking-wider mb-1.5 ${tone === 'warning' ? 'text-amber-400' : 'text-retro-text-dim'}`}>
        {title}
      </p>
      <ul className="list-disc pl-4 space-y-1 text-xs text-retro-text">
        {items.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}
      </ul>
    </div>
  )
}
