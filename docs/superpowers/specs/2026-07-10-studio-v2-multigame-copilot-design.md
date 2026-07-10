# Retro-Dreamer Studio v2 — Multi-Game Workspaces + Resident Copilot

**Date:** 2026-07-10
**Status:** Approved in dialogue (Schuyler + Claude), consolidated here
**Supersedes:** extends `2026-07-09-grand-prix-design.md` (M5 "second game" milestone becomes this)

## Vision

Anyone with a GPU and a ROM can train a DreamerV3 agent on a retro game entirely
from the dashboard: import ROM → discover RAM variables → structure rewards →
capture save states → train → watch — with an LLM copilot resident in the studio
doing the cognitive gruntwork. Multiple games coexist; switching is atomic and
lossless; no brain is ever lost or mis-selected.

## Non-negotiable constraints (from F-Zero scar tissue)

1. **No brain selection by filesystem mtime.** Resumable heads are explicit
   catalog pointers, scoped by game + lineage.
2. **Never train past a replay-buffer reset.** Fork-weights-onto-fresh-buffer
   collapsed once (world model overfits survivor-only data). Reward/done config
   changes default to **keep replay** (reward head recalibrates; observed to work
   twice). Fresh-buffer forking exists but is the explicitly-dangerous option.
3. **Reward changes never ship unprobed.** The constant-action probe +
   degenerate-strategy checks are a first-class studio feature, not a script.
4. **Training must survive unattended runs**: graceful suspend, watchdog that
   resumes *what was running* (from catalog), eval bundles cheap enough to keep
   (rolling keep-N once deleted a peak brain), hourly archive links.
5. **RAM over vision** for the copilot: tools return RAM values alongside frames;
   vision confirms, RAM measures.
6. User supplies the ROM. We never ship or fetch ROMs.

## Architecture

### Training catalog (SQLite, stdlib only)
Tables (from the external agent's proposal, adopted):
- `games(id, display_name, active_lineage_id)`
- `lineages(id, game_id, name, parent_snapshot_id, compatibility_hash, status, created_at)`
- `sessions(id, lineage_id, run_dir, started_at, ended_at, start_step, end_step, status, exit_reason, resolved_config, git_commit, rom_hash)`
- `snapshots(id, session_id, step, checkpoint_path, replay_path, kind, validation_status, config_hash, metrics_json, created_at)`

`snapshots.kind` ∈ {`resume`, `eval`, `archive`}:
- **resume** — full training state (weights, optimizers, ratio counters, buffer ref).
- **eval** — weights + resolved config only; small; retained generously.
- **archive** — resume + frozen replay copy; manual/milestone.

SQLite owns pointers/metadata; artifacts stay files.

### Storage layout
```
retro-dreamer/training-state/
├── catalog.sqlite
├── control/<session-id>/            # checkpoint-request / checkpoint-complete.json
└── games/<game_id>/lineages/<name>/
    ├── lineage.json                 # human-readable mirror of catalog row
    ├── replay/                      # STABLE buffer home (memmaps live here, not in run dirs)
    ├── snapshots/snap-<step>/       # promoted resume snapshots + manifest.json
    └── exports/                     # best.ckpt, latest-eval.ckpt (catalog-pointed)
```
SheepRL run dirs remain as session logs; durable state moves out of them.

### Compatibility fingerprint
Hash over: ROM hash, dreamer impl version, model dims, obs keys/shapes/preproc,
action count+mappings, num_envs, replay schema, reward-config hash, done-config
hash. Change classification:
- **Fully compatible** (replay ratio, entropy, cadences): resume normally.
- **Reward/done changed**: resume with SAME replay + loud warning (constraint 2).
- **Incompatible** (action dims, obs shape, model size): fresh lineage or
  explicit weights-only fork with double-confirm.

### Graceful suspend
File control channel (`checkpoint-request` → loop checks between iterations →
final resume snapshot → `checkpoint-complete.json`), then SIGTERM. `stop()`
keeps SIGTERM-only as the fallback path. Switch state machine:
RUNNING → SUSPEND_REQUESTED → CHECKPOINTING → VALIDATING → SUSPENDED →
STARTING_NEXT → RUNNING, behind a global training lock.

### Tool layer (harness-agnostic, HTTP)
All per-game craft ships as studio endpoints, drivable by human UI, Claude, or
the resident copilot identically:
`/api/tools/ram_capture_diff`, `run_reward_probe`, `record_episode` (frames +
per-step RAM), `build_state` (button-plan menu walk), `run_walker`
(agent-earns-states), `read_metrics`, `scaffold_game`, `write_game_config`
(validation-gated). Fat tools, distilled outputs.

### Resident copilot
- Model: Qwen 3.6 27B on the 2×3090 box. Vision REQUIRES the :8082 proxy
  (direct :6789 silently drops images). Reasoning model: 600-900s timeouts,
  never cap output tokens. Attribution header off.
- Harness: `claude-local` headless (Claude Code CLI against the proxy), spawned
  and lifecycle-managed by the studio backend; skills run natively.
- UI: chat panel in dashboard streams the session ("thinking…" state expected).
- Context hygiene: lean sessions; tools return summaries, not dumps (the 3090
  box comfortably holds ~2 heavyweight conversations).
- Role: proposes/probes/explains; human approves reward changes and theory
  calls. Outputs are leads to verify — the tool layer enforces verification.
- Escalation path for diagnosis-grade puzzles: frontier model (Claude session /
  GPT-5.6 harness), by human invocation.
- Skills library (shipped in repo): `onboard-a-game`, `audit-a-reward`,
  `diagnose-a-run` (split/evolved from retro-integrator v1).

### UI
- Game/lineage picker: status (never-trained/running/suspended/error), active
  lineage, latest step, snapshot age, best eval, disk usage. Switch = one action.
- All views (metrics, videos, checkpoints, logs, TensorBoard) scoped by
  game/lineage/session, not "whatever the trainer is doing".
- Watch tab reads states + labels from `games/<id>/metadata.json`
  (schema gains `states: [{file, label, group, default}]`) — kills the
  hardcoded F-Zero TRACKS list.
- Onboarding wizard: ROM import (hash, scaffold) → RAM workbench (mark events
  while playing/watching → full-RAM diff/intersect → candidate addresses →
  adopt into data.json) → reward builder (training.json editor; save runs the
  probe and shows deviation + degenerate-strategy report) → state capture
  (manual + walker) → launch training.

## Phases

**A — Foundation** (buildable while F-Zero trains; adoption at its next restart)
1. Catalog + schema + retroactive registration of the F-Zero XL lineage
   (crawl existing run dirs → sessions/snapshots; head = newest valid ckpt).
2. Kill every mtime-latest lookup (trainer, live player, recorder, walker) →
   catalog heads scoped by game+lineage.
3. Graceful suspend protocol (vendored-loop patch + /api/training/suspend).
4. Stable lineage replay dir; current buffer migrates at next restart.
5. Watchdog v2 reads catalog (resumes the active session's own params).
Acceptance: suspend/resume F-Zero via API with zero step loss; correct brain
resolution per game with a second (scaffolded) game present.

**B — Switching + dashboard**
1. /api/training/switch state machine.
2. Game/lineage picker + status UI.
3. Scope all data endpoints/views by game/lineage.
4. Metadata-driven Watch tab.
Acceptance: F-Zero → test game → F-Zero round trip from the dashboard,
both brains intact and correct.

**C — Onboarding pipeline + tool layer** (game two rides through this)
1. HTTP tool layer (list above).
2. ROM import UI. 3. RAM workbench. 4. Reward builder w/ probe-on-save.
5. State capture UI + walker feature.
6. **Onboard game two end-to-end through the product** (Claude driving the
   tools; every gap found becomes a tool fix).
Acceptance: game two training launched with zero scratchpad scripts.

**D — Copilot residency**
1. Headless claude-local session manager + chat panel.
2. Skills split/shipped. 3. Supervised copilot run of an onboarding workflow
   (game three, or replay of game two's) with approval gates.
Acceptance: copilot completes RAM-discovery + reward-draft via HTTP tools only.

**E — Product polish (later):** lineage forking UI, archive/retention policies,
VRAM-based model-size advisor, embedded tool-loop replacing the CLI harness,
packaging/distribution.

## Continuity during the build

- F-Zero XL keeps training through A and B; it adopts the catalog + stable
  replay path at its next natural restart (batched with the queued reverse-flag
  penalty and any other pending env changes).
- Standing side-quests unaffected: full-league GP attempt (M4) whenever the
  driver looks ready; per-track adeptness watch continues via heartbeats.

## Open items

- Game two selection (Schuyler deciding).
- Reverse-flag penalty: verify `reverse` RAM semantics per mode, then ship with
  the next restart (queued, approved direction).
- Observation-resolution experiment (64→96) only if cross-track blending
  persists after substantially more training.
