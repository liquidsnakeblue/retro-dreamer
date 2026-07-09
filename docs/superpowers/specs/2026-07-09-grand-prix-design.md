# Retro-Dreamer: Grand Prix Design

**Date**: 2026-07-09
**Goal**: DreamerV3 wins the F-Zero (SNES) Knight League Grand Prix, trained and operated
through the retro-dreamer studio — then the proven pipeline generalizes to other SNES/NES games.
**Research basis**: `docs/research/2026-07-09-sota-deep-research.md` (49-agent deep-research
pass, 2026-07-09).

## Vision & strategy

Agent first, studio second. One undeniable result (win the Knight League) proves the
pipeline; generalization to other games comes after, reusing the same machinery. We stay on
DreamerV3 (Nature 2025, verified minimal-tuning claims) via the vendored SheepRL 0.5.8.dev
and stable-retro (upgraded to 1.0.1). Port to a modern base (NM512/r2dreamer) only if we hit
a demonstrated ceiling — that is the designated escape hatch, not a first move.

Success milestones:

| # | Milestone | Exit criterion |
|---|-----------|----------------|
| M0 | Pipeline green | Debug run completes end-to-end: checkpoint + video + dashboard metrics |
| M1 | Lap parity | New studio reproduces the legacy "good lap" on Mute City I |
| M2 | Full race | Agent finishes a 5-lap race |
| M3 | Win one GP race | 1st place in a single Grand Prix race vs rivals |
| M4 | Win Knight League | 1st place across all 5 Knight League tracks |
| M5 | Second game | Another SNES game scaffolded + trained through the same pipeline |

Only two hyperparameters are tuned per game (verified research finding): **model size**
(iterate at S ~12-25M; league pushes at M/L ~50-100M) and **replay_ratio**. Everything else
stays at DreamerV3 defaults.

## Section 1 — Foundation

**Problem**: retro-dreamer has no Python environment, has never completed a training run,
and carries dead code from the F-Zero-only era. Rewards are double-stacked (retro's
scenario.json reward + wrapper's training.json shaping are summed with different weights).

**Work**:

1. **Runtime rebuild**: fresh venv at `~/retro-dreamer/venv` — Python 3.12, torch ≥2.10
   (cu128, native sm_120), stable-retro 1.0.1, gymnasium, lightning; vendored sheeprl
   installed editable (`pip install -e sheeprl/`). Patch vendored SheepRL for modern
   gymnasium/numpy where needed; patches stay minimal and documented in
   `docs/sheeprl-patches.md`.
2. **Reward consolidation**: `training.json` becomes the single reward source of truth.
   `scenario.json` reward/done variables are zeroed out (kept structurally for retro
   compatibility). The wrapper ignores retro's base reward (`base_reward` dropped from
   `_calculate_reward`, or scenario emits 0 — both, belt and suspenders).
3. **Dead code removal** (verify each is unreferenced first): `fzero*.py` env wrappers
   (~900 lines), `dreamer_v3_fzero.yaml`, `TrainingConfig.to_sheeprl_config()`,
   `MetricsCollector`, `TensorBoardCallback`, `EpisodeRenderer`, `_TrainerWSBridge`,
   WebSocket broadcaster plumbing, `train_ratio` legacy field. Frontend keeps HTTP polling.
4. **Run discipline**:
   - `buffer.checkpoint=True` + memmap dir preserved — replay buffer survives restarts.
   - Resume flow (`checkpoint.resume_from` + identical config) is a tested, first-class path.
   - **Run ledger**: `backend/runs/ledger.jsonl` — every training start appends
     `{run_id, timestamp, game_id, config_snapshot, git_sha, sheeprl_cmd, notes}`;
     API endpoint exposes it; outcome/notes appendable per run.
5. **obs_size / misc config hygiene** from the code review (field factory, ROM validation
   moved to Section 3).

**Exit criterion (M0+M1)**: a debug-size run completes end-to-end, then a real run
reproduces legacy lap quality on Mute City I from the `go` state.

## Section 2 — Grand Prix machinery

**Problem**: the current setup can learn "drive fast" but cannot represent "win the league":
no rank/lap/finish RAM variables, no GP-mode save states, episodes end only on death or
reversing, reward is speed-centric.

**Work**:

1. **RAM map expansion** (`data.json`): find and verify addresses for `rank` (current
   place 1-20), `lap` (current lap), `race_end`/finished flag; nice-to-have: track id, lap
   time. Method: stable-retro integration-UI RAM search cross-checked against community
   F-Zero RAM maps. Addresses are verified empirically in-emulator before use — never
   assumed. Deliverable includes a `docs/fzero-ram-map.md` recording evidence per address.
2. **Reward v2** (`training.json` schema extension, backward compatible):
   - `pos` → `mode: "delta"` — per-step progress along track, the **primary** signal
     (Linesight-style progress-over-reference; clipped to kill teleport/lap-wrap spikes).
   - `rank` → `mode: "rank_delta"` — bonus for passing, penalty for being passed.
   - `mode: "terminal_placement"` — one-time end-of-race bonus scaled by finish position
     (Mario Kart Wii BTR template), fired on the race-finish flag.
   - `health` penalty retained; `speed` quadratic retained but **optional/off by default**.
   - Reward-component logging: each component's per-episode sum is logged so reward hacking
     is diagnosable from TensorBoard, not just video review.
3. **Episode design**: time-limit truncation (`truncate.max_steps` in training.json;
   Gymnasium `truncated=True`, not `terminated`), done on race finish (rank-aware terminal
   bonus), done on death/reverse as today.
4. **GP-mode save states**: record Knight League states in Grand Prix mode (rivals + rank
   live): race start per track, plus lap-4/lap-5 late-race states for reverse curriculum.
   Existing practice states stay for early curriculum stages.
5. **Curriculum engine**:
   - `curriculum.json` per game: ordered stages, each with a state pool and promotion rule
     `{metric: success_rate, threshold, window}`; success is game-defined (e.g. finish ≥
     required rank).
   - Wrapper samples the episode start state on `reset()` from the active stage
     (reverse-curriculum weighting: recent-failure states sampled more).
   - Episode outcomes append to a per-run JSONL; a `CurriculumManager` aggregates it to
     decide stage promotion. Works with `num_envs ≥ 1` (append-only file, atomic lines).
   - Multi-track joint training is the same mechanism: a stage whose state pool spans all
     five Knight League tracks.
   - Dashboard shows current stage + success rates (read-only view first; editor later).

## Section 3 — Studio & ops

**Problem**: multi-day runs die silently and lose progress; the studio has unwired
infrastructure and missing validations.

**Work**:

1. **Crash watchdog**: `_monitor_process` on unexpected nonzero exit auto-relaunches with
   resume (checkpoint + buffer), max 3 retries per 6h window, loud logging + run-ledger
   entries. Manual stop never triggers it.
2. **Scheduled eval**: every N hours (config, default 6) run `sheeprl_eval.py` against the
   latest checkpoint with video capture → full-race video in the dashboard episode player.
   Brief GPU sharing with the training run is acceptable (num_envs=1, short).
3. **Validation & health**: ROM existence + SHA check before training start; TensorBoard
   subprocess health check (port bound) with dashboard warning; config PUT endpoints
   validate JSON schema before write.
4. **API honesty**: pause/resume return HTTP 501; game list refreshes after scaffold.
5. Frontend polling stays; the checkbox actions editor stays untouched.

## Section 4 — Testing & guardrails

1. **Unit tests**: reward v2 calculation (each mode, incl. clipping and terminal bonus),
   curriculum promotion logic, config schema validation. Pytest, `backend/tests/`.
2. **Smoke-run script**: `scripts/smoke_run.sh` — 1k-step debug-size training run; must be
   green before any long run starts. This is the pre-flight check.
3. **Determinism check**: same state + same action script ⇒ identical info trajectory
   (guards RAM-address and emulator regressions).
4. **Anti-reward-hacking ritual**: before trusting any metric milestone, watch the latest
   eval videos; reward-component logs reviewed for any component dominating unexpectedly.
   (Verified finding: every dense reward term eventually gets exploited.)

## Data flow (end to end)

```
games/<id>/{metadata,actions,training,curriculum,data}.json + states/*.state
        │ (GameManager merges, validates, registers retro integration)
        ▼
DreamerV3Trainer._launch_sheeprl() ── subprocess ──► SheepRL CLI (dreamer_v3 + retro exp)
        │                                               │ creates N × RetroDreamerWrapper
        │ stdout parse → _MetricsTracker                │   ├─ reset(): CurriculumManager picks start state
        │ watchdog → auto-resume                        │   ├─ step(): reward v2 from RAM deltas
        ▼                                               │   └─ outcomes → run JSONL
FastAPI /api/* ──► React dashboard (polling)            ▼
        ▲                                    logs/runs/... {checkpoints, memmap buffer,
        └── run ledger, eval videos ◄─────── tfevents, train/eval videos}
```

## Error handling

- SheepRL launch failure → ERROR state with last stdout lines surfaced in dashboard (exists).
- Unexpected subprocess death → watchdog resume path (new).
- Missing ROM / bad config → rejected at API layer before subprocess launch (new).
- RAM variable absent from info dict → wrapper logs once per variable and skips (exists,
  keep) — plus a startup assertion that all training.json variables exist in data.json.
- Curriculum file corrupt/missing → fall back to `initial_state`, warn loudly.

## Risks & mitigations

- **Vendored SheepRL vs modern deps**: unknown patch surface. Mitigation: M0 is scoped
  narrowly to prove it; if patching balloons, pin older deps in the venv instead (isolated,
  documented), since the venv exists only for this project.
- **Rank RAM address hard to find**: fall back to screen-region OCR of the rank digit in
  eval only, while training on progress+finish rewards without rank_delta (terminal
  placement still works from finish order).
- **DreamerV3 stalls on full-GP horizon** (nobody has published a full-game DreamerV3
  clear): curriculum shortens effective horizon; if M3 stalls after honest tuning of the
  two knobs, the designated escape hatch is porting the wrapper to r2dreamer — not
  redesigning rewards forever.
- **Reward hacking**: component logging + video ritual (Section 4).

## Out of scope (this cycle)

- PPO/PufferLib parallel path (revisit only at a demonstrated Dreamer ceiling).
- Curriculum UI editor (JSON first; read-only dashboard view only).
- WebSocket streaming, pause/resume support, multi-GPU, cloud training.
- Deleting `~/fzero-dreamer/` (17GB) — archival decision belongs to Schuyler, not this spec.
