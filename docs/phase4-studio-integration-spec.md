# Phase 4 — Studio Integration Spec (copilot slice + episode-report engine)

Author: god (Michael). Date: 2026-07-14. Builder: Jim.
Status: DISPATCHED — this is the build contract for the phase-4 card.

## Provenance (what's already decided — do not re-litigate)

- **Layers 1+2 first slice** was designed by Jim and approved on the hive board
  (`copilot-central-driver-options-jim.md`, "Minimal first slice"). Build was PARKED at
  savepoint `3895a5c` (tag `savepoint-copilot-predesign-20260713`) pending the copilot
  capability test. That test is done; results changed the *emphasis*, not the architecture.
- **Capability-test verdict** (docs/pam-capability-profile.md + board): the model out-read
  the frontier orchestrator when grounded, and confabulation came from *vision/narrative*,
  not from tool use. Design consequence: **enable the model's hands; make watch-the-brain
  the reflex; surface modal (not peak) metrics; keep the training trigger gated.** Do not
  add judgment scaffolding beyond what's below.
- **Episode-report engine** is COMPLETE and proven through 4 gates (god build → Jim
  hardening → Pam usability PASS → Jim round 2). `scratch/episode_report.py`, 590 lines,
  numpy-only, zero game-specific code. Frozen fixtures + verified outputs in
  `scratch/fixtures/` and `docs/episode-report-hardening-results.md`.

## Scope: two workstreams, one vertical product slice

### Workstream A — copilot layers 1+2 (build exactly the parked slice)

Per Jim's own design (options doc §3, items 1–5). Decisions locked:

1. **StudioStateBuilder** (read-only, code owns ALL joins) + `GET /api/studio/state?focus_game_id=…`.
   Revision + per-section `observed_at` timestamps. Compact slice ~2–4 KB; full projection
   queryable through the same builder (one code path, never two truths).
2. **CopilotPanel** gets `selectedGame` + `status` props; `send` includes `focus_game_id`;
   the compact slice is injected **every turn**; UI shows observed time/revision.
3. **`POST /api/training/plan`** — read-only planner for already-onboarded custom games:
   decides new/resume/switch, applies advisor + TrainingConfig dependent defaults
   (fix the known wart: UI hardcodes num_envs=6 — presets are debug=4/small=6/medium=8/large=6/xl=4),
   enumerates state labels/descriptions, validates compatibility, returns a typed
   `training_start_proposal` bound to the studio revision. Qwen narrates + asks only the
   unresolved preference questions.
4. **Immutable proposal CARD** in Copilot (game, mode/head, model, state(s), replay_ratio,
   num_envs, batch, consequences, exact request body). Confirm/Cancel is the ONLY mutation
   point → UI-bound broker calls the existing start/switch routes via
   `POST /api/training/plans/{id}/confirm` (one-time plan, server revalidates revision,
   rejects stale). Cancel changes nothing.
5. On success: refresh ambient status + emit the reversible `open_tab(metrics)` intent.

### Workstream B — episode-report engine into the studio ("watch the brain")

6. **Promote the engine**: `scratch/episode_report.py` → `backend/episode_report.py`
   (module, importable; keep a `python3 backend/episode_report.py <npz> <training.json>`
   CLI). Move `scratch/fixtures/` → `backend/tests/fixtures/` (or `tests/fixtures/`) and
   add a regression test that asserts the three verified outputs still hold:
   LM 21 events / 2 deaths @ playerPage=6 + roomPos=1, life segments 477/511/412;
   Mario 6 events / TERMINAL @436 / time=timer; F-Zero 19 events / damage @ pos +198..+208
   rel / health margin 368. The engine file itself DOES NOT CHANGE in this phase — it is
   accepted; any defect found becomes a bug-sweep card, not a drive-by edit.
7. **`POST /api/tools/watch_brain`** `{game_id, state, steps=1400, checkpoint="latest"}` —
   one composed job: resolve the checkpoint through the catalog (same game-scoped
   `get_resumable_head` path record_episode now uses — 404 unknown game, 409 no checkpoint),
   run `_retro_ram_capture.py` (existing: brain replays checkpoint, full-RAM npz per step),
   then run the engine on npz + the game's training.json. Job result = `{npz_path,
   report_path, report_text}`; persist the report next to the capture under
   `training-state/tools/`. CPU-pinned like every other tool. Reuse the existing job
   manager; do not build a new one.
8. **Primer update** (`backend/copilot_primer.md`) — three additions, keep them short:
   - **Watch-the-brain reflex**: for "how did it go / is it stuck / any problems" questions,
     call watch_brain and read the report BEFORE answering. TB/metrics summarize; the
     report is ground truth for *what the brain actually does*.
   - **Anti-confabulation contract** (verbatim discipline Pam proved): every claim about
     gameplay traces to a report line; unknowns stated as unknowns; honest-unknown beats
     plausible-guess.
   - **Modal, not peak**: characterize runs by typical episodes, not best-ever return.

## Hard constraints (violating any of these fails the build)

- **Injection goes in the USER message**, wrapped in a delimiter the dashboard UI strips
  from display — NEVER the system prompt and never any header/prefix that varies per turn
  ahead of the conversation. This protects Qwen's prompt cache (same failure class as the
  attribution-header KV killer). Envelope at the TOP of the user message, user text after.
- **The model cannot start/stop/switch training.** The proposal/confirm broker is UI-bound;
  the copilot never receives a confirm credential. (Known caveat stands: Bash is still
  permission-bypassed on this box, so the gate is architectural, not airtight — the
  sandboxing pass is Phase E, out of scope here.)
- **Engine stays game-agnostic**: no game names, no per-game branches, shape+config
  matching only. numpy-only, system python3 (no TF/torch imports in the report path —
  PEP-668 blocks installs).
- **GPU is for training only** — watch_brain and everything under /api/tools stays CPU-pinned.
- Follow existing code idioms (job manager, route style, frontend hooks). No new frameworks.

## Acceptance (god verifies each, E2E, before the card closes)

A. With an onboarded game selected: "continue training this" → correct game+head identified
   from ambient context (no discovery round-trip), plan narrated with code-derived settings,
   proposal card rendered, NOTHING mutates before Confirm, Confirm invokes the existing
   route, fresh status reflects the run. Stale-revision plan is rejected on confirm.
B. Injected envelope visible in `/api/copilot/events` raw stream but NOT rendered in the
   dashboard chat; system prompt byte-identical across turns (cache check: second turn's
   prompt-eval count in llama logs stays ~0 for the prefix).
C. `watch_brain` on LittleMermaid returns a report the copilot quotes accurately — deaths
   located by variable values, done-margin line present; and on FZero-Snes — damage
   clustered by position, health margin stated. (These double as live checks on the two
   bug fixes: LM episodes now TERMINAL at lives==0; fzero speed reward no longer flat.)
D. Engine regression test green on the three frozen fixtures.
E. Copilot answers "how is the brain doing at <game>" using watch_brain + modal metrics,
   every claim traceable to a report line.

## Non-goals (defer; do not build)

Full onboarding flow, arbitrary config edits, stop/promote approval flows, durable chat
transport/session resume, generic UI/DOM automation, Bash sandboxing (Phase E), engine
feature work (hi/lo byte-pair fusion stays deferred), retention policies.

## Build order & fences

1. B6 (engine promotion + regression test) — smallest, unblocks C/D early.
2. A1–A2 (builder + envelope injection) — verify acceptance B before layering the planner.
3. B7–B8 (watch_brain + primer) — verify acceptance C/E.
4. A3–A5 (planner, card, broker, intent) — verify acceptance A.
Commit per milestone, not one megacommit. `games/LittleMermaid-Nes-v0/` is git-UNTRACKED
(ROMs) — never `git add` it. Bugs found along the way: fix if in-scope+small, otherwise
new card on the sweep — Schuyler's standing order is active.
