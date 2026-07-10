# Studio v2 — As-Built Notes & Known Warts (2026-07-10)

For the audit/cleanup crew. Design spec: `docs/superpowers/specs/2026-07-10-studio-v2-multigame-copilot-design.md`.
Built in one day (commits 3f16860 → 3d6433f) while F-Zero trained continuously;
expect fast-build roughness. This file lists what exists, how to verify it, and
every wart the builder knows about. Findings beyond this list are the point of
your audit.

## Architecture map

- `backend/catalog.py` — SQLite catalog (training-state/catalog.sqlite):
  games/lineages/sessions/snapshots; `get_resumable_head()` (per game+lineage,
  existence-checked, re-crawl-before-resolve via `register_existing_runs`);
  `get_watch_head()` (running session first). CLI: `python3 backend/catalog.py
  [active_run_dir]` re-crawls + prints.
- `backend/training/trainer.py` — SheepRL subprocess orchestration + catalog
  integration (session rows on launch, snapshot rows on checkpoint log lines,
  crashed/ended on exit) + graceful `stop(graceful=True)` via control channel
  (`training-state/control/<session>/checkpoint-request` → ack JSON) + stable
  replay override (`buffer.memmap_dir`).
- `sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` — vendored-loop patches:
  suspend hook at the checkpoint block (~line 765); `cfg.buffer.memmap_dir`.
- `backend/api/routes.py` — /training/{start,stop,suspend,switch}, /workspaces,
  /advisor/model_size, /games/import (multipart ROM), plus pre-existing routes.
- `backend/tools.py` — job manager + /api/tools/{reward_probe,ram_capture,
  ram_diff,build_state,run_walker,record_episode,jobs}. Scripts in `sheeprl/`
  (`_retro_tool_probe.py`, `_retro_ram_diff.py`, `_retro_build_state.py`,
  `_retro_walker.py`; last stdout line = `RESULT <json>`).
- `backend/copilot.py` + `backend/copilot_primer.md` — headless claude-local
  (Qwen 3.6 27B) session; /api/copilot/{start,send,events,stop}.
- `.claude/skills/{onboard-a-game,audit-a-reward,diagnose-a-run}` — copilot skills.
- `scripts/overnight_watchdog.py` — auto-resume (reads
  training-state/last_start_request.json) + hourly ckpt hard-link archive.
- Frontend: `GameSelector` (workspace status + New Game ROM modal),
  `TrainingControls` (⇄ Switch + VRAM advisor hint), `LivePlay` (metadata-driven
  tracks), `GameConfigEditor` (🧪 Validate = reward_probe), `CopilotPanel`.

## Verified acceptance (reproduce these before trusting a refactor)

1. Zero-loss suspend/resume: POST /api/training/suspend → ack {path, step};
   POST /training/start (no fresh_start) → resumes from EXACTLY that step.
2. Switch round trip: switch to FZero-Test (fresh brain), switch back —
   F-Zero resumes its own head; catalog heads never cross games.
3. Tool chain: POST /api/tools/reward_probe {"game_id":"FZero-Snes",
   "states":["go"],"steps":150,"actions":"0"} → poll jobs → result.ok true,
   deviation 0.0.
4. Copilot smoke: /api/copilot/start, send "curl the workspaces endpoint and
   summarize" → autonomous Bash curl + correct one-sentence answer (~25s).

## Known warts (builder-confessed)

1. **Tool jobs are in-memory only** — `backend/tools.py` `_jobs` dict dies with
   the server; docstring mentions jobs.json persistence that was NEVER
   implemented. Logs survive on disk (training-state/tools/<id>/output.log).
2. **Copilot events are poll-based** (1s) not SSE/WS; ring buffer capped at
   500 events; `raw` arg of `_emit` unused. Session doesn't survive server
   restart and there's no --resume support yet (StartReq.resume is a stub).
3. **last_start_request.json is overwritten by every start** — replaying it
   blindly "to switch back" restarts the CURRENT game (bit the builder once).
   The watchdog uses it correctly (resume-what-was-running) but a
   game-switch immediately followed by a crash resumes the NEW game — correct
   but worth knowing.
4. **Snapshot rows for rolled-off checkpoints linger** (existence check hides
   them; nothing deletes rows). No retention policy anywhere yet; the
   overnight archive (keep 8) is the only eval-bundle mechanism. The
   three-tier snapshot taxonomy (resume/eval/archive) exists in schema but
   only 'resume' rows are ever written.
5. **Lineage naming**: retroactive lineages are named after their root run
   dirname (ugly); only the active chain gets 'main'. Forking/naming UI is
   Phase E. `compatibility_hash` column exists but is never populated —
   fingerprints are designed, not implemented.
6. **TrainingControls model-size dropdown doesn't reflect the RUNNING
   lineage's architecture** — it's an input, and arch is locked-on-resume
   anyway (resume merges old config; locked keys only WARN in cli.py).
   Selecting 'small' and resuming an XL lineage resumes XL with a warning.
7. **games list**: 'retro-dreamer' junk game (M0-era runs) kept in catalog
   because its old checkpoints still exist on disk. FZero-Test is deliberate.
   Failed-launch run dirs (2026-07-10_10-09*, Airstriker) still on disk under
   sheeprl/logs — rows cleaned, dirs not.
8. **RAM workbench has no UI** — tools only (ram_capture/ram_diff via curl or
   copilot). The record_episode tool ignores its game_id param (the recorder
   infers game from the checkpoint's run config).
   ALSO: no "attach ROM to a built-in integration" flow — /games/import only
   creates custom workspaces, and the games list doesn't show which of the
   1006 built-ins are ROM-ready (user hit this 2026-07-10; worked around
   with a bulk `python -m retro.import <dir>` — 207 NES games now ROM-ready).
   Needed: ROM-ready badge in the list + per-game attach-ROM,
   and /games/import should sanitize game_id (no validation today).
9. **The suspend ack path is relative** (child cwd) — absolutized when the
   trainer registers it; anything else consuming the raw ack must handle that.
10. **FZero-Test buffer** lives at training-state/games/FZero-Test/... (~debug
    size). F-Zero's 5.8GB replay was MOVED to its stable home with a symlink
    left at the old 21-42-51 run-dir path — do not "clean up" that symlink;
    restored buffers resolve through it.
11. **Copilot runs --dangerously-skip-permissions** on a local model with
    full Bash. Fine for this box, unacceptable for the product — sandboxing
    is Phase E work.
12. **Monitor/watchdog interplay**: intentional switches/suspends emit
    TERMINAL/idle noise events on the health monitor; the watchdog is killed+
    relaunched manually around server bounces (pkill self-match footgun: never
    `pkill -f` a pattern that appears in your own command line).
    CONFIRMED IN THE WILD 2026-07-10: the user clicked the UI Stop button and
    the watchdog auto-resumed training ~2min later — it cannot distinguish a
    user-initiated stop from a crash. Real fix: /training/stop should record
    stop intent (e.g. a stopped-by-user flag in status or a marker file) and
    the watchdog must honor it; until then the watchdog is manually disarmed
    whenever a human is driving the UI.

## Current live state (2026-07-10 ~11:00)

F-Zero XL, lineage main, ~738k steps, 5-track practice rotation + heal reward.
Suspended/resumed 6+ times today with zero loss. Watchdog + 30-min monitor
armed. FZero-Test parked @ 12,138. GPU: training only; all tools CPU-pinned.
