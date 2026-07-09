# Vendored SheepRL patches

The `sheeprl/` tree is vendored (0.5.8.dev, frozen ~mid-2024, written for gymnasium 0.29).
Runtime is gymnasium 1.2.3 / numpy 2.4 / torch 2.10. Every divergence from the vendored
state is recorded here.

## 2026-07-09 — gymnasium 1.x compatibility

1. **`sheeprl/utils/env.py` — TransformObservation signature.**
   gymnasium ≥ 1.0 requires the transformed `observation_space` as a constructor arg
   (0.29 allowed post-hoc mutation). We now build the Dict space with the resized CNN-key
   Boxes up front and pass it in.

2. **`sheeprl/utils/env.py` — RecordVideo.**
   `gym.experimental.wrappers.RecordVideoV0` was removed in gymnasium 1.0; replaced with
   `gym.wrappers.RecordVideo` (same args). Dropped the manual `render_fps` metadata poke
   (`frames_per_sec` no longer exists; RecordVideo reads `env.metadata["render_fps"]`).

3. **`sheeprl/algos/dreamer_v3/dreamer_v3.py` — vector-env autoreset semantics.**
   gymnasium ≥ 1.0 defaults to NEXT_STEP autoreset and removed
   `infos["final_observation"]`. SheepRL's collection loop assumes 0.29 same-step resets.
   Patch: construct Sync/AsyncVectorEnv with `autoreset_mode=AutoresetMode.SAME_STEP`
   and read `infos["final_obs"]` (1.x name) with fallback to `final_observation`.
   Episode stats (`info["episode"]["r"]`) are scalars in 1.x (arrays in 0.29) —
   wrapped in `np.ravel` before indexing.

4. **`backend/training/trainer.py` — torch.load monkey-patch (template for `_retro_run.py`).**
   torch ≥ 2.6 defaults `weights_only=True` and lightning passes it explicitly; our
   checkpoints contain pickled sheeprl buffer objects. The wrapper now force-overrides
   `kwargs['weights_only'] = False` (we only ever load our own checkpoints). The old
   lambda both duplicated the kwarg (crash on resume) and predates the lightning change.

## Studio-side fixes (same session, backend/ not sheeprl/)

- `trainer.py`: **child stdout block-buffering** — the root cause of the studio "going
  blind" mid-run. The subprocess's stdout is a pipe, so Python block-buffers it (~8KB);
  once the agent stops dying, episode-end prints become sparse (minutes-to-hours apart)
  and sit unflushed — status/logs freeze in a way indistinguishable from a hang. Fix:
  `PYTHONUNBUFFERED=1` in the child env. (The reader-thread hardening committed just
  before this was good hygiene but was NOT the actual cause; evidence: checkpoint files
  appeared on disk while their "Saving checkpoint" lines never crossed the pipe.)

- `trainer.py`: `SHEEPRL_DIR` pointed at the inner package dir — `import sheeprl` could
  never resolve; the studio had never successfully launched a subprocess on this machine.
- `trainer.py`: launch command now pins `root_dir=dreamer_v3/<game_id>` (SheepRL defaults
  to `${algo.name}/${env.id}` = "dreamer_v3/retro-dreamer"), so checkpoint/video/TB
  discovery in the studio matches where SheepRL actually writes.
- `server.py`: port configurable via `RETRO_DREAMER_PORT` (8080 collides with another
  local service on this box; verified E2E on 8091).

5. **`sheeprl/utils/memmap.py` — unpickling scheduled deletion of live buffer files.**
   `MemmapArray.__setstate__` wrapped the buffer's backing file in
   `_TemporaryFileWrapper(delete=True)`, so ANY process that unpickled a checkpoint
   (eval, inspection) deleted the replay-buffer memmaps on exit — which crashed a LIVE
   training run writing to those same files (checkpoint chains carry absolute paths).
   Fixed to `delete=False`; `__getstate__` already strips ownership. This was also the
   source of the ubiquitous benign-looking `NoneType has no attribute 'close'` tempfile
   spam. **Rule regardless: prefer eval against a buffer-stripped checkpoint copy.**

6. **`sheeprl/cli.py` — resume clobbers `buffer.checkpoint` CLI override.**
   `resume_from_checkpoint()` merges the old run's config over the CLI config except a
   whitelist; added `buffer.checkpoint` to the whitelist so "resume without restoring
   the buffer" (recovery path) is expressible. Paired with a `"rb" in state` guard in
   `dreamer_v3.py` for buffer-stripped checkpoints, and `resume_prefill` in the studio
   API (`buffer.checkpoint=false` + `algo.learning_starts=N` on resume).

7. **`sheeprl/utils/env.py` — video recording starves on long episodes.**
   `RecordVideo` was episode-trigger-only (every 10 episodes on env 0). Once the agent
   stops dying, episodes run to the 10k-step TimeLimit (~50 min wall each on env 0) →
   next video ~8 h away; the dashboard's Episode Replays froze for hours. Added a
   `step_trigger` (every `env.video_step_freq` env-0 steps, default 10k) alongside the
   episode trigger — gymnasium runs both; whichever fires first starts a recording.
   ⚠️ OmegaConf gotcha: missing keys return **None** instead of raising, so
   `getattr(cfg.env, "key", default)` NEVER uses the default — must `or`-fallback.
   The first version did `step % None` → worker crash masked by two more bugs (below).

8. **`sheeprl/envs/wrappers.py` — RestartOnException masks crashes whose message
   contains `%`.** `gym.logger.warn(msg)` treats msg as a %-format string; an exception
   message containing `%` (e.g. `unsupported operand type(s) for %: ...`) raises
   `TypeError: not enough arguments for format string` INSIDE the crash handler,
   killing the async worker and hiding the real error. Escaped `%` → `%%` in both
   handlers. Debug recipe that surfaced it: rerun with `env.sync_env=true
   env.num_envs=1` so exceptions propagate instead of dying in Worker-N.

9. **`sheeprl/utils/memmap.py` — silence exit-time tempfile spam from patch 5.**
   The `delete=False` wrapper holds `file=None`; interpreter-exit cleanup called
   `None.close()` → "Exception ignored" tracebacks polluting every eval. Set
   `_closer.close_called = True` (nothing to close; deletion semantics unchanged).

Studio-side fix in the same session: `trainer.list_videos()` now scans ALL run dirs
(replays survive resume, which starts a new run dir) and returns a unique `id`
(run-relative path) per video plus `source` (train/eval) — bare filenames collide
because every eval writes its own `rl-video-episode-0.mp4`, which made the dashboard
serve the wrong video on click. `/api/videos/{id:path}` serves by id.

10. **`sheeprl/cli.py` — resume also clobbers `env.wrapper.initial_state`.**
    Added to the resume whitelist so a resume can start from a different save
    state (fine-tuning the practice-mode brain on the Grand Prix state;
    save-state curricula later). Without it the CLI override silently reverted
    to the old run's state.

GP era notes (same session): `games/FZero-Snes/states/gp_knight_beginner.state`
recorded via scripted menu navigation from power-on (`_retro_gp_probe.py` runs a
checkpoint on it for one CPU episode). training.json done threshold health<2048 →
health<100: 2048 is FULL health, so the old rule ended the episode on any contact —
fine on an empty practice track (it taught perfect wall avoidance), fatal in a
19-rival pack (probe: dead in 103 steps at health 2024). <100 matches the game's
own power-empty retire; the health penalty still prices contact.

## Known-not-patched (watchlist)

- Other algos (p2e, dreamer_v1/v2, PPO, SAC…) still carry 0.29 assumptions — we only
  maintain the `dreamer_v3` path.
- `sheeprl_eval.py` path not yet exercised against gymnasium 1.x (M0 scope).
- `_retro_run.py` (repo root of sheeprl/) monkey-patches `torch.load(weights_only=False)`
  for lightning checkpoint loads under torch ≥ 2.6 defaults.
