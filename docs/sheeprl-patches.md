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

- `trainer.py`: `SHEEPRL_DIR` pointed at the inner package dir — `import sheeprl` could
  never resolve; the studio had never successfully launched a subprocess on this machine.
- `trainer.py`: launch command now pins `root_dir=dreamer_v3/<game_id>` (SheepRL defaults
  to `${algo.name}/${env.id}` = "dreamer_v3/retro-dreamer"), so checkpoint/video/TB
  discovery in the studio matches where SheepRL actually writes.
- `server.py`: port configurable via `RETRO_DREAMER_PORT` (8080 collides with another
  local service on this box; verified E2E on 8091).

## Known-not-patched (watchlist)

- Other algos (p2e, dreamer_v1/v2, PPO, SAC…) still carry 0.29 assumptions — we only
  maintain the `dreamer_v3` path.
- `sheeprl_eval.py` path not yet exercised against gymnasium 1.x (M0 scope).
- `_retro_run.py` (repo root of sheeprl/) monkey-patches `torch.load(weights_only=False)`
  for lightning checkpoint loads under torch ≥ 2.6 defaults.
