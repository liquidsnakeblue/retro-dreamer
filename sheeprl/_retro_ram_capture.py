"""Replay a checkpoint deterministically while capturing FULL console RAM
every step — the raw material for dynamic RAM diffing (finding game-state
addresses like "race over" empirically instead of guessing).

Wrapper done-conditions are disabled during capture so the run continues
past the moment we want to study (e.g. the YOU LOST screen).

Saves a compressed .npz with:
  ram      (T, N) uint8 — full RAM per step
  offsets  block base addresses (retro address space)
  sizes    block lengths (flat index -> address via these)
  health/pos/speed/reverse (T,) — env vars for locating events

Usage (checkpoint mode — drives a TRAINED brain):
  python _retro_ram_capture.py <checkpoint.ckpt> <initial_state> <max_steps> <out.npz>

Usage (random mode — NO checkpoint needed; drives uniform-random actions, so a
game with no trained brain can still be captured for config validation):
  python _retro_ram_capture.py <checkpoint.ckpt|random> <initial_state> <max_steps> <out.npz> random <template_config.yaml> <game_id> <game_dir>

  `random` may be passed as argv[1] (sentinel) OR as argv[5] (flag); the
  remaining game/template args follow. Random mode builds cfg from a TEMPLATE
  config.yaml (any existing run's config works — env keys are overridden by the
  explicit game_id/game_dir/initial_state) instead of from the checkpoint's run
  dir, and skips agent/fabric construction entirely. Checkpoint-mode behavior is
  byte-for-byte unchanged when the flag is absent.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import pyglet

pyglet.options["shadow_window"] = False

import numpy as np
import torch

_original_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_torch_load

from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import prepare_obs
from sheeprl.envs.retro_dreamer import RetroDreamerWrapper
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict

# --- argv parsing: additive random mode (sentinel argv[1]=='random' or flag argv[5]) ---
_random_mode = (len(sys.argv) > 1 and sys.argv[1] == "random")
if _random_mode:
    # argv: random <initial_state> <max_steps> <out.npz> <template_config.yaml> <game_id> <game_dir>
    initial_state = sys.argv[2]
    max_steps = int(sys.argv[3])
    out_path = Path(sys.argv[4]).resolve()
    template_cfg_path = Path(sys.argv[5]).resolve()
    _game_id = sys.argv[6]
    _game_dir = Path(sys.argv[7]).resolve()
    ckpt_path = None  # no brain in random mode
else:
    ckpt_path = Path(sys.argv[1]).resolve()
    initial_state = sys.argv[2]
    max_steps = int(sys.argv[3])
    out_path = Path(sys.argv[4]).resolve()

if ckpt_path is not None:
    cfg = dotdict(
        OmegaConf.to_container(
            OmegaConf.load(ckpt_path.parent.parent / "config.yaml"), resolve=True
        )
    )
else:
    # Random mode: cfg from a TEMPLATE config (any run's config.yaml), overridden
    # by the explicit game_id/game_dir so no run dir for the target game is
    # required. Only the env.* keys used by make_env need to be sane.
    cfg = dotdict(
        OmegaConf.to_container(
            OmegaConf.load(template_cfg_path), resolve=True
        )
    )
    cfg.setdefault("seed", 42)
    cfg.env.wrapper.game_id = _game_id
    cfg.env.wrapper.game_dir = str(_game_dir)
    # Random capture is an authoring workflow for the current workspace, not
    # checkpoint evaluation. Never inherit an unrelated template manifest.
    cfg.env.wrapper.action_manifest = ""
    cfg.env.wrapper.action_manifest_hash = ""
    cfg.env.wrapper.allow_mutable_actions = True

cfg.env.num_envs = 1
cfg.env.capture_video = False
cfg.env.wrapper.initial_state = initial_state

torch.set_num_threads(6)

if ckpt_path is not None:
    fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
    fabric.launch()
    state = fabric.load(str(ckpt_path))

env = make_env(cfg, cfg.seed, 0, "logs/ram_capture", "capture")()

if ckpt_path is not None:
    action_space = env.action_space
    actions_dim = tuple(
        action_space.nvec.tolist()
        if hasattr(action_space, "nvec")
        else [action_space.n]
    )
    _, _, _, _, player = build_agent(
        fabric, actions_dim, False, cfg, env.observation_space,
        state["world_model"], state["actor"],
    )
else:
    # Random driver: uniform sample over the MultiBinary action space each step.
    action_space = env.action_space
    _act_shape = action_space.shape
    _act_dtype = action_space.dtype
    _rng = np.random.default_rng(cfg.seed)
    def player_act():
        return _rng.integers(0, 2, size=_act_shape, dtype=_act_dtype)

inner = env
while not isinstance(inner, RetroDreamerWrapper):
    inner = inner.env
retro_env = inner._env
# capture must outlive the buggy/incomplete done rules we're here to fix
inner.training_config = {**inner.training_config, "done": {"variables": {}}}

blocks = retro_env.data.memory.blocks
offsets = np.array(sorted(blocks), dtype=np.int64)
sizes = np.array([len(blocks[o]) for o in sorted(blocks)], dtype=np.int64)
n = int(sizes.sum())
print(f"RAM blocks: {[(hex(o), s) for o, s in zip(offsets, sizes)]} total={n}", flush=True)

ram = np.zeros((max_steps, n), dtype=np.uint8)
vars_log = None  # keyed off the game's actual info vars at first step

obs = env.reset(seed=cfg.seed)[0]
if ckpt_path is not None:
    player.num_envs = 1
    player.init_states()
t0 = time.perf_counter()
steps_done = 0
for step in range(max_steps):
    if ckpt_path is not None:
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        real_actions = player.get_actions(
            torch_obs, True, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        )
        real_actions = torch.stack([a.argmax(dim=-1) for a in real_actions], -1).cpu().numpy()
    else:
        real_actions = player_act()
    obs, reward, terminated, truncated, info = env.step(
        real_actions.reshape(env.action_space.shape)
    )
    ram[step] = retro_env.get_ram()
    if vars_log is None:
        vars_log = {
            k: np.zeros(max_steps, dtype=np.float64)
            for k, v in info.items() if isinstance(v, (int, float))
        }
    for k in vars_log:
        vars_log[k][step] = info.get(k, np.nan)
    steps_done = step + 1
    if steps_done % 500 == 0:
        print(f"step={steps_done} vars={ {k: info.get(k) for k in list(vars_log)[:5]} } "
              f"({steps_done / (time.perf_counter() - t0):.0f} steps/s)",
              flush=True)
    if terminated or truncated:
        print(f"env ended early at step {steps_done} term={terminated} trunc={truncated}", flush=True)
        break

np.savez_compressed(
    out_path,
    ram=ram[:steps_done],
    offsets=offsets,
    sizes=sizes,
    ckpt=("random" if ckpt_path is None else str(ckpt_path.name)),
    state=initial_state,
    **{k: v[:steps_done] for k, v in (vars_log or {}).items()},
)
print(f"saved {out_path} steps={steps_done}", flush=True)
env.close()
import json as _json

print("RESULT " + _json.dumps({
    "npz": str(out_path), "steps": int(steps_done),
    "vars_logged": sorted(vars_log) if vars_log else [],
}), flush=True)
