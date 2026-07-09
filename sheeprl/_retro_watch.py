"""Watch a trained agent play live in a real game window (WSLg/X11).

Loads a checkpoint, opens the emulator window, and the agent drives with
the controls — like loading up the game and handing it the pad. Runs on
CPU so it never disturbs a training run. Ctrl-C to stop; plays episodes
back-to-back until then.

Usage:
  python _retro_watch.py <checkpoint.ckpt> [initial_state]

Defaults to the newest checkpoint across all runs when no path is given:
  python _retro_watch.py latest [initial_state]
"""
import sys
import time
from pathlib import Path

import torch

_original_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_torch_load

import gymnasium as gym
from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import prepare_obs
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict

SHEEPRL_DIR = Path(__file__).parent


def find_latest_checkpoint() -> Path:
    ckpts = sorted(
        SHEEPRL_DIR.glob("logs/runs/dreamer_v3/*/*/version_*/checkpoint/*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise SystemExit("No checkpoints found under logs/runs/dreamer_v3/")
    return ckpts[-1]


arg = sys.argv[1] if len(sys.argv) > 1 else "latest"
ckpt_path = find_latest_checkpoint() if arg == "latest" else Path(arg).resolve()
initial_state = sys.argv[2] if len(sys.argv) > 2 else None

cfg = dotdict(
    OmegaConf.to_container(
        OmegaConf.load(ckpt_path.parent.parent / "config.yaml"), resolve=True
    )
)
cfg.env.num_envs = 1
cfg.env.capture_video = False
cfg.env.wrapper.render_mode = "human"
if initial_state:
    cfg.env.wrapper.initial_state = initial_state

fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
fabric.launch()
state = fabric.load(str(ckpt_path))

env = make_env(cfg, cfg.seed, 0, str(SHEEPRL_DIR / "logs" / "watch"), "watch")()
action_space = env.action_space
is_continuous = isinstance(action_space, gym.spaces.Box)
is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
actions_dim = tuple(
    action_space.shape
    if is_continuous
    else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
)
_, _, _, _, player = build_agent(
    fabric,
    actions_dim,
    is_continuous,
    cfg,
    env.observation_space,
    state["world_model"],
    state["actor"],
)

frame_skip = int(getattr(cfg.env.wrapper, "frame_skip", 4) or 4)
target_dt = frame_skip / 60.0  # real-time pacing: one action = frame_skip SNES frames

print(f"WATCH ckpt={ckpt_path.name} state={cfg.env.wrapper.initial_state}", flush=True)
print("Window open — Ctrl-C to stop.", flush=True)
try:
    episode = 0
    while True:
        obs = env.reset(seed=cfg.seed + episode)[0]
        player.num_envs = 1
        player.init_states()
        done, step, cum_rew = False, 0, 0.0
        t_last = time.perf_counter()
        while not done:
            torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
            real_actions = player.get_actions(
                torch_obs, True, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
            )
            real_actions = (
                torch.stack(real_actions, -1).cpu().numpy()
                if player.actor.is_continuous
                else torch.stack([a.argmax(dim=-1) for a in real_actions], -1).cpu().numpy()
            )
            obs, reward, terminated, truncated, _ = env.step(
                real_actions.reshape(env.action_space.shape)
            )
            cum_rew += reward
            step += 1
            done = terminated or truncated
            # pace to real time (no-op when inference is the bottleneck)
            now = time.perf_counter()
            sleep_for = target_dt - (now - t_last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            t_last = time.perf_counter()
        episode += 1
        print(f"episode {episode}: steps={step} reward={cum_rew:.1f}", flush=True)
except KeyboardInterrupt:
    print("\nstopped.", flush=True)
finally:
    env.close()
