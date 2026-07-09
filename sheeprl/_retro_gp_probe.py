"""Run a trained checkpoint on an alternate save state (default: the GP
Knight League state) for ONE greedy episode on CPU, logging env vars every
step. Answers "what actually happens" before committing the GPU to a
fine-tune: does the agent survive the starting grid, when does done fire,
what does health do in traffic.

Usage:
  python _retro_gp_probe.py <checkpoint.ckpt> [initial_state] [out_dir]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import pyglet

pyglet.options["shadow_window"] = False

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

ckpt_path = Path(sys.argv[1]).resolve()
initial_state = sys.argv[2] if len(sys.argv) > 2 else "gp_knight_beginner"
out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("logs/gp_probe")
out_dir.mkdir(parents=True, exist_ok=True)

cfg = dotdict(
    OmegaConf.to_container(
        OmegaConf.load(ckpt_path.parent.parent / "config.yaml"), resolve=True
    )
)
cfg.env.num_envs = 1
cfg.env.capture_video = True
cfg.env.wrapper.initial_state = initial_state

fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
fabric.launch()
state = fabric.load(str(ckpt_path))

env = make_env(cfg, cfg.seed, 0, str(out_dir), "gp_probe", vector_env_idx=0)()
if not isinstance(env.observation_space, gym.spaces.Dict):
    raise RuntimeError(f"Unexpected observation space {env.observation_space}")
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

print(f"PROBE ckpt={ckpt_path.name} state={initial_state}", flush=True)
obs = env.reset(seed=cfg.seed)[0]
player.num_envs = 1
player.init_states()
done, step, cum_rew = False, 0, 0.0
while not done and step < 12000:
    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
    real_actions = player.get_actions(
        torch_obs, True, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
    )
    real_actions = (
        torch.stack(real_actions, -1).cpu().numpy()
        if player.actor.is_continuous
        else torch.stack([a.argmax(dim=-1) for a in real_actions], -1).cpu().numpy()
    )
    obs, reward, terminated, truncated, info = env.step(
        real_actions.reshape(env.action_space.shape)
    )
    cum_rew += reward
    step += 1
    if step % 500 == 0 or terminated or truncated:
        print(
            f"step={step} health={info.get('health')} pos={info.get('pos')} "
            f"speed={info.get('speed')} rew={cum_rew:.1f} "
            f"term={terminated} trunc={truncated}",
            flush=True,
        )
    done = terminated or truncated
print(f"EPISODE END step={step} cum_rew={cum_rew:.1f}", flush=True)
env.close()
