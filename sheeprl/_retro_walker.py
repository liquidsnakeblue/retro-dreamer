"""Generalized 'agent earns the states' walker (productized league_walker).

Drive a checkpoint's greedy policy from a start state until the episode
really ends (a chosen RAM flag leaves its live value), then walk the
between-levels screens with scripted button taps until the flag goes live
again, snapshot that moment as a new save state, and repeat — chaining
through a game's progression using nothing but the agent's own skill.

Failure detection: if the wrapper's done fires while the live-flag still
reads live (e.g. health rule mid-level), that attempt FAILED — retry with a
new seed up to max_retries.

Last stdout line: RESULT {captures: [{name, path, screenshot, vars}...]}

Usage:
  python _retro_walker.py <game_id> <game_dir> <ckpt|head> <start_state> \
      <n_captures> <workdir> [--flag race_on] [--live 1] [--tap START] \
      [--max-drive 6000] [--max-menu 10800] [--retries 4] [--prefix capture]
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "1")
import pyglet

pyglet.options["shadow_window"] = False

import torch

_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
torch.set_num_threads(6)

import numpy as np
from PIL import Image
import gymnasium as gym
from lightning import Fabric
from omegaconf import OmegaConf

SHEEPRL_DIR = Path(__file__).parent
sys.path.insert(0, str(SHEEPRL_DIR))
sys.path.insert(0, str(SHEEPRL_DIR.parent))
os.chdir(SHEEPRL_DIR)

from sheeprl.algos.dreamer_v3.agent import build_agent
from sheeprl.algos.dreamer_v3.utils import prepare_obs
from sheeprl.envs.retro_dreamer import RetroDreamerWrapper
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict

ap = argparse.ArgumentParser()
ap.add_argument("game_id")
ap.add_argument("game_dir")
ap.add_argument("ckpt")
ap.add_argument("start_state")
ap.add_argument("n_captures", type=int)
ap.add_argument("workdir")
ap.add_argument("--flag", default="race_on")
ap.add_argument("--live", type=float, default=1)
ap.add_argument("--tap", default="START")
ap.add_argument("--max-drive", type=int, default=6000)
ap.add_argument("--max-menu", type=int, default=10800)
ap.add_argument("--retries", type=int, default=4)
ap.add_argument("--prefix", default="capture")
args = ap.parse_args()

workdir = Path(args.workdir)
workdir.mkdir(parents=True, exist_ok=True)
states_dir = Path(args.game_dir) / "states"

if args.ckpt == "head":
    from backend import catalog as _catalog

    con = _catalog.connect()
    head = _catalog.get_resumable_head(con, args.game_id)
    con.close()
    if not head:
        raise SystemExit(f"no catalog head for {args.game_id}")
    ckpt_path = Path(head["checkpoint_path"])
else:
    ckpt_path = Path(args.ckpt).resolve()

cfg = dotdict(OmegaConf.to_container(
    OmegaConf.load(ckpt_path.parent.parent / "config.yaml"), resolve=True))
cfg.env.num_envs = 1
cfg.env.capture_video = False

fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
fabric.launch()
agent_state = fabric.load(str(ckpt_path))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


log(f"walker ckpt={ckpt_path.name} start={args.start_state} flag={args.flag} live={args.live}")

current_state = args.start_state
player = None
captures = []
capture_idx, retries = 1, 0
TAP_EVERY, TAP_HOLD = 45, 2

while capture_idx <= args.n_captures and retries <= args.retries:
    cfg.env.wrapper.initial_state = current_state
    env = make_env(cfg, cfg.seed + retries * 1000, 0, str(workdir / "envlogs"), "walker")()
    if player is None:
        actions_dim = tuple(
            env.action_space.nvec.tolist()
            if isinstance(env.action_space, gym.spaces.MultiDiscrete)
            else [env.action_space.n]
        )
        _, _, _, _, player = build_agent(
            fabric, actions_dim, False, cfg, env.observation_space,
            agent_state["world_model"], agent_state["actor"],
        )
    inner = env
    while not isinstance(inner, RetroDreamerWrapper):
        inner = inner.env
    retro_env = inner._env

    # Phase A: policy drives until done
    obs = env.reset(seed=cfg.seed + capture_idx + retries * 1000)[0]
    player.num_envs = 1
    player.init_states()
    done, step = False, 0
    while not done and step < args.max_drive:
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        acts = player.get_actions(
            torch_obs, True, {k: v for k, v in torch_obs.items() if k.startswith("mask")}
        )
        acts = torch.stack([a.argmax(dim=-1) for a in acts], -1).cpu().numpy()
        obs, reward, term, trunc, info = env.step(acts.reshape(env.action_space.shape))
        done = term or trunc
        step += 1
        if step % 500 == 0:
            log(f"  attempt {capture_idx}.{retries}: step={step} vars={{{args.flag}: {info.get(args.flag)}}}")
    end_vars = dict(retro_env.data.lookup_all())
    Image.fromarray(retro_env.img).save(workdir / f"attempt{capture_idx}_{retries}_end.png")
    log(f"  attempt over: steps={step} {args.flag}={end_vars.get(args.flag)}")
    if step >= args.max_drive:
        log("  BAIL: drive cap hit")
        env.close()
        break
    if end_vars.get(args.flag) == args.live:
        log("  FAILED mid-level (flag still live) — retrying")
        env.close()
        retries += 1
        continue

    # Phase B: raw-emulator tap walk until the flag goes live again
    buttons = retro_env.buttons
    tap_mask = np.zeros(len(buttons), dtype=np.uint8)
    tap_mask[buttons.index(args.tap)] = 1
    noop = np.zeros(len(buttons), dtype=np.uint8)
    frame, captured, off_streak = 0, False, 0
    while frame < args.max_menu:
        retro_env.em.set_button_mask(tap_mask if (frame % TAP_EVERY) < TAP_HOLD else noop, 0)
        retro_env.em.step()
        retro_env.data.update_ram()
        frame += 1
        vars_now = retro_env.data.lookup_all()
        if vars_now.get(args.flag) != args.live:
            off_streak += 1
            continue
        if off_streak < 30:
            off_streak = 0
            continue
        name = f"{args.prefix}_{capture_idx}"
        raw = retro_env.em.get_state()
        for dest in (workdir / f"{name}.state", states_dir / f"{name}.state"):
            with gzip.open(dest, "wb") as fh:
                fh.write(raw)
        shot = workdir / f"{name}.png"
        Image.fromarray(retro_env.em.get_screen()).save(shot)
        log(f"  CAPTURED {name} at menu frame {frame}")
        captures.append({
            "name": name, "path": str(states_dir / f"{name}.state"),
            "screenshot": str(shot),
            "vars": {k: v for k, v in vars_now.items()},
        })
        current_state = name
        captured = True
        break
    env.close()
    if not captured:
        log("  BAIL: flag never re-lit during menu walk")
        break
    capture_idx += 1
    retries = 0

print("RESULT " + json.dumps({"captures": captures}), flush=True)
