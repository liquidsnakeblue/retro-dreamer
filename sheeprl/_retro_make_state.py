"""Generate a mid-level practice savestate by playing the newest checkpoint
until Mario reaches a target x position, then dumping the emulator state.

Usage:
  python _retro_make_state.py <game_dir> <base_state> <target_x> <out_name> [max_episodes]

Example (8-1 long-gap practice state):
  python _retro_make_state.py games/SuperMarioBros-Nes-v0 Level8-1 3600 Level8-1-gap

The saved state drops into <game_dir>/states/<out_name>.state and can be
used directly (initial_state=<out_name>) or in a rotation
(initial_state="Level8-1+Level8-1-gap").

SMB-specific: position = playerPage(0x6D)*256 + playerX(0x86). Runs on CPU
(same as _retro_record.py) so it never contends with a live training run.
"""

import gzip
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "1")

import pyglet

pyglet.options["shadow_window"] = False

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

_original_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = _patched_torch_load

from lightning.fabric import Fabric  # noqa: E402

SHEEPRL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SHEEPRL_DIR))

from sheeprl.algos.dreamer_v3.agent import build_agent  # noqa: E402
from sheeprl.algos.dreamer_v3.utils import prepare_obs  # noqa: E402
from sheeprl.envs.retro_dreamer import RetroDreamerWrapper  # noqa: E402
from sheeprl.utils.env import make_env  # noqa: E402
from sheeprl.utils.utils import dotdict  # noqa: E402


def find_latest_checkpoint() -> Path:
    sys.path.insert(0, str(SHEEPRL_DIR.parent))
    from backend import catalog

    con = catalog.connect()
    game_id = Path(sys.argv[1]).name
    head = catalog.get_resumable_head(con, game_id)
    con.close()
    if head:
        return Path(head["checkpoint_path"])
    raise SystemExit(f"no resumable head for {game_id}")


def main():
    game_dir = Path(sys.argv[1]).resolve()
    base_state = sys.argv[2]
    target_x = int(sys.argv[3])
    out_name = sys.argv[4]
    max_episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 5

    ckpt_path = find_latest_checkpoint()
    cfg = dotdict(
        OmegaConf.to_container(
            OmegaConf.load(ckpt_path.parent.parent / "config.yaml"), resolve=True
        )
    )
    cfg.env.num_envs = 1
    cfg.env.capture_video = False
    cfg.env.wrapper.initial_state = base_state

    torch.set_num_threads(6)
    fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
    fabric.launch()
    state = fabric.load(str(ckpt_path))

    env = make_env(cfg, cfg.seed, 0, str(SHEEPRL_DIR / "logs" / "watch"), "mkstate")()
    action_space = env.action_space
    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape
        if is_continuous
        else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    _, _, _, _, player = build_agent(
        fabric, actions_dim, is_continuous, cfg,
        env.observation_space, state["world_model"], state["actor"],
    )

    inner = env
    while not isinstance(inner, RetroDreamerWrapper):
        inner = inner.env
    retro_env = inner._env.unwrapped

    print(f"MAKE-STATE ckpt={ckpt_path.name} base={base_state} target_x={target_x} -> {out_name}.state", flush=True)
    t0 = time.perf_counter()
    for ep in range(max_episodes):
        obs = env.reset(seed=cfg.seed + ep)[0]
        player.num_envs = 1
        player.init_states()
        done, step, best_x = False, 0, 0
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
            obs, _, terminated, truncated, _ = env.step(
                real_actions.reshape(env.action_space.shape)
            )
            done = terminated or truncated
            step += 1
            ram = retro_env.get_ram()
            x = int(ram[0x6D]) * 256 + int(ram[0x86])
            best_x = max(best_x, x)
            if x >= target_x:
                out = game_dir / "states" / f"{out_name}.state"
                out.write_bytes(gzip.compress(retro_env.em.get_state()))
                print(f"SAVED {out} at x={x} (episode {ep + 1}, step {step}, "
                      f"{time.perf_counter() - t0:.0f}s)", flush=True)
                env.close()
                return 0
        print(f"episode {ep + 1}: died, best_x={best_x}", flush=True)
    env.close()
    print(f"FAILED: never reached x={target_x} in {max_episodes} episodes", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
