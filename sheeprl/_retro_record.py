"""Record the newest checkpoint playing one episode to a normal MP4 —
real 60fps video, real emulator sound, 5x crisp upscale — as fast as the
machine can generate it (no real-time pacing). The result is a static
file: playback in the browser is flawless, seekable, replayable.

Writes progress to <out>.progress.json so a UI can show percent done.

Usage:
  python _retro_record.py <checkpoint|latest> <media_seconds|full> <out.mp4> [initial_state]
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
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
from sheeprl.envs.retro_dreamer import RetroDreamerWrapper
from sheeprl.utils.env import make_env
from sheeprl.utils.utils import dotdict

SHEEPRL_DIR = Path(__file__).parent


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def find_latest_checkpoint() -> Path:
    ckpts = sorted(
        SHEEPRL_DIR.glob("logs/runs/dreamer_v3/*/*/version_*/checkpoint/*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise SystemExit("No checkpoints found")
    return ckpts[-1]


ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else "latest"
length_arg = sys.argv[2] if len(sys.argv) > 2 else "60"
out_path = Path(sys.argv[3] if len(sys.argv) > 3 else "recording.mp4").resolve()
initial_state = sys.argv[4] if len(sys.argv) > 4 else None

ckpt_path = find_latest_checkpoint() if ckpt_arg == "latest" else Path(ckpt_arg).resolve()
progress_path = out_path.with_suffix(".progress.json")
FPS = 60
# "full" = run to episode end (TimeLimit caps it); otherwise media seconds
max_frames = None if length_arg == "full" else int(float(length_arg) * FPS)


def write_progress(frames, done, error=None):
    total = max_frames or 10000 * 4  # full episode worst case (TimeLimit)
    progress_path.write_text(
        json.dumps(
            {
                "frames": frames,
                "target_frames": total,
                "percent": min(100, round(100 * frames / total)),
                "done": done,
                "ckpt": ckpt_path.name,
                "error": error,
            }
        )
    )


write_progress(0, False)

cfg = dotdict(
    OmegaConf.to_container(
        OmegaConf.load(ckpt_path.parent.parent / "config.yaml"), resolve=True
    )
)
cfg.env.num_envs = 1
cfg.env.capture_video = False
if initial_state:
    cfg.env.wrapper.initial_state = initial_state

torch.set_num_threads(6)
fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
fabric.launch()
state = fabric.load(str(ckpt_path))

env = make_env(cfg, cfg.seed, 0, str(SHEEPRL_DIR / "logs" / "watch"), "record")()
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

inner = env
while not isinstance(inner, RetroDreamerWrapper):
    inner = inner.env
retro_env = inner._env
audio_rate = int(retro_env.em.get_audio_rate())
h, w = retro_env.observation_space.shape[:2]

v_r, v_w = os.pipe()
a_r, a_w = os.pipe()
tmp_out = out_path.with_suffix(".tmp.mp4")
ffmpeg = subprocess.Popen(
    [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", f"pipe:{v_r}",
        "-f", "s16le", "-ar", str(audio_rate), "-ac", "2", "-i", f"pipe:{a_r}",
        "-vf", f"scale={w * 5}:{h * 5}:flags=neighbor",
        # a file, not a live stream: better compression, browser-seekable
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(tmp_out),
    ],
    stdin=subprocess.DEVNULL,
    pass_fds=(v_r, a_r),
)
os.close(v_r)
os.close(a_r)

_qv: "queue.Queue[bytes | None]" = queue.Queue(maxsize=600)
_qa: "queue.Queue[bytes | None]" = queue.Queue(maxsize=600)


def _writer(fd, q):
    f = os.fdopen(fd, "wb", buffering=0)
    try:
        while True:
            b = q.get()
            if b is None:
                break
            f.write(b)
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            f.close()
        except Exception:
            pass


threading.Thread(target=_writer, args=(v_w, _qv), daemon=True).start()
threading.Thread(target=_writer, args=(a_w, _qa), daemon=True).start()

frames = 0


def on_frame(frame, audio):
    global frames
    _qv.put(frame.tobytes())
    _qa.put(audio.tobytes())
    frames += 1


inner.frame_callback = on_frame

log(f"RECORD ckpt={ckpt_path.name} state={cfg.env.wrapper.initial_state} "
    f"len={length_arg} -> {out_path.name}")

t0 = time.perf_counter()
try:
    obs = env.reset(seed=cfg.seed)[0]
    player.num_envs = 1
    player.init_states()
    done, step, cum_rew = False, 0, 0.0
    while not done and (max_frames is None or frames < max_frames):
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
        if step % 50 == 0:
            write_progress(frames, False)
    _qv.put(None)
    _qa.put(None)
    ffmpeg.wait(timeout=120)
    tmp_out.rename(out_path)
    write_progress(frames, True)
    gen_speed = (frames / FPS) / (time.perf_counter() - t0)
    log(f"done: {frames} frames ({frames / FPS:.1f}s media) reward={cum_rew:.1f} "
        f"gen_speed={gen_speed:.2f}x realtime")
except Exception as e:  # noqa: BLE001
    write_progress(frames, True, error=str(e))
    raise
finally:
    try:
        ffmpeg.terminate()
    except Exception:
        pass
    env.close()
