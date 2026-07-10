"""Live A/V player: the newest checkpoint plays the game, and an HLS live
stream (h264 + aac — real 60fps video with real emulator sound, 5x
nearest-neighbor upscaled) is written as segments to RETRO_LIVE_HLS_DIR.
The live sidecar serves the segment files; the browser plays them with
hls.js, which handles live buffering robustly (a raw progressive MP4
download into <video> stutters unpredictably in Chrome).

Usage:
  RETRO_LIVE_HLS_DIR=/tmp/retro-dreamer-live python _retro_live_player.py [checkpoint|latest] [initial_state]
"""
import os
import subprocess
import sys
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
    # Catalog first (studio-v2): the running session's lineage head — never a
    # different game's brain just because its file is newer. Legacy mtime scan
    # only as a fallback for catalog-less installs.
    try:
        sys.path.insert(0, str(SHEEPRL_DIR.parent))
        from backend import catalog as _catalog

        con = _catalog.connect()
        head = _catalog.get_watch_head(con)
        con.close()
        if head:
            return Path(head["checkpoint_path"])
    except Exception as exc:
        log(f"catalog watch-head lookup failed ({exc}); falling back to mtime scan")
    ckpts = sorted(
        SHEEPRL_DIR.glob("logs/runs/dreamer_v3/*/*/version_*/checkpoint/*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise SystemExit("No checkpoints found")
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
if initial_state:
    cfg.env.wrapper.initial_state = initial_state

# GPU inference when available: the S model is tiny next to a training run,
# and CPU-only pacing sits at exactly 1.0x real-time with zero headroom —
# any hiccup drains the viewer's buffer and stutters.
_accel = "cuda" if torch.cuda.is_available() and os.environ.get("RETRO_LIVE_GPU", "1") != "0" else "cpu"
if _accel == "cpu":
    torch.set_num_threads(4)
fabric = Fabric(accelerator=_accel, devices=1, num_nodes=1)
fabric.launch()
state = fabric.load(str(ckpt_path))

env = make_env(cfg, cfg.seed, 0, str(SHEEPRL_DIR / "logs" / "live"), "live")()
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

# Find our wrapper in the chain to install the raw A/V tap
inner = env
while not isinstance(inner, RetroDreamerWrapper):
    inner = inner.env
retro_env = inner._env
audio_rate = int(retro_env.em.get_audio_rate())
h, w = retro_env.observation_space.shape[:2]
fps = 60

# Hand ffmpeg the read-ends of two pipes we create: raw video on one,
# raw audio on the other; muxed fMP4 comes out on our stdout.
# Each pipe gets its own writer THREAD — a single writer deadlocks: a raw
# frame (172KB) overfills the 64KB pipe while ffmpeg's muxer waits for
# audio to interleave, so the writer never reaches the audio write.
v_r, v_w = os.pipe()
a_r, a_w = os.pipe()
hls_dir = Path(os.environ.get("RETRO_LIVE_HLS_DIR", "/tmp/retro-dreamer-live"))
hls_dir.mkdir(parents=True, exist_ok=True)
# only stream files — glob("*") would unlink the sidecar's open player.log
for old in hls_dir.glob("live*"):
    try:
        old.unlink()
    except OSError:
        pass

ffmpeg = subprocess.Popen(
    [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", f"pipe:{v_r}",
        "-f", "s16le", "-ar", str(audio_rate), "-ac", "2", "-i", f"pipe:{a_r}",
        # 5x nearest-neighbor upscale at the SOURCE: browsers ignore
        # image-rendering:pixelated on <video>, so crisp pixels must be
        # baked into the stream itself.
        "-vf", f"scale={w * 5}:{h * 5}:flags=neighbor",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-g", str(fps), "-crf", "21",
        "-c:a", "aac", "-b:a", "128k",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "30",
        "-hls_flags", "delete_segments+independent_segments",
        str(hls_dir / "live.m3u8"),
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    pass_fds=(v_r, a_r),
)
os.close(v_r)
os.close(a_r)

import queue
import threading

_qv: "queue.Queue[bytes | None]" = queue.Queue(maxsize=600)
_qa: "queue.Queue[bytes | None]" = queue.Queue(maxsize=600)


def _writer(fd: int, q: "queue.Queue[bytes | None]"):
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


_stats = {"frames": 0, "samples": 0, "t0": time.perf_counter(), "last_log": 0.0}


def on_frame(frame, audio):
    _qv.put(frame.tobytes())
    _qa.put(audio.tobytes())
    _stats["frames"] += 1
    _stats["samples"] += len(audio)
    wall = time.perf_counter() - _stats["t0"]
    if wall - _stats["last_log"] >= 10:
        _stats["last_log"] = wall
        media = _stats["frames"] / 60.0
        log(
            f"pace: wall={wall:.1f}s media={media:.1f}s ratio={media / wall:.3f} "
            f"audio_rate={_stats['samples'] / media:.0f}Hz qv={_qv.qsize()} qa={_qa.qsize()}"
        )


inner.frame_callback = on_frame

frame_skip = int(getattr(cfg.env.wrapper, "frame_skip", 4) or 4)
target_dt = frame_skip / fps

log(f"LIVE ckpt={ckpt_path.name} state={cfg.env.wrapper.initial_state} "
    f"{w}x{h}@{fps} audio={audio_rate}Hz")

try:
    episode = 0
    while ffmpeg.poll() is None:
        obs = env.reset(seed=cfg.seed + episode)[0]
        player.num_envs = 1
        player.init_states()
        done, step, cum_rew = False, 0, 0.0
        t_last = time.perf_counter()
        while not done and ffmpeg.poll() is None:
            torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
            real_actions = player.get_actions(
                torch_obs, True,
                {k: v for k, v in torch_obs.items() if k.startswith("mask")},
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
            now = time.perf_counter()
            sleep_for = target_dt - (now - t_last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            t_last = time.perf_counter()
        episode += 1
        log(f"episode {episode}: steps={step} reward={cum_rew:.1f}")
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    try:
        _qv.put(None)
        _qa.put(None)
    except Exception:
        pass
    ffmpeg.terminate()
    env.close()
    log("live player stopped")
