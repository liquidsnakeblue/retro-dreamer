"""DreamerV3 training orchestrator — launches SheepRL as a subprocess.

Instead of reimplementing DreamerV3's training loop (and fighting every
edge case SheepRL already solved), we run SheepRL training as a child
process and monitor its TensorBoard logs, checkpoints, and videos.

Our dashboard observes and controls the process.
"""

import os
import sys
import time
import subprocess
import threading
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from .config import TrainingConfig
from .callbacks import TensorBoardCallback, WebSocketBroadcaster, EpisodeRenderer


# The vendored SheepRL tree lives at PROJECT_ROOT/sheeprl/ (repo root, containing the
# `sheeprl` Python package). The subprocess runs with this as cwd so `import sheeprl`
# resolves against the vendored copy without a pip install.
SHEEPRL_DIR = Path(__file__).resolve().parent.parent.parent / "sheeprl"

# Use the same Python interpreter that is running us (we share a venv)
SHEEPRL_PYTHON = sys.executable


def _sheeprl_logs(game_id: str) -> Path:
    """Return the SheepRL log directory for a given game."""
    return SHEEPRL_DIR / "logs" / "runs" / "dreamer_v3" / game_id


# ffprobe is ~100ms per file; without a cache the dashboard's 5s poll runs it
# on every video every time, stalling the whole (single-loop) server.
_VIDEO_META_CACHE: dict = {}  # path -> (mtime, duration)


# Wrapper script content written to SHEEPRL_DIR at launch time
_WRAPPER_SCRIPT = (
    "import os\n"
    "os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')\n"
    "os.environ.setdefault('PYGLET_HEADLESS', '1')\n"
    "import pyglet; pyglet.options['shadow_window'] = False\n"
    "import torch\n"
    "original_load = torch.load\n"
    "def _patched_torch_load(*args, **kwargs):\n"
    "    kwargs['weights_only'] = False\n"
    "    return original_load(*args, **kwargs)\n"
    "torch.load = _patched_torch_load\n"
    "try:\n"
    "    import lightning.fabric.utilities.cloud_io as cloud_io\n"
    "    def patched_load(path, map_location=None):\n"
    "        return original_load(path, map_location=map_location, weights_only=False)\n"
    "    cloud_io._load = patched_load\n"
    "except: pass\n"
    "from sheeprl.cli import run\n"
    "run()\n"
)


class TrainingState(str, Enum):
    IDLE = "idle"
    TRAINING = "training"
    STOPPING = "stopping"
    ERROR = "error"


class TrainingStatus:
    def __init__(self, state=TrainingState.IDLE, current_step=0, current_episode=0,
                 elapsed_time=0.0, steps_per_second=0.0, avg_return=0.0,
                 avg_length=0.0, max_return=0.0, gpu_memory_used=0.0,
                 gpu_memory_total=0.0, error_message="", game_id=""):
        self.state = state
        self.current_step = current_step
        self.current_episode = current_episode
        self.elapsed_time = elapsed_time
        self.steps_per_second = steps_per_second
        self.avg_return = avg_return
        self.avg_length = avg_length
        self.max_return = max_return
        self.gpu_memory_used = gpu_memory_used
        self.gpu_memory_total = gpu_memory_total
        self.error_message = error_message
        self.game_id = game_id


class DreamerV3Trainer:
    """Manages SheepRL DreamerV3 training as a subprocess.

    - start(): launches SheepRL training via its CLI
    - stop(): kills the subprocess
    - status: reads process state + metrics parsed from SheepRL output
    - Videos and checkpoints are read from SheepRL's output directories
    """

    def __init__(
        self,
        config: TrainingConfig,
        game_manager=None,
        ws_broadcaster: Optional[WebSocketBroadcaster] = None,
    ):
        self.config = config
        self.game_manager = game_manager
        self.state = TrainingState.IDLE
        self.ws_broadcaster = ws_broadcaster or WebSocketBroadcaster()
        self.renderer = EpisodeRenderer(config.episode_dir)

        # Metrics tracking (parsed from SheepRL stdout)
        self.metrics = _MetricsTracker()

        # TensorBoard callback (our own TB instance pointing at SheepRL logs)
        self.tb_callback = TensorBoardCallback(config.logdir)

        # Subprocess
        self._process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._error_message = ""
        self._fresh_start = False

        # Active run directory (set when training starts)
        self._run_dir: Optional[Path] = None

        # Ring buffer of recent log lines (200 lines)
        self._log_lines: deque[str] = deque(maxlen=200)

        # World model placeholder (for API compatibility)
        self._world_model = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def status(self) -> TrainingStatus:
        gpu_mem_used, gpu_mem_total = 0.0, 0.0
        try:
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / 1e9
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass

        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        return TrainingStatus(
            state=self.state,
            current_step=self.metrics.current_step,
            current_episode=self.metrics.current_episode,
            elapsed_time=elapsed,
            steps_per_second=self.metrics.steps_per_second,
            avg_return=self.metrics.avg_return,
            avg_length=self.metrics.avg_length,
            max_return=self.metrics.max_return,
            gpu_memory_used=gpu_mem_used,
            gpu_memory_total=gpu_mem_total,
            error_message=self._error_message,
            game_id=self.config.game_id,
        )

    def start(self, config: Optional[TrainingConfig] = None, fresh_start: bool = False):
        """Launch SheepRL training as a subprocess."""
        if self.state == TrainingState.TRAINING:
            return
        if config:
            self.config = config

        self.state = TrainingState.TRAINING
        self._error_message = ""
        self._start_time = time.time()
        self._fresh_start = fresh_start
        self.metrics = _MetricsTracker()

        # Register the game with retro if a game_manager is available
        if self.game_manager is not None:
            try:
                self.game_manager.setup_retro_integration(self.config.game_id)
            except Exception as exc:
                print(f"[Trainer] Warning: retro integration setup failed: {exc}")

        try:
            self._launch_sheeprl()
        except Exception as e:
            self._error_message = f"{type(e).__name__}: {e}"
            self.state = TrainingState.ERROR
            raise

    def stop(self):
        """Kill the SheepRL subprocess."""
        self.state = TrainingState.STOPPING
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass
        self._process = None
        self.state = TrainingState.IDLE

    def pause(self):
        """Not supported in subprocess mode."""
        pass

    def resume(self):
        """Not supported in subprocess mode."""
        pass

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _launch_sheeprl(self):
        """Build and launch the SheepRL training command."""
        cfg = self.config
        game_id = cfg.game_id
        initial_state = cfg.initial_state or cfg.env_state or "go"

        # Resolve game directory (absolute path)
        if self.game_manager is not None:
            game_dir = self.game_manager.games_dir / game_id
        else:
            # Fall back to PROJECT_ROOT/games/<game_id>
            game_dir = SHEEPRL_DIR.parent.parent / "games" / game_id
        abs_game_dir = str(game_dir.resolve())

        # Map our model sizes to SheepRL algo names
        model_map = {
            "debug": "dreamer_v3_XS",
            "small": "dreamer_v3_S",
            "medium": "dreamer_v3_M",
            "large": "dreamer_v3_L",
            "xl": "dreamer_v3_XL",
        }
        algo = model_map.get(cfg.model_size, "dreamer_v3_S")

        # Write the torch.load patch wrapper script into SHEEPRL_DIR
        wrapper_path = SHEEPRL_DIR / "_retro_run.py"
        SHEEPRL_DIR.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(_WRAPPER_SCRIPT)

        # Build command
        cmd = [
            SHEEPRL_PYTHON, str(wrapper_path),
            "exp=dreamer_v3_retro",
            f"algo={algo}",
            # SheepRL defaults root_dir to ${algo.name}/${env.id} ("dreamer_v3/retro-dreamer");
            # pin it to the game id so _sheeprl_logs()/videos/checkpoints discovery matches
            f"root_dir=dreamer_v3/{game_id}",
            f"env.wrapper.game_id={game_id}",
            f"env.wrapper.game_dir={abs_game_dir}",
            f"env.wrapper.initial_state={initial_state}",
            f"env.num_envs={cfg.num_envs}",
            f"env.sync_env={'true' if cfg.num_envs == 1 else 'false'}",
            f"algo.per_rank_batch_size={cfg.batch_size}",
            f"algo.replay_ratio={cfg.replay_ratio}",
            f"algo.learning_starts={cfg.prefill_steps}",
            f"algo.per_rank_sequence_length={cfg.batch_length}",
            "env.capture_video=true",
            "env.video_freq=10",
            f"metric.log_every={cfg.log_every}",
            f"checkpoint.every={cfg.checkpoint_every}",
        ]

        # Resume from checkpoint if not a fresh start
        if self._fresh_start:
            print("[Trainer] Fresh start — skipping checkpoint resume")
        else:
            resume_ckpt = self._find_latest_checkpoint()
            if resume_ckpt:
                cmd.append(f'checkpoint.resume_from="{resume_ckpt}"')
                if cfg.resume_prefill > 0:
                    # Buffer lost/corrupt: skip restoring it and re-collect
                    # experience with the current policy before training resumes.
                    cmd.append("buffer.checkpoint=false")
                    cmd.append(f"algo.learning_starts={cfg.resume_prefill}")
                    print(f"[Trainer] Resuming from: {resume_ckpt} (fresh buffer, prefill={cfg.resume_prefill})")
                else:
                    cmd.append("algo.learning_starts=0")
                    print(f"[Trainer] Resuming from: {resume_ckpt}")

        env = os.environ.copy()
        # Child stdout is a pipe → Python block-buffers it. During long-episode
        # phases output is sparse and sits unflushed for hours, freezing the
        # dashboard (looks identical to a hang). Force per-line flushing.
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["PYGLET_HEADLESS"] = "1"

        print(
            f"[Trainer] Launching SheepRL "
            f"({algo}, game={game_id}, state={initial_state}, batch={cfg.batch_size})..."
        )
        print(f"[Trainer] Command: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            cwd=str(SHEEPRL_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Start log monitoring thread
        self._log_thread = threading.Thread(target=self._monitor_process, daemon=True)
        self._log_thread.start()

    def _monitor_process(self):
        """Monitor subprocess output and detect completion/errors."""
        proc = self._process
        if proc is None:
            return

        last_line = ""
        # Read until EOF. A single bad line (decode error, parse bug) must never
        # kill this thread: it is also responsible for detecting process exit.
        while True:
            try:
                line = proc.stdout.readline()
            except Exception as exc:
                print(f"[Trainer] stdout read error (continuing): {exc!r}")
                if proc.poll() is not None:
                    break
                continue
            if line == "":  # EOF — process closed stdout
                break
            line = line.rstrip()
            if not line:
                continue
            last_line = line
            try:
                self.metrics.parse_log_line(line)
            except Exception as exc:
                print(f"[Trainer] metric parse error on {line!r}: {exc!r}")
            self._log_lines.append(line)
            print(f"[SheepRL] {line}")

        exit_code = proc.wait()
        if exit_code != 0 and self.state == TrainingState.TRAINING:
            self._error_message = (
                f"SheepRL exited with code {exit_code}. Last output: {last_line}"
            )
            self.state = TrainingState.ERROR
            print(f"[Trainer] ERROR: {self._error_message}")
        elif self.state == TrainingState.TRAINING:
            self.state = TrainingState.IDLE
            print("[Trainer] Training completed normally.")

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent SheepRL checkpoint for the current game.

        Searches ALL run directories, newest checkpoint file wins: an aborted
        launch leaves an empty run dir that must not shadow older checkpoints
        (that failure mode silently degraded a resume into a fresh start).
        """
        logs = _sheeprl_logs(self.config.game_id)
        if not logs.exists():
            return None

        ckpts = sorted(
            logs.glob("*/version_*/checkpoint/ckpt_*.ckpt"),
            key=lambda p: p.stat().st_mtime,
        )
        if not ckpts:
            return None

        return str(ckpts[-1])

    # ------------------------------------------------------------------
    # Video / checkpoint scanning
    # ------------------------------------------------------------------

    def get_run_dir(self) -> Optional[Path]:
        """Get the current/latest SheepRL run directory for the active game."""
        logs = _sheeprl_logs(self.config.game_id)
        if not logs.exists():
            return None
        run_dirs = sorted(logs.glob("*/"), key=lambda p: p.stat().st_mtime)
        if not run_dirs:
            return None
        latest = run_dirs[-1]
        versions = sorted(latest.glob("version_*/"))
        if versions:
            return versions[-1]
        return latest

    def list_videos(self) -> list[dict]:
        """Scan SheepRL's video directories for training videos.

        Scans ALL run dirs for the active game (not just the latest) so
        replays survive checkpoint resumes, which start a new run dir.
        Every video gets a unique ``id`` (path relative to the game's log
        dir) — bare filenames collide: each eval writes its own
        rl-video-episode-0.mp4.
        """
        logs = _sheeprl_logs(self.config.game_id)
        if not logs.exists():
            return []

        videos = []
        for run_dir in logs.glob("*/version_*/"):
            for video_dir in [run_dir / "train_videos", run_dir / "videos"]:
                if video_dir.exists():
                    for mp4 in video_dir.glob("*.mp4"):
                        videos.append(self._video_info(mp4, logs, "train"))
            eval_dir = run_dir / "evaluation"
            if eval_dir.exists():
                for vdir in eval_dir.glob("version_*/test_videos"):
                    for mp4 in vdir.glob("*.mp4"):
                        videos.append(self._video_info(mp4, logs, "eval"))

        videos.sort(key=lambda v: v["modified"], reverse=True)
        return videos[:20]  # most recent 20

    @staticmethod
    def _video_info(mp4: Path, base_dir: Path, source: str) -> dict:
        """Extract metadata from a video file."""
        import re
        stat = mp4.stat()
        info = {
            "id": str(mp4.relative_to(base_dir)),
            "source": source,
            "filename": mp4.name,
            "path": str(mp4),
            "size_mb": stat.st_size / 1e6,
            "modified": stat.st_mtime,
            "step": 0,
            "duration": 0.0,
            "frames": 0,
        }
        m = re.search(r"step-(\d+)", mp4.name) or re.search(r"episode-(\d+)", mp4.name)
        if m:
            info["step"] = int(m.group(1))
        cached = _VIDEO_META_CACHE.get(info["path"])
        if cached is not None and cached[0] == stat.st_mtime:
            info["duration"] = cached[1]
            return info
        try:
            import json as _json
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(mp4)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                fmt = _json.loads(result.stdout).get("format", {})
                info["duration"] = float(fmt.get("duration", 0))
            _VIDEO_META_CACHE[info["path"]] = (stat.st_mtime, info["duration"])
        except Exception:
            pass
        return info

    def resolve_video(self, video_id: str) -> Optional[str]:
        """Resolve a video id (path relative to the game's log dir) to a file path.

        Direct resolution — no directory scan, no ffprobe — so serving video
        bytes doesn't pay the metadata tax. Refuses paths that escape the
        game's log dir."""
        logs = _sheeprl_logs(self.config.game_id).resolve()
        try:
            path = (logs / video_id).resolve()
            path.relative_to(logs)
        except (ValueError, OSError):
            return None
        if path.suffix == ".mp4" and path.is_file():
            return str(path)
        return None

    def list_checkpoints(self) -> list[dict]:
        """Scan SheepRL's checkpoint directory for the active game."""
        checkpoints = []
        run_dir = self.get_run_dir()
        if not run_dir:
            return checkpoints

        ckpt_dir = run_dir / "checkpoint"
        if not ckpt_dir.exists():
            return checkpoints

        for ckpt in sorted(
            ckpt_dir.glob("ckpt_*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            checkpoints.append({
                "filename": ckpt.name,
                "path": str(ckpt),
                "size_mb": ckpt.stat().st_size / 1e6,
                "modified": ckpt.stat().st_mtime,
            })

        return checkpoints

    def get_tensorboard_logdir(self) -> str:
        """Return the directory TensorBoard should point at for the active game."""
        return str(_sheeprl_logs(self.config.game_id))

    # ------------------------------------------------------------------
    # Compatibility stubs
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str):
        """Not directly supported — SheepRL handles checkpointing."""
        print("[Trainer] Checkpoint loading handled by SheepRL on next start")


class _MetricsTracker:
    """Parses SheepRL console output to extract training metrics."""

    def __init__(self):
        self.current_step = 0
        self.current_episode = 0
        self.steps_per_second = 0.0
        self.avg_return = 0.0
        self.avg_length = 0.0
        self.max_return = 0.0
        self._start_time = time.time()
        self._all_returns: list[float] = []

    def parse_log_line(self, line: str):
        """Extract metrics from SheepRL's console output."""
        # SheepRL prints: "Rank-0: policy_step=1234, reward_env_0=56.7"
        if "policy_step=" in line:
            try:
                parts = line.split("policy_step=")[1]
                step_str = parts.split(",")[0].strip()
                self.current_step = int(step_str)
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    self.steps_per_second = self.current_step / elapsed
            except (ValueError, IndexError):
                pass

            if "reward_env_" in line:
                try:
                    reward_str = line.split("reward_env_")[1].split("=")[1].strip()
                    reward_str = reward_str.strip("[]")
                    reward = float(reward_str)
                    self.current_episode += 1
                    self._all_returns.append(reward)
                    if reward > self.max_return:
                        self.max_return = reward
                    recent = self._all_returns[-100:]
                    self.avg_return = sum(recent) / len(recent)
                except (ValueError, IndexError):
                    pass

        # SheepRL checkpoint messages
        if "Saving checkpoint" in line:
            try:
                if "policy_step=" in line:
                    step_str = line.split("policy_step=")[1].split(",")[0].strip()
                    self.current_step = int(step_str)
            except (ValueError, IndexError):
                pass

    def get_recent_metrics(self, n: int = 100) -> dict:
        """Compatibility with old MetricsCollector interface."""
        return {
            "current_step": self.current_step,
            "current_episode": self.current_episode,
            "elapsed_time": time.time() - self._start_time,
            "steps_per_second": self.steps_per_second,
            "avg_return": self.avg_return,
            "avg_length": self.avg_length,
            "max_return": self.max_return,
        }
