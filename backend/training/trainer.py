"""DreamerV3 training orchestrator — launches SheepRL as a subprocess.

Instead of reimplementing DreamerV3's training loop (and fighting every
edge case SheepRL already solved), we run SheepRL training as a child
process and monitor its TensorBoard logs, checkpoints, and videos.

Our dashboard observes and controls the process.
"""

import json
import os
import re
import shutil
import sys
import time
import subprocess
import tempfile
import threading
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import yaml

from .config import TrainingConfig
from .callbacks import TensorBoardCallback, WebSocketBroadcaster, EpisodeRenderer
from backend.action_manifest import (
    build_action_manifest,
    load_action_manifest,
    write_action_manifest,
)
from backend import catalog as _catalog

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_DIR = PROJECT_ROOT / "training-state"


# The vendored SheepRL tree lives at PROJECT_ROOT/sheeprl/ (repo root, containing the
# `sheeprl` Python package). The subprocess runs with this as cwd so `import sheeprl`
# resolves against the vendored copy without a pip install.
SHEEPRL_DIR = Path(__file__).resolve().parent.parent.parent / "sheeprl"

# Use the same Python interpreter that is running us (we share a venv)
SHEEPRL_PYTHON = sys.executable


def _sheeprl_logs(game_id: str) -> Path:
    """Return the SheepRL log directory for a given game."""
    return SHEEPRL_DIR / "logs" / "runs" / "dreamer_v3" / game_id


def _atomic_write_json(path: Path, value: dict):
    """Durably replace a small JSON reference file in its own directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, sort_keys=True, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp_path.unlink(missing_ok=True)


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
    "_frac = os.environ.get('RETRO_CUDA_MEM_FRACTION')\n"
    "if _frac and torch.cuda.is_available():\n"
    "    torch.cuda.set_per_process_memory_fraction(float(_frac), 0)\n"
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
        self._action_manifest_hash: Optional[str] = None
        self._action_manifest_path: Optional[Path] = None

        # Active run directory (set when training starts)
        self._run_dir: Optional[Path] = None
        self._active_run: Optional[tuple[subprocess.Popen, Path]] = None

        # Studio-v2: graceful-suspend channel + catalog session tracking
        self._control_dir: Optional[Path] = None
        self._catalog_session_id: Optional[int] = None
        self._session_run_dir: Optional[Path] = None

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

    @staticmethod
    def _workspace_action_manifest(game_id: str, game_dir: Path) -> dict:
        path = game_dir / "actions.json"
        try:
            actions_data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot build action manifest from {path}: {exc}") from exc
        return build_action_manifest(game_id, actions_data)

    @staticmethod
    def _checkpoint_action_manifest(checkpoint: Path, game_id: str) -> tuple[Path, dict]:
        config_path = checkpoint.parent.parent / "config.yaml"
        try:
            resolved = yaml.safe_load(config_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"resumed checkpoint config is unreadable: {exc}") from exc
        wrapper = ((resolved.get("env") or {}).get("wrapper") or {})
        manifest_text = wrapper.get("action_manifest")
        expected_hash = wrapper.get("action_manifest_hash")
        if not manifest_text or not expected_hash:
            raise ValueError(
                "legacy checkpoint has no immutable action manifest; action ordering "
                "is unprovable, so start fresh or perform an explicit audited legacy seal"
            )
        manifest_path = Path(manifest_text)
        if not manifest_path.is_absolute():
            manifest_path = (config_path.parent / manifest_path).resolve()
        try:
            manifest = load_action_manifest(
                manifest_path,
                expected_game_id=game_id,
                expected_hash=expected_hash,
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"resumed checkpoint action manifest is invalid: {exc}") from exc
        return manifest_path, manifest

    def _prepare_action_manifest(
        self,
        game_dir: Path,
        lineage_dir: Path,
        resume_ckpt: Optional[str],
        *,
        expected_catalog_hash: Optional[str] = None,
        require_catalog_hash: bool = False,
    ) -> tuple[Path, dict]:
        """Bind this launch to immutable ordered actions before any replay mutation."""
        current = self._workspace_action_manifest(self.config.game_id, game_dir)
        requested_hash = self.config.action_manifest_hash
        if requested_hash is not None and requested_hash != current["sha256"]:
            raise ValueError(
                "training plan is stale: ordered actions changed before launch "
                f"(planned {requested_hash}, current {current['sha256']})"
            )
        if resume_ckpt:
            saved_path, saved = self._checkpoint_action_manifest(
                Path(resume_ckpt), self.config.game_id
            )
            if require_catalog_hash and not expected_catalog_hash:
                raise ValueError(
                    "resumable catalog snapshot has no immutable action-manifest "
                    "binding; start fresh or perform an explicit audited legacy seal"
                )
            if (
                expected_catalog_hash is not None
                and saved["sha256"] != expected_catalog_hash
            ):
                raise ValueError(
                    "resumed config action manifest conflicts with the catalog's "
                    f"write-once binding: catalog {expected_catalog_hash}, "
                    f"config {saved['sha256']}"
                )
            if saved["sha256"] != current["sha256"]:
                raise ValueError(
                    "ordered actions changed since the resumed checkpoint: "
                    f"saved {saved['sha256']}, current {current['sha256']}; "
                    "same-count reorders and remaps cannot resume"
                )
            return saved_path, saved
        path = write_action_manifest(current, lineage_dir)
        return Path(path), current

    def start(self, config: Optional[TrainingConfig] = None, fresh_start: bool = False):
        """Launch SheepRL training as a subprocess."""
        if self.state == TrainingState.TRAINING:
            return
        if config:
            self.config = config
        self.config.validate()
        self._active_run = None
        self._action_manifest_hash = None
        self._action_manifest_path = None
        self._resume_catalog_action_hash: Optional[str] = None

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

    def stop(self, graceful: bool = True, timeout: float = 90.0) -> Optional[dict]:
        """Stop training. Graceful path (default): request a final checkpoint
        via the control channel, wait for the ack, then terminate — no steps
        lost. Falls back to plain SIGTERM if the channel is absent or slow.

        Returns the suspend ack ({checkpoint_path, step}) when one was written.
        """
        self.state = TrainingState.STOPPING
        ack = None
        alive = self._process and self._process.poll() is None
        if graceful and alive and self._control_dir:
            try:
                (self._control_dir / "checkpoint-request").write_text("1")
                complete = self._control_dir / "checkpoint-complete.json"
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if complete.exists():
                        ack = json.loads(complete.read_text())
                        print(f"[Trainer] Graceful suspend: final checkpoint {ack.get('checkpoint_path')}")
                        break
                    if self._process.poll() is not None:
                        break
                    time.sleep(0.5)
                else:
                    print("[Trainer] Suspend ack timed out — falling back to SIGTERM")
            except Exception as exc:
                print(f"[Trainer] Graceful suspend failed ({exc}) — falling back to SIGTERM")
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
        self._active_run = None
        self._catalog_session_end("ended", "stopped by studio")
        if ack:
            self._catalog_register_snapshot_path(ack.get("checkpoint_path"), ack.get("step"))
        self.state = TrainingState.IDLE
        return ack

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

        lineage_dir = STATE_DIR / "games" / game_id / "lineages" / "main"
        retention_manifest = lineage_dir / "checkpoint-retention.json"
        retention_root = _sheeprl_logs(game_id)
        wrapper_path = SHEEPRL_DIR / "_retro_run.py"

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
            f"checkpoint.keep_last={cfg.checkpoint_keep_last}",
            f"checkpoint.milestone_every={cfg.checkpoint_milestone_every}",
            f"checkpoint.keep_milestones={cfg.checkpoint_keep_milestones}",
            f"checkpoint.retention_manifest={retention_manifest}",
            f"checkpoint.retention_root={retention_root}",
        ]

        # Stable lineage-owned replay home (used when the buffer is created
        # fresh; a restored buffer keeps its own paths). New lineages never
        # write replay inside run dirs.
        replay_dir = lineage_dir / "replay"

        # Resume from checkpoint if not a fresh start
        resume_ckpt = None
        if self._fresh_start:
            print("[Trainer] Fresh start — skipping checkpoint resume")
        else:
            resume_ckpt = self._catalog_resumable_head(game_id)
            if resume_ckpt:
                print(f"[Trainer] Catalog head: {resume_ckpt}")
            else:
                # Catalog empty for this game — legacy mtime scan (still
                # game-scoped) so old installs keep resuming.
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

        # Seal and validate the exact ordered actions before touching replay
        # state or spawning the child. A resumed run uses its checkpoint's
        # immutable object and additionally requires the current workspace to
        # match, so equal-count reorders cannot silently reinterpret policy
        # logits or replay actions.
        manifest_path, manifest = self._prepare_action_manifest(
            game_dir,
            lineage_dir,
            resume_ckpt,
            expected_catalog_hash=self._resume_catalog_action_hash,
            require_catalog_hash=resume_ckpt is not None,
        )
        self._action_manifest_path = manifest_path
        self._action_manifest_hash = manifest["sha256"]
        cfg.action_manifest_hash = manifest["sha256"]
        cmd.extend((
            f'env.wrapper.action_manifest="{manifest_path}"',
            f"env.wrapper.action_manifest_hash={manifest['sha256']}",
        ))

        # Replay-buffer hygiene. SheepRL memmaps raw .memmap files with
        # whatever shapes THIS run assumes — no metadata on disk. A fresh
        # buffer created over a stale dir whose arrays were written with a
        # different action count mmaps garbage and has hard-killed the whole
        # WSL VM (2026-07-10: two instant crashes; same command over a clean
        # dir trains fine). Rule: buffer built fresh → wipe the dir first;
        # buffer restored → verify shapes via our meta file.
        buffer_restored = resume_ckpt is not None and cfg.resume_prefill == 0
        buffer_meta_path = replay_dir.parent / "buffer-meta.json"
        current_meta = {
            "format": "retro-dreamer-buffer-meta-v2",
            "num_envs": cfg.num_envs,
            "action_count": len(manifest["actions"]),
            "action_manifest_hash": manifest["sha256"],
        }
        if not buffer_restored:
            if replay_dir.exists():
                print(f"[Trainer] Wiping stale replay buffer: {replay_dir}")
                shutil.rmtree(replay_dir)
        elif buffer_meta_path.exists():
            saved_meta = json.loads(buffer_meta_path.read_text())
            if saved_meta != current_meta:
                raise ValueError(
                    f"Replay buffer at {replay_dir} was written with "
                    f"{saved_meta}, but this run is {current_meta} — resuming "
                    f"would corrupt the buffer (ordered actions or num_envs "
                    f"changed since). Start fresh, or restore the old config."
                )
        else:
            raise ValueError(
                f"Legacy replay buffer at {replay_dir} has no immutable v2 "
                "buffer-meta.json; action ordering is unprovable. Start fresh "
                "or perform an explicit audited legacy seal."
            )
        replay_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(buffer_meta_path, current_meta)
        cmd.append(f"buffer.memmap_dir={replay_dir}")

        env = os.environ.copy()
        # Child stdout is a pipe → Python block-buffers it. During long-episode
        # phases output is sparse and sits unflushed for hours, freezing the
        # dashboard (looks identical to a hang). Force per-line flushing.
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYOPENGL_PLATFORM"] = "egl"
        env["PYGLET_HEADLESS"] = "1"
        # DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
        # Added 2026-07-10 (a8352b4) for XL fragmentation headroom, it turned
        # out to be the one env-var difference between backend launches and
        # bare-CLI launches of the same trainer command: with it, every fresh
        # L run (4/4) hard-panicked the WSL2 guest VM at the first train
        # batch; without it, identical CLI runs trained fine. Expandable
        # segments use CUDA VMM mapping, which on WSL rides the dxg GPU-PV
        # path — see microsoft/WSL#40732 for the panic-instead-of-clean-OOM
        # failure class. Fragmentation OOM on multi-hour XL runs is the
        # accepted tradeoff; the RETRO_CUDA_MEM_FRACTION cap below keeps any
        # OOM a clean in-process error instead of a VM death.
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        # WSL VM-death guard: under WDDM/GPU-PV, allocating past FREE VRAM
        # (the Windows desktop holds several GB) demand-pages GPU memory
        # instead of raising OOM, which can hard-kill the whole WSL VM. Cap
        # the allocator just under what's free so overflow surfaces as a
        # clean CUDA OOM in the trainer log instead.
        try:
            free_mib, total_mib = map(int, subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"], timeout=10
            ).decode().split("\n")[0].split(","))
            fraction = max(0.1, round((free_mib - 1536) / total_mib, 3))
            env["RETRO_CUDA_MEM_FRACTION"] = str(fraction)
            print(f"[Trainer] VRAM guard: {free_mib}MiB free of {total_mib} — "
                  f"allocator capped at {fraction} of card")
        except Exception as exc:
            print(f"[Trainer] VRAM guard skipped (nvidia-smi failed: {exc})")

        # Graceful-suspend control channel (studio-v2)
        self._control_dir = STATE_DIR / "control" / f"session-{int(time.time())}"
        self._control_dir.mkdir(parents=True, exist_ok=True)
        env["RETRO_CONTROL_DIR"] = str(self._control_dir)

        # Write the small static launcher only after every compatibility check
        # has succeeded. It is repository scaffolding, not launch identity.
        SHEEPRL_DIR.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(_WRAPPER_SCRIPT)

        print(
            f"[Trainer] Launching SheepRL "
            f"({algo}, game={game_id}, state={initial_state}, batch={cfg.batch_size})..."
        )
        print(f"[Trainer] Command: {' '.join(cmd)}")

        launch_ts = time.time()
        known_run_dirs = {
            path.resolve()
            for path in _sheeprl_logs(game_id).glob("*/version_0")
        }
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
        # Register the session in the catalog once the run dir materializes
        threading.Thread(
            target=self._catalog_register_session,
            args=(
                game_id,
                launch_ts,
                algo.rsplit("_", 1)[1],
                self._process,
                known_run_dirs,
            ),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Catalog integration (best-effort: catalog failures must never take
    # down training itself)
    # ------------------------------------------------------------------

    def _catalog_resumable_head(self, game_id: str) -> Optional[str]:
        con = None
        try:
            con = _catalog.connect()
            # Re-crawl before resolving: checkpoints written while no live
            # registrar was running (old code, crashes, manual runs) would
            # otherwise leave the catalog behind the filesystem — that
            # near-missed a 545k-step rewind on 2026-07-10. Idempotent+fast.
            _catalog.register_existing_runs(con, game_filter=game_id)
            head = _catalog.get_resumable_head(con, game_id)
            self._resume_catalog_action_hash = head["config_hash"] if head else None
            return head["checkpoint_path"] if head else None
        except Exception as exc:
            print(f"[Trainer] catalog head lookup failed: {exc}")
            return None
        finally:
            if con is not None:
                con.close()

    def _catalog_register_session(
        self,
        game_id: str,
        launch_ts: float,
        size_label: str = "",
        process: Optional[subprocess.Popen] = None,
        known_run_dirs: Optional[set[Path]] = None,
    ):
        """Wait for the child's run dir to appear, then record the session."""
        logs = _sheeprl_logs(game_id)
        known_run_dirs = known_run_dirs or set()
        run_dir = None
        for _ in range(240):  # up to 2 min (model load can be slow)
            candidates = [
                d for d in logs.glob("*/version_0")
                if d.resolve() not in known_run_dirs
            ] if logs.exists() else []
            if candidates:
                run_dir = max(candidates, key=lambda d: d.stat().st_mtime)
                break
            if (
                process is None
                or self._process is not process
                or process.poll() is not None
            ):
                return
            time.sleep(0.5)
        if run_dir is None:
            print("[Trainer] catalog: run dir never appeared; session unregistered")
            return
        if self._process is not process or process.poll() is not None:
            return
        # Storage telemetry is launch-owned, not catalog-owned: database
        # failure must not hide a live run, and a stale registrar must never
        # attach its directory to a newer process.
        self._active_run = (process, run_dir)
        self._update_tb_view(game_id, size_label, run_dir)
        try:
            # Lazy import: backend.server imports this module at load time.
            from backend import server as _server
            _server.repoint_tensorboard(str(run_dir))
        except Exception as exc:
            print(f"[Trainer] tb repoint failed: {exc}")
        try:
            con = _catalog.connect()
            con.execute(
                "INSERT OR IGNORE INTO games (id, display_name) VALUES (?,?)",
                (game_id, game_id),
            )
            row = con.execute(
                "SELECT active_lineage_id FROM games WHERE id=?", (game_id,)
            ).fetchone()
            lineage_id = row["active_lineage_id"] if row else None
            if lineage_id is None:
                con.execute(
                    """INSERT OR IGNORE INTO lineages (game_id, name, status, created_at)
                       VALUES (?,?,'active',?)""",
                    (game_id, "main", time.time()),
                )
                lineage_id = con.execute(
                    "SELECT id FROM lineages WHERE game_id=? AND name='main'", (game_id,)
                ).fetchone()["id"]
                con.execute(
                    "UPDATE games SET active_lineage_id=? WHERE id=?", (lineage_id, game_id)
                )
            con.execute(
                """INSERT OR IGNORE INTO sessions
                   (lineage_id, run_dir, started_at, status, resolved_config)
                   VALUES (?,?,?,'running',?)""",
                (lineage_id, str(run_dir), launch_ts, str(run_dir / "config.yaml")),
            )
            self._catalog_session_id = con.execute(
                "SELECT id FROM sessions WHERE run_dir=?", (str(run_dir),)
            ).fetchone()["id"]
            self._session_run_dir = run_dir
            con.commit()
            con.close()
            print(f"[Trainer] catalog: session {self._catalog_session_id} = {run_dir}")
        except Exception as exc:
            print(f"[Trainer] catalog session registration failed: {exc}")

    @staticmethod
    def _update_tb_view(game_id: str, size_label: str, version0_dir: Path):
        """Maintain the TensorBoard symlink view: logs/tb/<game>/<SIZE>_<stamp>
        → the run's version_0. Real run dirs keep SheepRL's timestamp naming
        (resume discovery, the catalog, and the F-Zero buffer symlink all
        depend on those paths staying put); TensorBoard points at logs/tb
        instead, where every run is game- and model-tagged so the dashboard
        regex filter can slice by either. scripts/rebuild_tb_view.py rebuilds
        the whole view for pre-existing runs."""
        try:
            stamp = version0_dir.parent.name[:19]
            link = SHEEPRL_DIR / "logs" / "tb" / game_id / f"{size_label or 'UNK'}_{stamp}"
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.is_symlink():
                link.symlink_to(os.path.relpath(version0_dir, link.parent))
                print(f"[Trainer] tb view: {link.parent.name}/{link.name}")
        except Exception as exc:
            print(f"[Trainer] tb view symlink failed: {exc}")

    def _catalog_register_snapshot_step(self, step: int):
        if self._catalog_session_id is None or self._session_run_dir is None:
            return
        path = self._session_run_dir / "checkpoint" / f"ckpt_{step}_0.ckpt"
        self._catalog_register_snapshot_path(str(path), step)

    def _catalog_register_snapshot_path(self, path: Optional[str], step: Optional[int]):
        if not path or self._catalog_session_id is None:
            return
        # The child reports log_dir-relative paths (its cwd is SHEEPRL_DIR);
        # the catalog stores absolutes so existence checks work from anywhere.
        if not os.path.isabs(path):
            path = str((SHEEPRL_DIR / path).resolve())
        try:
            con = _catalog.connect()
            replay = STATE_DIR / "games" / self.config.game_id / "lineages" / "main" / "replay"
            _catalog.register_snapshot(
                con, self._catalog_session_id, int(step or 0), str(path),
                replay_path=str(replay) if replay.exists() else None,
                config_hash=self._action_manifest_hash,
            )
            con.execute(
                "UPDATE sessions SET end_step=? WHERE id=?",
                (int(step or 0), self._catalog_session_id),
            )
            con.commit()
            con.close()
        except Exception as exc:
            print(f"[Trainer] catalog snapshot registration failed: {exc}")

    def _catalog_session_end(self, status: str, reason: str):
        session_id = self._catalog_session_id
        self._catalog_session_id = None
        self._session_run_dir = None
        if session_id is None:
            return
        try:
            con = _catalog.connect()
            con.execute(
                "UPDATE sessions SET status=?, exit_reason=?, ended_at=? WHERE id=?",
                (status, reason, time.time(), session_id),
            )
            con.commit()
            con.close()
        except Exception as exc:
            print(f"[Trainer] catalog session end failed: {exc}")

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
            if "Saving checkpoint at policy_step=" in line:
                try:
                    step = int(line.split("policy_step=")[1].split(",")[0])
                    self._catalog_register_snapshot_step(step)
                except (ValueError, IndexError):
                    pass
            self._log_lines.append(line)
            print(f"[SheepRL] {line}")

        exit_code = proc.wait()
        active_run = self._active_run
        is_current = active_run is not None and active_run[0] is proc
        if is_current:
            self._active_run = None
        if not is_current:
            # A superseded child (e.g. the game we switched AWAY from) exited
            # while a NEW run is already TRAINING. Its (expected) SIGTERM must
            # not be reported as the new run's crash — this exact race marked
            # a healthy fresh launch as ERROR while it kept training.
            print(f"[Trainer] Stale monitor: superseded child exited with "
                  f"code {exit_code} (ignored; a newer run owns the state)")
            return
        if exit_code != 0 and self.state == TrainingState.TRAINING:
            self._error_message = (
                f"SheepRL exited with code {exit_code}. Last output: {last_line}"
            )
            self.state = TrainingState.ERROR
            self._catalog_session_end("crashed", self._error_message[:300])
            print(f"[Trainer] ERROR: {self._error_message}")
        elif self.state == TrainingState.TRAINING:
            self.state = TrainingState.IDLE
            self._catalog_session_end("ended", "completed")
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

    @property
    def active_run_dir(self) -> Optional[Path]:
        """The registered live run only; never falls back to historical runs."""
        process = self._process
        active_run = self._active_run
        if (
            self.state != TrainingState.TRAINING
            or process is None
            or process.poll() is not None
            or active_run is None
            or active_run[0] is not process
        ):
            return None
        return active_run[1]

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
        """Return the directory TensorBoard should point at for the active game.

        Only the newest run's version_0 — Schuyler wants just the latest run
        in the Metrics tab, not every historical run dir. Older runs stay on
        disk; the trainer repoints TB when a start/resume creates a new one.
        """
        logs = _sheeprl_logs(self.config.game_id)
        if logs.exists():
            candidates = sorted(
                logs.glob("*/version_0"), key=lambda p: p.stat().st_mtime
            )
            if candidates:
                return str(candidates[-1])
        return str(logs)

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
        self._all_lengths: list[float] = []

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
                    # SheepRL prints (dreamer_v3.py): "reward_env_0=56.7,
                    # length_env_0=1234" (+ optional " track=go"). Parse both
                    # with regex so the length_env_ suffix can't corrupt the
                    # reward value (the old split-on-'=' broke once length was
                    # added, because it introduced a second '=' on the line).
                    rm = re.search(r"reward_env_\d+=(-?[0-9eE.]+)", line)
                    lm = re.search(r"length_env_\d+=(-?[0-9eE.]+)", line)
                    if rm:
                        reward = float(rm.group(1))
                        self.current_episode += 1
                        self._all_returns.append(reward)
                        if reward > self.max_return:
                            self.max_return = reward
                        recent = self._all_returns[-100:]
                        self.avg_return = sum(recent) / len(recent)
                    if lm:
                        length = float(lm.group(1))
                        self._all_lengths.append(length)
                        recent_len = self._all_lengths[-100:]
                        self.avg_length = sum(recent_len) / len(recent_len)
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
