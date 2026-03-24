"""Training callbacks for metrics, TensorBoard, WebSocket, and rendering."""

import time
import json
import asyncio
import imageio
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Optional


class MetricsCollector:
    """Collects and aggregates training metrics."""

    def __init__(self):
        self.step_metrics: list[dict] = []
        self.episode_metrics: list[dict] = []
        self._current_step = 0
        self._current_episode = 0
        self._start_time = time.time()

    def log_step(self, metrics: dict):
        """Log per-training-step metrics."""
        self._current_step += 1
        entry = {
            "step": self._current_step,
            "timestamp": time.time() - self._start_time,
            **metrics,
        }
        self.step_metrics.append(entry)

    def log_episode(self, metrics: dict):
        """Log per-episode metrics."""
        self._current_episode += 1
        entry = {
            "episode": self._current_episode,
            "step": self._current_step,
            "timestamp": time.time() - self._start_time,
            **metrics,
        }
        self.episode_metrics.append(entry)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def current_episode(self) -> int:
        return self._current_episode

    @property
    def elapsed_time(self) -> float:
        return time.time() - self._start_time

    @property
    def steps_per_second(self) -> float:
        elapsed = self.elapsed_time
        return self._current_step / elapsed if elapsed > 0 else 0

    def get_recent_metrics(self, n: int = 100) -> dict:
        """Get summary of recent metrics."""
        recent_eps = self.episode_metrics[-n:] if self.episode_metrics else []
        recent_steps = self.step_metrics[-n:] if self.step_metrics else []

        summary = {
            "current_step": self._current_step,
            "current_episode": self._current_episode,
            "elapsed_time": self.elapsed_time,
            "steps_per_second": self.steps_per_second,
        }

        # Average recent episode returns
        if recent_eps:
            returns = [e.get("episode_return", 0) for e in recent_eps]
            lengths = [e.get("episode_length", 0) for e in recent_eps]
            summary["avg_return"] = sum(returns) / len(returns)
            summary["avg_length"] = sum(lengths) / len(lengths)
            summary["max_return"] = max(returns)

        # Average recent losses
        if recent_steps:
            for key in ["world_model_loss", "actor_loss", "critic_loss", "kl_loss"]:
                vals = [s.get(key, 0) for s in recent_steps if key in s]
                if vals:
                    summary[f"avg_{key}"] = sum(vals) / len(vals)

        return summary


class TensorBoardCallback:
    """Writes metrics to TensorBoard."""

    def __init__(self, logdir: str):
        Path(logdir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logdir)

    def on_step(self, step: int, metrics: dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, step)

    def on_episode(self, episode: int, step: int, metrics: dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"episode/{key}", value, step)

    def log_video(self, tag: str, frames: list[np.ndarray], step: int, fps: int = 30):
        """Log a video to TensorBoard."""
        if not frames:
            return
        # TensorBoard expects (N, T, C, H, W) format
        video = np.stack(frames)  # (T, H, W, C)
        video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W)
        video = np.expand_dims(video, 0)  # (1, T, C, H, W)
        self.writer.add_video(tag, video, step, fps=fps)

    def log_image(self, tag: str, image: np.ndarray, step: int):
        """Log an image to TensorBoard."""
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))
        self.writer.add_image(tag, image, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


class WebSocketBroadcaster:
    """Broadcasts metrics to connected WebSocket clients."""

    def __init__(self):
        self._connections: set = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def add_connection(self, ws):
        self._connections.add(ws)

    def remove_connection(self, ws):
        self._connections.discard(ws)

    def broadcast(self, data: dict):
        """Broadcast data to all connected clients (thread-safe)."""
        if not self._connections or not self._loop:
            return
        message = json.dumps(data)
        for ws in list(self._connections):
            try:
                asyncio.run_coroutine_threadsafe(ws.send_text(message), self._loop)
            except Exception:
                self._connections.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._connections)


class EpisodeRenderer:
    """Saves rendered episodes as MP4 files."""

    def __init__(self, output_dir: str, fps: int = 60):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._rendered_episodes: list[dict] = []

    def save_episode(
        self,
        frames: list[np.ndarray],
        episode_num: int,
        metadata: dict | None = None,
    ) -> str:
        """Save frames as MP4. Returns the file path."""
        if not frames:
            return ""

        filename = f"episode_{episode_num:06d}.mp4"
        filepath = self.output_dir / filename

        imageio.mimwrite(str(filepath), frames, fps=self.fps, quality=8)

        # Save a thumbnail (middle frame)
        thumb_path = self.output_dir / f"episode_{episode_num:06d}_thumb.jpg"
        from PIL import Image
        img = Image.fromarray(frames[len(frames) // 2])
        img.thumbnail((160, 120))
        img.save(str(thumb_path), quality=85)

        episode_info = {
            "episode": episode_num,
            "filename": filename,
            "thumbnail": f"episode_{episode_num:06d}_thumb.jpg",
            "frame_count": len(frames),
            "duration": len(frames) / self.fps,
            **(metadata or {}),
        }
        self._rendered_episodes.append(episode_info)

        return str(filepath)

    def list_episodes(self) -> list[dict]:
        """List all rendered episodes."""
        return list(reversed(self._rendered_episodes))
