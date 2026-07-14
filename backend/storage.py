"""Cheap filesystem and active-training-run storage telemetry."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional


class StorageUsageSampler:
    """Sample filesystem space and cache the only recursive operation."""

    def __init__(
        self,
        filesystem_path: Path,
        *,
        run_size_ttl: float = 30.0,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        disk_usage: Callable = shutil.disk_usage,
    ) -> None:
        self.filesystem_path = Path(filesystem_path)
        self.run_size_ttl = run_size_ttl
        self._clock = clock
        self._monotonic = monotonic
        self._disk_usage = disk_usage
        self._run_cache: tuple[Path, float, float, Optional[int]] | None = None
        self._run_cache_lock = threading.Lock()

    def sample(self, active_run_dir: Path | None) -> dict:
        sampled_at = self._clock()
        try:
            usage = self._disk_usage(self.filesystem_path)
            total = int(usage.total)
            free = int(usage.free)
            filesystem = {
                "total_bytes": total,
                "free_bytes": free,
                "free_percent": (free / total * 100.0) if total else None,
            }
        except OSError:
            filesystem = {
                "total_bytes": None,
                "free_bytes": None,
                "free_percent": None,
            }

        active_run_bytes, active_run_sampled_at = self._cached_run_size(active_run_dir)
        return {
            "sampled_at": sampled_at,
            "filesystem": filesystem,
            # Scope is intentionally the active version_0 run directory. The
            # lineage-owned replay buffer lives elsewhere and is excluded.
            "active_run_bytes": active_run_bytes,
            "active_run_sampled_at": active_run_sampled_at,
        }

    def _cached_run_size(
        self, run_dir: Path | None
    ) -> tuple[Optional[int], Optional[float]]:
        if run_dir is None:
            return None, None
        path = Path(run_dir)
        with self._run_cache_lock:
            monotonic_now = self._monotonic()
            if self._run_cache is not None:
                cached_path, cached_at, sampled_at, cached_size = self._run_cache
                if (
                    cached_path == path
                    and monotonic_now - cached_at < self.run_size_ttl
                ):
                    return cached_size, sampled_at

            size = self._directory_size(path)
            monotonic_now = self._monotonic()
            sampled_at = self._clock()
            self._run_cache = (path, monotonic_now, sampled_at, size)
            return size, sampled_at

    @staticmethod
    def _directory_size(root: Path) -> Optional[int]:
        if not root.is_dir():
            return None
        total = 0
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError:
                # A partial total is more misleading than an unavailable one.
                return None
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    # Runs are written concurrently; disappearing files are
                    # expected and do not make the remaining total partial.
                    continue
                except OSError:
                    return None
        return total
