"""Bounded, future-write-only checkpoint retention.

The policy deliberately never scans a checkpoint directory. It can unlink
only paths recorded after a successful save in this process or in its own
lineage manifest, so enabling it cannot adopt or delete checkpoints created
before deployment.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path


_STEP_RE = re.compile(r"^ckpt_(\d+)(?:_|$)")
_MANIFEST_FORMAT = "retro-dreamer-checkpoint-retention-v1"


@dataclass(frozen=True)
class _ManagedCheckpoint:
    path: Path
    step: int | None
    milestone: bool


class CheckpointRetentionPolicy:
    """Retain recent writes plus a bounded set of milestone-bucket writes."""

    def __init__(
        self,
        keep_last: int | None = None,
        milestone_every: int | None = None,
        keep_milestones: int | None = None,
        manifest_path: str | Path | None = None,
        managed_root: str | Path | None = None,
    ) -> None:
        for name, value in (
            ("keep_last", keep_last),
            ("milestone_every", milestone_every),
            ("keep_milestones", keep_milestones),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        self.keep_last = keep_last
        self.milestone_every = milestone_every
        self.keep_milestones = keep_milestones
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.managed_root = Path(managed_root).resolve() if managed_root else None
        if self.manifest_path is not None and self.managed_root is None:
            raise ValueError("managed_root is required when manifest_path is set")
        self._managed = self._load_manifest()

    @property
    def managed_paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self._managed)

    def record_successful_write(self, path: str | Path) -> None:
        """Register one durable write, then prune only older managed writes."""
        checkpoint = Path(path).resolve()
        if not self._is_managed_path(checkpoint):
            warnings.warn(
                f"Checkpoint {checkpoint} is outside retention root; preserving it",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self._managed = [item for item in self._managed if item.path != checkpoint]

        match = _STEP_RE.match(checkpoint.stem)
        step = int(match.group(1)) if match else None
        milestone = False
        if (
            step is not None
            and self.milestone_every
            and self.keep_milestones
            and step >= self.milestone_every
        ):
            bucket = step // self.milestone_every
            previous_step = self._managed[-1].step if self._managed else None
            milestone = (
                previous_step is None
                or previous_step // self.milestone_every != bucket
            )

        self._managed.append(_ManagedCheckpoint(checkpoint, step, milestone))
        # First publish the expanded ownership ledger. If this fails, preserve
        # every file; deleting before the new path is durable in the manifest
        # could leak it forever after a crash.
        if not self._persist_manifest():
            return
        self._prune()
        self._persist_manifest()

    def _is_managed_path(self, path: Path) -> bool:
        if self.managed_root is None:
            return True
        try:
            path.relative_to(self.managed_root)
        except ValueError:
            return False
        return path.suffix == ".ckpt"

    def _load_manifest(self) -> list[_ManagedCheckpoint]:
        if self.manifest_path is None or not self.manifest_path.exists():
            return []
        try:
            payload = json.loads(self.manifest_path.read_text())
            if not isinstance(payload, dict):
                raise ValueError("manifest must be an object")
            if payload.get("format") != _MANIFEST_FORMAT:
                raise ValueError("unrecognized manifest format")
            records = payload.get("checkpoints")
            if not isinstance(records, list):
                raise ValueError("manifest checkpoints must be a list")

            managed: list[_ManagedCheckpoint] = []
            seen: set[Path] = set()
            for record in records:
                if not isinstance(record, dict):
                    continue
                path = Path(str(record.get("path", ""))).resolve()
                step = record.get("step")
                milestone = record.get("milestone")
                if (
                    path in seen
                    or not self._is_managed_path(path)
                    or isinstance(step, bool)
                    or not isinstance(step, int)
                    or not isinstance(milestone, bool)
                ):
                    continue
                # Missing manifest-owned files were already removed elsewhere;
                # forget them without ever scanning for replacements.
                if not path.is_file():
                    continue
                managed.append(_ManagedCheckpoint(path, step, milestone))
                seen.add(path)
            return managed
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"Could not load checkpoint retention manifest "
                f"{self.manifest_path}: {exc}; preserving all existing files",
                RuntimeWarning,
                stacklevel=2,
            )
            return []

    def _persist_manifest(self) -> bool:
        if self.manifest_path is None:
            return True
        payload = {
            "format": _MANIFEST_FORMAT,
            "checkpoints": [
                {
                    "path": str(item.path),
                    "step": item.step,
                    "milestone": item.milestone,
                }
                for item in self._managed
            ],
        }
        temp_path = self.manifest_path.with_name(
            f".{self.manifest_path.name}.{os.getpid()}.{id(self)}.tmp"
        )
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(json.dumps(payload, separators=(",", ":")))
            os.replace(temp_path, self.manifest_path)
            return True
        except OSError as exc:
            warnings.warn(
                f"Could not persist checkpoint retention manifest "
                f"{self.manifest_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _prune(self) -> None:
        milestone_enabled = bool(self.milestone_every) and self.keep_milestones != 0
        if not self.keep_last and not milestone_enabled:
            return

        recent = self._managed[-self.keep_last :] if self.keep_last else []
        milestones: list[_ManagedCheckpoint] = []
        if milestone_enabled:
            milestones = [item for item in self._managed if item.milestone]
            if self.keep_milestones is not None:
                milestones = milestones[-self.keep_milestones :]
        retained = {item.path for item in recent + milestones}

        remaining: list[_ManagedCheckpoint] = []
        for item in self._managed:
            # Malformed names are never deletion candidates: ambiguity fails
            # closed and protects the checkpoint.
            if item.path in retained or item.step is None:
                remaining.append(item)
                continue
            try:
                item.path.unlink(missing_ok=True)
            except OSError as exc:
                warnings.warn(
                    f"Could not prune managed checkpoint {item.path}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                remaining.append(item)
        self._managed = remaining
