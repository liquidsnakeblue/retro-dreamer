"""Immutable, content-addressed action manifests for Retro Dreamer.

This module deliberately uses only the Python standard library.  SheepRL
imports it normally, while the backend loads the same file by path so both
processes agree on the exact bytes covered by the manifest hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union


ACTION_MANIFEST_FORMAT = "retro-dreamer-action-manifest-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GAME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PathLike = Union[str, os.PathLike[str]]


def _canonical_json(value: Any) -> bytes:
    """Return the one JSON encoding used for hashing and persistence."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action manifest contains non-JSON data: {exc}") from exc
    return encoded.encode("utf-8")


def _normalized_payload(game_id: str, actions_data_or_rows: Any) -> dict:
    if (
        not isinstance(game_id, str)
        or not _GAME_ID_RE.fullmatch(game_id)
        or game_id in {".", ".."}
    ):
        raise ValueError(
            "action manifest game_id must use only letters, numbers, dot, "
            "underscore, and hyphen, and cannot be '.' or '..'"
        )

    if isinstance(actions_data_or_rows, Mapping):
        if "actions" not in actions_data_or_rows:
            raise ValueError("action manifest source must contain an 'actions' list")
        rows = actions_data_or_rows["actions"]
    else:
        rows = actions_data_or_rows

    if not isinstance(rows, list) or not rows:
        raise ValueError("action manifest actions must be a non-empty list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"action manifest actions[{index}] must be a JSON object"
            )
        if set(row) != {"name", "buttons"}:
            raise ValueError(
                f"action manifest actions[{index}] must contain exactly "
                "'name' and 'buttons'"
            )
        name = row.get("name")
        buttons = row.get("buttons")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"action manifest actions[{index}].name must be a non-empty string"
            )
        if not isinstance(buttons, list):
            raise ValueError(
                f"action manifest actions[{index}].buttons must be a list"
            )
        names_only = all(isinstance(button, str) for button in buttons)
        bits_only = all(
            isinstance(button, int) and not isinstance(button, bool)
            and button in (0, 1)
            for button in buttons
        )
        if not names_only and not bits_only:
            raise ValueError(
                f"action manifest actions[{index}].buttons must contain either "
                "button names or 0/1 integers"
            )

    payload = {
        "format": ACTION_MANIFEST_FORMAT,
        "game_id": game_id,
        "actions": rows,
    }
    # A canonical JSON round trip both proves that the rows are JSON-safe,
    # normalizes them to JSON-native values, and detaches the manifest from a
    # mutable actions.json object held by callers.
    return json.loads(_canonical_json(payload).decode("utf-8"))


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_action_manifest(game_id: str, actions_data_or_rows: Any) -> dict:
    """Build a detached v1 envelope from exact, ordered authoring rows."""
    payload = _normalized_payload(game_id, actions_data_or_rows)
    return {**payload, "sha256": _payload_hash(payload)}


def action_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Validate an envelope and return its embedded content hash."""
    if not isinstance(manifest, Mapping):
        raise ValueError("action manifest must be a JSON object")

    allowed = {"format", "game_id", "actions", "sha256"}
    missing = allowed - set(manifest)
    unknown = set(manifest) - allowed
    if missing:
        raise ValueError(f"action manifest is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"action manifest has unknown fields: {sorted(unknown)}")
    if manifest["format"] != ACTION_MANIFEST_FORMAT:
        raise ValueError(
            "unsupported action manifest format "
            f"{manifest['format']!r}; expected {ACTION_MANIFEST_FORMAT!r}"
        )

    # Reuse source validation, then hash the normalized payload.
    payload = _normalized_payload(manifest["game_id"], manifest["actions"])
    embedded = manifest["sha256"]
    if not isinstance(embedded, str) or not _SHA256_RE.fullmatch(embedded):
        raise ValueError("action manifest sha256 must be 64 lowercase hex characters")
    computed = _payload_hash(payload)
    if embedded != computed:
        raise ValueError(
            "action manifest hash mismatch: "
            f"embedded {embedded}, computed {computed}; the manifest is corrupt or tampered"
        )
    return embedded


def load_action_manifest(
    path: _PathLike,
    expected_game_id: Optional[str] = None,
    expected_hash: Optional[str] = None,
) -> dict:
    """Load and verify an immutable manifest and optional launch bindings."""
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot read action manifest {manifest_path}: {exc}"
        ) from exc
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"action manifest {manifest_path} is not valid JSON: {exc}"
        ) from exc

    digest = action_manifest_hash(manifest)
    if expected_game_id is not None and manifest["game_id"] != expected_game_id:
        raise ValueError(
            f"action manifest game mismatch: expected {expected_game_id!r}, "
            f"found {manifest['game_id']!r} in {manifest_path}"
        )
    if expected_hash is not None:
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            raise ValueError("expected action manifest hash must be 64 lowercase hex characters")
        if digest != expected_hash:
            raise ValueError(
                f"action manifest launch hash mismatch: expected {expected_hash}, "
                f"found {digest} in {manifest_path}"
            )
    return manifest


def validate_resume_action_binding(
    env_id: str,
    saved_wrapper: Mapping[str, Any],
    requested_wrapper: Mapping[str, Any],
) -> None:
    """Enforce Retro Dreamer's hard resume binding without affecting other envs."""
    if env_id != "retro-dreamer":
        return
    if not isinstance(saved_wrapper, Mapping) or not isinstance(
        requested_wrapper, Mapping
    ):
        raise ValueError("Retro Dreamer resume has no wrapper action-manifest binding")
    saved_hash = saved_wrapper.get("action_manifest_hash")
    saved_path = saved_wrapper.get("action_manifest")
    requested_hash = requested_wrapper.get("action_manifest_hash")
    if not saved_hash or not saved_path:
        raise ValueError(
            "Legacy checkpoint has no immutable action manifest; action ordering "
            "is unprovable. Start fresh or explicitly migrate it."
        )
    if not requested_hash or requested_hash != saved_hash:
        raise ValueError(
            "Resume action manifest mismatch: checkpoint uses "
            f"{saved_hash}, requested launch uses {requested_hash or 'none'}."
        )


def write_action_manifest(manifest: Mapping[str, Any], lineage_dir: _PathLike) -> Path:
    """Atomically publish a manifest at ``action-manifests/<hash>.json``.

    A complete, fsynced temporary inode is hard-linked into place, so readers
    can never observe a partial file.  An existing content-addressed file is
    verified rather than overwritten.
    """
    digest = action_manifest_hash(manifest)
    manifest_dir = Path(lineage_dir) / "action-manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / f"{digest}.json"

    if target.is_symlink():
        raise ValueError(f"action manifest target must not be a symlink: {target}")
    if target.exists():
        if not target.is_file():
            raise ValueError(f"action manifest target is not a regular file: {target}")
        load_action_manifest(
            target,
            expected_game_id=manifest["game_id"],
            expected_hash=digest,
        )
        return target

    # Normalize the persisted envelope as well as the hashed payload.  The
    # trailing newline is outside the JSON value and does not affect hashing.
    detached = json.loads(_canonical_json(dict(manifest)).decode("utf-8"))
    data = _canonical_json(detached) + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".tmp", dir=manifest_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            # Another launcher won the race.  Its bytes must describe the
            # exact same content or the lineage store is not trustworthy.
            if target.is_symlink() or not target.is_file():
                raise ValueError(
                    f"action manifest race produced a non-regular target: {target}"
                )
            load_action_manifest(
                target,
                expected_game_id=manifest["game_id"],
                expected_hash=digest,
            )
        else:
            try:
                directory_fd = os.open(manifest_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Some filesystems do not support directory fsync.  The file
                # itself is already complete and atomically visible.
                pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


__all__ = [
    "ACTION_MANIFEST_FORMAT",
    "action_manifest_hash",
    "build_action_manifest",
    "load_action_manifest",
    "validate_resume_action_binding",
    "write_action_manifest",
]
