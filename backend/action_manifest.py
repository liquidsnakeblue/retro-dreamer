"""Backend bridge to SheepRL's dependency-free action manifest contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "sheeprl"
    / "sheeprl"
    / "action_manifest.py"
)
_SPEC = importlib.util.spec_from_file_location("_retro_dreamer_action_manifest", _SOURCE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib contract guard
    raise ImportError(f"cannot load action manifest implementation from {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


ACTION_MANIFEST_FORMAT = _MODULE.ACTION_MANIFEST_FORMAT
action_manifest_hash = _MODULE.action_manifest_hash
build_action_manifest = _MODULE.build_action_manifest
load_action_manifest = _MODULE.load_action_manifest
validate_resume_action_binding = _MODULE.validate_resume_action_binding
write_action_manifest = _MODULE.write_action_manifest

__all__ = [
    "ACTION_MANIFEST_FORMAT",
    "action_manifest_hash",
    "build_action_manifest",
    "load_action_manifest",
    "validate_resume_action_binding",
    "write_action_manifest",
]
