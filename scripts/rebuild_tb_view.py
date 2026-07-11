#!/usr/bin/env python3
"""Rebuild the TensorBoard symlink view under sheeprl/logs/tb/.

Layout: logs/tb/<game_id>/<SIZE>_<launch-timestamp> -> ../../runs/.../version_N

Real run dirs keep SheepRL's timestamp naming (resume discovery, the
catalog, and the F-Zero buffer symlink depend on those paths staying put).
TensorBoard should be pointed at sheeprl/logs/tb, where every run is game-
and model-tagged, so the dashboard's run regex filter can slice by game
(`FZero-Snes`), by model (`/XL_`), or both (`SuperMario.*L_`).

Idempotent: prunes dangling links, skips runs with no event data, never
touches the real run dirs. The trainer adds links for new runs live; this
script exists for backfill and for cleaning up after manual deletions.
"""
import os
import re
import sys
from pathlib import Path

SHEEPRL = Path(__file__).resolve().parent.parent / "sheeprl"
RUNS = SHEEPRL / "logs" / "runs" / "dreamer_v3"
TB = SHEEPRL / "logs" / "tb"

# dense_units in the resolved config identifies the SheepRL model size.
SIZE_BY_DENSE = {1024: "XL", 768: "L", 640: "M", 512: "S", 256: "XS"}


def model_size(version_dir: Path) -> str:
    for cfg in (version_dir / "config.yaml", version_dir.parent / ".hydra" / "config.yaml"):
        try:
            m = re.search(r"^\s*dense_units:\s*(\d+)", cfg.read_text(), re.M)
            if m:
                return SIZE_BY_DENSE.get(int(m.group(1)), f"D{m.group(1)}")
        except OSError:
            continue
    return "UNK"


def has_event_data(version_dir: Path) -> bool:
    return any(
        f.stat().st_size > 0 for f in version_dir.glob("events.out.tfevents.*")
    )


def main() -> int:
    pruned = linked = kept = skipped = 0
    for link in TB.glob("*/*"):
        if link.is_symlink() and not link.exists():
            link.unlink()
            pruned += 1
    for version_dir in sorted(RUNS.glob("*/*/version_*")):
        game = version_dir.parent.parent.name
        if not has_event_data(version_dir):
            skipped += 1
            continue
        stamp = version_dir.parent.name[:19]
        suffix = "" if version_dir.name == "version_0" else f"_{version_dir.name}"
        link = TB / game / f"{model_size(version_dir)}_{stamp}{suffix}"
        if link.is_symlink():
            kept += 1
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(version_dir, link.parent))
        linked += 1
    for game_dir in TB.glob("*"):
        if game_dir.is_dir() and not any(game_dir.iterdir()):
            game_dir.rmdir()
    print(f"tb view: +{linked} linked, {kept} kept, {pruned} pruned, {skipped} no-data skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
