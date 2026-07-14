#!/usr/bin/env python3
"""Compose a trained-brain RAM capture and the frozen episode-report CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHEEPRL_DIR = PROJECT_ROOT / "sheeprl"
CAPTURE_SCRIPT = SHEEPRL_DIR / "_retro_ram_capture.py"
REPORT_ENGINE = PROJECT_ROOT / "backend" / "episode_report.py"
DEFAULT_REPORT_PYTHON = (
    "/usr/bin/python3" if Path("/usr/bin/python3").is_file() else "python3"
)


def _forward(prefix: str, output: str) -> None:
    """Log child output without exposing a nested line to RESULT parsing."""
    for line in output.splitlines():
        print(f"[{prefix}] {line}", flush=True)


def run_watch_brain(
    checkpoint: Path,
    state: str,
    steps: int,
    npz_path: Path,
    training_config: Path,
    report_path: Path,
    *,
    capture_python: str = sys.executable,
    report_python: str | None = None,
) -> dict:
    npz_path = npz_path.resolve()
    report_path = report_path.resolve()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    capture = subprocess.run(
        [
            capture_python,
            str(CAPTURE_SCRIPT),
            str(checkpoint.resolve()),
            state,
            str(steps),
            str(npz_path),
        ],
        cwd=str(SHEEPRL_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _forward("capture", capture.stdout or "")
    if capture.returncode != 0:
        raise RuntimeError(f"RAM capture failed with exit code {capture.returncode}")
    if not npz_path.is_file():
        raise RuntimeError(f"RAM capture succeeded without creating {npz_path}")

    report = subprocess.run(
        [
            report_python
            or os.environ.get("RETRO_REPORT_PYTHON", DEFAULT_REPORT_PYTHON),
            str(REPORT_ENGINE),
            str(npz_path),
            str(training_config.resolve()),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _forward("report", report.stdout or "")
    _forward("report stderr", report.stderr or "")
    if report.returncode != 0:
        raise RuntimeError(f"episode report failed with exit code {report.returncode}")

    report_text = report.stdout
    report_path.write_text(report_text)
    return {
        "npz_path": str(npz_path),
        "report_path": str(report_path),
        "report_text": report_text,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 6:
        print(
            "usage: watch_brain_job.py <checkpoint> <state> <steps> "
            "<capture.npz> <training.json> <report.txt>",
            file=sys.stderr,
        )
        return 2
    try:
        result = run_watch_brain(
            Path(args[0]), args[1], int(args[2]), Path(args[3]),
            Path(args[4]), Path(args[5]),
        )
    except Exception as exc:
        print(f"watch_brain failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
