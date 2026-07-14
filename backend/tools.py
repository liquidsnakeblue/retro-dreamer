"""Studio tool layer: long-running CPU jobs (probes, captures, walkers,
state builds) exposed as HTTP endpoints, drivable identically by the
dashboard UI, Claude, or the resident copilot. Harness-agnostic on purpose —
the tools ARE the studio's per-game craft; whoever calls them is swappable.

Job contract: each tool script's last stdout line is "RESULT <json>".
Jobs run as subprocesses; stdout streams to a log file; status + parsed
result served from memory (and jobs.json for post-restart inspection).
"""

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHEEPRL_DIR = PROJECT_ROOT / "sheeprl"
GAMES_DIR = PROJECT_ROOT / "games"
JOBS_DIR = PROJECT_ROOT / "training-state" / "tools"
PYTHON = sys.executable

router = APIRouter(prefix="/api/tools")

_jobs: dict = {}
_lock = threading.Lock()
_report_served_callback = None


def set_report_served_callback(callback):
    """Register a non-blocking observer for completed watch_brain reports."""
    global _report_served_callback
    with _lock:
        _report_served_callback = callback


def _run_job(job_id: str, cmd: list, cwd: Path):
    with _lock:
        job = _jobs[job_id]
        log_path = Path(job["log"])
    env = os.environ.copy()
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("PYGLET_HEADLESS", "1")
    # Tools are CPU-only, even when the server itself was launched with a GPU
    # selection in its environment. setdefault() would leak that selection.
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            with _lock:
                job["pid"] = proc.pid
            result = None
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                if line.startswith("RESULT "):
                    try:
                        result = json.loads(line[len("RESULT "):])
                    except json.JSONDecodeError:
                        pass
            code = proc.wait()
        with _lock:
            job["status"] = "done" if code == 0 else "failed"
            job["exit_code"] = code
            job["result"] = result
            job["ended_at"] = time.time()
    except Exception as exc:
        with _lock:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["ended_at"] = time.time()


def submit(
    tool: str,
    cmd: list,
    cwd: Path = SHEEPRL_DIR,
    job_id: Optional[str] = None,
) -> str:
    job_id = job_id or f"{tool}-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with _lock:
        _jobs[job_id] = {
            "id": job_id, "tool": tool, "status": "running",
            "cmd": [str(c) for c in cmd], "log": str(job_dir / "output.log"),
            "workdir": str(job_dir), "started_at": time.time(),
            "result": None,
        }
    threading.Thread(target=_run_job, args=(job_id, cmd, cwd), daemon=True).start()
    return job_id


def snapshot_jobs() -> list[dict]:
    """Bounded job facts for ambient studio state (no commands or log paths)."""
    with _lock:
        jobs = [dict(job) for job in _jobs.values()]
    out = []
    for job in sorted(jobs, key=lambda item: item["started_at"], reverse=True):
        result = job.get("result")
        if result is not None and len(json.dumps(result, default=str)) > 1000:
            result = {"keys": sorted(result) if isinstance(result, dict) else None,
                      "detail": "result omitted from compact studio state"}
        out.append({
            "id": job["id"],
            "tool": job["tool"],
            "status": job["status"],
            "started_at": job["started_at"],
            "ended_at": job.get("ended_at"),
            "result": result,
            "error": job.get("error"),
        })
    return out


def _game_dir(game_id: str) -> Path:
    d = GAMES_DIR / game_id
    if not (d / "data.json").exists():
        raise HTTPException(404, f"game '{game_id}' has no custom integration dir")
    return d


# ------------------------------------------------------------------
# Tool endpoints — each returns {"job_id": ...}; poll /jobs/{id}
# ------------------------------------------------------------------

class ProbeReq(BaseModel):
    game_id: str
    states: list[str]
    steps: int = 400
    actions: str = "all"  # or "0,2,4"


@router.post("/reward_probe")
def reward_probe(req: ProbeReq):
    gd = _game_dir(req.game_id)
    return {"job_id": submit("reward_probe", [
        PYTHON, str(SHEEPRL_DIR / "_retro_tool_probe.py"),
        req.game_id, str(gd), ",".join(req.states), str(req.steps), req.actions,
    ])}


class RamCaptureReq(BaseModel):
    game_id: str
    state: str
    steps: int = 2000
    checkpoint: str = "head"


@router.post("/ram_capture")
def ram_capture(req: RamCaptureReq):
    ckpt = req.checkpoint
    if ckpt == "head":
        from backend import catalog as _catalog

        con = _catalog.connect()
        head = _catalog.get_resumable_head(con, req.game_id)
        con.close()
        if not head:
            raise HTTPException(409, f"no trained brain for {req.game_id}; pass an explicit checkpoint")
        ckpt = head["checkpoint_path"]
    job_id = f"ramcap-{uuid.uuid4().hex[:6]}"
    out = JOBS_DIR / "captures" / f"{req.game_id}-{req.state}-{job_id}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    return {"job_id": submit("ram_capture", [
        PYTHON, str(SHEEPRL_DIR / "_retro_ram_capture.py"),
        ckpt, req.state, str(req.steps), str(out),
    ])}


class WatchBrainReq(BaseModel):
    game_id: str
    state: str
    steps: int = Field(default=1400, ge=1, le=100_000)
    checkpoint: str = "latest"


@router.post("/watch_brain")
def watch_brain(req: WatchBrainReq):
    """Replay one game-scoped brain and turn its RAM trace into a report."""
    game_dir = _game_dir(req.game_id)  # 404 for an unknown/non-custom game

    training_config = game_dir / "training.json"
    try:
        parsed_training = json.loads(training_config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            409, f"game '{req.game_id}' has no valid training.json: {exc}"
        ) from exc
    if not isinstance(parsed_training, dict):
        raise HTTPException(
            409, f"game '{req.game_id}' has no valid training.json object"
        )

    available_states = {
        path.stem for path in (game_dir / "states").glob("*.state")
    }
    if req.state not in available_states:
        raise HTTPException(
            404, f"state '{req.state}' not found for game '{req.game_id}'"
        )

    checkpoint = req.checkpoint
    if checkpoint == "latest":
        from backend import catalog as _catalog

        con = _catalog.connect()
        try:
            head = _catalog.get_resumable_head(con, req.game_id)
        finally:
            con.close()
        if not head or not head["checkpoint_path"]:
            raise HTTPException(
                409, f"no resumable checkpoint for game '{req.game_id}'"
            )
        checkpoint = head["checkpoint_path"]
    elif not Path(checkpoint).is_file():
        raise HTTPException(404, f"checkpoint not found: {checkpoint}")

    # Preselect the managed job id so output.log and both artifacts live in
    # one directory owned by the existing job manager.
    job_id = f"watch_brain-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    npz_path = job_dir / "capture.npz"
    report_path = job_dir / "report.txt"
    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "backend" / "watch_brain_job.py"),
        str(checkpoint),
        req.state,
        str(req.steps),
        str(npz_path),
        str(training_config),
        str(report_path),
    ]
    return {
        "job_id": submit(
            "watch_brain", cmd, cwd=PROJECT_ROOT, job_id=job_id
        )
    }


class RamDiffBoundaryReq(BaseModel):
    window: int = 60
    captures: list[dict]  # [{"npz": path, "event_step": int}]


@router.post("/ram_diff")
def ram_diff(req: RamDiffBoundaryReq):
    specs = [f"{c['npz']}:{c['event_step']}" for c in req.captures]
    return {"job_id": submit("ram_diff", [
        PYTHON, str(SHEEPRL_DIR / "_retro_ram_diff.py"), "boundary", str(req.window), *specs,
    ])}


class BuildStateReq(BaseModel):
    game_id: str
    plan: list  # [[wait_frames, "BUTTONS"], ...]
    out_state_name: str
    start_state: Optional[str] = None


@router.post("/build_state")
def build_state(req: BuildStateReq):
    gd = _game_dir(req.game_id)
    job_id = f"buildstate-{uuid.uuid4().hex[:6]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    plan_path = job_dir / "plan.json"
    plan_path.write_text(json.dumps(req.plan))
    cmd = [
        PYTHON, str(SHEEPRL_DIR / "_retro_build_state.py"),
        req.game_id, str(gd), str(plan_path), req.out_state_name, str(job_dir / "shots"),
    ]
    if req.start_state:
        cmd.append(req.start_state)
    return {"job_id": submit("build_state", cmd)}


class WalkerReq(BaseModel):
    game_id: str
    start_state: str
    n_captures: int = 1
    checkpoint: str = "head"
    flag: str = "race_on"
    live_value: float = 1
    tap_button: str = "START"
    prefix: str = "capture"


@router.post("/run_walker")
def run_walker(req: WalkerReq):
    gd = _game_dir(req.game_id)
    job_id = f"walker-{uuid.uuid4().hex[:6]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return {"job_id": submit("run_walker", [
        PYTHON, str(SHEEPRL_DIR / "_retro_walker.py"),
        req.game_id, str(gd), req.checkpoint, req.start_state,
        str(req.n_captures), str(job_dir),
        "--flag", req.flag, "--live", str(req.live_value),
        "--tap", req.tap_button, "--prefix", req.prefix,
    ])}


class RecordReq(BaseModel):
    game_id: str
    state: str
    seconds: int = 60
    checkpoint: str = "latest"


@router.post("/record_episode")
def record_episode(req: RecordReq):
    # game_id was previously accepted but IGNORED: the recorder inferred the
    # game from the checkpoint, and checkpoint="latest" resolved GLOBALLY
    # (catalog watch-head, else a cross-game mtime scan), so a caller asking
    # to record game A could silently record game B's brain. Honor game_id:
    # validate the game exists, and scope "latest" to THAT game's head.
    _game_dir(req.game_id)  # 404 if no custom integration dir
    ckpt = req.checkpoint
    if ckpt == "latest":
        from backend import catalog as _catalog
        con = _catalog.connect()
        try:
            head = _catalog.get_resumable_head(con, req.game_id)
        finally:
            con.close()
        if not head or not head["checkpoint_path"]:
            raise HTTPException(
                409, f"no resumable checkpoint for game '{req.game_id}'"
            )
        ckpt = head["checkpoint_path"]
    job_id = f"record-{uuid.uuid4().hex[:6]}"
    out = JOBS_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)
    return {"job_id": submit("record_episode", [
        PYTHON, str(SHEEPRL_DIR / "_retro_record.py"),
        ckpt, str(req.seconds), str(out / "episode.mp4"), req.state,
    ])}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, log_tail: int = 20):
    # The worker publishes terminal status + result under this same lock. Copy
    # them atomically so a completed response can never miss its report tap.
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        out = dict(job)
        callback = _report_served_callback
    try:
        lines = Path(out["log"]).read_text().splitlines()
        out["log_tail"] = lines[-log_tail:]
    except OSError:
        out["log_tail"] = []
    result = out.get("result")
    if (
        out.get("tool") == "watch_brain"
        and out.get("status") == "done"
        and isinstance(result, dict)
        and isinstance(result.get("report_text"), str)
        and callback is not None
    ):
        try:
            callback(job_id, result["report_text"])
        except Exception as exc:
            # Grounding is observability, never a reason to break the tool API.
            print(f"[grounding] report tap failed for {job_id}: {exc}")
    return out


@router.get("/jobs")
def jobs_list():
    with _lock:
        jobs = [dict(job) for job in _jobs.values()]
    return {"jobs": sorted(jobs, key=lambda j: j["started_at"], reverse=True)}
