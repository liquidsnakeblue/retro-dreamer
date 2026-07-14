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
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHEEPRL_DIR = PROJECT_ROOT / "sheeprl"
GAMES_DIR = PROJECT_ROOT / "games"
JOBS_DIR = PROJECT_ROOT / "training-state" / "tools"
PYTHON = sys.executable

router = APIRouter(prefix="/api/tools")

_jobs: dict = {}
_lock = threading.Lock()


def _run_job(job_id: str, cmd: list, cwd: Path):
    job = _jobs[job_id]
    log_path = Path(job["log"])
    env = os.environ.copy()
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("PYGLET_HEADLESS", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # tools are CPU-only, never fight training
    try:
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
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


def submit(tool: str, cmd: list, cwd: Path = SHEEPRL_DIR) -> str:
    job_id = f"{tool}-{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _jobs[job_id] = {
        "id": job_id, "tool": tool, "status": "running",
        "cmd": [str(c) for c in cmd], "log": str(job_dir / "output.log"),
        "workdir": str(job_dir), "started_at": time.time(),
        "result": None,
    }
    threading.Thread(target=_run_job, args=(job_id, cmd, cwd), daemon=True).start()
    return job_id


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
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    out = dict(job)
    try:
        lines = Path(job["log"]).read_text().splitlines()
        out["log_tail"] = lines[-log_tail:]
    except OSError:
        out["log_tail"] = []
    return out


@router.get("/jobs")
def jobs_list():
    return {"jobs": sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)}
