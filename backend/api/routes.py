"""REST API routes for training control, data access, and game management."""

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

router = APIRouter(prefix="/api")


class TrainingStartRequest(BaseModel):
    model_size: str = "small"
    batch_size: Optional[int] = None
    replay_ratio: Optional[float] = None
    num_envs: Optional[int] = None
    fresh_start: bool = False
    game_id: Optional[str] = None
    initial_state: Optional[str] = None
    resume_prefill: Optional[int] = None


class CreateGameRequest(BaseModel):
    game_id: str
    display_name: str
    system: str


# Injected by server.py
_trainer = None
_game_manager = None


def set_dependencies(trainer, game_manager):
    global _trainer, _game_manager
    _trainer = trainer
    _game_manager = game_manager


# ------------------------------------------------------------------
# Training endpoints
# ------------------------------------------------------------------

VALID_MODEL_SIZES = ("debug", "small", "medium", "large", "xl")


@router.post("/training/start")
async def start_training(req: TrainingStartRequest):
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    if req.model_size not in VALID_MODEL_SIZES:
        # from_preset() and the launcher both silently fall back to small —
        # reject bad names here instead of echoing them back as accepted
        raise HTTPException(400, f"Unknown model_size {req.model_size!r}; valid: {list(VALID_MODEL_SIZES)}")

    from backend.training.config import TrainingConfig
    config = TrainingConfig.from_preset(req.model_size)

    # Override game / state if provided
    if req.game_id is not None:
        # Unknown game ids must fail HERE with a 404, not as a hydra stack
        # trace from the child process two seconds later.
        if _game_manager is not None:
            try:
                _game_manager.get_game(req.game_id)
            except FileNotFoundError:
                raise HTTPException(404, f"Unknown game '{req.game_id}'")
        config.game_id = req.game_id
    if req.initial_state is not None:
        config.initial_state = req.initial_state
        config.env_state = req.initial_state
    elif _game_manager is not None:
        # Fall back to the game's declared default_state (previously never
        # consulted; the hardcoded "go" default only happens to be right for
        # F-Zero). get_game() covers custom games (metadata.json) AND
        # built-in stable-retro games (their first shipped state).
        try:
            default_state = (_game_manager.get_game(config.game_id) or {}).get("default_state")
            if default_state:
                config.initial_state = default_state
                config.env_state = default_state
        except Exception as exc:
            print(f"[API] default_state lookup failed: {exc}")

    # Override numeric hyperparams if provided
    if req.batch_size is not None:
        config.batch_size = req.batch_size
    if req.replay_ratio is not None:
        config.replay_ratio = req.replay_ratio
    if req.num_envs is not None:
        config.num_envs = req.num_envs
    if req.resume_prefill is not None:
        config.resume_prefill = req.resume_prefill

    _trainer.start(config, fresh_start=req.fresh_start)

    # Persist the exact start request so the watchdog (or a server restart)
    # can resume WHAT WAS RUNNING instead of hardcoding parameters.
    try:
        from pathlib import Path
        import json as _json

        state_dir = Path(__file__).resolve().parent.parent.parent / "training-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        body = req.model_dump()
        body["fresh_start"] = False  # a watchdog resume must never fresh-start
        (state_dir / "last_start_request.json").write_text(_json.dumps(body, indent=2))
    except Exception as exc:
        print(f"[API] failed to persist last_start_request: {exc}")

    return {
        "status": "started",
        "model_size": req.model_size,
        "game_id": config.game_id,
        "initial_state": config.initial_state,
        "fresh_start": req.fresh_start,
    }


@router.post("/training/switch")
async def switch_training(req: TrainingStartRequest):
    """Atomic game/lineage switch: gracefully suspend whatever is training
    (final loss-free checkpoint registered as its lineage head), then start
    the requested game — resuming its own head if it has one, fresh brain if
    it never trained."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    suspended = None
    if _trainer.status.state == "training":
        suspended = _trainer.stop(graceful=True, timeout=180.0)
    result = await start_training(req)
    result["status"] = "switched"
    result["suspended_snapshot"] = suspended
    return result


@router.post("/training/stop")
async def stop_training():
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    ack = _trainer.stop()  # graceful by default; SIGTERM fallback inside
    return {"status": "stopped", "final_snapshot": ack}


@router.post("/training/suspend")
async def suspend_training():
    """Graceful suspend: force a final checkpoint via the control channel,
    register it as the lineage's resumable head, then stop. The returned
    snapshot is guaranteed loss-free (no steps discarded)."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    ack = _trainer.stop(graceful=True, timeout=180.0)
    return {"status": "suspended", "final_snapshot": ack}


@router.get("/advisor/model_size")
async def model_size_advisor():
    """Recommend a DreamerV3 size for this machine's GPU. Measured VRAM at
    batch 16 / seq 64 / AMP on an RTX 5090: XL ~31GB, L ~19GB, S ~7GB."""
    import torch

    if not torch.cuda.is_available():
        return {"gpu": None, "vram_gb": 0, "recommended": "debug",
                "note": "No CUDA GPU visible — debug size only (CPU training is impractical)."}
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    tiers = [("xl", 32.0), ("large", 20.0), ("medium", 12.0), ("small", 8.0), ("debug", 0.0)]
    rec = next(name for name, need in tiers if vram >= need)
    return {
        "gpu": props.name,
        "vram_gb": round(vram, 1),
        "recommended": rec,
        "fits": [name for name, need in tiers if vram >= need],
        "note": "Bigger models score higher AND need less data (DreamerV3 Fig 6c) — "
                "run the largest size that fits.",
    }


@router.get("/workspaces")
async def workspaces():
    """Games + lineages + resumable heads, from the training catalog."""
    from backend import catalog as _catalog

    con = _catalog.connect()
    out = []
    for g in con.execute("SELECT * FROM games"):
        lineages = []
        for ln in con.execute(
            "SELECT * FROM lineages WHERE game_id=? ORDER BY created_at", (g["id"],)
        ):
            head = _catalog.get_resumable_head(con, g["id"], ln["name"])
            running = con.execute(
                "SELECT COUNT(*) c FROM sessions WHERE lineage_id=? AND status='running'",
                (ln["id"],),
            ).fetchone()["c"]
            lineages.append({
                "name": ln["name"],
                "status": ln["status"],
                "running": bool(running),
                "head_step": head["step"] if head else None,
                "head_checkpoint": head["checkpoint_path"] if head else None,
            })
        out.append({
            "game_id": g["id"],
            "display_name": g["display_name"],
            "lineages": lineages,
        })
    con.close()
    return {"workspaces": out}


@router.post("/training/pause")
async def pause_training():
    """Not supported in subprocess mode."""
    return {"status": "not_supported", "detail": "Pause not available in subprocess mode"}


@router.post("/training/resume")
async def resume_training():
    """Not supported in subprocess mode."""
    return {"status": "not_supported", "detail": "Resume not available in subprocess mode"}


@router.get("/training/status")
async def training_status():
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    status = _trainer.status
    return {
        "state": status.state,
        "game_id": status.game_id,
        "current_step": status.current_step,
        "current_episode": status.current_episode,
        "elapsed_time": status.elapsed_time,
        "steps_per_second": status.steps_per_second,
        "avg_return": status.avg_return,
        "avg_length": status.avg_length,
        "max_return": status.max_return,
        "gpu_memory_used": status.gpu_memory_used,
        "gpu_memory_total": status.gpu_memory_total,
        "error_message": status.error_message,
    }


@router.get("/config")
async def get_config():
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    return _trainer.config.to_dict()


# ------------------------------------------------------------------
# Video / checkpoint endpoints
# ------------------------------------------------------------------

# Sync (def) routes on purpose: FastAPI runs them in the threadpool, so the
# directory scan / ffprobe / file IO never block the event loop that is also
# streaming video bytes and answering status polls.

@router.get("/videos")
def list_videos():
    """List training videos from SheepRL's output directories."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    return _trainer.list_videos()


@router.get("/videos/{video_id:path}")
def get_video(video_id: str):
    """Serve a training video by its unique id (run-relative path).

    Bare filenames collide — every eval run writes its own
    rl-video-episode-0.mp4 — so id is the reliable key; filename
    match kept as a fallback for old clients.
    """
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    path = _trainer.resolve_video(video_id)
    if path:
        return FileResponse(path, media_type="video/mp4")
    for v in _trainer.list_videos():
        if v["filename"] == video_id:
            return FileResponse(v["path"], media_type="video/mp4")
    raise HTTPException(404, "Video not found")


@router.get("/episodes")
def list_episodes():
    """List training videos (alias for /videos)."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    return _trainer.list_videos()


@router.get("/checkpoints")
async def list_checkpoints():
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    return _trainer.list_checkpoints()


# ------------------------------------------------------------------
# Logs / metrics endpoints
# ------------------------------------------------------------------

@router.get("/training/logs")
async def training_logs(n: int = 50):
    """Return the last N lines of SheepRL output."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    lines = list(_trainer._log_lines)
    return {"lines": lines[-n:]}


@router.get("/metrics/history")
async def metrics_history():
    """Get current metrics summary."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    return _trainer.metrics.get_recent_metrics()


@router.get("/tensorboard/logdir")
async def tensorboard_logdir():
    """Return the TensorBoard log directory for SheepRL runs."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    return {"logdir": _trainer.get_tensorboard_logdir()}


# ------------------------------------------------------------------
# Game management endpoints
# ------------------------------------------------------------------

@router.get("/games")
async def list_games():
    """List all available games."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    return _game_manager.list_games()


@router.get("/games/{game_id}")
async def get_game(game_id: str):
    """Get full metadata and available states for a game."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    try:
        return _game_manager.get_game(game_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/games/{game_id}/states")
async def list_states(game_id: str):
    """List available save states for a game."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    try:
        states = _game_manager.list_states(game_id)
        return {"game_id": game_id, "states": states}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@router.get("/games/{game_id}/config/{filename}")
async def get_game_config(game_id: str, filename: str):
    """Read a config file for a game."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    try:
        return _game_manager.read_config(game_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.put("/games/{game_id}/config/{filename}")
async def put_game_config(game_id: str, filename: str, data: dict):
    """Write a config file for a game."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    try:
        _game_manager.write_config(game_id, filename, data)
        return {"status": "ok", "game_id": game_id, "filename": filename}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/games/promote")
async def promote_game(game_id: str):
    """Promote a ROM-ready built-in game into a full custom workspace: copies
    the stock integration (RAM map, scenario, states) + imported ROM into
    games/<id>/ so the standard reward/probe/train pipeline applies."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    try:
        return {"status": "promoted", "game_id": game_id,
                **_game_manager.promote_game(game_id)}
    except FileExistsError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/games/import")
async def import_game(
    game_id: str,
    display_name: str,
    system: str,
    rom: "UploadFile" = None,
):
    """One-shot game onboarding entry point: scaffold a workspace and drop
    the user's ROM into it. The RAM workbench / reward builder take it from
    there. We never ship or fetch ROMs — the file comes from the user."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    if rom is None:
        raise HTTPException(400, "multipart 'rom' file is required")
    import hashlib
    from pathlib import Path as _P

    ext = _P(rom.filename or "rom.bin").suffix.lower() or ".bin"
    try:
        game_dir = _game_manager.create_game(game_id, display_name, system)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    body = await rom.read()
    (game_dir / f"rom{ext}").write_bytes(body)
    (game_dir / "rom.sha").write_text(hashlib.sha1(body).hexdigest() + "\n")
    return {
        "status": "imported",
        "game_id": game_id,
        "game_dir": str(game_dir),
        "rom_bytes": len(body),
        "rom_sha1": hashlib.sha1(body).hexdigest(),
        "next_steps": [
            "define RAM variables in data.json (RAM workbench / ram_capture + ram_diff tools)",
            "define reward + done in training.json (reward builder; validate with reward_probe)",
            "capture at least one save state (build_state tool)",
            "start training",
        ],
    }


@router.post("/games")
async def create_game(req: CreateGameRequest):
    """Scaffold a new game directory."""
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    try:
        game_dir = _game_manager.create_game(req.game_id, req.display_name, req.system)
        return {
            "status": "created",
            "game_id": req.game_id,
            "game_dir": str(game_dir),
        }
    except FileExistsError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/games/{game_id}/scaffold")
async def scaffold_builtin(game_id: str):
    """Create custom config files for a built-in retro game.

    Copies data.json/scenario.json/metadata.json from built-in retro data,
    creates training.json and actions.json with system defaults.
    """
    if _game_manager is None:
        raise HTTPException(500, "GameManager not initialized")
    if not _game_manager.is_builtin(game_id):
        raise HTTPException(404, f"'{game_id}' is not a built-in retro game")
    try:
        game_dir = _game_manager.scaffold_from_builtin(game_id)
        return {
            "status": "scaffolded",
            "game_id": game_id,
            "game_dir": str(game_dir),
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))
