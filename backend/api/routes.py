"""REST API routes for training control, data access, and game management."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

router = APIRouter(prefix="/api")


class TrainingStartRequest(BaseModel):
    model_size: str = "small"
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
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

@router.post("/training/start")
async def start_training(req: TrainingStartRequest):
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")

    from backend.training.config import TrainingConfig
    config = TrainingConfig.from_preset(req.model_size)

    # Override game / state if provided
    if req.game_id is not None:
        config.game_id = req.game_id
    if req.initial_state is not None:
        config.initial_state = req.initial_state
        config.env_state = req.initial_state

    # Override numeric hyperparams if provided
    if req.batch_size is not None:
        config.batch_size = req.batch_size
    if req.learning_rate is not None:
        config.learning_rate = req.learning_rate
    if req.replay_ratio is not None:
        config.replay_ratio = req.replay_ratio
    if req.num_envs is not None:
        config.num_envs = req.num_envs
    if req.resume_prefill is not None:
        config.resume_prefill = req.resume_prefill

    _trainer.start(config, fresh_start=req.fresh_start)
    return {
        "status": "started",
        "model_size": req.model_size,
        "game_id": config.game_id,
        "initial_state": config.initial_state,
        "fresh_start": req.fresh_start,
    }


@router.post("/training/stop")
async def stop_training():
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    _trainer.stop()
    return {"status": "stopped"}


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

@router.get("/videos")
async def list_videos():
    """List training videos from SheepRL's output directories."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    return _trainer.list_videos()


@router.get("/videos/{filename}")
async def get_video(filename: str):
    """Serve a training video file."""
    if _trainer is None:
        raise HTTPException(500, "Trainer not initialized")
    videos = _trainer.list_videos()
    for v in videos:
        if v["filename"] == filename:
            return FileResponse(v["path"], media_type="video/mp4")
    raise HTTPException(404, "Video not found")


@router.get("/episodes")
async def list_episodes():
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
