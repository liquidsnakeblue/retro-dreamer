"""FastAPI server — main entry point for Retro Dreamer.

Starts:
- FastAPI on port 8080 (API + frontend)
- TensorBoard subprocess on port 6006
- Training controlled via dashboard / REST API
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path so "backend.*" imports work when running directly
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.games.manager import GameManager
from backend.training.trainer import DreamerV3Trainer
from backend.training.config import TrainingConfig
from backend.training.callbacks import WebSocketBroadcaster
from backend.api.routes import router as api_router, set_dependencies
from backend.tools import router as tools_router
from backend.copilot import router as copilot_router
from backend.api.ws import ConnectionManager

# Global state
trainer: DreamerV3Trainer | None = None
game_manager: GameManager | None = None
ws_manager = ConnectionManager()
tb_process: subprocess.Popen | None = None


def start_tensorboard(logdir: str, port: int = 6006):
    """Start TensorBoard as a subprocess."""
    global tb_process
    logdir_path = Path(logdir)
    logdir_path.mkdir(parents=True, exist_ok=True)

    try:
        import shutil
        tb_bin = shutil.which("tensorboard") or str(Path(sys.executable).parent / "tensorboard")
        tb_process = subprocess.Popen(
            [
                tb_bin,
                "--logdir", str(logdir_path),
                "--port", str(port),
                "--host", "0.0.0.0",
                "--reload_interval", "10",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[Server] TensorBoard started on port {port}")
    except Exception as e:
        print(f"[Server] Failed to start TensorBoard: {e}")


def stop_tensorboard():
    global tb_process
    if tb_process is not None:
        try:
            tb_process.terminate()
            tb_process.wait(timeout=5)
        except Exception:
            pass
        tb_process = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global trainer, game_manager

    # Create GameManager pointing at PROJECT_ROOT/games/
    game_manager = GameManager(PROJECT_ROOT / "games")

    # Create trainer with default config
    config = TrainingConfig.from_preset("small")
    broadcaster = WebSocketBroadcaster()
    broadcaster.set_loop(asyncio.get_event_loop())
    trainer = DreamerV3Trainer(config, game_manager=game_manager, ws_broadcaster=broadcaster)

    # Inject both into routes
    set_dependencies(trainer, game_manager)

    # Point TensorBoard at SheepRL's log directory for the default game
    tb_logdir = trainer.get_tensorboard_logdir()
    start_tensorboard(tb_logdir)

    print("[Server] Retro Dreamer ready")
    print("[Server] Dashboard: http://localhost:8080")
    print("[Server] TensorBoard: http://localhost:6006")
    print("[Server] API docs: http://localhost:8080/docs")

    yield

    # Cleanup
    if trainer and trainer.state.value in ("training", "paused"):
        trainer.stop()
    stop_tensorboard()


class _TrainerWSBridge:
    """Bridges the trainer's callback system to FastAPI's WebSocket manager."""

    def __init__(self, ws_manager: ConnectionManager):
        self._ws_manager = ws_manager
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop):
        self._loop = loop

    def broadcast(self, data: dict):
        if self._ws_manager.client_count == 0:
            return
        self._ws_manager.broadcast_sync(data)

    @property
    def client_count(self):
        return self._ws_manager.client_count


# Create FastAPI app
app = FastAPI(
    title="Retro Dreamer",
    description="DreamerV3 training studio for any gym-retro ROM",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router)
app.include_router(tools_router)
app.include_router(copilot_router)


# WebSocket endpoint
@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; accept any client messages (e.g. ping)
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# Serve frontend (built React app)
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="frontend-assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        """Serve the React SPA — all non-API routes go to index.html."""
        file_path = frontend_dist / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dist / "index.html"))
else:
    @app.get("/")
    async def no_frontend():
        return {
            "message": "Retro Dreamer",
            "note": "Frontend not built. Run: cd frontend && npm run build",
            "api_docs": "/docs",
            "tensorboard": "http://localhost:6006",
        }


def main():
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("RETRO_DREAMER_PORT", "8080")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
