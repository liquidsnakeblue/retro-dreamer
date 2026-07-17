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

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

# Add project root to path so "backend.*" imports work when running directly
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.games.manager import GameManager
from backend.training.trainer import DreamerV3Trainer
from backend.training.config import TrainingConfig
from backend.training.callbacks import WebSocketBroadcaster
from backend.api.routes import (
    router as api_router,
    set_dependencies,
    set_studio_state_builder as set_api_state_builder,
    set_training_planner as set_api_training_planner,
)
from backend.tools import (
    router as tools_router,
    set_report_served_callback,
)
from backend.copilot import (
    cache_served_watch_report,
    router as copilot_router,
    set_studio_state_builder as set_copilot_state_builder,
)
from backend.studio_state import StudioStateBuilder
from backend.training.planner import TrainingPlanner
from backend.api.ws import ConnectionManager

# Global state
trainer: DreamerV3Trainer | None = None
game_manager: GameManager | None = None
ws_manager = ConnectionManager()
tb_process: subprocess.Popen | None = None
studio_state_builder: StudioStateBuilder | None = None
training_planner: TrainingPlanner | None = None


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
                # Served through this app's /tensorboard/ reverse proxy so the
                # dashboard works on a single hostname (Cloudflare tunnel).
                "--path_prefix", "/tensorboard",
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
    global trainer, game_manager, studio_state_builder, training_planner

    # Create GameManager pointing at PROJECT_ROOT/games/
    game_manager = GameManager(PROJECT_ROOT / "games")

    # Create trainer with default config
    config = TrainingConfig.from_preset("small")
    broadcaster = WebSocketBroadcaster()
    broadcaster.set_loop(asyncio.get_event_loop())
    trainer = DreamerV3Trainer(config, game_manager=game_manager, ws_broadcaster=broadcaster)

    # Inject both into routes
    set_dependencies(trainer, game_manager)
    studio_state_builder = StudioStateBuilder(game_manager, trainer)
    set_api_state_builder(studio_state_builder)
    set_copilot_state_builder(studio_state_builder)
    set_report_served_callback(cache_served_watch_report)
    training_planner = TrainingPlanner(studio_state_builder)
    set_api_training_planner(training_planner)

    # A previous server that crashed/bounced mid-training leaves its session
    # row 'running' forever, which poisons every running-state readout
    from backend import catalog as _catalog
    _con = _catalog.connect()
    _orphans = _catalog.close_orphaned_sessions(_con)
    _con.close()
    if _orphans:
        print(f"[Server] closed {_orphans} orphaned training session row(s)")

    # Point TensorBoard at SheepRL's log directory for the default game
    tb_logdir = trainer.get_tensorboard_logdir()
    start_tensorboard(tb_logdir)

    print("[Server] Retro Dreamer ready")
    print("[Server] Dashboard: http://localhost:8080")
    print("[Server] TensorBoard: http://localhost:6006/tensorboard/ (proxied at /tensorboard/)")
    print("[Server] API docs: http://localhost:8080/docs")

    yield

    # Cleanup
    if trainer and trainer.state.value in ("training", "paused"):
        trainer.stop()
    stop_tensorboard()
    if _proxy_client is not None:
        await _proxy_client.aclose()


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


# ---- Reverse proxies: live-play sidecar + TensorBoard ----
# Remote access (retro.schuyler.ai via Cloudflare tunnel) exposes only this
# port, so the browser can't reach :8092/:6006 directly. Everything the
# frontend needs is funneled same-origin through these routes; they must be
# registered before the SPA catch-all below.
LIVE_ORIGIN = "http://127.0.0.1:8092"
TB_ORIGIN = "http://127.0.0.1:6006"

_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
_proxy_client: httpx.AsyncClient | None = None


def _get_proxy_client() -> httpx.AsyncClient:
    global _proxy_client
    if _proxy_client is None:
        # read=None: HLS/MP4 streams stay open indefinitely
        _proxy_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=None)
        )
    return _proxy_client


async def _proxy(request: Request, target: str) -> Response:
    client = _get_proxy_client()
    if request.url.query:
        target = f"{target}?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
    upstream = client.build_request(
        request.method, target, headers=headers, content=await request.body()
    )
    try:
        resp = await client.send(upstream, stream=True)
    except httpx.ConnectError:
        return Response(status_code=502, content=b"upstream not running")
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("connection", "keep-alive", "transfer-encoding")
    }
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=resp_headers,
        background=BackgroundTask(resp.aclose),
    )


@app.api_route("/live/{path:path}", methods=["GET", "HEAD"])
async def proxy_live(request: Request, path: str):
    """Same-origin proxy to the live-play sidecar (backend/live_server.py)."""
    return await _proxy(request, f"{LIVE_ORIGIN}/{path}")


@app.api_route("/tensorboard{path:path}", methods=["GET", "HEAD", "POST"])
async def proxy_tensorboard(request: Request, path: str):
    """Same-origin proxy to TensorBoard (spawned with --path_prefix=/tensorboard)."""
    return await _proxy(request, f"{TB_ORIGIN}/tensorboard{path}")


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
