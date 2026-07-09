"""Live Play sidecar — manages a live-play session (the newest checkpoint
playing the game) and serves its HLS stream for the dashboard's Live Play tab.

Runs separately from the main studio server so it can be deployed/restarted
without touching a live training run.

  GET /start        spawn a session with the newest checkpoint (kills any old one)
  GET /stop         kill the session
  GET /status       {"running": bool, "playlist_ready": bool}
  GET /live/<file>  HLS playlist + segments (live.m3u8, live*.ts)

Start:  RETRO_LIVE_PORT=8092 python backend/live_server.py
"""
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("RETRO_LIVE_PORT", "8092"))
PROJECT_ROOT = Path(__file__).parent.parent
SHEEPRL_DIR = PROJECT_ROOT / "sheeprl"
HLS_DIR = Path(os.environ.get("RETRO_LIVE_HLS_DIR", "/tmp/retro-dreamer-live"))
PYTHON = os.environ.get(
    "RETRO_LIVE_PYTHON", str(Path.home() / "fzero-dreamer" / "venv" / "bin" / "python")
)
IDLE_KILL_SECONDS = 90

_lock = threading.Lock()
_session: subprocess.Popen | None = None
_last_access = 0.0


def _stop_session_locked():
    global _session
    if _session is not None and _session.poll() is None:
        try:
            _session.terminate()
            _session.wait(timeout=5)
        except Exception:
            try:
                _session.kill()
            except Exception:
                pass
    _session = None


def _start_session_locked():
    global _session, _last_access
    _stop_session_locked()
    # Clear leftovers from the previous session BEFORE spawning: a stale
    # finalized playlist otherwise reports playlist_ready immediately and the
    # client plays a dead VOD, hits 'ended', and tears down the new session.
    HLS_DIR.mkdir(parents=True, exist_ok=True)
    for old in HLS_DIR.glob("live*"):
        try:
            old.unlink()
        except OSError:
            pass
    env = os.environ.copy()
    env["RETRO_LIVE_HLS_DIR"] = str(HLS_DIR)
    # CPU inference: a busy training run saturates the GPU and inference
    # queues behind its kernels (measured 0.93x real-time — stutters at the
    # live edge). CPU holds 1.0x. Set RETRO_LIVE_GPU=1 to override when no
    # training is running.
    env.setdefault("RETRO_LIVE_GPU", "0")
    _session = subprocess.Popen(
        [PYTHON, str(SHEEPRL_DIR / "_retro_live_player.py"), "latest"],
        cwd=str(SHEEPRL_DIR),
        stdout=subprocess.DEVNULL,
        stderr=open(HLS_DIR / "player.log", "wb"),
        env=env,
    )
    _last_access = time.time()


def _reaper():
    global _last_access
    while True:
        time.sleep(10)
        with _lock:
            if (
                _session is not None
                and _session.poll() is None
                and time.time() - _last_access > IDLE_KILL_SECONDS
            ):
                _stop_session_locked()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        global _last_access
        path = self.path.split("?")[0]

        if path == "/start":
            with _lock:
                _start_session_locked()
            self._send(200, b'{"started": true}')
            return

        if path == "/stop":
            with _lock:
                _stop_session_locked()
            self._send(200, b'{"stopped": true}')
            return

        if path == "/status":
            running = _session is not None and _session.poll() is None
            if running:
                # status polls during startup (checkpoint loading) count as
                # activity — don't reap a session the client is waiting on
                _last_access = time.time()
            ready = (HLS_DIR / "live.m3u8").exists()
            self._send(
                200,
                (
                    '{"running": %s, "playlist_ready": %s}'
                    % (str(running).lower(), str(ready and running).lower())
                ).encode(),
            )
            return

        if path.startswith("/live/"):
            name = os.path.basename(path)
            f = HLS_DIR / name
            if not f.is_file():
                self._send(404, b"")
                return
            _last_access = time.time()
            ctype = (
                "application/vnd.apple.mpegurl" if name.endswith(".m3u8") else "video/mp2t"
            )
            self._send(200, f.read_bytes(), ctype)
            return

        self._send(404, b"")


def main():
    HLS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_reaper, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[LiveServer] serving on :{PORT}, hls dir {HLS_DIR}", flush=True)

    def shutdown(*_):
        with _lock:
            _stop_session_locked()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
