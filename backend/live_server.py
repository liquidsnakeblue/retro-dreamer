"""Live Play sidecar — serves a real-time A/V stream of the newest checkpoint
playing the game, for the dashboard's Live Play tab.

Runs separately from the main studio server so it can be deployed/restarted
without touching a live training run.

  GET /stream.mp4   start (or restart) a session with the newest checkpoint
                    and stream fragmented MP4 (h264+aac) until disconnect
  GET /status       {"running": bool}

Start:  RETRO_LIVE_PORT=8092 python backend/live_server.py
"""
import os
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("RETRO_LIVE_PORT", "8092"))
PROJECT_ROOT = Path(__file__).parent.parent
SHEEPRL_DIR = PROJECT_ROOT / "sheeprl"
PYTHON = os.environ.get(
    "RETRO_LIVE_PYTHON", str(Path.home() / "fzero-dreamer" / "venv" / "bin" / "python")
)

_lock = threading.Lock()
_session: subprocess.Popen | None = None


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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")

    def do_GET(self):
        global _session
        if self.path.startswith("/status"):
            running = _session is not None and _session.poll() is None
            body = ('{"running": %s}' % ("true" if running else "false")).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if not self.path.startswith("/stream.mp4"):
            self.send_response(404)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        # One session at a time; a new viewer takes over with the newest ckpt
        with _lock:
            _stop_session_locked()
            env = os.environ.copy()
            _session = subprocess.Popen(
                [PYTHON, str(SHEEPRL_DIR / "_retro_live_player.py"), "latest"],
                cwd=str(SHEEPRL_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            session = _session

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "video/mp4")
        # no Content-Length: stream until disconnect (chunked by HTTP/1.1)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = session.stdout.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(b"%X\r\n" % len(chunk))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
            except Exception:
                pass
            with _lock:
                if _session is session:
                    _stop_session_locked()


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[LiveServer] serving on :{PORT}", flush=True)

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
