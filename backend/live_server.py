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
_session_state: str | None = None  # initial_state the running session booted with
_last_access = 0.0
_recorder: subprocess.Popen | None = None
REC_PATH = None  # set in main() after HLS_DIR exists


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


def _start_session_locked(initial_state: str | None = None):
    global _session, _session_state, _last_access
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
    cmd = [PYTHON, str(SHEEPRL_DIR / "_retro_live_player.py"), "latest"]
    if initial_state:
        cmd.append(initial_state)
    _session = subprocess.Popen(
        cmd,
        cwd=str(SHEEPRL_DIR),
        stdout=subprocess.DEVNULL,
        stderr=open(HLS_DIR / "player.log", "wb"),
        env=env,
    )
    _session_state = initial_state
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


def _start_recorder_locked(seconds: str, initial_state: str | None):
    global _recorder
    if _recorder is not None and _recorder.poll() is None:
        _recorder.terminate()
        try:
            _recorder.wait(timeout=5)
        except Exception:
            _recorder.kill()
    rec = HLS_DIR / "recording.mp4"
    prog = HLS_DIR / "recording.progress.json"
    for f in (rec, prog, HLS_DIR / "recording.tmp.mp4"):
        try:
            f.unlink()
        except OSError:
            pass
    env = os.environ.copy()
    cmd = [PYTHON, str(SHEEPRL_DIR / "_retro_record.py"), "latest", seconds, str(rec)]
    if initial_state:
        cmd.append(initial_state)
    _recorder = subprocess.Popen(
        cmd,
        cwd=str(SHEEPRL_DIR),
        stdout=subprocess.DEVNULL,
        stderr=open(HLS_DIR / "recorder.log", "wb"),
        env=env,
    )


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
            from urllib.parse import parse_qs, urlparse

            requested_state = parse_qs(urlparse(self.path).query).get("state", [None])[0]
            with _lock:
                # reuse a live session: killing a healthy stream to cold-start
                # another (~60-90s CPU checkpoint load) punishes an impatient
                # second click; the idle reaper bounds staleness to ~90s anyway.
                # Only reuse when it's playing the SAME save state — a track
                # switch must restart the player.
                running = _session is not None and _session.poll() is None
                if (
                    running
                    and (HLS_DIR / "live.m3u8").exists()
                    and requested_state == _session_state
                ):
                    _last_access = time.time()
                    self._send(200, b'{"started": true, "reused": true}')
                    return
                _start_session_locked(requested_state)
            self._send(200, b'{"started": true, "reused": false}')
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

        if path.startswith("/record"):
            if path == "/record_status":
                prog = HLS_DIR / "recording.progress.json"
                alive = _recorder is not None and _recorder.poll() is None
                if prog.exists():
                    body = prog.read_bytes()
                elif alive:
                    body = b'{"frames": 0, "percent": 0, "done": false}'
                else:
                    body = b'{"done": false, "error": "no recorder running"}'
                self._send(200, body)
                return
            # /record?seconds=60[&state=gp_knight_beginner]
            from urllib.parse import parse_qs, urlparse

            q = parse_qs(urlparse(self.path).query)
            seconds = q.get("seconds", ["60"])[0]
            if seconds != "full":
                seconds = str(max(5, min(300, int(float(seconds)))))
            initial_state = q.get("state", [None])[0]
            with _lock:
                _start_recorder_locked(seconds, initial_state)
            self._send(200, b'{"recording": true}')
            return

        if path == "/rec/recording.mp4":
            f = HLS_DIR / "recording.mp4"
            if not f.is_file():
                self._send(404, b"")
                return
            data = f.read_bytes()
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                try:
                    spec = rng.split("=", 1)[1].split("-")
                    start = int(spec[0]) if spec[0] else 0
                    end = int(spec[1]) if len(spec) > 1 and spec[1] else len(data) - 1
                    end = min(end, len(data) - 1)
                    chunk = data[start : end + 1]
                    self.send_response(206)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                except (ValueError, IndexError):
                    pass
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
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
