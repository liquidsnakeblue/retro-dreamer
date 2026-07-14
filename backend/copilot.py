"""Resident copilot: a headless claude-local (Claude Code CLI against the
local Qwen proxy) session managed by the studio, streamed to the dashboard's
chat panel. The copilot drives the SAME HTTP tool layer as the UI and Claude —
the harness is a body, the tools are the studio's.

Qwen realities baked in (operational notes, 2026-07):
- vision requires the :8082 image-fix proxy (direct :6789 silently drops images)
- reasoning model: minutes-long thinking is normal — 900s API timeout, no
  output-token caps
- CLAUDE_CODE_ATTRIBUTION_HEADER=0 or the per-request header kills KV cache
- lean context: tools return distilled results; the 3090 box holds ~2 heavy
  conversations
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMER_PATH = PROJECT_ROOT / "backend" / "copilot_primer.md"
CONFIG_DIR = Path.home() / ".claude-local"
PROXY_SCRIPT = Path.home() / "lmstudio-proxy" / "proxy.py"
PROXY_ENV = Path.home() / "lmstudio-proxy" / ".env"
STUDIO_STATE_START = "<STUDIO_STATE>"
STUDIO_STATE_END = "</STUDIO_STATE>"

router = APIRouter(prefix="/api/copilot")

_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_events: list = []  # display events plus typed proposal cards
_seq = 0
_studio_state_builder = None


def set_studio_state_builder(builder):
    global _studio_state_builder
    _studio_state_builder = builder


def compose_user_message(text: str, studio_state: dict) -> str:
    envelope = json.dumps(studio_state, separators=(",", ":"), ensure_ascii=False)
    return f"{STUDIO_STATE_START}\n{envelope}\n{STUDIO_STATE_END}\n{text}"


def _emit(kind: str, text: str, detail: str = None):
    global _seq
    with _lock:
        _seq += 1
        ev = {"seq": _seq, "ts": time.time(), "kind": kind, "text": text}
        if detail:
            ev["detail"] = detail
        _events.append(ev)
        del _events[:-500]


def emit_proposal(proposal: dict):
    """Publish code-authored proposal data without routing it through Qwen."""
    global _seq
    # Round-trip through JSON so neither the caller nor a dashboard consumer
    # can mutate the planner's copy by retaining a shared nested object.
    safe_proposal = json.loads(json.dumps(proposal))
    with _lock:
        _seq += 1
        _events.append({
            "seq": _seq,
            "ts": time.time(),
            "kind": "proposal",
            "proposal": safe_proposal,
        })
        del _events[:-500]


def _ensure_proxy():
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:8082/", timeout=2)
        return
    except Exception:
        pass
    env = os.environ.copy()
    if PROXY_ENV.exists():
        for line in PROXY_ENV.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    subprocess.Popen(
        ["python3", str(PROXY_SCRIPT)], env=env,
        stdout=open("/tmp/lmstudio-proxy.log", "ab"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1.5)


def _reader(proc: subprocess.Popen):
    """Parse the CLI's stream-json output into display events."""
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            _emit("raw", line[:500])
            continue
        t = ev.get("type")
        if t == "assistant":
            for block in (ev.get("message") or {}).get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    _emit("assistant", block["text"])
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    inp = block.get("input") or {}
                    # One readable line up front; full input as expandable detail
                    if name == "Bash":
                        label = inp.get("description") or inp.get("command", "").split("\n")[0][:120]
                        detail = inp.get("command", "")
                    elif "file_path" in inp:
                        label = inp["file_path"]
                        detail = "" if name == "Read" else json.dumps(inp, indent=2)[:4000]
                    else:
                        blob = json.dumps(inp)
                        label = blob[:120]
                        detail = json.dumps(inp, indent=2)[:4000] if len(blob) > 120 else ""
                    _emit("tool", f"{name} — {label}", detail=detail)
        elif t == "result":
            _emit("meta", f"turn done ({ev.get('num_turns', '?')} turns, "
                          f"{ev.get('duration_ms', 0) / 1000:.0f}s)")
        elif t == "system" and ev.get("subtype") == "init":
            _emit("meta", f"session ready (model {ev.get('model', '?')})")
    _emit("meta", "copilot session ended")


class StartReq(BaseModel):
    resume: bool = False  # future: resume a prior session id


@router.post("/start")
def start(req: StartReq = None):
    global _proc, _events
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return {"status": "already_running"}
        # Clear history but do NOT reset _seq: pollers track last-seen seq,
        # and a rewind makes them silently ignore the new session.
        _events = []
    _ensure_proxy()
    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = str(CONFIG_DIR)
    env["ANTHROPIC_BASE_URL"] = "http://localhost:8082"
    env["ANTHROPIC_API_KEY"] = "sk-no-key-required"
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    env["API_TIMEOUT_MS"] = "900000"  # reasoning model thinks in minutes
    cmd = [
        "claude",
        "--setting-sources", "user",
        "--model", "qwen3.6-27b",
        "--disallowedTools", "WebSearch,WebFetch",
        "--append-system-prompt-file", str(PRIMER_PATH),
        "--dangerously-skip-permissions",
        "-p",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=open("/tmp/retro-copilot.err.log", "ab"),
        text=True, bufsize=1,
    )
    with _lock:
        _proc = proc
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    _emit("meta", "copilot starting (Qwen 3.6 27B via claude-local, headless)")
    return {"status": "started", "pid": proc.pid}


class SendReq(BaseModel):
    text: str
    focus_game_id: Optional[str] = None
    active_tab: Optional[str] = None


@router.post("/send")
def send(req: SendReq):
    if _proc is None or _proc.poll() is not None:
        raise HTTPException(409, "copilot not running — POST /api/copilot/start first")
    if _studio_state_builder is None:
        raise HTTPException(500, "Studio state builder not initialized")
    studio_state = _studio_state_builder.build(
        req.focus_game_id, active_tab=req.active_tab, projection="compact"
    )
    enveloped_text = compose_user_message(req.text, studio_state)
    msg = {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": enveloped_text}]},
    }
    # The API event is the auditable raw user payload; the dashboard strips
    # only the anchored studio-state envelope before rendering the bubble.
    _emit("user", enveloped_text)
    try:
        _proc.stdin.write(json.dumps(msg) + "\n")
        _proc.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise HTTPException(500, f"copilot stdin write failed: {exc}")
    return {
        "status": "sent",
        "studio_revision": studio_state["revision"],
        "observed_at": studio_state["generated_at"],
    }


@router.get("/events")
def events(since: int = 0):
    with _lock:
        evs = [e for e in _events if e["seq"] > since]
        running = _proc is not None and _proc.poll() is None
        last_seq = _seq
    return {"running": running, "events": evs, "last_seq": last_seq}


@router.post("/stop")
def stop():
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _proc.kill()
        _proc = None
    _emit("meta", "copilot stopped")
    return {"status": "stopped"}
