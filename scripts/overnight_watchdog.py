#!/usr/bin/env python3
"""Overnight training watchdog for retro-dreamer.

Every POLL seconds:
  - If training state is 'error', or was 'training' and silently went 'idle',
    resume training via the API (buffer-preserving), max MAX_RESTARTS times
    with a growing backoff. A restart that itself dies within 10 min counts
    against the cap — a persistent crash won't be hammered all night.
  - Every ARCHIVE_EVERY seconds, hard-link the newest checkpoint into
    ARCHIVE_DIR (keeps KEEP_ARCHIVES newest) so the rolling keep-10 window
    can't delete an overnight peak brain.

Kill it in the morning: pkill -f overnight_watchdog
Log: tail scripts/overnight_watchdog.log
"""
import json
import os
import shutil
import time
import urllib.request
from glob import glob
from pathlib import Path

API = "http://localhost:8091/api"
POLL = 60
ARCHIVE_EVERY = 3600
MAX_RESTARTS = 5
KEEP_ARCHIVES = 8

ROOT = Path(__file__).resolve().parent.parent
LAST_REQUEST = ROOT / "training-state" / "last_start_request.json"
STOP_MARKER = ROOT / "training-state" / "stopped_by_user.json"


def user_stopped_after(t: float) -> bool:
    """True iff /training/stop stamped a marker newer than time t.

    Lets the watchdog tell a USER-initiated stop (manual UI Stop) from a
    crash: the API writes stopped_by_user.json on every /training/stop. If
    that marker postdates our last 'training' sighting, the idle is
    intentional and must NOT be auto-resumed."""
    try:
        m = json.loads(STOP_MARKER.read_text())
        return float(m.get("ts", 0)) > t
    except Exception:
        return False


def resume_body() -> bytes:
    """Resume WHAT WAS RUNNING: the API persists every start request to
    training-state/last_start_request.json (fresh_start forced False)."""
    body = json.loads(LAST_REQUEST.read_text())
    body["fresh_start"] = False
    return json.dumps(body).encode()
CKPT_GLOB = str(ROOT / "sheeprl/logs/runs/dreamer_v3/*/*/version_*/checkpoint/*.ckpt")
ARCHIVE_DIR = ROOT / "sheeprl/logs/overnight_ckpt_archive"
LOG = Path(__file__).with_suffix(".log")


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def api(path, body=None):
    req = urllib.request.Request(
        f"{API}{path}", data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def archive_newest():
    ckpts = sorted(glob(CKPT_GLOB), key=os.path.getmtime)
    if not ckpts:
        return
    newest = Path(ckpts[-1])
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{int(time.time())}_{newest.name}"
    if any(a.name.endswith(newest.name) for a in ARCHIVE_DIR.iterdir()):
        return  # this checkpoint already archived
    try:
        os.link(newest, dest)  # hard link: instant, no extra space until source deleted
    except OSError:
        shutil.copy2(newest, dest)
    log(f"archived {newest.name}")
    archives = sorted(ARCHIVE_DIR.iterdir(), key=os.path.getmtime)
    for old in archives[:-KEEP_ARCHIVES]:
        old.unlink()


def main():
    restarts = 0
    last_restart_t = 0.0
    seen_training = False
    last_training_seen_t = 0.0
    last_archive = 0.0
    log(f"watchdog up (poll {POLL}s, archive {ARCHIVE_EVERY}s, max restarts {MAX_RESTARTS})")
    while True:
        try:
            st = api("/training/status")
            state = st.get("state")
            if state == "training":
                seen_training = True
                last_training_seen_t = time.time()
                # a restart that survived 10+ min resets nothing; only note health
            bad = state == "error" or (seen_training and state == "idle")
            if bad:
                # User-initiated stop? /training/stop dropped a fresh marker;
                # if it postdates our last 'training' sighting, do NOT restart.
                if user_stopped_after(last_training_seen_t):
                    log("idle/error after user /training/stop — not restarting; "
                        "will resume only on the next explicit /training/start")
                    try:
                        STOP_MARKER.unlink()
                    except OSError:
                        pass
                    seen_training = False
                    last_training_seen_t = 0.0
                    time.sleep(POLL)
                    continue
                if restarts >= MAX_RESTARTS:
                    log(f"state={state} but restart cap reached — giving up. "
                        f"error={st.get('error_message','')[:200]}")
                    time.sleep(POLL * 5)
                    continue
                since_last = time.time() - last_restart_t
                if last_restart_t and since_last < 600:
                    log(f"previous restart died within {since_last:.0f}s — counting double")
                    restarts += 1
                backoff = min(60 * (2 ** restarts), 600)
                log(f"state={state} (error={st.get('error_message','')[:200]}) — "
                    f"restart {restarts + 1}/{MAX_RESTARTS} in {backoff}s")
                time.sleep(backoff)
                api("/training/stop", b"{}")
                time.sleep(5)
                res = api("/training/start", resume_body())
                restarts += 1
                last_restart_t = time.time()
                seen_training = False
                log(f"resume requested: {res}")
            if time.time() - last_archive >= ARCHIVE_EVERY:
                archive_newest()
                last_archive = time.time()
        except Exception as exc:
            log(f"watchdog error (server down?): {exc}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
