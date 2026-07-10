"""Training catalog: SQLite-backed registry of games, lineages, sessions,
and snapshots. The studio's single source of truth for "which brain is the
resumable head of game X" — replacing newest-checkpoint-by-mtime scanning.

Design: docs/superpowers/specs/2026-07-10-studio-v2-multigame-copilot-design.md
SQLite owns pointers and metadata; large artifacts stay files.
"""

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "training-state"
DB_PATH = STATE_DIR / "catalog.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    active_lineage_id INTEGER REFERENCES lineages(id)
);
CREATE TABLE IF NOT EXISTS lineages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL REFERENCES games(id),
    name TEXT NOT NULL,
    parent_snapshot_id INTEGER REFERENCES snapshots(id),
    compatibility_hash TEXT,
    status TEXT NOT NULL DEFAULT 'parked',   -- active | parked | dead
    created_at REAL NOT NULL,
    UNIQUE(game_id, name)
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lineage_id INTEGER NOT NULL REFERENCES lineages(id),
    run_dir TEXT NOT NULL UNIQUE,
    started_at REAL,
    ended_at REAL,
    start_step INTEGER,
    end_step INTEGER,
    status TEXT NOT NULL DEFAULT 'ended',    -- running | ended | crashed
    exit_reason TEXT,
    resolved_config TEXT,                    -- path to the run's config.yaml
    git_commit TEXT,
    rom_hash TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    step INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL UNIQUE,
    replay_path TEXT,
    kind TEXT NOT NULL DEFAULT 'resume',     -- resume | eval | archive
    validation_status TEXT NOT NULL DEFAULT 'unvalidated',
    config_hash TEXT,
    metrics_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id, step);
CREATE INDEX IF NOT EXISTS idx_sessions_lineage ON sessions(lineage_id);
"""


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


# ------------------------------------------------------------------
# Head resolution — THE replacement for mtime scanning
# ------------------------------------------------------------------

def get_resumable_head(con, game_id: str, lineage_name: str = None) -> Optional[sqlite3.Row]:
    """Newest existing resume snapshot of the game's active (or named) lineage.

    Skips snapshots whose checkpoint file has vanished (rolling keep-N) so a
    stale row can never resolve to a missing file.
    """
    if lineage_name is None:
        row = con.execute(
            "SELECT active_lineage_id FROM games WHERE id=?", (game_id,)
        ).fetchone()
        if not row or row["active_lineage_id"] is None:
            return None
        lineage_id = row["active_lineage_id"]
    else:
        row = con.execute(
            "SELECT id FROM lineages WHERE game_id=? AND name=?",
            (game_id, lineage_name),
        ).fetchone()
        if not row:
            return None
        lineage_id = row["id"]

    for snap in con.execute(
        """SELECT s.* FROM snapshots s JOIN sessions sess ON s.session_id=sess.id
           WHERE sess.lineage_id=? AND s.kind='resume'
           ORDER BY s.step DESC""",
        (lineage_id,),
    ):
        if Path(snap["checkpoint_path"]).exists():
            return snap
    return None


def get_watch_head(con) -> Optional[sqlite3.Row]:
    """Best brain to WATCH right now: the running session's lineage head if
    training is live, else the newest existing snapshot across active lineages."""
    row = con.execute(
        """SELECT l.game_id, l.name FROM sessions sess
           JOIN lineages l ON sess.lineage_id = l.id
           WHERE sess.status='running' ORDER BY sess.started_at DESC LIMIT 1"""
    ).fetchone()
    if row:
        head = get_resumable_head(con, row["game_id"], row["name"])
        if head:
            return head
    for snap in con.execute(
        """SELECT s.* FROM snapshots s
           JOIN sessions x ON s.session_id=x.id
           JOIN lineages l ON x.lineage_id=l.id
           WHERE l.status='active' AND s.kind='resume'
           ORDER BY s.created_at DESC"""
    ):
        if Path(snap["checkpoint_path"]).exists():
            return snap
    return None


def register_snapshot(con, session_id: int, step: int, checkpoint_path: str,
                      replay_path: str = None, kind: str = "resume",
                      metrics: dict = None) -> int:
    cur = con.execute(
        """INSERT OR IGNORE INTO snapshots
           (session_id, step, checkpoint_path, replay_path, kind, metrics_json, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (session_id, step, checkpoint_path, replay_path, kind,
         json.dumps(metrics) if metrics else None, time.time()),
    )
    con.commit()
    return cur.lastrowid


# ------------------------------------------------------------------
# Retroactive registration: crawl existing SheepRL run dirs and build
# lineage chains by following checkpoint.resume_from references.
# ------------------------------------------------------------------

RUNS_GLOB = "sheeprl/logs/runs/dreamer_v3/*/*/version_0"
_DIRNAME_TS = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")
_CKPT_STEP = re.compile(r"ckpt_(\d+)_\d+\.ckpt$")


def _run_started_at(run_dir: Path) -> Optional[float]:
    m = _DIRNAME_TS.search(run_dir.parent.name)
    if not m:
        return None
    return time.mktime(time.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H-%M-%S"))


def _resume_from(run_dir: Path) -> Optional[str]:
    """Parse checkpoint.resume_from out of the run's saved config (cheap grep —
    OmegaConf load would drag hydra imports into the server)."""
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return None
    in_ckpt_block = False
    for line in cfg.read_text().splitlines():
        if re.match(r"^checkpoint:\s*$", line):
            in_ckpt_block = True
            continue
        if in_ckpt_block:
            if line and not line.startswith(" "):
                break
            m = re.match(r"\s+resume_from:\s*(.+?)\s*$", line)
            if m:
                val = m.group(1).strip("'\"")
                return None if val in ("null", "None", "") else val
    return None


def register_existing_runs(con, game_filter: str = None, active_run_dir: str = None):
    """Idempotently register historical run dirs as lineages/sessions/snapshots.

    Chains are built via resume_from; a chain root becomes a lineage named after
    its root run timestamp; the chain containing `active_run_dir` is named
    'main' and marked active. Superseded/contaminated checkpoint dirs (renamed
    checkpoint_*) contribute no snapshots and register as status='dead' hints
    only via their sessions' exit_reason.
    """
    runs = {}
    for vdir in sorted(PROJECT_ROOT.glob(RUNS_GLOB)):
        game_id = vdir.parent.parent.name if vdir.parent.parent.name != "dreamer_v3" else vdir.parent.name
        # layout: logs/runs/dreamer_v3/<game>/<run_name>/version_0
        game_id = vdir.parent.parent.name
        if game_filter and game_id != game_filter:
            continue
        ckpt_dir = vdir / "checkpoint"
        ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
        steps = sorted(
            int(_CKPT_STEP.search(p.name).group(1))
            for p in ckpts if _CKPT_STEP.search(p.name)
        )
        runs[str(vdir)] = {
            "game_id": game_id,
            "run_dir": str(vdir),
            "started_at": _run_started_at(vdir),
            "resume_from": _resume_from(vdir),
            "ckpts": {int(_CKPT_STEP.search(p.name).group(1)): str(p)
                      for p in ckpts if _CKPT_STEP.search(p.name)},
            "steps": steps,
            "superseded": bool(list(vdir.glob("checkpoint_*"))) and not ckpts,
        }

    # map: ckpt path -> owning run dir (to follow resume_from across runs)
    ckpt_owner = {}
    for r in runs.values():
        for p in r["ckpts"].values():
            ckpt_owner[p] = r["run_dir"]

    def chain_parent(r):
        rf = r["resume_from"]
        if not rf:
            return None
        return ckpt_owner.get(rf.strip('"'))

    # find each run's chain root
    def root_of(run_dir):
        seen = set()
        cur = run_dir
        while True:
            if cur in seen:
                return cur
            seen.add(cur)
            parent = chain_parent(runs[cur])
            if parent is None or parent not in runs:
                return cur
            cur = parent

    chains = {}
    for run_dir in runs:
        chains.setdefault(root_of(run_dir), []).append(run_dir)

    active_chain_root = root_of(str(active_run_dir)) if active_run_dir and str(active_run_dir) in runs else None

    for root, members in chains.items():
        r0 = runs[root]
        game_id = r0["game_id"]
        con.execute(
            "INSERT OR IGNORE INTO games (id, display_name) VALUES (?,?)",
            (game_id, game_id),
        )
        is_main = root == active_chain_root
        lineage_name = "main" if is_main else Path(root).parent.name
        con.execute(
            """INSERT OR IGNORE INTO lineages (game_id, name, status, created_at)
               VALUES (?,?,?,?)""",
            (game_id, lineage_name, "active" if is_main else "parked",
             r0["started_at"] or time.time()),
        )
        lineage_id = con.execute(
            "SELECT id FROM lineages WHERE game_id=? AND name=?",
            (game_id, lineage_name),
        ).fetchone()["id"]
        if is_main:
            con.execute("UPDATE games SET active_lineage_id=? WHERE id=?",
                        (lineage_id, game_id))

        for member in sorted(members, key=lambda m: runs[m]["started_at"] or 0):
            r = runs[member]
            running = active_run_dir and member == str(active_run_dir)
            con.execute(
                """INSERT OR IGNORE INTO sessions
                   (lineage_id, run_dir, started_at, start_step, end_step,
                    status, exit_reason, resolved_config)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (lineage_id, member, r["started_at"],
                 r["steps"][0] if r["steps"] else None,
                 r["steps"][-1] if r["steps"] else None,
                 "running" if running else "ended",
                 "superseded (checkpoints retired)" if r["superseded"] else None,
                 str(Path(member) / "config.yaml")),
            )
            session_id = con.execute(
                "SELECT id FROM sessions WHERE run_dir=?", (member,)
            ).fetchone()["id"]
            memmap = Path(member) / "memmap_buffer"
            for step, path in sorted(r["ckpts"].items()):
                con.execute(
                    """INSERT OR IGNORE INTO snapshots
                       (session_id, step, checkpoint_path, replay_path, kind, created_at)
                       VALUES (?,?,?,?,'resume',?)""",
                    (session_id, step, path,
                     str(memmap) if memmap.exists() else None,
                     Path(path).stat().st_mtime),
                )
    con.commit()


if __name__ == "__main__":
    import sys

    con = connect()
    active = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else None
    register_existing_runs(con, active_run_dir=active)
    for g in con.execute("SELECT * FROM games"):
        print(f"game {g['id']} (active lineage id {g['active_lineage_id']})")
        for ln in con.execute("SELECT * FROM lineages WHERE game_id=?", (g["id"],)):
            n_sess = con.execute(
                "SELECT COUNT(*) c FROM sessions WHERE lineage_id=?", (ln["id"],)
            ).fetchone()["c"]
            n_snap = con.execute(
                """SELECT COUNT(*) c FROM snapshots s JOIN sessions x ON s.session_id=x.id
                   WHERE x.lineage_id=?""", (ln["id"],)
            ).fetchone()["c"]
            head = get_resumable_head(con, g["id"], ln["name"])
            head_s = f"step {head['step']}" if head else "NONE"
            print(f"  lineage {ln['name']:28s} [{ln['status']:6s}] "
                  f"sessions={n_sess} snapshots={n_snap} head={head_s}")
