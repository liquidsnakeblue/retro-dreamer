"""One code-owned projection of the studio for the API and copilot envelope."""

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend import catalog


def _observed_at(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _path_fingerprint(path_value, *, include_text: bool = False):
    """Material identity without leaking an absolute path into compact state."""
    if not path_value:
        return None
    path = Path(path_value)
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False}
    fingerprint = {
        "exists": True,
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_text and path.is_file():
        try:
            fingerprint["content"] = _hash(path.read_text())
        except (OSError, UnicodeError):
            fingerprint["content"] = None
    return fingerprint


def model_size_advice() -> dict:
    """Existing GPU advice as a reusable code-owned fact source."""
    import torch

    if not torch.cuda.is_available():
        return {
            "gpu": None,
            "vram_gb": 0,
            "recommended": "debug",
            "fits": ["debug"],
            "note": "No CUDA GPU visible — debug size only (CPU training is impractical).",
        }
    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    tiers = [("xl", 32.0), ("large", 20.0), ("medium", 12.0),
             ("small", 8.0), ("debug", 0.0)]
    recommended = next(name for name, need in tiers if vram >= need)
    return {
        "gpu": props.name,
        "vram_gb": round(vram, 1),
        "recommended": recommended,
        "fits": [name for name, need in tiers if vram >= need],
        "note": "Bigger models score higher AND need less data (DreamerV3 Fig 6c) — "
                "run the largest size that fits.",
    }


class StudioStateBuilder:
    """Join live, catalog, game, config, advisor, and job state once."""

    def __init__(
        self,
        game_manager,
        trainer,
        *,
        catalog_connect: Callable = catalog.connect,
        jobs_provider: Optional[Callable[[], list[dict]]] = None,
        advisor_provider: Callable[[], dict] = model_size_advice,
        inventory_ttl: float = 30.0,
    ):
        self.game_manager = game_manager
        self.trainer = trainer
        self.catalog_connect = catalog_connect
        self.jobs_provider = jobs_provider or self._default_jobs
        self.advisor_provider = advisor_provider
        self.inventory_ttl = inventory_ttl
        self._inventory_cache: tuple[float, list[dict]] | None = None
        self._cache_lock = threading.Lock()

    @staticmethod
    def _default_jobs() -> list[dict]:
        from backend.tools import snapshot_jobs
        return snapshot_jobs()

    def invalidate(self):
        with self._cache_lock:
            self._inventory_cache = None

    def _inventory(self) -> tuple[float, list[dict]]:
        now = time.time()
        with self._cache_lock:
            if self._inventory_cache and now - self._inventory_cache[0] < self.inventory_ttl:
                return self._inventory_cache
        games = self.game_manager.list_games()
        with self._cache_lock:
            self._inventory_cache = (now, games)
        return now, games

    @staticmethod
    def _compact_game(game: dict) -> dict:
        return {
            "game_id": game.get("game_id"),
            "display_name": game.get("display_name") or game.get("game_id"),
            "system": game.get("system"),
            "source": game.get("source"),
            "rom_ready": bool(game.get("rom_ready")),
            "has_custom_config": bool(game.get("has_custom_config")),
        }

    def _training(self, observed_at: str) -> dict:
        status = self.trainer.status
        state = getattr(status.state, "value", status.state)
        config = self.trainer.config
        return {
            "observed_at": observed_at,
            "state": state,
            "game_id": status.game_id,
            "current_step": status.current_step,
            "current_episode": status.current_episode,
            "elapsed_time": round(status.elapsed_time, 1),
            "steps_per_second": round(status.steps_per_second, 2),
            "avg_return": status.avg_return,
            "avg_length": status.avg_length,
            "max_return": status.max_return,
            "error": status.error_message or None,
            "effective_config": {
                "model_size": config.model_size,
                "initial_state": config.initial_state,
                "batch_size": config.batch_size,
                "batch_length": config.batch_length,
                "replay_ratio": config.replay_ratio,
                "num_envs": config.num_envs,
                "resume_prefill": config.resume_prefill,
            },
        }

    def _catalog_focus(self, game_id: str, *, full: bool) -> dict:
        con = self.catalog_connect()
        try:
            game = con.execute("SELECT * FROM games WHERE id=?", (game_id,)).fetchone()
            if not game:
                result = {"has_brain": False, "active_lineage": None,
                          "head": None, "lineage_count": 0}
                if full:
                    result["lineages"] = []
                return result
            lineages = []
            for lineage in con.execute(
                "SELECT * FROM lineages WHERE game_id=? ORDER BY created_at", (game_id,)
            ):
                head = catalog.get_resumable_head(con, game_id, lineage["name"])
                running = con.execute(
                    "SELECT COUNT(*) AS n FROM sessions WHERE lineage_id=? AND status='running'",
                    (lineage["id"],),
                ).fetchone()["n"]
                item = {
                    "name": lineage["name"],
                    "status": lineage["status"],
                    "running": bool(running),
                    "head": None,
                }
                if head:
                    session = con.execute(
                        "SELECT resolved_config FROM sessions WHERE id=?",
                        (head["session_id"],),
                    ).fetchone()
                    resolved_config = session["resolved_config"] if session else None
                    replay_path = head["replay_path"]
                    replay_available = bool(replay_path and Path(replay_path).exists())
                    buffer_meta = (
                        Path(replay_path).parent / "buffer-meta.json"
                        if replay_path else None
                    )
                    compatibility_revision = _hash({
                        "checkpoint": _path_fingerprint(head["checkpoint_path"]),
                        "replay": _path_fingerprint(replay_path),
                        "buffer_meta": _path_fingerprint(buffer_meta, include_text=True),
                        "resolved_config": _path_fingerprint(
                            resolved_config, include_text=True
                        ),
                        "validation_status": head["validation_status"],
                        "config_hash": head["config_hash"],
                    })
                    item["head"] = {
                        "snapshot_id": head["id"],
                        "step": head["step"],
                        "created_at": head["created_at"],
                        "replay_available": replay_available,
                        "validation_status": head["validation_status"],
                        "action_manifest_hash": head["config_hash"],
                        "compatibility_revision": compatibility_revision,
                    }
                    if full:
                        item["head"].update({
                            "checkpoint_path": head["checkpoint_path"],
                            "replay_path": replay_path,
                            "resolved_config": resolved_config,
                        })
                lineages.append(item)
            active_id = game["active_lineage_id"]
            active = next((lineage["name"] for lineage in con.execute(
                "SELECT id,name FROM lineages WHERE game_id=?", (game_id,)
            ) if lineage["id"] == active_id), None)
            active_row = next((item for item in lineages if item["name"] == active), None)
            result = {
                "has_brain": bool(active_row and active_row["head"]),
                "active_lineage": active,
                "head": active_row["head"] if active_row else None,
                "lineage_count": len(lineages),
            }
            if full:
                result["lineages"] = lineages
            return result
        finally:
            con.close()

    def _focused_game(self, game: dict, observed_at: str, *, full: bool) -> dict:
        game_id = game["game_id"]
        detail = self.game_manager.get_game(game_id)
        configs = {}
        for filename in ("data.json", "actions.json", "training.json", "metadata.json"):
            try:
                configs[filename] = self.game_manager.read_config(game_id, filename)
            except (FileNotFoundError, ValueError):
                configs[filename] = None
        data = configs["data.json"] or {}
        actions = configs["actions.json"] or {}
        training = configs["training.json"] or {}
        annotated = detail.get("annotated_states") or [
            {"file": state, "label": state, "group": "other"}
            for state in detail.get("states", [])
        ]
        states = [{key: state.get(key) for key in
                   ("file", "label", "group", "description", "objective")
                   if state.get(key) is not None} for state in annotated]
        games_dir = getattr(self.game_manager, "games_dir", None)
        state_manifest = []
        for state in states:
            filename = state.get("file")
            artifact = None
            if games_dir is not None and filename:
                state_name = filename if filename.endswith(".state") else f"{filename}.state"
                artifact = _path_fingerprint(
                    Path(games_dir) / game_id / "states" / state_name
                )
            state_manifest.append({"definition": state, "artifact": artifact})
        config_files = set(detail.get("config_files", []))
        blockers = []
        if game.get("source") != "custom":
            blockers.append("game needs a custom workspace")
        if not game.get("rom_ready"):
            blockers.append("ROM is not ready")
        for filename in ("data.json", "actions.json", "training.json"):
            if filename not in config_files:
                blockers.append(f"missing {filename}")
        if not states:
            blockers.append("no save states")
        compact_states = states if full else states[:8]
        result = {
            "observed_at": observed_at,
            **self._compact_game(game),
            "default_state": detail.get("default_state"),
            "state_count": len(states),
            "states": compact_states,
            "states_truncated": len(compact_states) < len(states),
            "config": {
                "revision": _hash({"configs": configs, "states": state_manifest}),
                "files": sorted(config_files),
                "ram_variable_count": len(data.get("info") or {}),
                "action_count": len(actions.get("actions") or []),
                "reward_variables": sorted(
                    (((training.get("reward") or {}).get("variables")) or {}).keys()
                ),
                "done_variables": sorted(
                    (((training.get("done") or {}).get("variables")) or {}).keys()
                ),
            },
            "brain": self._catalog_focus(game_id, full=full),
            "readiness": {"trainable": not blockers, "blockers": blockers},
        }
        if full:
            result["config"]["ram_variables"] = sorted((data.get("info") or {}).keys())
            result["configs"] = configs
            result["metadata"] = detail
        return result

    def build(
        self,
        focus_game_id: str | None = None,
        *,
        active_tab: str | None = None,
        projection: str = "compact",
    ) -> dict:
        if projection not in {"compact", "full"}:
            raise ValueError("projection must be 'compact' or 'full'")
        full = projection == "full"
        now = time.time()
        observed_at = _observed_at(now)
        inventory_seen, raw_games = self._inventory()
        games = [self._compact_game(game) for game in raw_games]
        custom = [game for game in games if game["source"] == "custom"]
        builtins = [game for game in games if game["source"] == "builtin"]
        inventory = {
            "observed_at": _observed_at(inventory_seen),
            "revision": _hash(games),
            "custom_games": custom,
            "builtin_count": len(builtins),
            "promotable_builtin_count": sum(game["rom_ready"] for game in builtins),
        }
        if full:
            inventory["builtins"] = builtins
        selected = next((game for game in games if game["game_id"] == focus_game_id), None)
        if focus_game_id and selected is None:
            raise FileNotFoundError(f"Game '{focus_game_id}' not found")
        focused = self._focused_game(selected, observed_at, full=full) if selected else None
        training = self._training(observed_at)
        advisor = {"observed_at": observed_at, **self.advisor_provider()}
        jobs = self.jobs_provider()
        tools = {"observed_at": observed_at, "jobs": jobs[:20] if full else jobs[:8]}
        capabilities = {
            "observed_at": observed_at,
            "can_plan_training": bool(focused and focused["readiness"]["trainable"]),
            "can_start": bool(focused and focused["readiness"]["trainable"]
                              and training["state"] != "training"),
            "can_switch": bool(focused and focused["readiness"]["trainable"]
                               and training["state"] == "training"
                               and training["game_id"] != focus_game_id),
            "needs_onboarding": bool(selected and selected["source"] != "custom"),
        }
        revision_material = {
            "focus_game_id": focus_game_id,
            "inventory_revision": inventory["revision"],
            "focused_config": focused["config"]["revision"] if focused else None,
            "focused_head": {
                key: (focused["brain"].get("head") or {}).get(key)
                for key in ("snapshot_id", "step", "compatibility_revision")
            } if focused else None,
            "training": {
                "state": training["state"],
                "game_id": training["game_id"],
                "effective_config": training["effective_config"],
            },
        }
        return {
            "schema_version": 1,
            "projection": projection,
            "revision": _hash(revision_material),
            "generated_at": observed_at,
            "focus": {
                "observed_at": observed_at,
                "game_id": focus_game_id,
                "active_tab": active_tab,
            },
            "training": training,
            "inventory": inventory,
            "focused_game": focused,
            "advisor": advisor,
            "tools": tools,
            "capabilities": capabilities,
        }
