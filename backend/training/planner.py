"""Deterministic, human-gated training proposals.

The planner may inspect studio state and remember immutable proposal payloads,
but it never mutates training itself.  Confirmation receives an executor from
the API layer so the existing start/switch implementation remains the sole
mutation path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yaml

from backend.training.config import TrainingConfig


APPROVAL_COOKIE_NAME = "retro_training_approval"
VALID_MODEL_SIZES = ("debug", "small", "medium", "large", "xl")
_ARCHITECTURE_BY_RECURRENT_SIZE = {
    256: "debug",
    512: "small",
    1024: "medium",
    2048: "large",
    4096: "xl",
}


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PlannerError(Exception):
    """An API-safe planner rejection with an intended HTTP status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class PlanOptions:
    game_id: str
    model_size: Optional[str] = None
    states: Optional[tuple[str, ...]] = None
    replay_ratio: Optional[float] = None
    num_envs: Optional[int] = None
    batch_size: Optional[int] = None
    batch_length: Optional[int] = None
    resume_prefill: Optional[int] = None
    fresh_start: bool = False

    @classmethod
    def from_mapping(cls, value: dict) -> "PlanOptions":
        states = value.get("states")
        return cls(
            game_id=(value.get("game_id") or "").strip(),
            model_size=value.get("model_size"),
            states=tuple(states) if states is not None else None,
            replay_ratio=value.get("replay_ratio"),
            num_envs=value.get("num_envs"),
            batch_size=value.get("batch_size"),
            batch_length=value.get("batch_length"),
            resume_prefill=value.get("resume_prefill"),
            fresh_start=bool(value.get("fresh_start", False)),
        )


@dataclass(frozen=True)
class _PlanPayload:
    """Immutable plan content; JSON strings prevent shared-dict mutation."""

    plan_id: str
    game_id: str
    studio_revision: str
    created_at: float
    expires_at: float
    proposal_json: str
    exact_request_json: str


@dataclass
class _StoredPlan:
    payload: _PlanPayload
    status: str = "pending"


class TrainingPlanner:
    """Build and broker one-time training proposals from StudioStateBuilder."""

    def __init__(
        self,
        state_builder,
        *,
        clock: Callable[[], float] = time.time,
        plan_id_factory: Callable[[], str] | None = None,
        token_factory: Callable[[], str] | None = None,
        plan_ttl: float = 15 * 60,
        approval_ttl: float = 4 * 60 * 60,
    ):
        self.state_builder = state_builder
        self.clock = clock
        self.plan_id_factory = plan_id_factory or (lambda: secrets.token_urlsafe(12))
        self.token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self.plan_ttl = plan_ttl
        self.approval_ttl = approval_ttl
        self._plans: dict[str, _StoredPlan] = {}
        self._approval_sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Browser approval capability
    # ------------------------------------------------------------------

    @staticmethod
    def _token_digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_approval_session(self) -> str:
        """Return a raw capability for the route to place only in HttpOnly cookie."""
        now = self.clock()
        token = self.token_factory()
        digest = self._token_digest(token)
        with self._lock:
            self._approval_sessions = {
                key: expires for key, expires in self._approval_sessions.items()
                if expires > now
            }
            self._approval_sessions[digest] = now + self.approval_ttl
        return token

    def _authorize_locked(self, token: Optional[str], now: float):
        if not token:
            raise PlannerError(403, "browser approval session required")
        digest = self._token_digest(token)
        expires = self._approval_sessions.get(digest)
        if expires is None or expires <= now:
            self._approval_sessions.pop(digest, None)
            raise PlannerError(403, "browser approval session is missing or expired")

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def create_plan(self, request: PlanOptions | dict) -> dict:
        options = request if isinstance(request, PlanOptions) else PlanOptions.from_mapping(request)
        if not options.game_id:
            raise PlannerError(400, "game_id is required")

        try:
            studio = self.state_builder.build(options.game_id, projection="full")
        except FileNotFoundError as exc:
            raise PlannerError(404, str(exc)) from exc

        focused = studio.get("focused_game")
        if not focused:
            raise PlannerError(404, f"Game '{options.game_id}' not found")
        if focused.get("source") != "custom":
            raise PlannerError(409, "training plans require an onboarded custom game")
        readiness = focused.get("readiness") or {}
        if not readiness.get("trainable"):
            blockers = "; ".join(readiness.get("blockers") or ["game is not trainable"])
            raise PlannerError(409, f"game is not trainable: {blockers}")
        self._validate_workspace(focused)

        live = studio.get("training") or {}
        live_state = str(live.get("state") or "idle")
        live_game = live.get("game_id")
        if live_state == "training" and live_game == options.game_id:
            raise PlannerError(409, f"{options.game_id} is already training")
        if live_state == "training" and not live_game:
            raise PlannerError(409, "live training game is unknown; cannot plan a safe switch")
        if live_state == "stopping":
            raise PlannerError(409, "training is stopping; wait for idle before planning")
        switching = live_state == "training" and live_game != options.game_id

        brain = focused.get("brain") or {}
        head = brain.get("head")
        resume = bool(head) and not options.fresh_start
        if resume:
            effective = self._resolved_resume_settings(head, focused, options)
        else:
            effective = self._new_settings(studio, focused, options)

        selected_states, available_states = self._resolve_states(
            focused, effective.pop("initial_states"), locked=resume
        )
        initial_state = "+".join(state["file"] for state in selected_states)
        body = {
            "model_size": effective["model_size"],
            "batch_size": effective["batch_size"],
            "batch_length": effective["batch_length"],
            "replay_ratio": effective["replay_ratio"],
            "num_envs": effective["num_envs"],
            "fresh_start": options.fresh_start,
            "game_id": options.game_id,
            "initial_state": initial_state,
            "resume_prefill": effective["resume_prefill"],
        }
        route = "/api/training/switch" if switching else "/api/training/start"
        mode = "switch" if switching else ("resume" if resume else "new")
        consequences = self._consequences(
            mode=mode,
            resume=resume,
            explicit_fresh=options.fresh_start,
            live_game=live_game,
            head=head,
        )
        warnings = []
        if switching:
            warnings.append("Confirm gracefully suspends the currently running game before starting this plan.")

        now = self.clock()
        plan_id = self.plan_id_factory()
        head_summary = None
        if head:
            head_summary = {
                "snapshot_id": head.get("snapshot_id"),
                "step": head.get("step"),
                "lineage": brain.get("active_lineage"),
            }
        proposal = {
            "type": "training_start_proposal",
            "id": plan_id,
            "studio_revision": studio["revision"],
            "created_at": _iso(now),
            "expires_at": _iso(now + self.plan_ttl),
            "game": {
                "id": options.game_id,
                "display_name": focused.get("display_name") or options.game_id,
            },
            "mode": mode,
            "head": head_summary,
            "model": {"size": effective["model_size"]},
            "states": selected_states,
            "available_states": available_states,
            "replay_ratio": effective["replay_ratio"],
            "num_envs": effective["num_envs"],
            "batch_size": effective["batch_size"],
            "batch_length": effective["batch_length"],
            "consequences": consequences,
            "warnings": warnings,
            "exact_request": {"route": route, "body": copy.deepcopy(body)},
        }
        payload = _PlanPayload(
            plan_id=plan_id,
            game_id=options.game_id,
            studio_revision=studio["revision"],
            created_at=now,
            expires_at=now + self.plan_ttl,
            proposal_json=_canonical(proposal),
            exact_request_json=_canonical({"route": route, "body": body}),
        )
        with self._lock:
            if plan_id in self._plans:
                raise PlannerError(500, "plan id collision")
            self._plans[plan_id] = _StoredPlan(payload=payload)
        return json.loads(payload.proposal_json)

    @staticmethod
    def _validate_workspace(focused: dict):
        configs = focused.get("configs") or {}
        data = configs.get("data.json") or {}
        actions = configs.get("actions.json") or {}
        training = configs.get("training.json") or {}
        errors = []
        if not isinstance(data.get("info"), dict) or not data["info"]:
            errors.append("data.json has no RAM variables")
        action_rows = actions.get("actions")
        if not isinstance(action_rows, list) or not action_rows:
            errors.append("actions.json has no actions")
        elif any(not isinstance(row, dict) or not row.get("name")
                 or not isinstance(row.get("buttons"), list) for row in action_rows):
            errors.append("actions.json contains an invalid action")
        reward_vars = ((training.get("reward") or {}).get("variables"))
        if not isinstance(reward_vars, dict) or not reward_vars:
            errors.append("training.json has no reward variables")
        if errors:
            raise PlannerError(409, "invalid training workspace: " + "; ".join(errors))

    def _new_settings(self, studio: dict, focused: dict, options: PlanOptions) -> dict:
        advisor = studio.get("advisor") or {}
        model_size = options.model_size or advisor.get("recommended")
        if model_size not in VALID_MODEL_SIZES:
            raise PlannerError(400, f"unknown model architecture '{model_size}'")
        fits = advisor.get("fits") or []
        if fits and model_size not in fits:
            raise PlannerError(409, f"model architecture '{model_size}' does not fit this GPU")
        config = TrainingConfig.from_preset(model_size)
        values = {
            "model_size": model_size,
            "batch_size": options.batch_size if options.batch_size is not None else config.batch_size,
            "batch_length": options.batch_length if options.batch_length is not None else config.batch_length,
            "replay_ratio": options.replay_ratio if options.replay_ratio is not None else config.replay_ratio,
            "num_envs": options.num_envs if options.num_envs is not None else config.num_envs,
            "resume_prefill": options.resume_prefill if options.resume_prefill is not None else 0,
            "initial_states": options.states or self._default_states(focused),
        }
        self._validate_numeric(values)
        return values

    def _resolved_resume_settings(
        self, head: dict, focused: dict, options: PlanOptions
    ) -> dict:
        path_text = head.get("resolved_config")
        if not path_text or not Path(path_text).is_file():
            raise PlannerError(409, "resumable head has no readable resolved config")
        try:
            resolved = yaml.safe_load(Path(path_text).read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise PlannerError(409, f"resumable head config is invalid: {exc}") from exc

        algo = resolved.get("algo") or {}
        env = resolved.get("env") or {}
        wrapper = env.get("wrapper") or {}
        recurrent_size = (((algo.get("world_model") or {}).get("recurrent_model") or {})
                          .get("recurrent_state_size"))
        model_size = _ARCHITECTURE_BY_RECURRENT_SIZE.get(recurrent_size)
        if model_size is None:
            raise PlannerError(
                409, f"unknown resumed architecture (recurrent_state_size={recurrent_size!r})"
            )
        initial = wrapper.get("initial_state")
        values = {
            "model_size": model_size,
            "batch_size": algo.get("per_rank_batch_size"),
            "batch_length": algo.get("per_rank_sequence_length"),
            "replay_ratio": algo.get("replay_ratio"),
            "num_envs": env.get("num_envs"),
            "resume_prefill": 0,
            "initial_states": self._split_states(initial),
        }
        if not values["initial_states"]:
            raise PlannerError(409, "resumable head config has no initial state")
        self._validate_numeric(values)
        self._reject_locked_override("model_size", options.model_size, model_size)
        self._reject_locked_override("batch_size", options.batch_size, values["batch_size"])
        self._reject_locked_override("batch_length", options.batch_length, values["batch_length"])
        self._reject_locked_override("replay_ratio", options.replay_ratio, values["replay_ratio"])
        self._reject_locked_override("num_envs", options.num_envs, values["num_envs"])
        self._reject_locked_override("resume_prefill", options.resume_prefill, 0)
        if options.states is not None and tuple(options.states) != tuple(values["initial_states"]):
            raise PlannerError(409, "states are locked by the resumable head config")
        if not head.get("replay_available"):
            raise PlannerError(409, "resumable head replay is unavailable; a fresh plan is required")
        replay_path = head.get("replay_path")
        if not replay_path:
            raise PlannerError(409, "resumable head has no replay path")
        buffer_meta_path = Path(replay_path).parent / "buffer-meta.json"
        try:
            buffer_meta = json.loads(buffer_meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PlannerError(
                409, "resumable head has no valid buffer-meta.json; compatibility is unknown"
            ) from exc
        action_rows = (((focused.get("configs") or {}).get("actions.json") or {})
                       .get("actions") or [])
        expected_meta = {
            "num_envs": values["num_envs"],
            "action_count": len(action_rows),
        }
        if buffer_meta != expected_meta:
            raise PlannerError(
                409,
                f"replay buffer is incompatible: saved {buffer_meta}, current {expected_meta}",
            )
        return values

    @staticmethod
    def _reject_locked_override(name: str, requested, locked):
        if requested is None:
            return
        equal = (
            math.isclose(float(requested), float(locked), rel_tol=1e-9, abs_tol=1e-12)
            if isinstance(requested, (int, float)) and isinstance(locked, (int, float))
            else requested == locked
        )
        if not equal:
            raise PlannerError(409, f"{name} is locked to {locked!r} by the resumable head")

    @staticmethod
    def _validate_numeric(values: dict):
        checks = (
            ("batch_size", values.get("batch_size"), 1, 256),
            ("batch_length", values.get("batch_length"), 1, 2048),
            ("num_envs", values.get("num_envs"), 1, 16),
            ("replay_ratio", values.get("replay_ratio"), 0, 1),
            ("resume_prefill", values.get("resume_prefill"), 0, 10_000_000),
        )
        for name, value, lower, upper in checks:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PlannerError(409, f"resolved {name} is invalid")
            if name == "replay_ratio":
                valid = lower < value <= upper
            else:
                valid = lower <= value <= upper and float(value).is_integer()
            if not valid:
                raise PlannerError(400, f"{name} must be in the supported range")

    @staticmethod
    def _split_states(value) -> tuple[str, ...]:
        if not isinstance(value, str):
            return ()
        return tuple(part.strip().removesuffix(".state") for part in re.split(r"[+,]", value)
                     if part.strip())

    @staticmethod
    def _default_states(focused: dict) -> tuple[str, ...]:
        default = focused.get("default_state")
        if default:
            return (default,)
        states = focused.get("states") or []
        return (states[0]["file"],) if states else ()

    @staticmethod
    def _resolve_states(
        focused: dict, selected: tuple[str, ...], *, locked: bool
    ) -> tuple[list[dict], list[dict]]:
        available = []
        by_file = {}
        for raw in focused.get("states") or []:
            item = {
                "file": raw.get("file"),
                "label": raw.get("label") or raw.get("file"),
                "description": raw.get("description") or "Unknown",
            }
            if item["file"]:
                available.append(item)
                by_file[item["file"]] = item
        if not selected:
            raise PlannerError(409, "game has no usable initial state")
        missing = [state for state in selected if state not in by_file]
        if missing:
            prefix = "resumable head references" if locked else "request includes"
            raise PlannerError(409, f"{prefix} unknown state(s): {', '.join(missing)}")
        return [copy.deepcopy(by_file[state]) for state in selected], available

    @staticmethod
    def _consequences(*, mode: str, resume: bool, explicit_fresh: bool,
                      live_game: Optional[str], head: Optional[dict]) -> list[str]:
        result = []
        if mode == "switch":
            result.append(f"Training for {live_game} will be gracefully suspended first.")
        if resume:
            result.append(f"Resume the active lineage from checkpoint step {head.get('step')}.")
            result.append("Model architecture and replay-compatible settings stay locked to that head.")
        elif explicit_fresh and head:
            result.append("The existing head will not be resumed; a new brain and replay buffer will start.")
        else:
            result.append("No resumable head will be used; training starts a new brain.")
        result.append("Nothing changes until the browser Confirm control is used.")
        return result

    # ------------------------------------------------------------------
    # One-time broker
    # ------------------------------------------------------------------

    def _claim(self, plan_id: str, approval_token: Optional[str], action: str) -> _PlanPayload:
        now = self.clock()
        with self._lock:
            self._authorize_locked(approval_token, now)
            stored = self._plans.get(plan_id)
            if stored is None:
                raise PlannerError(404, "training plan not found")
            if stored.status != "pending":
                raise PlannerError(409, f"training plan is already {stored.status}")
            if stored.payload.expires_at <= now:
                stored.status = "expired"
                raise PlannerError(409, "training plan expired")
            stored.status = action
            return stored.payload

    def cancel(self, plan_id: str, approval_token: Optional[str]) -> dict:
        payload = self._claim(plan_id, approval_token, "cancelled")
        return {"status": "cancelled", "plan_id": payload.plan_id}

    async def confirm(
        self,
        plan_id: str,
        approval_token: Optional[str],
        executor: Callable[[str, dict], Awaitable[dict]],
    ) -> dict:
        payload = self._claim(plan_id, approval_token, "validating")
        try:
            current = self.state_builder.build(payload.game_id, projection="compact")
        except Exception as exc:
            self._set_status(plan_id, "failed")
            if isinstance(exc, FileNotFoundError):
                raise PlannerError(409, "training plan focus no longer exists") from exc
            raise
        if current.get("revision") != payload.studio_revision:
            self._set_status(plan_id, "stale")
            raise PlannerError(409, "training plan is stale; create a fresh proposal")

        self._set_status(plan_id, "confirming", expected="validating")
        exact = json.loads(payload.exact_request_json)
        try:
            execution = await executor(exact["route"], copy.deepcopy(exact["body"]))
        except BaseException:
            self._set_status(plan_id, "failed")
            raise
        self._set_status(plan_id, "confirmed", expected="confirming")
        state_warning = None
        try:
            studio_state = self.state_builder.build(payload.game_id, projection="compact")
        except Exception:
            # The mutation receipt remains authoritative. A failed ambient read
            # must not turn a successful start into an apparent failed action.
            studio_state = None
            state_warning = "Training executed, but fresh studio state is temporarily unavailable."
        result = {
            "status": "confirmed",
            "plan_id": payload.plan_id,
            "execution": execution,
            "studio_state": studio_state,
            "intent": {"type": "open_tab", "tab": "metrics"},
        }
        if state_warning:
            result["warning"] = state_warning
        return result

    def _set_status(self, plan_id: str, status: str, expected: Optional[str] = None):
        with self._lock:
            stored = self._plans[plan_id]
            if expected is not None and stored.status != expected:
                raise PlannerError(409, f"training plan is already {stored.status}")
            stored.status = status
