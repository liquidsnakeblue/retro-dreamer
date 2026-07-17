import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import Response
from pydantic import ValidationError

from backend import copilot
from backend.action_manifest import build_action_manifest, write_action_manifest
from backend.api import routes
from backend.training.planner import PlannerError, TrainingPlanner


class FakeClock:
    def __init__(self, value=1_800_000_000.0):
        self.value = value

    def __call__(self):
        return self.value


class FakeStateBuilder:
    def __init__(self):
        self.revision = "studio-rev-1"
        self.raise_missing = False
        self.state = {
            "training": {"state": "idle", "game_id": "Old-Game"},
            "advisor": {
                "recommended": "medium",
                "fits": ["debug", "small", "medium", "large", "xl"],
            },
            "focused_game": {
                "game_id": "Focus-Game",
                "display_name": "Focus Game",
                "source": "custom",
                "default_state": "start",
                "states": [
                    {"file": "start", "label": "Start"},
                    {"file": "hard", "label": "Hard", "description": "Hard section"},
                ],
                "readiness": {"trainable": True, "blockers": []},
                "configs": {
                    "data.json": {"info": {"score": {"type": "<u2"}}},
                    "actions.json": {
                        "actions": [{"name": "NoOp", "buttons": []}]
                    },
                    "training.json": {
                        "reward": {"variables": {"score": {"reward": 1}}},
                        "done": {"variables": {}},
                    },
                    "metadata.json": {"default_state": "start"},
                },
                "brain": {
                    "has_brain": False,
                    "active_lineage": None,
                    "head": None,
                },
            },
        }

    def build(self, game_id, *, projection="compact", **_kwargs):
        if self.raise_missing or game_id != "Focus-Game":
            raise FileNotFoundError(f"Game '{game_id}' not found")
        value = copy.deepcopy(self.state)
        value.update({"revision": self.revision, "projection": projection})
        return value


class TrainingPlannerTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = FakeClock()
        self.builder = FakeStateBuilder()
        self.current_manifest = build_action_manifest(
            "Focus-Game",
            self.builder.state["focused_game"]["configs"]["actions.json"],
        )
        self.plan_number = 0

        def next_plan_id():
            self.plan_number += 1
            return f"plan-{self.plan_number}"

        self.planner = TrainingPlanner(
            self.builder,
            clock=self.clock,
            plan_id_factory=next_plan_id,
            token_factory=lambda: "browser-secret",
        )

    def tearDown(self):
        self.temp.cleanup()

    def add_head(self, recurrent_size=2048):
        replay_path = Path(self.temp.name) / "lineage" / "replay"
        replay_path.mkdir(parents=True)
        manifest_path = write_action_manifest(
            self.current_manifest, replay_path.parent
        )
        self.head_manifest_path = manifest_path
        config_path = Path(self.temp.name) / "config.yaml"
        config_path.write_text(
            "algo:\n"
            "  per_rank_batch_size: 32\n"
            "  per_rank_sequence_length: 64\n"
            "  replay_ratio: 0.25\n"
            "  world_model:\n"
            "    recurrent_model:\n"
            f"      recurrent_state_size: {recurrent_size}\n"
            "env:\n"
            "  num_envs: 6\n"
            "  wrapper:\n"
            "    initial_state: start+hard\n"
            f"    action_manifest: {manifest_path}\n"
            f"    action_manifest_hash: {self.current_manifest['sha256']}\n"
        )
        (replay_path.parent / "buffer-meta.json").write_text(
            json.dumps({
                "format": "retro-dreamer-buffer-meta-v2",
                "num_envs": 6,
                "action_count": 1,
                "action_manifest_hash": self.current_manifest["sha256"],
            })
        )
        self.builder.state["focused_game"]["brain"] = {
            "has_brain": True,
            "active_lineage": "main",
            "head": {
                "snapshot_id": 7,
                "step": 12345,
                "action_manifest_hash": self.current_manifest["sha256"],
                "replay_available": True,
                "replay_path": str(replay_path),
                "resolved_config": str(config_path),
            },
        }
        return config_path

    async def test_new_plan_uses_code_presets_and_is_immutable(self):
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(proposal["type"], "training_start_proposal")
        self.assertEqual(proposal["generation"], 1)
        self.assertEqual(proposal["superseded_plan_ids"], [])
        self.assertEqual(proposal["mode"], "new")
        self.assertEqual(proposal["launch"], {
            "strategy": "new",
            "initial_state": "start",
            "fresh_start": False,
            "action_manifest_hash": self.current_manifest["sha256"],
        })
        self.assertEqual(proposal["model"], {"size": "medium"})
        self.assertEqual(proposal["num_envs"], 8)
        self.assertEqual(proposal["batch_size"], 16)
        self.assertEqual(proposal["batch_length"], 64)
        self.assertEqual(proposal["replay_ratio"], 0.125)
        self.assertEqual(proposal["states"], [
            {"file": "start", "label": "Start", "description": "Unknown"}
        ])
        exact = proposal["exact_request"]
        self.assertEqual(exact["route"], "/api/training/start")
        self.assertFalse(exact["body"]["fresh_start"])
        self.assertEqual(exact["body"]["batch_length"], 64)
        self.assertEqual(
            exact["body"]["action_manifest_hash"],
            self.current_manifest["sha256"],
        )

        # Mutating the caller's returned object cannot alter the stored body.
        proposal["exact_request"]["body"]["model_size"] = "xl"
        token = self.planner.create_approval_session()
        seen = []

        async def execute(route, body):
            seen.append((route, body))
            return {"status": "started"}

        result = await self.planner.confirm("plan-1", token, execute)
        self.assertEqual(seen[0][1]["model_size"], "medium")
        self.assertEqual(result["intent"], {"type": "open_tab", "tab": "metrics"})

    async def test_state_rotation_override_reaches_the_stored_executor_request(self):
        proposal = self.planner.create_plan({
            "game_id": "Focus-Game",
            "states": ["hard", "start"],
        })
        self.assertEqual(
            [state["file"] for state in proposal["states"]],
            ["hard", "start"],
        )
        self.assertEqual(proposal["launch"]["initial_state"], "hard+start")
        self.assertEqual(
            proposal["exact_request"]["body"]["initial_state"],
            "hard+start",
        )
        token = self.planner.create_approval_session()
        seen = []

        async def execute(route, body):
            seen.append((route, body))
            return {"status": "started"}

        await self.planner.confirm(proposal["id"], token, execute)
        self.assertEqual(seen[0][0], "/api/training/start")
        self.assertEqual(seen[0][1]["initial_state"], "hard+start")

    async def test_plus_joined_default_state_splits_into_rotation(self):
        # Regression: metadata default_state may itself be a '+'-joined
        # rotation (Zelda "Overworld+SwordCavePre+PostSwordExit"). The fresh
        # path must split it like the resume path does — before the fix,
        # _default_states returned the combined string as ONE state and every
        # fresh plan using the default 409'd as "unknown state(s)".
        self.builder.state["focused_game"]["default_state"] = "start+hard"
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(
            [state["file"] for state in proposal["states"]],
            ["start", "hard"],
        )
        self.assertEqual(proposal["launch"]["initial_state"], "start+hard")
        self.assertEqual(
            proposal["exact_request"]["body"]["initial_state"],
            "start+hard",
        )

    def test_resume_locks_every_effective_setting_from_resolved_config(self):
        self.add_head()
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(proposal["mode"], "resume")
        self.assertEqual(proposal["launch"], {
            "strategy": "resume",
            "initial_state": "start+hard",
            "fresh_start": False,
            "action_manifest_hash": self.current_manifest["sha256"],
        })
        self.assertEqual(proposal["head"], {
            "snapshot_id": 7, "step": 12345, "lineage": "main"
        })
        self.assertEqual(proposal["model"], {"size": "large"})
        self.assertEqual(proposal["batch_size"], 32)
        self.assertEqual(proposal["batch_length"], 64)
        self.assertEqual(proposal["replay_ratio"], 0.25)
        self.assertEqual(proposal["num_envs"], 6)
        self.assertEqual([state["file"] for state in proposal["states"]], ["start", "hard"])

        with self.assertRaisesRegex(PlannerError, "model_size is locked"):
            self.planner.create_plan({"game_id": "Focus-Game", "model_size": "small"})
        with self.assertRaisesRegex(PlannerError, "states are locked"):
            self.planner.create_plan({"game_id": "Focus-Game", "states": ["start"]})

    def test_explicit_fresh_plan_marks_existing_head_unused(self):
        self.add_head()
        proposal = self.planner.create_plan({
            "game_id": "Focus-Game",
            "states": ["hard"],
            "fresh_start": True,
        })
        self.assertEqual(proposal["mode"], "new")
        self.assertIsNotNone(proposal["head"])
        self.assertEqual(proposal["launch"], {
            "strategy": "fresh",
            "initial_state": "hard",
            "fresh_start": True,
            "action_manifest_hash": self.current_manifest["sha256"],
        })

    def test_resume_rejects_incompatible_buffer_metadata_before_proposal(self):
        self.add_head()
        replay_path = Path(
            self.builder.state["focused_game"]["brain"]["head"]["replay_path"]
        )
        (replay_path.parent / "buffer-meta.json").write_text(
            json.dumps({
                "format": "retro-dreamer-buffer-meta-v2",
                "num_envs": 6,
                "action_count": 99,
                "action_manifest_hash": self.current_manifest["sha256"],
            })
        )
        with self.assertRaisesRegex(PlannerError, "replay buffer is incompatible") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

    def test_resume_rejects_same_count_action_reorder(self):
        self.add_head()
        rows = self.builder.state["focused_game"]["configs"]["actions.json"]["actions"]
        rows.append({"name": "Forward", "buttons": ["B"]})
        # Reseal the saved two-action order, then reverse only the workspace.
        saved = build_action_manifest("Focus-Game", {"actions": list(rows)})
        manifest_path = write_action_manifest(saved, Path(self.temp.name) / "saved")
        config_path = Path(
            self.builder.state["focused_game"]["brain"]["head"]["resolved_config"]
        )
        text = config_path.read_text()
        text = text.replace(
            f"action_manifest: {self.head_manifest_path}",
            f"action_manifest: {manifest_path}",
        ).replace(
            f"action_manifest_hash: {self.current_manifest['sha256']}",
            f"action_manifest_hash: {saved['sha256']}",
        )
        config_path.write_text(text)
        replay_path = Path(
            self.builder.state["focused_game"]["brain"]["head"]["replay_path"]
        )
        (replay_path.parent / "buffer-meta.json").write_text(json.dumps({
            "format": "retro-dreamer-buffer-meta-v2",
            "num_envs": 6,
            "action_count": 2,
            "action_manifest_hash": saved["sha256"],
        }))
        self.builder.state["focused_game"]["brain"]["head"][
            "action_manifest_hash"
        ] = saved["sha256"]
        rows.reverse()

        with self.assertRaisesRegex(PlannerError, "same-count reorders") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

    def test_unknown_resumed_architecture_is_rejected(self):
        self.add_head(recurrent_size=777)
        with self.assertRaisesRegex(PlannerError, "unknown resumed architecture") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

    def test_legacy_resume_without_manifest_fails_closed(self):
        config_path = self.add_head()
        config_path.write_text("\n".join(
            line for line in config_path.read_text().splitlines()
            if "action_manifest" not in line
        ) + "\n")

        with self.assertRaisesRegex(PlannerError, "legacy checkpoint") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

        fresh = self.planner.create_plan({
            "game_id": "Focus-Game",
            "fresh_start": True,
        })
        self.assertEqual("new", fresh["mode"])
        self.assertEqual("fresh", fresh["launch"]["strategy"])

    def test_resume_rejects_config_that_conflicts_with_catalog_binding(self):
        self.add_head()
        self.builder.state["focused_game"]["brain"]["head"][
            "action_manifest_hash"
        ] = "b" * 64

        with self.assertRaisesRegex(PlannerError, "write-once binding") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

    def test_switch_is_selected_and_same_game_active_is_rejected(self):
        self.builder.state["training"] = {"state": "training", "game_id": "Other-Game"}
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(proposal["mode"], "switch")
        self.assertEqual(proposal["launch"]["strategy"], "new")
        self.assertEqual(proposal["exact_request"]["route"], "/api/training/switch")
        self.builder.state["training"]["game_id"] = "Focus-Game"
        with self.assertRaisesRegex(PlannerError, "already training") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

    def test_switch_keeps_resume_strategy_visible(self):
        self.add_head()
        self.builder.state["training"] = {
            "state": "training", "game_id": "Other-Game",
        }
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(proposal["mode"], "switch")
        self.assertEqual(proposal["launch"]["strategy"], "resume")
        self.assertEqual(proposal["launch"]["initial_state"], "start+hard")

    def test_unknown_builtin_nontrainable_and_invalid_are_rejected(self):
        self.builder.raise_missing = True
        with self.assertRaises(PlannerError) as missing:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(missing.exception.status_code, 404)
        self.builder.raise_missing = False

        focused = self.builder.state["focused_game"]
        focused["source"] = "builtin"
        with self.assertRaisesRegex(PlannerError, "onboarded custom"):
            self.planner.create_plan({"game_id": "Focus-Game"})
        focused["source"] = "custom"
        focused["readiness"] = {"trainable": False, "blockers": ["missing ROM"]}
        with self.assertRaisesRegex(PlannerError, "missing ROM"):
            self.planner.create_plan({"game_id": "Focus-Game"})
        focused["readiness"] = {"trainable": True, "blockers": []}
        focused["configs"]["training.json"]["reward"]["variables"] = {}
        with self.assertRaisesRegex(PlannerError, "no reward source"):
            self.planner.create_plan({"game_id": "Focus-Game"})

    async def test_stale_and_one_time_guards_prevent_every_mutation(self):
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        token = self.planner.create_approval_session()
        self.builder.revision = "studio-rev-2"
        calls = []

        async def execute(route, body):
            calls.append((route, body))
            return {}

        with self.assertRaisesRegex(PlannerError, "stale") as stale:
            await self.planner.confirm(proposal["id"], token, execute)
        self.assertEqual(stale.exception.status_code, 409)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(PlannerError, "already stale"):
            await self.planner.confirm(proposal["id"], token, execute)
        self.assertEqual(calls, [])

    async def test_new_plan_atomically_supersedes_the_prior_pending_plan(self):
        first = self.planner.create_plan({"game_id": "Focus-Game"})
        second = self.planner.create_plan({
            "game_id": "Focus-Game", "states": ["hard"],
        })
        self.assertEqual(second["generation"], first["generation"] + 1)
        self.assertEqual(second["superseded_plan_ids"], [first["id"]])
        token = self.planner.create_approval_session()
        calls = []

        async def execute(route, body):
            calls.append((route, body))
            return {"status": "started"}

        with self.assertRaisesRegex(PlannerError, "superseded") as denied:
            await self.planner.confirm(first["id"], token, execute)
        self.assertEqual(denied.exception.status_code, 409)
        with self.assertRaisesRegex(PlannerError, "superseded"):
            self.planner.cancel(first["id"], token)
        self.assertEqual(calls, [])

        result = await self.planner.confirm(second["id"], token, execute)
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["initial_state"], "hard")

    async def test_superseded_status_does_not_bypass_browser_authorization(self):
        first = self.planner.create_plan({"game_id": "Focus-Game"})
        self.planner.create_plan({"game_id": "Focus-Game", "states": ["hard"]})

        async def forbidden(_route, _body):
            self.fail("superseded plan reached the executor")

        with self.assertRaises(PlannerError) as denied:
            await self.planner.confirm(first["id"], None, forbidden)
        self.assertEqual(denied.exception.status_code, 403)

    async def test_failed_replan_and_id_collision_leave_pending_plan_actionable(self):
        first = self.planner.create_plan({"game_id": "Focus-Game"})
        with self.assertRaisesRegex(PlannerError, "unknown state"):
            self.planner.create_plan({
                "game_id": "Focus-Game", "states": ["not-a-state"],
            })
        self.planner.plan_id_factory = lambda: first["id"]
        with self.assertRaisesRegex(PlannerError, "collision"):
            self.planner.create_plan({"game_id": "Focus-Game"})

        token = self.planner.create_approval_session()
        calls = []

        async def execute(route, body):
            calls.append((route, body))
            return {"status": "started"}

        await self.planner.confirm(first["id"], token, execute)
        self.assertEqual(len(calls), 1)

    async def test_failed_replacement_serialization_is_atomic(self):
        first = self.planner.create_plan({"game_id": "Focus-Game"})
        with patch(
            "backend.training.planner._canonical",
            side_effect=TypeError("forced serialization failure"),
        ):
            with self.assertRaisesRegex(TypeError, "forced serialization failure"):
                self.planner.create_plan({
                    "game_id": "Focus-Game", "states": ["hard"],
                })
        self.assertEqual(self.planner._plan_generation, first["generation"])
        self.assertNotIn("plan-2", self.planner._plans)

        token = self.planner.create_approval_session()
        calls = []

        async def execute(route, body):
            calls.append((route, body))
            return {"status": "started"}

        await self.planner.confirm(first["id"], token, execute)
        self.assertEqual(len(calls), 1)

    async def test_confirm_lock_allows_exactly_one_executor(self):
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        token = self.planner.create_approval_session()
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def execute(route, body):
            calls.append((route, body))
            entered.set()
            await release.wait()
            return {"status": "started"}

        first = asyncio.create_task(self.planner.confirm(proposal["id"], token, execute))
        await entered.wait()
        with self.assertRaisesRegex(PlannerError, "confirmation is already in progress"):
            self.planner.create_plan({"game_id": "Focus-Game", "states": ["hard"]})
        with self.assertRaisesRegex(PlannerError, "already confirming"):
            await self.planner.confirm(proposal["id"], token, execute)
        release.set()
        await first
        self.assertEqual(len(calls), 1)

    async def test_post_execution_state_failure_keeps_confirmed_receipt(self):
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        token = self.planner.create_approval_session()

        async def execute(_route, _body):
            self.builder.raise_missing = True
            return {"status": "started"}

        result = await self.planner.confirm(proposal["id"], token, execute)
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["execution"], {"status": "started"})
        self.assertIsNone(result["studio_state"])
        self.assertIn("fresh studio state", result["warning"])
        with self.assertRaisesRegex(PlannerError, "already confirmed"):
            await self.planner.confirm(proposal["id"], token, execute)

    async def test_cancel_is_one_time_and_has_zero_training_mutations(self):
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        token = self.planner.create_approval_session()
        self.assertEqual(
            self.planner.cancel(proposal["id"], token),
            {"status": "cancelled", "plan_id": proposal["id"]},
        )

        async def forbidden(_route, _body):
            self.fail("cancelled plan executed a training mutation")

        with self.assertRaisesRegex(PlannerError, "already cancelled"):
            await self.planner.confirm(proposal["id"], token, forbidden)

    async def test_missing_browser_credential_is_forbidden_without_consuming_plan(self):
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})

        async def execute(_route, _body):
            return {}

        with self.assertRaises(PlannerError) as denied:
            await self.planner.confirm(proposal["id"], None, execute)
        self.assertEqual(denied.exception.status_code, 403)
        token = self.planner.create_approval_session()
        await self.planner.confirm(proposal["id"], token, execute)


class TrainingPlannerRouteTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.builder = FakeStateBuilder()
        self.planner = TrainingPlanner(
            self.builder,
            clock=FakeClock(),
            plan_id_factory=lambda: "route-plan",
            token_factory=lambda: "route-browser-secret",
        )
        self.old_planner = routes._training_planner
        self.old_events = copilot._events
        self.old_seq = copilot._seq
        routes._training_planner = self.planner
        copilot._events = []
        copilot._seq = 0

    def tearDown(self):
        routes._training_planner = self.old_planner
        copilot._events = self.old_events
        copilot._seq = self.old_seq

    async def test_route_emits_typed_proposal_and_confirm_reuses_start_semantics(self):
        proposal = routes.plan_training(routes.TrainingPlanRequest(
            game_id="Focus-Game", states=["hard"],
        ))
        event = copilot.events(0)["events"][0]
        self.assertEqual(event, {
            "seq": 1,
            "ts": event["ts"],
            "kind": "proposal",
            "proposal": proposal,
        })
        self.assertNotIn("approval", json.dumps(event).lower())

        response = Response()
        self.assertEqual(routes.training_approval_session(response), {"status": "ready"})
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertIn("Path=/api/training/plans", cookie)
        self.assertNotIn("route-browser-secret", json.dumps({"status": "ready"}))

        start = AsyncMock(return_value={"status": "started", "game_id": "Focus-Game"})
        with patch.object(routes, "start_training", start):
            result = await routes.confirm_training_plan(
                proposal["id"], "route-browser-secret"
            )
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["intent"], {"type": "open_tab", "tab": "metrics"})
        sent = start.await_args.args[0]
        self.assertEqual(sent.game_id, "Focus-Game")
        self.assertEqual(sent.model_size, "medium")
        self.assertEqual(sent.batch_length, 64)
        self.assertEqual(sent.initial_state, "hard")
        self.assertEqual(proposal["launch"]["initial_state"], "hard")

    def test_plan_request_and_primer_expose_states_not_initial_state(self):
        request = routes.TrainingPlanRequest(
            game_id="Focus-Game", states=["hard"],
        )
        self.assertEqual(request.states, ["hard"])
        with self.assertRaises(ValidationError):
            routes.TrainingPlanRequest(
                game_id="Focus-Game", initial_state="hard",
            )

        primer = copilot.PRIMER_PATH.read_text()
        self.assertIn('"states":["BBP1"]', primer)
        self.assertIn("Never send `initial_state` to `/training/plan`", primer)
        self.assertIn("`metadata.json.default_state` as a launch workaround", primer)


if __name__ == "__main__":
    unittest.main()
