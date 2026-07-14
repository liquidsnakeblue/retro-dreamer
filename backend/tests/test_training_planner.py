import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import Response

from backend import copilot
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
        )
        replay_path = Path(self.temp.name) / "lineage" / "replay"
        replay_path.mkdir(parents=True)
        (replay_path.parent / "buffer-meta.json").write_text(
            json.dumps({"num_envs": 6, "action_count": 1})
        )
        self.builder.state["focused_game"]["brain"] = {
            "has_brain": True,
            "active_lineage": "main",
            "head": {
                "snapshot_id": 7,
                "step": 12345,
                "replay_available": True,
                "replay_path": str(replay_path),
                "resolved_config": str(config_path),
            },
        }
        return config_path

    async def test_new_plan_uses_code_presets_and_is_immutable(self):
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(proposal["type"], "training_start_proposal")
        self.assertEqual(proposal["mode"], "new")
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

    def test_resume_locks_every_effective_setting_from_resolved_config(self):
        self.add_head()
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(proposal["mode"], "resume")
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

    def test_resume_rejects_incompatible_buffer_metadata_before_proposal(self):
        self.add_head()
        replay_path = Path(
            self.builder.state["focused_game"]["brain"]["head"]["replay_path"]
        )
        (replay_path.parent / "buffer-meta.json").write_text(
            json.dumps({"num_envs": 6, "action_count": 99})
        )
        with self.assertRaisesRegex(PlannerError, "replay buffer is incompatible") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

    def test_unknown_resumed_architecture_is_rejected(self):
        self.add_head(recurrent_size=777)
        with self.assertRaisesRegex(PlannerError, "unknown resumed architecture") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

    def test_switch_is_selected_and_same_game_active_is_rejected(self):
        self.builder.state["training"] = {"state": "training", "game_id": "Other-Game"}
        proposal = self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(proposal["mode"], "switch")
        self.assertEqual(proposal["exact_request"]["route"], "/api/training/switch")
        self.builder.state["training"]["game_id"] = "Focus-Game"
        with self.assertRaisesRegex(PlannerError, "already training") as caught:
            self.planner.create_plan({"game_id": "Focus-Game"})
        self.assertEqual(caught.exception.status_code, 409)

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
        with self.assertRaisesRegex(PlannerError, "no reward variables"):
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
        proposal = routes.plan_training(routes.TrainingPlanRequest(game_id="Focus-Game"))
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


if __name__ == "__main__":
    unittest.main()
