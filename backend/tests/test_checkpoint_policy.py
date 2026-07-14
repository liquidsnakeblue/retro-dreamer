import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import ValidationError

from backend.api import routes
from backend.api.routes import TrainingStartRequest
from backend.training.config import TrainingConfig


POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "sheeprl"
    / "sheeprl"
    / "utils"
    / "checkpoint_policy.py"
)
SPEC = importlib.util.spec_from_file_location("checkpoint_policy_under_test", POLICY_PATH)
assert SPEC and SPEC.loader
POLICY_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY_MODULE
SPEC.loader.exec_module(POLICY_MODULE)
CheckpointRetentionPolicy = POLICY_MODULE.CheckpointRetentionPolicy


class TrainingCheckpointConfigTest(unittest.TestCase):
    def test_storage_safe_defaults(self):
        config = TrainingConfig().validate()
        self.assertEqual(config.checkpoint_every, 10_000)
        self.assertEqual(config.checkpoint_keep_last, 3)
        self.assertEqual(config.checkpoint_milestone_every, 50_000)
        self.assertEqual(config.checkpoint_keep_milestones, 5)

    def test_invalid_checkpoint_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            TrainingConfig(checkpoint_keep_last=True).validate()
        with self.assertRaisesRegex(ValueError, "both be zero"):
            TrainingConfig(checkpoint_keep_milestones=0).validate()
        with self.assertRaisesRegex(ValueError, "at least checkpoint_every"):
            TrainingConfig(
                checkpoint_every=60_000,
                checkpoint_milestone_every=50_000,
            ).validate()


class CheckpointRetentionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, step: int) -> Path:
        path = self.root / f"ckpt_{step}_0.ckpt"
        path.write_text(str(step))
        return path

    def test_preexisting_checkpoint_is_never_adopted_or_deleted(self):
        preexisting = self.write(1)
        policy = CheckpointRetentionPolicy(
            keep_last=2, milestone_every=50_000, keep_milestones=1
        )
        for step in (10_000, 20_000, 30_000, 40_000):
            policy.record_successful_write(self.write(step))
        self.assertTrue(preexisting.exists())
        self.assertEqual(preexisting.read_text(), "1")

    def test_recent_plus_bounded_non_divisible_milestones(self):
        policy = CheckpointRetentionPolicy(
            keep_last=2, milestone_every=50_000, keep_milestones=2
        )
        for step in (49_998, 50_004, 60_006, 100_002, 110_004, 150_006, 160_008):
            policy.record_successful_write(self.write(step))

        retained = {int(path.stem.split("_")[1]) for path in self.root.glob("*.ckpt")}
        # First writes in the newest two positive buckets, union newest two.
        self.assertEqual(retained, {100_002, 150_006, 160_008})
        self.assertLessEqual(len(retained), 4)

    def test_new_policy_instance_does_not_adopt_prior_process_writes(self):
        old_write = self.write(10_000)
        policy = CheckpointRetentionPolicy(keep_last=1)
        policy.record_successful_write(self.write(20_000))
        policy.record_successful_write(self.write(30_000))
        self.assertTrue(old_write.exists())
        self.assertFalse((self.root / "ckpt_20000_0.ckpt").exists())

    def test_manifest_bounds_only_future_writes_across_processes(self):
        logs = self.root / "logs"
        manifest = self.root / "state" / "retention.json"
        preexisting = logs / "old-run" / "checkpoint" / "ckpt_1_0.ckpt"
        preexisting.parent.mkdir(parents=True)
        preexisting.write_text("historical")

        def record(run_name, steps):
            policy = CheckpointRetentionPolicy(
                keep_last=2,
                milestone_every=50_000,
                keep_milestones=1,
                manifest_path=manifest,
                managed_root=logs,
            )
            checkpoint_dir = logs / run_name / "checkpoint"
            checkpoint_dir.mkdir(parents=True)
            for step in steps:
                path = checkpoint_dir / f"ckpt_{step}_0.ckpt"
                path.write_text(str(step))
                policy.record_successful_write(path)

        record("run-1", (10_000, 20_000, 55_000))
        record("run-2", (65_000, 110_000))
        record("run-3", (120_000, 160_000))

        retained_future = [
            path for path in logs.glob("**/*.ckpt") if path != preexisting
        ]
        self.assertLessEqual(len(retained_future), 3)
        self.assertTrue(preexisting.exists())
        self.assertEqual(preexisting.read_text(), "historical")

    def test_manifest_cannot_authorize_deletion_outside_managed_root(self):
        logs = self.root / "logs"
        logs.mkdir()
        outside = self.root / "outside.ckpt"
        outside.write_text("protected")
        manifest = self.root / "retention.json"
        manifest.write_text(json.dumps({
            "format": "retro-dreamer-checkpoint-retention-v1",
            "checkpoints": [{
                "path": str(outside), "step": 1, "milestone": False,
            }],
        }))
        policy = CheckpointRetentionPolicy(
            keep_last=1,
            manifest_path=manifest,
            managed_root=logs,
        )
        managed = logs / "ckpt_10_0.ckpt"
        managed.write_text("new")
        policy.record_successful_write(managed)
        self.assertTrue(outside.exists())

    def test_step_reset_starts_a_new_milestone_sequence(self):
        logs = self.root / "logs"
        manifest = self.root / "retention.json"

        first = CheckpointRetentionPolicy(
            keep_last=1,
            milestone_every=50_000,
            keep_milestones=3,
            manifest_path=manifest,
            managed_root=logs,
        )
        run_one = logs / "run-one"
        run_one.mkdir(parents=True)
        for step in (50_000, 60_000, 100_000):
            path = run_one / f"ckpt_{step}_0.ckpt"
            path.write_text("one")
            first.record_successful_write(path)

        second = CheckpointRetentionPolicy(
            keep_last=1,
            milestone_every=50_000,
            keep_milestones=3,
            manifest_path=manifest,
            managed_root=logs,
        )
        run_two = logs / "run-two"
        run_two.mkdir()
        for step in (10_000, 50_000):
            path = run_two / f"ckpt_{step}_0.ckpt"
            path.write_text("two")
            second.record_successful_write(path)

        self.assertTrue((run_one / "ckpt_50000_0.ckpt").exists())
        self.assertTrue((run_two / "ckpt_50000_0.ckpt").exists())

    def test_zero_milestone_count_does_not_retain_milestones(self):
        policy = CheckpointRetentionPolicy(
            keep_last=1, milestone_every=10, keep_milestones=0
        )
        for step in (10, 20, 30, 40, 50):
            policy.record_successful_write(self.write(step))
        self.assertEqual(
            {path.name for path in self.root.glob("*.ckpt")},
            {"ckpt_50_0.ckpt"},
        )

    def test_malformed_managed_name_fails_closed(self):
        malformed = self.root / "manual.ckpt"
        malformed.write_text("brain")
        policy = CheckpointRetentionPolicy(keep_last=1)
        policy.record_successful_write(malformed)
        policy.record_successful_write(self.write(10_000))
        policy.record_successful_write(self.write(20_000))
        self.assertTrue(malformed.exists())


class CheckpointPolicyApiTest(unittest.TestCase):
    def setUp(self):
        self.old_trainer = routes._trainer
        self.old_game_manager = routes._game_manager
        self.old_training_state_dir = routes.TRAINING_STATE_DIR
        self.state_temp = tempfile.TemporaryDirectory()
        self.trainer = SimpleNamespace(config=None)

        def start(config, fresh_start=False):
            self.trainer.config = config
            self.trainer.fresh_start = fresh_start

        self.trainer.start = start
        routes._trainer = self.trainer
        routes._game_manager = None
        routes.TRAINING_STATE_DIR = Path(self.state_temp.name)

    def tearDown(self):
        routes._trainer = self.old_trainer
        routes._game_manager = self.old_game_manager
        routes.TRAINING_STATE_DIR = self.old_training_state_dir
        self.state_temp.cleanup()

    def test_start_request_applies_checkpoint_overrides(self):
        request = TrainingStartRequest(
            model_size="small",
            checkpoint_every=20_000,
            checkpoint_keep_last=4,
            checkpoint_milestone_every=100_000,
            checkpoint_keep_milestones=2,
        )
        asyncio.run(routes.start_training(request))
        self.assertEqual(self.trainer.config.checkpoint_every, 20_000)
        self.assertEqual(self.trainer.config.checkpoint_keep_last, 4)
        self.assertEqual(self.trainer.config.checkpoint_milestone_every, 100_000)
        self.assertEqual(self.trainer.config.checkpoint_keep_milestones, 2)
        persisted = json.loads(
            (routes.TRAINING_STATE_DIR / "last_start_request.json").read_text()
        )
        self.assertEqual(persisted["checkpoint_every"], 20_000)

    def test_start_request_rejects_bool_and_invalid_cross_field_policy(self):
        with self.assertRaises(ValidationError):
            TrainingStartRequest(checkpoint_every=True)
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(routes.start_training(TrainingStartRequest(
                checkpoint_every=60_000,
                checkpoint_milestone_every=50_000,
            )))
        self.assertEqual(caught.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
