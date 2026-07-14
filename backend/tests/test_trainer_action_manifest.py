import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# The lightweight backend test environment intentionally does not install the
# multi-gigabyte training stack. Stub only the import surface used while this
# test exercises pre-launch compatibility logic; no model code is run.
try:  # pragma: no cover - production/training environments have real torch
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - branch depends on test image
    torch_module = types.ModuleType("torch")
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: False,
        memory_allocated=lambda: 0,
        get_device_properties=lambda _index: SimpleNamespace(total_memory=0),
    )
    torch_utils = types.ModuleType("torch.utils")
    torch_tensorboard = types.ModuleType("torch.utils.tensorboard")
    torch_tensorboard.SummaryWriter = object
    sys.modules.update({
        "torch": torch_module,
        "torch.utils": torch_utils,
        "torch.utils.tensorboard": torch_tensorboard,
    })

try:  # pragma: no cover - production environments have rendering dependencies
    import backend.training.callbacks  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - branch depends on test image
    callbacks_module = types.ModuleType("backend.training.callbacks")
    callbacks_module.TensorBoardCallback = object
    callbacks_module.WebSocketBroadcaster = object
    callbacks_module.EpisodeRenderer = object
    sys.modules["backend.training.callbacks"] = callbacks_module

from backend.action_manifest import (
    build_action_manifest,
    load_action_manifest,
    write_action_manifest,
)
from backend.training.config import TrainingConfig
from backend.training.trainer import DreamerV3Trainer


class TrainerActionManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.game_dir = self.root / "games" / "Example-Nes-v0"
        self.game_dir.mkdir(parents=True)
        self.rows = [
            {"name": "NoOp", "buttons": []},
            {"name": "Forward", "buttons": ["B"]},
        ]
        self._write_actions(self.rows)
        self.lineage = self.root / "state" / "games" / "Example-Nes-v0" / "lineages" / "main"

    def tearDown(self):
        self.temp.cleanup()

    def _write_actions(self, rows):
        (self.game_dir / "actions.json").write_text(json.dumps({"actions": rows}))

    def _trainer(self, expected_hash=None):
        trainer = DreamerV3Trainer.__new__(DreamerV3Trainer)
        trainer.config = TrainingConfig(
            game_id="Example-Nes-v0",
            action_manifest_hash=expected_hash,
        )
        return trainer

    def _checkpoint(self, manifest_path=None, manifest_hash=None):
        run_dir = self.root / "run" / "version_0"
        checkpoint = run_dir / "checkpoint" / "ckpt_10_0.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        wrapper = "    initial_state: start\n"
        if manifest_path is not None:
            wrapper += f"    action_manifest: {manifest_path}\n"
            wrapper += f"    action_manifest_hash: {manifest_hash}\n"
        (run_dir / "config.yaml").write_text("env:\n  wrapper:\n" + wrapper)
        return checkpoint

    def test_fresh_launch_publishes_frozen_content_addressed_actions(self):
        current = build_action_manifest("Example-Nes-v0", {"actions": self.rows})
        trainer = self._trainer(current["sha256"])

        path, sealed = trainer._prepare_action_manifest(
            self.game_dir, self.lineage, None
        )
        self._write_actions(list(reversed(self.rows)))

        self.assertEqual(current["sha256"], sealed["sha256"])
        self.assertEqual(self.rows, load_action_manifest(path)["actions"])
        self.assertEqual(
            self.lineage / "action-manifests" / f"{current['sha256']}.json",
            path,
        )

    def test_stale_planned_hash_fails_before_manifest_store_write(self):
        planned = build_action_manifest("Example-Nes-v0", {"actions": self.rows})
        self._write_actions(list(reversed(self.rows)))
        trainer = self._trainer(planned["sha256"])

        with self.assertRaisesRegex(ValueError, "training plan is stale"):
            trainer._prepare_action_manifest(self.game_dir, self.lineage, None)
        self.assertFalse(self.lineage.exists())

    def test_resume_uses_checkpoint_manifest_when_workspace_matches(self):
        saved = build_action_manifest("Example-Nes-v0", {"actions": self.rows})
        saved_path = write_action_manifest(saved, self.lineage)
        checkpoint = self._checkpoint(saved_path, saved["sha256"])
        trainer = self._trainer(saved["sha256"])

        path, manifest = trainer._prepare_action_manifest(
            self.game_dir, self.lineage, str(checkpoint)
        )

        self.assertEqual(saved_path, path)
        self.assertEqual(saved, manifest)

    def test_resume_same_count_reorder_fails_before_replay_or_process_mutation(self):
        saved = build_action_manifest("Example-Nes-v0", {"actions": self.rows})
        saved_path = write_action_manifest(saved, self.lineage)
        checkpoint = self._checkpoint(saved_path, saved["sha256"])
        reordered = list(reversed(self.rows))
        self._write_actions(reordered)
        current = build_action_manifest("Example-Nes-v0", {"actions": reordered})
        replay = self.lineage / "replay"
        replay.mkdir(parents=True)
        sentinel = replay / "keep.memmap"
        sentinel.write_bytes(b"untouched")
        meta = self.lineage / "buffer-meta.json"
        meta.write_text("legacy sentinel")

        trainer = DreamerV3Trainer.__new__(DreamerV3Trainer)
        trainer.config = TrainingConfig(
            game_id="Example-Nes-v0",
            action_manifest_hash=current["sha256"],
        )
        trainer.game_manager = SimpleNamespace(games_dir=self.game_dir.parent)
        trainer._fresh_start = False
        trainer._resume_catalog_action_hash = saved["sha256"]
        trainer._catalog_resumable_head = lambda _game_id: str(checkpoint)

        with (
            patch("backend.training.trainer.STATE_DIR", self.root / "state"),
            patch("backend.training.trainer.SHEEPRL_DIR", self.root / "sheeprl"),
            patch("backend.training.trainer.subprocess.Popen") as popen,
            patch(
                "backend.training.trainer.subprocess.check_output",
                return_value=b"10000,20000\n",
            ),
            patch("backend.training.trainer.threading.Thread"),
            self.assertRaisesRegex(ValueError, "same-count reorders"),
        ):
            trainer._launch_sheeprl()

        popen.assert_not_called()
        self.assertEqual(b"untouched", sentinel.read_bytes())
        self.assertEqual("legacy sentinel", meta.read_text())

    def test_legacy_checkpoint_without_manifest_fails_clearly(self):
        checkpoint = self._checkpoint()
        current = build_action_manifest("Example-Nes-v0", {"actions": self.rows})
        trainer = self._trainer(current["sha256"])

        with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
            trainer._prepare_action_manifest(
                self.game_dir, self.lineage, str(checkpoint)
            )

    def test_resume_rejects_config_that_conflicts_with_catalog_binding(self):
        saved = build_action_manifest("Example-Nes-v0", {"actions": self.rows})
        saved_path = write_action_manifest(saved, self.lineage)
        checkpoint = self._checkpoint(saved_path, saved["sha256"])
        trainer = self._trainer(saved["sha256"])

        with self.assertRaisesRegex(ValueError, "write-once binding"):
            trainer._prepare_action_manifest(
                self.game_dir,
                self.lineage,
                str(checkpoint),
                expected_catalog_hash="b" * 64,
                require_catalog_hash=True,
            )


if __name__ == "__main__":
    unittest.main()
