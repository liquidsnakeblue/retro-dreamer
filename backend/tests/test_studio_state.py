import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend import catalog
from backend.studio_state import StudioStateBuilder
from backend.training.config import TrainingConfig


class FakeGameManager:
    def __init__(self):
        self.training = {
            "reward": {"variables": {"score": {"reward": 1}}},
            "done": {"variables": {"lives": {"op": "equal", "reference": 0}}},
        }

    def list_games(self):
        return [
            {
                "game_id": "Focus-Game",
                "display_name": "Focus Game",
                "system": "Nes",
                "source": "custom",
                "has_custom_config": True,
                "rom_ready": True,
            },
            {
                "game_id": "Builtin-Game-v0",
                "display_name": "Builtin Game",
                "system": "Nes",
                "source": "builtin",
                "has_custom_config": False,
                "rom_ready": True,
            },
        ]

    def get_game(self, game_id):
        if game_id != "Focus-Game":
            raise FileNotFoundError(game_id)
        return {
            "game_id": game_id,
            "display_name": "Focus Game",
            "source": "custom",
            "default_state": "start",
            "states": ["start"],
            "annotated_states": [
                {"file": "start", "label": "Start", "group": "training"}
            ],
            "config_files": ["data.json", "actions.json", "training.json", "metadata.json"],
        }

    def read_config(self, game_id, filename):
        configs = {
            "data.json": {"info": {"score": {"address": 1}}},
            "actions.json": {"actions": [{"name": "NoOp", "buttons": []}]},
            "training.json": self.training,
            "metadata.json": {"game_id": game_id, "default_state": "start"},
        }
        return copy.deepcopy(configs[filename])


class StudioStateBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        checkpoint = root / "head.ckpt"
        checkpoint.write_text("fixture")
        replay = root / "replay"
        replay.mkdir()
        resolved = root / "config.yaml"
        resolved.write_text("algo:\n  recurrent_state_size: 512\n")
        db_path = root / "catalog.sqlite"
        con = catalog.connect(db_path)
        con.execute("INSERT INTO games (id,display_name) VALUES (?,?)", ("Focus-Game", "Focus Game"))
        lineage_id = con.execute(
            "INSERT INTO lineages (game_id,name,status,created_at) VALUES (?,?,?,?)",
            ("Focus-Game", "main", "active", 1.0),
        ).lastrowid
        con.execute("UPDATE games SET active_lineage_id=? WHERE id=?", (lineage_id, "Focus-Game"))
        session_id = con.execute(
            "INSERT INTO sessions (lineage_id,run_dir,status,resolved_config) VALUES (?,?,?,?)",
            (lineage_id, str(root / "run"), "ended", str(resolved)),
        ).lastrowid
        con.execute(
            "INSERT INTO snapshots (session_id,step,checkpoint_path,replay_path,created_at) "
            "VALUES (?,?,?,?,?)",
            (session_id, 123, str(checkpoint), str(replay), 2.0),
        )
        con.commit()
        con.close()
        self.manager = FakeGameManager()
        self.status = SimpleNamespace(
            state="idle", game_id="Focus-Game", current_step=123,
            current_episode=4, elapsed_time=10.0, steps_per_second=2.0,
            avg_return=3.0, avg_length=20.0, max_return=5.0,
            error_message="",
        )
        self.trainer = SimpleNamespace(
            status=self.status,
            config=TrainingConfig.from_preset("small"),
        )
        self.builder = StudioStateBuilder(
            self.manager,
            self.trainer,
            catalog_connect=lambda: catalog.connect(db_path),
            jobs_provider=lambda: [{"id": "job-1", "tool": "probe", "status": "running"}],
            advisor_provider=lambda: {
                "gpu": "fixture", "vram_gb": 32, "recommended": "xl", "fits": ["xl"]
            },
            inventory_ttl=3600,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_compact_and_full_share_stable_material_revision(self):
        compact = self.builder.build("Focus-Game", active_tab="copilot")
        full = self.builder.build("Focus-Game", projection="full")
        self.assertEqual(compact["revision"], full["revision"])
        self.assertNotIn("builtins", compact["inventory"])
        self.assertIn("builtins", full["inventory"])
        self.assertLessEqual(len(json.dumps(compact, separators=(",", ":"))), 4096)
        for section in ("focus", "training", "inventory", "focused_game",
                        "advisor", "tools", "capabilities"):
            self.assertIn("observed_at", compact[section])

        self.status.current_step += 100
        self.status.elapsed_time += 30
        self.status.avg_return = 99
        self.assertEqual(compact["revision"], self.builder.build("Focus-Game")["revision"])

        self.manager.training["reward"]["variables"]["bonus"] = {"reward": 2}
        changed = self.builder.build("Focus-Game")
        self.assertNotEqual(compact["revision"], changed["revision"])

    def test_unknown_focus_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.builder.build("missing")


if __name__ == "__main__":
    unittest.main()
