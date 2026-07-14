import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import catalog


class ResumableHeadTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.con = catalog.connect(self.root / "catalog.sqlite")
        self.con.execute(
            "INSERT INTO games (id, display_name) VALUES (?, ?)",
            ("game", "Game"),
        )
        lineage_id = self.con.execute(
            """INSERT INTO lineages (game_id, name, status, created_at)
               VALUES (?, ?, ?, ?)""",
            ("game", "main", "active", 1.0),
        ).lastrowid
        self.con.execute(
            "UPDATE games SET active_lineage_id=? WHERE id=?",
            (lineage_id, "game"),
        )
        self.lineage_id = lineage_id
        self.con.commit()

    def tearDown(self):
        self.con.close()
        self.temp_dir.cleanup()

    def _session(self, name: str, started_at: float) -> int:
        return self.con.execute(
            """INSERT INTO sessions (lineage_id, run_dir, started_at, status)
               VALUES (?, ?, ?, ?)""",
            (self.lineage_id, str(self.root / name), started_at, "ended"),
        ).lastrowid

    def _snapshot(self, session_id: int, step: int, *, exists: bool = True) -> Path:
        checkpoint = self.root / f"session-{session_id}-step-{step}.ckpt"
        if exists:
            checkpoint.write_bytes(b"checkpoint")
        self.con.execute(
            """INSERT INTO snapshots
               (session_id, step, checkpoint_path, kind, created_at)
               VALUES (?, ?, ?, 'resume', ?)""",
            (session_id, step, str(checkpoint), float(step)),
        )
        self.con.commit()
        return checkpoint

    def test_newer_fresh_session_beats_older_higher_step(self):
        old_session = self._session("old-run", 100.0)
        old_checkpoint = self._snapshot(old_session, 756_696)
        fresh_session = self._session("fresh-run", 200.0)
        fresh_checkpoint = self._snapshot(fresh_session, 52_000)

        head = catalog.get_resumable_head(self.con, "game")

        self.assertEqual(Path(head["checkpoint_path"]), fresh_checkpoint)
        self.assertNotEqual(Path(head["checkpoint_path"]), old_checkpoint)

    def test_missing_newest_file_falls_back_in_session_then_lineage(self):
        old_session = self._session("old-run", 100.0)
        old_checkpoint = self._snapshot(old_session, 756_696)
        fresh_session = self._session("fresh-run", 200.0)
        fresh_checkpoint = self._snapshot(fresh_session, 51_000)
        self._snapshot(fresh_session, 52_000, exists=False)

        head = catalog.get_resumable_head(self.con, "game", "main")
        self.assertEqual(Path(head["checkpoint_path"]), fresh_checkpoint)

        fresh_checkpoint.unlink()
        head = catalog.get_resumable_head(self.con, "game", "main")
        self.assertEqual(Path(head["checkpoint_path"]), old_checkpoint)

    def test_snapshot_action_manifest_hash_is_write_once(self):
        session = self._session("run", 100.0)
        checkpoint = self.root / "bound.ckpt"
        checkpoint.write_bytes(b"checkpoint")
        digest = "a" * 64

        snapshot_id = catalog.register_snapshot(
            self.con,
            session,
            10,
            str(checkpoint),
            config_hash=digest,
        )
        row = self.con.execute(
            "SELECT config_hash FROM snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
        self.assertEqual(digest, row["config_hash"])

        with self.assertRaisesRegex(ValueError, "already bound"):
            catalog.register_snapshot(
                self.con,
                session,
                10,
                str(checkpoint),
                config_hash="b" * 64,
            )
        row = self.con.execute(
            "SELECT config_hash FROM snapshots WHERE id=?", (snapshot_id,)
        ).fetchone()
        self.assertEqual(digest, row["config_hash"])

    def test_catalog_recrawl_backfills_null_action_manifest_hash(self):
        run_dir = (
            self.root
            / "sheeprl/logs/runs/dreamer_v3/game"
            / "2026-07-14_01-02-03_dreamer_v3_retro-dreamer_42"
            / "version_0"
        )
        checkpoint = run_dir / "checkpoint" / "ckpt_10_0.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        config = run_dir / "config.yaml"
        config.write_text("env:\n  id: retro-dreamer\n")

        with patch.object(catalog, "PROJECT_ROOT", self.root):
            catalog.register_existing_runs(
                self.con, game_filter="game", active_run_dir=str(run_dir)
            )
            row = self.con.execute(
                "SELECT config_hash FROM snapshots WHERE checkpoint_path=?",
                (str(checkpoint),),
            ).fetchone()
            self.assertIsNone(row["config_hash"])

            digest = "c" * 64
            config.write_text(
                "env:\n  id: retro-dreamer\n  wrapper:\n"
                f"    action_manifest_hash: {digest}\n"
            )
            catalog.register_existing_runs(
                self.con, game_filter="game", active_run_dir=str(run_dir)
            )

        row = self.con.execute(
            "SELECT config_hash FROM snapshots WHERE checkpoint_path=?",
            (str(checkpoint),),
        ).fetchone()
        self.assertEqual(digest, row["config_hash"])


if __name__ == "__main__":
    unittest.main()
