import json
import tempfile
import unittest
from pathlib import Path

from backend.action_manifest import (
    ACTION_MANIFEST_FORMAT,
    action_manifest_hash,
    build_action_manifest,
    load_action_manifest,
    validate_resume_action_binding,
    write_action_manifest,
)


class ActionManifestTest(unittest.TestCase):
    def setUp(self):
        self.actions = {
            "actions": [
                {"name": "NoOp", "buttons": []},
                {"name": "Forward", "buttons": ["B"]},
                {"name": "Forward+Left", "buttons": ["B", "LEFT"]},
            ]
        }

    def test_build_is_canonical_and_detached_from_source(self):
        source_a = json.loads(json.dumps(self.actions, indent=4))
        source_b = json.loads(
            '{"actions":[{"buttons":[],"name":"NoOp"},'
            '{"buttons":["B"],"name":"Forward"},'
            '{"buttons":["B","LEFT"],"name":"Forward+Left"}]}'
        )

        manifest_a = build_action_manifest("Example-Nes-v0", source_a)
        manifest_b = build_action_manifest("Example-Nes-v0", source_b)

        self.assertEqual(ACTION_MANIFEST_FORMAT, manifest_a["format"])
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(64, len(action_manifest_hash(manifest_a)))

        source_a["actions"][0]["name"] = "mutated after launch"
        self.assertEqual("NoOp", manifest_a["actions"][0]["name"])

    def test_same_count_reorder_and_remap_change_hash(self):
        original = build_action_manifest("Example-Nes-v0", self.actions)
        reordered = build_action_manifest(
            "Example-Nes-v0",
            {"actions": list(reversed(self.actions["actions"]))},
        )
        remapped_rows = json.loads(json.dumps(self.actions["actions"]))
        remapped_rows[1]["buttons"] = ["A"]
        remapped = build_action_manifest("Example-Nes-v0", remapped_rows)

        self.assertNotEqual(original["sha256"], reordered["sha256"])
        self.assertNotEqual(original["sha256"], remapped["sha256"])

    def test_write_is_content_addressed_idempotent_and_load_is_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_action_manifest("Example-Nes-v0", self.actions)
            path = write_action_manifest(manifest, Path(temp) / "lineages" / "main")
            second = write_action_manifest(manifest, Path(temp) / "lineages" / "main")

            self.assertEqual(path, second)
            self.assertEqual(
                Path(temp)
                / "lineages"
                / "main"
                / "action-manifests"
                / f"{manifest['sha256']}.json",
                path,
            )
            loaded = load_action_manifest(
                path,
                expected_game_id="Example-Nes-v0",
                expected_hash=manifest["sha256"],
            )
            self.assertEqual(manifest, loaded)
            self.assertFalse(path.stat().st_mode & 0o222)

    def test_workspace_rewrite_does_not_change_reconstructed_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "actions.json"
            workspace.write_text(json.dumps(self.actions))
            manifest = build_action_manifest(
                "Example-Nes-v0", json.loads(workspace.read_text())
            )
            manifest_path = write_action_manifest(manifest, Path(temp) / "lineage")

            changed = json.loads(workspace.read_text())
            changed["actions"] = list(reversed(changed["actions"]))
            workspace.write_text(json.dumps(changed))

            first_wrapper_load = load_action_manifest(manifest_path)["actions"]
            second_wrapper_load = load_action_manifest(manifest_path)["actions"]
            self.assertEqual(self.actions["actions"], first_wrapper_load)
            self.assertEqual(first_wrapper_load, second_wrapper_load)
            self.assertNotEqual(
                json.loads(workspace.read_text())["actions"], second_wrapper_load
            )

    def test_load_rejects_tamper_and_wrong_launch_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_action_manifest("Example-Nes-v0", self.actions)
            path = write_action_manifest(manifest, temp)

            with self.assertRaisesRegex(ValueError, "game mismatch"):
                load_action_manifest(path, expected_game_id="Other-Nes-v0")
            with self.assertRaisesRegex(ValueError, "launch hash mismatch"):
                load_action_manifest(path, expected_hash="0" * 64)

            path.chmod(0o644)
            tampered = json.loads(path.read_text())
            tampered["actions"][0]["name"] = "Changed"
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_action_manifest(path)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                write_action_manifest(manifest, temp)

    def test_write_rejects_symlink_at_content_address(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_action_manifest("Example-Nes-v0", self.actions)
            lineage = Path(temp) / "lineage"
            store = lineage / "action-manifests"
            store.mkdir(parents=True)
            elsewhere = Path(temp) / "elsewhere.json"
            elsewhere.write_text(json.dumps(manifest))
            target = store / f"{manifest['sha256']}.json"
            target.symlink_to(elsewhere)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                write_action_manifest(manifest, lineage)

    def test_resume_binding_is_retro_only_and_hash_locked(self):
        # Generic SheepRL environments often have wrappers too; they must not
        # inherit Retro Dreamer's manifest contract.
        validate_resume_action_binding("dmc", {}, {})
        digest = "a" * 64
        saved = {"action_manifest": "/manifest.json", "action_manifest_hash": digest}
        validate_resume_action_binding(
            "retro-dreamer", saved, {"action_manifest_hash": digest}
        )
        with self.assertRaisesRegex(ValueError, "Legacy checkpoint"):
            validate_resume_action_binding("retro-dreamer", {}, {})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_resume_action_binding(
                "retro-dreamer", saved, {"action_manifest_hash": "b" * 64}
            )

    def test_rejects_invalid_sources_and_envelopes(self):
        with self.assertRaisesRegex(ValueError, "game_id"):
            build_action_manifest("../escape", self.actions)
        with self.assertRaisesRegex(ValueError, "non-empty list"):
            build_action_manifest("Example-Nes-v0", {"actions": []})
        with self.assertRaisesRegex(ValueError, "JSON object"):
            build_action_manifest(
                "Example-Nes-v0", [{"name": "ok", "buttons": []}, "bad"]
            )
        with self.assertRaisesRegex(ValueError, "buttons must be a list"):
            build_action_manifest(
                "Example-Nes-v0", [{"name": "bad", "buttons": "B"}]
            )
        with self.assertRaisesRegex(ValueError, "button names or 0/1"):
            build_action_manifest(
                "Example-Nes-v0", [{"name": "bad", "buttons": [2]}]
            )
        with self.assertRaisesRegex(ValueError, "exactly"):
            build_action_manifest(
                "Example-Nes-v0",
                [{"name": "bad", "buttons": [], "button": "A"}],
            )

        manifest = build_action_manifest("Example-Nes-v0", self.actions)
        manifest["extra"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            action_manifest_hash(manifest)


if __name__ == "__main__":
    unittest.main()
