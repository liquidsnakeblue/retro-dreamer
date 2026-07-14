import io
import json
import unittest

from backend import copilot


class FakeBuilder:
    def build(self, focus_game_id, *, active_tab=None, projection="compact"):
        return {
            "revision": "revision-fixture",
            "generated_at": "2026-07-14T05:00:00Z",
            "focus": {"game_id": focus_game_id, "active_tab": active_tab},
            "sentinel": "ambient-context",
        }


class FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()

    def poll(self):
        return None


class CopilotEnvelopeTest(unittest.TestCase):
    def setUp(self):
        self.old_proc = copilot._proc
        self.old_builder = copilot._studio_state_builder
        self.old_events = copilot._events
        self.old_seq = copilot._seq
        copilot._proc = FakeProcess()
        copilot._studio_state_builder = FakeBuilder()
        copilot._events = []
        copilot._seq = 0

    def tearDown(self):
        copilot._proc = self.old_proc
        copilot._studio_state_builder = self.old_builder
        copilot._events = self.old_events
        copilot._seq = self.old_seq

    def test_send_injects_one_user_envelope_and_exposes_receipt(self):
        primer_before = copilot.PRIMER_PATH.read_bytes()
        receipt = copilot.send(copilot.SendReq(
            text="continue training this",
            focus_game_id="Focus-Game",
            active_tab="copilot",
        ))
        wire = json.loads(copilot._proc.stdin.getvalue())
        blocks = wire["message"]["content"]
        self.assertEqual(wire["message"]["role"], "user")
        self.assertEqual(len(blocks), 1)
        text = blocks[0]["text"]
        self.assertTrue(text.startswith(copilot.STUDIO_STATE_START + "\n"))
        self.assertTrue(text.endswith("\ncontinue training this"))
        self.assertIn('"game_id":"Focus-Game"', text)
        self.assertIn('"sentinel":"ambient-context"', text)
        event = copilot.events(0)["events"][0]
        self.assertEqual(event["kind"], "user")
        self.assertEqual(event["text"], text)
        self.assertEqual(receipt["studio_revision"], "revision-fixture")
        self.assertEqual(receipt["observed_at"], "2026-07-14T05:00:00Z")
        self.assertEqual(primer_before, copilot.PRIMER_PATH.read_bytes())


if __name__ == "__main__":
    unittest.main()
