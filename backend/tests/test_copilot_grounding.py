import json
import subprocess
import sys
import unittest
from functools import cache
from pathlib import Path

from backend import copilot, episode_report


FIXTURES = Path(__file__).parent / "fixtures"
ENGINE = Path(episode_report.__file__).resolve()
JOB_ID = "watch_brain-deadbeef"
LM_JOB_ID = "watch_brain-cafebabe"
MARIO_JOB_ID = "watch_brain-feedface"


@cache
def fixture_report(name: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(ENGINE),
            str(FIXTURES / f"{name}.npz"),
            str(FIXTURES / f"{name}_training.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def valid_claim_payload() -> tuple[str, dict]:
    claim = "The capture recorded a significant health loss at relative position +198."
    quote = "step    20  loss           health 2048->1887 (-161, significant) @ pos=+198 rel"
    report = fixture_report("fzero")
    if quote not in report:
        raise AssertionError("frozen F-Zero evidence line changed")
    return claim, {
        "job_id": JOB_ID,
        "claims": [{
            "claim": claim,
            "evidence_quote": quote,
            "anchor": {"step": 20, "event": "loss"},
        }],
    }


class GroundingVocabularyTelemetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {
            name: fixture_report(name) for name in ("lm", "mario", "fzero")
        }

    def test_real_gate_e_confabulations_are_flagged(self):
        for text, term in (
            ("The brain could not avoid obstacles.", "avoid obstacles"),
            ("The brain sped into a wall.", "sped into a wall"),
            ("It went lap after lap.", "lap after lap"),
        ):
            with self.subTest(text=text):
                warnings = copilot._check_grounding(text, self.reports["fzero"])
                self.assertEqual(1, len(warnings))
                self.assertIn(term, warnings[0]["detail"])

    def test_legitimate_summaries_are_not_flagged(self):
        self.assertEqual([], copilot._check_grounding(
            "The brain died twice at playerPage 6.", self.reports["lm"]
        ))
        self.assertEqual([], copilot._check_grounding(
            "It stalled near the end.", self.reports["mario"]
        ))

    def test_planner_and_meta_language_have_zero_false_positives(self):
        samples = (
            "The training plan keeps replay_ratio at 0.125 and batch size 16.",
            "The resume strategy will avoid resetting the replay buffer.",
            "The approval decision remains yours.",
            "The training plan may add an obstacle-aware reward later.",
            "The training plan rewards the agent for completed laps.",
            "Add a penalty whenever the agent hits walls.",
            'Avoid saying "lap after lap" without evidence.',
            "A planner can avoid obstacles using a learned world model.",
            "The model should learn to avoid hazards.",
            "We should teach the agent to avoid obstacles.",
            "Train the agent to avoid obstacles.",
            "The agent needs to learn to dodge enemies.",
            "A future curriculum teaches the agent to avoid hazards.",
            "Our goal is to teach the brain to avoid walls.",
            "The report does not say whether the brain hit a wall.",
            "I cannot infer laps or obstacles from this capture.",
            "Do not claim the agent collided with an enemy.",
            "The report contains no evidence of hazards.",
        )
        for report in self.reports.values():
            for sample in samples:
                with self.subTest(sample=sample):
                    self.assertEqual([], copilot._check_grounding(sample, report))

    def test_fixture_accurate_diagnoses_have_no_warnings(self):
        summaries = {
            "lm": "The brain died twice at playerPage 6 and stalled after early progress.",
            "mario": (
                "The brain died twice; lives reached zero at step 436, the terminal "
                "fired, and progress stalled after step 382."
            ),
            "fzero": (
                "The brain survived the 1,400-step window with no terminal; health "
                "bottomed at 468 and recovered to 2018, while pos netted +202."
            ),
        }
        for name, report in self.reports.items():
            with self.subTest(name=name):
                self.assertEqual([], copilot._check_grounding(summaries[name], report))

    def test_decorative_quote_does_not_suppress_vocabulary_telemetry(self):
        report = self.reports["fzero"]
        text = 'The brain sped into a wall: "EVENT STREAM (19 events)".'
        self.assertEqual(1, len(copilot._check_grounding(text, report)))

    def test_disclaimer_cannot_launder_a_later_assertion(self):
        text = "The report does not establish the cause, but the brain sped into a wall."
        warnings = copilot._check_grounding(text, self.reports["fzero"])
        self.assertEqual(1, len(warnings))
        self.assertIn("sped into a wall", warnings[0]["message"])


class StructuredGroundingValidationTest(unittest.TestCase):
    def setUp(self):
        self.old_reports = copilot._served_reports
        self.old_report_meta = copilot._served_report_meta
        self.old_report_seq = copilot._served_report_seq
        copilot._served_reports = {}
        copilot._served_report_meta = {}
        copilot._served_report_seq = 0
        copilot.cache_served_watch_report(JOB_ID, fixture_report("fzero"))
        copilot.cache_served_watch_report(LM_JOB_ID, fixture_report("lm"))
        copilot.cache_served_watch_report(MARIO_JOB_ID, fixture_report("mario"))

    def tearDown(self):
        copilot._served_reports = self.old_reports
        copilot._served_report_meta = self.old_report_meta
        copilot._served_report_seq = self.old_report_seq

    def test_valid_verbatim_quote_and_structured_anchor_pass(self):
        claim, payload = valid_claim_payload()
        self.assertEqual(
            [],
            copilot._validate_grounding_claims(payload, claim, {JOB_ID}),
        )

    def test_structured_fixture_replay_covers_all_three_report_vocabularies(self):
        cases = (
            (
                LM_JOB_ID,
                "The report records a level reset at step 482.",
                "step   482  reset/loop     playerPage 6->0 (looped back)",
                {"step": 482, "event": "reset/loop"},
            ),
            (
                MARIO_JOB_ID,
                "The report records an objective increase at step 35.",
                "step    35  objective+     score +20 (=20)",
                {"step": 35, "event": "objective+"},
            ),
            (
                JOB_ID,
                *valid_claim_payload(),
            ),
        )
        for case in cases:
            job_id = case[0]
            if job_id == JOB_ID:
                claim, payload = case[1], case[2]
            else:
                claim, quote, anchor = case[1:]
                payload = {
                    "job_id": job_id,
                    "claims": [{
                        "claim": claim,
                        "evidence_quote": quote,
                        "anchor": anchor,
                    }],
                }
            with self.subTest(job_id=job_id):
                self.assertEqual(
                    [],
                    copilot._validate_grounding_claims(payload, claim, {job_id}),
                )

    def test_decorative_quote_cannot_borrow_a_different_anchor(self):
        claim, payload = valid_claim_payload()
        payload["claims"][0]["anchor"] = {"step": 350, "event": "loss"}
        warnings = copilot._validate_grounding_claims(payload, claim, {JOB_ID})
        self.assertTrue(any("does not belong" in item["message"] for item in warnings))

    def test_fabricated_quote_and_missing_anchor_are_reported(self):
        claim, payload = valid_claim_payload()
        payload["claims"][0]["evidence_quote"] = "the brain hit an invisible wall"
        payload["claims"][0]["anchor"] = {"step": 9999, "event": "collision"}
        warnings = copilot._validate_grounding_claims(payload, claim, {JOB_ID})
        messages = "\n".join(item["message"] for item in warnings)
        self.assertIn("is not verbatim", messages)
        self.assertIn("does not exist", messages)

    def test_claim_must_appear_in_diagnosis_and_job_must_match_turn(self):
        claim, payload = valid_claim_payload()
        warnings = copilot._validate_grounding_claims(payload, "Different prose.", {JOB_ID})
        self.assertTrue(any("not an exact diagnosis sentence" in item["message"] for item in warnings))
        warnings = copilot._validate_grounding_claims(payload, claim, set())
        self.assertTrue(any("outside this diagnosis turn" in item["message"] for item in warnings))

    def test_claim_cannot_be_a_substring_of_a_longer_diagnosis_sentence(self):
        claim, payload = valid_claim_payload()
        payload["claims"][0]["claim"] = "health loss"
        warnings = copilot._validate_grounding_claims(payload, claim, {JOB_ID})
        self.assertTrue(any("not an exact diagnosis sentence" in item["message"] for item in warnings))

    def test_claims_tail_is_parsed_and_removed_from_visible_text(self):
        claim, payload = valid_claim_payload()
        raw = (
            claim + "\n" + copilot.GROUNDING_CLAIMS_START + "\n"
            + json.dumps(payload) + "\n" + copilot.GROUNDING_CLAIMS_END
        )
        visible, parsed, error = copilot._split_grounding_claims(raw)
        self.assertEqual(claim, visible)
        self.assertEqual(payload, parsed)
        self.assertIsNone(error)

    def test_invalid_or_duplicate_machine_tails_stay_out_of_visible_chat(self):
        claim, payload = valid_claim_payload()
        block = (
            copilot.GROUNDING_CLAIMS_START + json.dumps(payload)
            + copilot.GROUNDING_CLAIMS_END
        )
        visible, parsed, error = copilot._split_grounding_claims(
            claim + "\n" + block + "\n" + block
        )
        self.assertEqual(claim, visible)
        self.assertIsNone(parsed)
        self.assertIn("single final block", error)
        self.assertNotIn(copilot.GROUNDING_CLAIMS_START, visible)

    def test_diagnosis_turn_requires_a_claims_block(self):
        claim, _ = valid_claim_payload()
        warnings = copilot._turn_grounding_warnings([claim], {JOB_ID}, [], [])
        self.assertTrue(any("omitted structured claims" in item["message"] for item in warnings))

    def test_exactly_one_final_claims_block_is_required(self):
        claim, payload = valid_claim_payload()
        warnings = copilot._turn_grounding_warnings(
            [claim, "Analysis complete."],
            {JOB_ID},
            [payload, payload],
            [],
            [0, 1],
        )
        self.assertTrue(any("expected exactly one" in item["message"] for item in warnings))
        warnings = copilot._turn_grounding_warnings(
            [claim, "Analysis complete."], {JOB_ID}, [payload], [], [0]
        )
        self.assertTrue(any("not the final" in item["message"] for item in warnings))
        warnings = copilot._turn_grounding_warnings(
            [claim], {JOB_ID}, [payload], [], [0], last_assistant_position=1
        )
        self.assertTrue(any("not the final" in item["message"] for item in warnings))

    def test_secondary_telemetry_scans_all_diagnosis_blocks(self):
        payload = {"job_id": JOB_ID, "claims": []}
        warnings = copilot._turn_grounding_warnings(
            ["The brain sped into a wall.", "Analysis complete."],
            {JOB_ID},
            [payload],
            [],
            [1],
        )
        self.assertTrue(any("vocabulary telemetry" in item["message"].lower() for item in warnings))

    def test_real_but_irrelevant_anchor_does_not_silence_secondary_telemetry(self):
        _, payload = valid_claim_payload()
        claim = "The brain sped into a wall."
        payload["claims"][0]["claim"] = claim
        warnings = copilot._turn_grounding_warnings(
            [claim], {JOB_ID}, [payload], [], [0]
        )
        self.assertTrue(any("vocabulary telemetry" in item["message"].lower() for item in warnings))

    def test_non_diagnosis_turn_is_out_of_scope(self):
        text = "The training plan may add an obstacle-aware reward later."
        self.assertEqual([], copilot._turn_grounding_warnings([text], set(), [], []))


class CopilotGroundingReaderTest(unittest.TestCase):
    def setUp(self):
        self.old_events = copilot._events
        self.old_seq = copilot._seq
        self.old_reports = copilot._served_reports
        self.old_report_meta = copilot._served_report_meta
        self.old_report_seq = copilot._served_report_seq
        copilot._events = []
        copilot._seq = 0
        copilot._served_reports = {}
        copilot._served_report_meta = {}
        copilot._served_report_seq = 0

    def tearDown(self):
        copilot._events = self.old_events
        copilot._seq = self.old_seq
        copilot._served_reports = self.old_reports
        copilot._served_report_meta = self.old_report_meta
        copilot._served_report_seq = self.old_report_seq

    def _run_reader(
        self,
        assistant_text: str,
        *,
        include_job=True,
        second_turn=None,
        serve_report=True,
        failed_result=False,
        literal_job_in_command=True,
    ):
        content = []
        report = fixture_report("fzero")
        if include_job:
            content.append({
                "type": "tool_use",
                "id": "bash-1",
                "name": "Bash",
                "input": {"command": (
                    f"curl /api/tools/jobs/{JOB_ID}"
                    if literal_job_in_command
                    else "curl /api/tools/jobs/$JOB_ID"
                )},
            })
        events = [(False, {"type": "assistant", "message": {"content": content}})]
        if include_job:
            result_content = json.dumps({
                "id": JOB_ID,
                "status": "done",
                "result": {"report_text": report},
            })
            events.append((serve_report, {
                "type": "user",
                "message": {"content": [{
                    "type": "tool_result",
                    "tool_use_id": "bash-1",
                    "content": result_content,
                    "is_error": failed_result,
                }]},
            }))
        events.extend((False, event) for event in [
            {"type": "assistant", "message": {"content": [{
                "type": "text", "text": assistant_text,
            }]}},
            {"type": "result", "num_turns": 3, "duration_ms": 1200},
        ])
        if second_turn is not None:
            events.extend((False, event) for event in [
                {"type": "assistant", "message": {"content": [{
                    "type": "text", "text": second_turn,
                }]}},
                {"type": "result", "num_turns": 1, "duration_ms": 100},
            ])

        def lines():
            for cache_before, event in events:
                if cache_before:
                    copilot.cache_served_watch_report(JOB_ID, report)
                yield json.dumps(event) + "\n"

        proc = type("FakeProc", (), {"stdout": lines()})()
        copilot._reader(proc)

    def test_reader_validates_job_keyed_claims_and_hides_machine_tail(self):
        claim, payload = valid_claim_payload()
        raw = (
            claim + "\n" + copilot.GROUNDING_CLAIMS_START + "\n"
            + json.dumps(payload) + "\n" + copilot.GROUNDING_CLAIMS_END
        )
        self._run_reader(raw, second_turn="The brain sped into a wall.")
        warnings = [e for e in copilot._events if e["kind"] == "grounding-warning"]
        self.assertEqual([], warnings)
        assistant = [e for e in copilot._events if e["kind"] == "assistant"][0]
        self.assertEqual(claim, assistant["text"])
        self.assertIn(copilot.GROUNDING_CLAIMS_START, assistant["detail"])

    def test_reader_emits_nonblocking_telemetry_for_missing_claims(self):
        claim, _ = valid_claim_payload()
        self._run_reader(claim)
        warnings = [e for e in copilot._events if e["kind"] == "grounding-warning"]
        self.assertTrue(any("omitted structured claims" in e["text"] for e in warnings))
        self.assertTrue(any(e["kind"] == "assistant" for e in copilot._events))

    def test_reader_does_not_check_planner_text_without_a_report_job(self):
        self._run_reader(
            "The training plan may add an obstacle-aware reward later.",
            include_job=False,
        )
        self.assertNotIn(
            "grounding-warning", [event["kind"] for event in copilot._events]
        )

    def test_reader_requires_a_successful_same_turn_report_serve(self):
        copilot.cache_served_watch_report(JOB_ID, fixture_report("fzero"))
        self._run_reader("The brain sped into a wall.", serve_report=False)
        self.assertNotIn(
            "grounding-warning", [event["kind"] for event in copilot._events]
        )
        copilot._events.clear()
        self._run_reader("The brain sped into a wall.", failed_result=True)
        self.assertNotIn(
            "grounding-warning", [event["kind"] for event in copilot._events]
        )

    def test_reader_can_associate_job_id_from_successful_result(self):
        claim, payload = valid_claim_payload()
        raw = (
            claim + "\n" + copilot.GROUNDING_CLAIMS_START + "\n"
            + json.dumps(payload) + "\n" + copilot.GROUNDING_CLAIMS_END
        )
        self._run_reader(raw, literal_job_in_command=False)
        self.assertNotIn(
            "grounding-warning", [event["kind"] for event in copilot._events]
        )


if __name__ == "__main__":
    unittest.main()
