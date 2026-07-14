import re
import subprocess
import sys
import unittest
from pathlib import Path

from backend import episode_report


FIXTURES = Path(__file__).parent / "fixtures"
ENGINE = Path(episode_report.__file__).resolve()


class EpisodeReportRegressionTest(unittest.TestCase):
    def run_fixture(self, name: str) -> str:
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
        declared = int(re.search(r"EVENT STREAM \((\d+) events\)", result.stdout).group(1))
        emitted = len(re.findall(r"^  step\s+\d+", result.stdout, re.MULTILINE))
        self.assertEqual(declared, emitted)
        return result.stdout

    def test_lm_verified_story(self):
        report = self.run_fixture("lm")
        self.assertIn("EVENT STREAM (21 events)", report)
        self.assertIn("OUTCOME: 2 death/fail · 7 damage events", report)
        self.assertIn("death @477: playerPage=6, roomPos=1, scrollY=422", report)
        self.assertIn("death @988: playerPage=6, roomPos=1, scrollY=418", report)
        self.assertIn("playerPage=6 (2/2); roomPos=1 (2/2)", report)
        self.assertIn("steps 0–477 (477)", report)
        self.assertIn("steps 477–988 (511)", report)
        self.assertIn("steps 988–1400 (412)", report)

    def test_mario_verified_story(self):
        report = self.run_fixture("mario")
        self.assertIn("EVENT STREAM (6 events)", report)
        self.assertIn("time                 timer", report)
        self.assertIn("step   436  TERMINAL", report)
        self.assertIn("done: lives equal 0", report)
        self.assertNotRegex(report, r"loss\s+time\b")

    def test_fzero_verified_story(self):
        report = self.run_fixture("fzero")
        self.assertIn("EVENT STREAM (19 events)", report)
        self.assertIn("speed                rewarded", report)
        positions = [
            int(value)
            for value in re.findall(r"loss\s+health.*?@ pos=\+(\d+) rel", report)
        ]
        self.assertEqual(positions, [198, 206, 204, 204, 204, 207, 208, 205, 204])
        self.assertIn("low 468 vs < 100 — not met (margin 368)", report)
        self.assertIn("survived the window (no death/terminal)", report)


if __name__ == "__main__":
    unittest.main()
