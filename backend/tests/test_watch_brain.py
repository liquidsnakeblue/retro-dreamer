import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from backend import tools
from backend import watch_brain_job


class _Connection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class WatchBrainEndpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.games = self.root / "games"
        self.jobs = self.root / "tools"
        self.game = self.games / "Example-Nes-v0"
        (self.game / "states").mkdir(parents=True)
        (self.game / "data.json").write_text("{}")
        (self.game / "training.json").write_text(
            json.dumps({"reward": {"variables": {}}, "done": {"variables": {}}})
        )
        (self.game / "states" / "Level1.state").write_bytes(b"state")
        self.checkpoint = self.root / "checkpoint.ckpt"
        self.checkpoint.write_bytes(b"checkpoint")
        self.dir_patches = [
            patch.object(tools, "GAMES_DIR", self.games),
            patch.object(tools, "JOBS_DIR", self.jobs),
        ]
        for item in self.dir_patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.dir_patches):
            item.stop()
        self.tmp.cleanup()

    def test_latest_head_builds_one_managed_artifact_directory(self):
        con = _Connection()
        submitted = {}

        def fake_submit(tool, cmd, cwd=tools.SHEEPRL_DIR, job_id=None):
            submitted.update(tool=tool, cmd=cmd, cwd=cwd, job_id=job_id)
            return job_id

        with (
            patch("backend.catalog.connect", return_value=con),
            patch(
                "backend.catalog.get_resumable_head",
                return_value={"checkpoint_path": str(self.checkpoint)},
            ) as get_head,
            patch.object(tools, "submit", side_effect=fake_submit),
        ):
            result = tools.watch_brain(
                tools.WatchBrainReq(game_id="Example-Nes-v0", state="Level1")
            )

        self.assertTrue(con.closed)
        get_head.assert_called_once_with(con, "Example-Nes-v0")
        self.assertEqual(result, {"job_id": submitted["job_id"]})
        self.assertEqual(submitted["tool"], "watch_brain")
        self.assertEqual(submitted["cwd"], tools.PROJECT_ROOT)
        self.assertTrue(submitted["job_id"].startswith("watch_brain-"))

        cmd = submitted["cmd"]
        self.assertEqual(cmd[0], tools.PYTHON)
        self.assertEqual(Path(cmd[1]).name, "watch_brain_job.py")
        self.assertEqual(cmd[2], str(self.checkpoint))
        self.assertEqual(cmd[3:5], ["Level1", "1400"])
        capture, training, report = map(Path, cmd[5:8])
        expected_dir = self.jobs / submitted["job_id"]
        self.assertEqual(capture, expected_dir / "capture.npz")
        self.assertEqual(report, expected_dir / "report.txt")
        self.assertEqual(training, self.game / "training.json")

    def test_unknown_game_is_404(self):
        with self.assertRaises(HTTPException) as caught:
            tools.watch_brain(
                tools.WatchBrainReq(game_id="Missing-Nes-v0", state="Level1")
            )
        self.assertEqual(caught.exception.status_code, 404)

    def test_missing_training_config_is_rejected_before_job(self):
        (self.game / "training.json").unlink()
        with self.assertRaises(HTTPException) as caught:
            tools.watch_brain(
                tools.WatchBrainReq(game_id="Example-Nes-v0", state="Level1")
            )
        self.assertEqual(caught.exception.status_code, 409)

    def test_unknown_state_is_404(self):
        with self.assertRaises(HTTPException) as caught:
            tools.watch_brain(
                tools.WatchBrainReq(game_id="Example-Nes-v0", state="Nope")
            )
        self.assertEqual(caught.exception.status_code, 404)

    def test_missing_game_scoped_head_is_409(self):
        con = _Connection()
        with (
            patch("backend.catalog.connect", return_value=con),
            patch("backend.catalog.get_resumable_head", return_value=None),
            patch.object(tools, "submit") as submit,
            self.assertRaises(HTTPException) as caught,
        ):
            tools.watch_brain(
                tools.WatchBrainReq(game_id="Example-Nes-v0", state="Level1")
            )
        self.assertEqual(caught.exception.status_code, 409)
        self.assertTrue(con.closed)
        submit.assert_not_called()


class JobManagerCpuPinTest(unittest.TestCase):
    def test_parent_gpu_selection_is_always_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "output.log"
            captured = {}

            class FakeProcess:
                pid = 123
                stdout = iter(())

                def wait(self):
                    return 0

            def fake_popen(*args, **kwargs):
                captured.update(kwargs)
                return FakeProcess()

            job_id = "cpu-test"
            job = {
                "id": job_id,
                "tool": "test",
                "status": "running",
                "cmd": ["true"],
                "log": str(log),
                "workdir": tmp,
                "started_at": 0,
                "result": None,
            }
            with (
                patch.dict(tools._jobs, {job_id: job}, clear=True),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}),
                patch.object(tools.subprocess, "Popen", side_effect=fake_popen),
            ):
                tools._run_job(job_id, ["true"], Path(tmp))

            self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(job["status"], "done")


class WatchBrainRunnerTest(unittest.TestCase):
    def test_capture_then_report_emits_one_final_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "brain.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            training = root / "training.json"
            training.write_text("{}")
            capture = root / "job" / "capture.npz"
            report = root / "job" / "report.txt"
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                if Path(cmd[1]).name == "_retro_ram_capture.py":
                    Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                    Path(cmd[-1]).write_bytes(b"npz")
                    return subprocess.CompletedProcess(
                        cmd, 0, 'saved\nRESULT {"npz": "nested"}\n'
                    )
                return subprocess.CompletedProcess(
                    cmd, 0, "EPISODE REPORT\nPOST-MORTEM: survived\n", ""
                )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(watch_brain_job.subprocess, "run", side_effect=fake_run),
                patch.dict(os.environ, {"RETRO_REPORT_PYTHON": "report-python"}),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = watch_brain_job.main([
                    str(checkpoint), "Level1", "1400", str(capture),
                    str(training), str(report),
                ])

            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(Path(calls[0][0][1]), watch_brain_job.CAPTURE_SCRIPT)
            self.assertEqual(calls[1][0][0], "report-python")
            self.assertEqual(Path(calls[1][0][1]), watch_brain_job.REPORT_ENGINE)

            output_lines = stdout.getvalue().splitlines()
            result_lines = [line for line in output_lines if line.startswith("RESULT ")]
            self.assertEqual(len(result_lines), 1)
            self.assertIn('[capture] RESULT {"npz": "nested"}', output_lines)
            parsed = json.loads(result_lines[0][len("RESULT "):])
            self.assertEqual(parsed, {
                "npz_path": str(capture.resolve()),
                "report_path": str(report.resolve()),
                "report_text": "EPISODE REPORT\nPOST-MORTEM: survived\n",
            })
            self.assertEqual(report.read_text(), parsed["report_text"])

    def test_failed_capture_never_emits_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "brain.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            training = root / "training.json"
            training.write_text("{}")
            stdout = io.StringIO()
            stderr = io.StringIO()
            failed = subprocess.CompletedProcess(
                ["capture"], 3, 'RESULT {"npz": "nested"}\n'
            )
            with (
                patch.object(watch_brain_job.subprocess, "run", return_value=failed),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = watch_brain_job.main([
                    str(checkpoint), "Level1", "1400",
                    str(root / "capture.npz"), str(training),
                    str(root / "report.txt"),
                ])
            self.assertEqual(code, 1)
            self.assertFalse(any(
                line.startswith("RESULT ") for line in stdout.getvalue().splitlines()
            ))
            self.assertIn("RAM capture failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
