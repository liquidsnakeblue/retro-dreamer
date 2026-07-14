import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from backend.api import routes
from backend.storage import StorageUsageSampler


class FakeClock:
    def __init__(self, value=1000.0):
        self.value = value

    def __call__(self):
        return self.value


class StorageUsageSamplerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock = FakeClock()
        self.disk_usage = Mock(return_value=SimpleNamespace(total=1000, free=125))
        self.sampler = StorageUsageSampler(
            self.root,
            run_size_ttl=30,
            clock=self.clock,
            monotonic=self.clock,
            disk_usage=self.disk_usage,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_filesystem_percent_and_recursive_run_size(self):
        run = self.root / "run"
        nested = run / "checkpoint"
        nested.mkdir(parents=True)
        (run / "config.yaml").write_bytes(b"1234")
        (nested / "brain.ckpt").write_bytes(b"123456")

        outside = self.root / "outside"
        outside.write_bytes(b"not counted")
        (run / "outside-link").symlink_to(outside)

        sample = self.sampler.sample(run)
        self.assertEqual(sample["filesystem"], {
            "total_bytes": 1000,
            "free_bytes": 125,
            "free_percent": 12.5,
        })
        self.assertEqual(sample["active_run_bytes"], 10)
        self.assertEqual(sample["active_run_sampled_at"], 1000.0)

    def test_run_size_is_cached_then_refreshed(self):
        run = self.root / "run"
        run.mkdir()
        (run / "one").write_bytes(b"1")
        self.assertEqual(self.sampler.sample(run)["active_run_bytes"], 1)
        (run / "two").write_bytes(b"22")
        self.assertEqual(self.sampler.sample(run)["active_run_bytes"], 1)
        self.clock.value += 31
        self.assertEqual(self.sampler.sample(run)["active_run_bytes"], 3)

    def test_run_path_change_bypasses_cache(self):
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        (first / "data").write_bytes(b"1")
        (second / "data").write_bytes(b"22")
        self.assertEqual(self.sampler.sample(first)["active_run_bytes"], 1)
        self.assertEqual(self.sampler.sample(second)["active_run_bytes"], 2)

    def test_inaccessible_child_returns_unknown_not_partial_total(self):
        run = self.root / "run"
        blocked = run / "blocked"
        blocked.mkdir(parents=True)
        (run / "visible").write_bytes(b"visible")
        real_scandir = os.scandir

        def selective_scandir(path):
            if Path(path) == blocked:
                raise PermissionError("blocked")
            return real_scandir(path)

        with patch("backend.storage.os.scandir", side_effect=selective_scandir):
            self.assertIsNone(self.sampler.sample(run)["active_run_bytes"])

    def test_inaccessible_file_metadata_returns_unknown(self):
        run = self.root / "run"
        run.mkdir()

        entry = Mock()
        entry.is_symlink.return_value = False
        entry.is_dir.return_value = False
        entry.is_file.return_value = True
        entry.stat.side_effect = PermissionError("blocked")
        with patch("backend.storage.os.scandir", return_value=[entry]):
            self.assertIsNone(self.sampler.sample(run)["active_run_bytes"])

    def test_missing_active_run_and_disk_error_fail_soft(self):
        sampler = StorageUsageSampler(
            self.root / "missing",
            clock=self.clock,
            disk_usage=Mock(side_effect=OSError("gone")),
        )
        sample = sampler.sample(self.root / "also-missing")
        self.assertEqual(sample["active_run_bytes"], None)
        self.assertEqual(sample["filesystem"]["free_percent"], None)


class StorageUsageRouteTest(unittest.TestCase):
    def setUp(self):
        self.old_trainer = routes._trainer
        self.old_sampler = routes._storage_sampler

    def tearDown(self):
        routes._trainer = self.old_trainer
        routes._storage_sampler = self.old_sampler

    def test_route_uses_only_registered_active_run(self):
        active = Path("/active/version_0")
        routes._trainer = SimpleNamespace(active_run_dir=active)
        routes._storage_sampler = Mock(
            sample=Mock(return_value={"active_run_bytes": 42})
        )
        self.assertEqual(routes.storage_usage(), {"active_run_bytes": 42})
        routes._storage_sampler.sample.assert_called_once_with(active)

    def test_route_requires_initialized_trainer(self):
        routes._trainer = None
        with self.assertRaises(HTTPException) as caught:
            routes.storage_usage()
        self.assertEqual(caught.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
