import tempfile
import unittest
from pathlib import Path

from rase.envs.task_catalog import LiberoPlusTask
from rase.eval.collapse import CollapseError, ResultManifest, run_tasks


def _tasks():
    return [
        LiberoPlusTask(
            suite="libero_spatial",
            task_id=1,
            name="camera",
            dimension="camera",
            difficulty=1,
            category="Camera Viewpoints",
        ),
        LiberoPlusTask(
            suite="libero_spatial",
            task_id=2,
            name="robot",
            dimension="robot",
            difficulty=5,
            category="Robot Initial States",
        ),
    ]


class CollapseManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_persists_each_task_and_resumes(self):
        tasks = _tasks()
        path = self.root / "manifest.json"
        provenance = {
            "git_sha": "abc",
            "env_lock_sha256": "def",
            "resolved_config": {},
        }
        manifest = ResultManifest.open_or_create(
            path,
            tasks,
            provenance,
        )
        calls = []

        def backend(task, output_dir, config):
            calls.append(task.key)
            return {"successes": 1, "episodes": config["episodes"]}

        run_tasks(manifest, tasks, backend, self.root, {"episodes": 1})
        self.assertEqual(manifest.pending(), ())
        self.assertTrue(
            all(
                result["status"] == "completed"
                for result in manifest.data["results"].values()
            )
        )

        resumed = ResultManifest.open_or_create(path, tasks, provenance)
        run_tasks(resumed, tasks, backend, self.root, {"episodes": 1})
        self.assertEqual(calls, [task.key for task in tasks])

        with self.assertRaisesRegex(CollapseError, "provenance/config differs"):
            ResultManifest.open_or_create(path, tasks, {"changed": True})

    def test_failed_task_is_retryable(self):
        tasks = _tasks()[:1]
        manifest = ResultManifest.open_or_create(
            self.root / "manifest.json", tasks, {}
        )

        def broken_backend(task, output_dir, config):
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            run_tasks(manifest, tasks, broken_backend, self.root, {})
        record = manifest.data["results"][tasks[0].key]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["attempts"], 1)
        self.assertEqual(manifest.pending(), (tasks[0].key,))

    def test_max_attempts_skips_stuck_running_task(self):
        tasks = _tasks()[:1]
        key = tasks[0].key
        manifest = ResultManifest.open_or_create(
            self.root / "manifest.json", tasks, {}
        )
        # Simulate a native crash: attempts already counted, status left running.
        manifest.data["results"][key].update(status="running", attempts=2)
        manifest.save()

        def backend(task, output_dir, config):
            raise AssertionError("should not run after max attempts")

        run_tasks(manifest, tasks, backend, self.root, {}, max_attempts=2)
        record = manifest.data["results"][key]
        self.assertEqual(record["status"], "skipped")
        self.assertEqual(manifest.pending(), ())


if __name__ == "__main__":
    unittest.main()
