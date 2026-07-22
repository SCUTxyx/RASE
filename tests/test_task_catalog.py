import json
import tempfile
import unittest
from pathlib import Path

from rase.envs.task_catalog import (
    LiberoPlusTaskCatalog,
    TaskCatalogError,
    parse_levels,
)


def _catalog(tmp_path: Path):
    records = {
        "libero_spatial": [
            {
                "id": 4,
                "name": "camera_four",
                "category": "Camera Viewpoints",
                "difficulty_level": 4,
            },
            {
                "id": 1,
                "name": "camera_one_a",
                "category": "Camera Viewpoints",
                "difficulty_level": 1,
            },
            {
                "id": 2,
                "name": "camera_one_b",
                "category": "Camera Viewpoints",
                "difficulty_level": 1,
            },
            {
                "id": 3,
                "name": "robot_one",
                "category": "Robot Initial States",
                "difficulty_level": 1,
            },
            {
                "id": 99,
                "name": "ignored",
                "category": "Language Instructions",
                "difficulty_level": None,
            },
        ],
        "libero_goal": [
            {
                "id": 7,
                "name": "goal_camera_one",
                "category": "Camera Viewpoints",
                "difficulty_level": 1,
            }
        ],
    }
    path = tmp_path / "task_classification.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return LiberoPlusTaskCatalog.load(path)


class TaskCatalogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_load_and_filter_camera_levels(self):
        selected = _catalog(self.root).select(
            dimensions=["camera"],
            levels=[1],
            suites=["libero_spatial"],
            profile="full",
        )
        self.assertEqual([task.task_id for task in selected], [1, 2])
        self.assertTrue(all(task.dimension == "camera" for task in selected))

    def test_smoke_is_one_deterministic_task_per_cell(self):
        selected = _catalog(self.root).select(levels=[1], profile="smoke")
        self.assertEqual(
            [task.key for task in selected],
            [
                "libero_goal:7:camera:L1",
                "libero_spatial:1:camera:L1",
                "libero_spatial:3:robot:L1",
            ],
        )

    def test_level_parser_and_invalid_filter(self):
        self.assertEqual(parse_levels("L1,L3-L5"), (1, 3, 4, 5))
        with self.assertRaises(TaskCatalogError):
            _catalog(self.root).select(dimensions=["lighting"])
        with self.assertRaises(TaskCatalogError):
            parse_levels("L0-L2")


if __name__ == "__main__":
    unittest.main()
