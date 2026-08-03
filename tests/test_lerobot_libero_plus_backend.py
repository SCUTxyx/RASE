import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from rase.backends.libero_plus_paths import (
    build_libero_plus_path_dict,
    ensure_libero_plus_paths,
    resolve_libero_plus_root,
)
from rase.backends.lerobot_libero_plus import (
    _resolve_local_vlm_path,
    catalog_task_to_suite_index,
)
from rase.eval.collapse import CollapseError, require_lerobot_backend


class LiberoPlusPathsTest(unittest.TestCase):
    def test_build_paths_from_fixture_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "libero" / "libero"
            for name in ("bddl_files", "init_files", "assets"):
                (package / name).mkdir(parents=True)
            paths = build_libero_plus_path_dict(root)
            self.assertEqual(paths["bddl_files"], str((package / "bddl_files").resolve()))
            self.assertEqual(paths["init_states"], str((package / "init_files").resolve()))
            self.assertEqual(paths["assets"], str((package / "assets").resolve()))

    def test_ensure_writes_dedicated_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "plus"
            package = root / "libero" / "libero"
            for name in ("bddl_files", "init_files", "assets"):
                (package / name).mkdir(parents=True)
            config_dir = Path(temporary) / "cfg"
            with mock.patch.dict(
                "os.environ",
                {"LIBERO_PLUS_ROOT": str(root), "LIBERO_CONFIG_PATH": str(config_dir)},
                clear=False,
            ):
                paths = ensure_libero_plus_paths()
            config_file = config_dir / "config.yaml"
            self.assertTrue(config_file.is_file())
            loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
            self.assertEqual(loaded["bddl_files"], paths["bddl_files"])

    def test_resolve_missing_root_raises(self):
        with self.assertRaises(CollapseError):
            resolve_libero_plus_root("/no/such/path/for/rase/test")


class BackendHelpersTest(unittest.TestCase):
    def test_catalog_task_index_is_zero_based(self):
        self.assertEqual(catalog_task_to_suite_index(1), 0)
        self.assertEqual(catalog_task_to_suite_index(683), 682)
        with self.assertRaises(CollapseError):
            catalog_task_to_suite_index(0)

    def test_require_lerobot_backend_returns_evaluate(self):
        try:
            import lerobot  # noqa: F401
        except ImportError:
            self.skipTest("lerobot not installed")
        backend = require_lerobot_backend()
        from rase.backends.lerobot_libero_plus import evaluate

        self.assertIs(backend, evaluate)

    def test_resolve_local_vlm_path_requires_config(self):
        self.assertIsNone(_resolve_local_vlm_path(None))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(CollapseError):
                _resolve_local_vlm_path(root)
            (root / "config.json").write_text("{}", encoding="utf-8")
            resolved = _resolve_local_vlm_path(root)
            self.assertEqual(resolved, str(root.resolve()))


if __name__ == "__main__":
    unittest.main()
