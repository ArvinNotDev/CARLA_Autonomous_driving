from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from utils.carla_installation import (
    find_compatible_wheel,
    has_importable_carla_module,
    is_carla_root,
    normalize_carla_root,
    prepend_carla_paths,
)


class TestCarlaInstallationHelpers(unittest.TestCase):
    def _make_root(self, base: Path) -> Path:
        root = base / "Carla-1.0-Win64-Shipping"
        planner = (
            root / "PythonAPI" / "carla" / "agents" / "navigation"
            / "global_route_planner.py"
        )
        planner.parent.mkdir(parents=True)
        planner.write_text("# test fixture", encoding="utf-8")
        return root

    def test_accepts_package_root_and_pythonapi_subfolders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_root(Path(temporary))
            self.assertEqual(normalize_carla_root(root / "PythonAPI"), root.resolve())
            self.assertEqual(
                normalize_carla_root(root / "PythonAPI" / "carla"), root.resolve()
            )
            self.assertTrue(is_carla_root(root))

    def test_finds_only_matching_python_and_platform_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            dist = Path(temporary)
            wrong_python = dist / "carla-1.0-cp311-cp311-win_amd64.whl"
            wrong_platform = dist / "carla-1.0-cp312-cp312-manylinux_2_27_x86_64.whl"
            matching = dist / "carla-1.0-cp312-cp312-win_amd64.whl"
            for wheel in (wrong_python, wrong_platform, matching):
                wheel.touch()

            self.assertEqual(
                find_compatible_wheel(dist, "cp312", "win_amd64"),
                matching,
            )

    def test_accepts_top_level_module_as_well_as_package_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            module_directory = root / "module-install"
            module_directory.mkdir()
            (module_directory / "carla.py").write_text(
                "CLIENT_VERSION = '1.0'\n", encoding="utf-8"
            )
            self.assertTrue(has_importable_carla_module(module_directory))

            package_directory = root / "package-install"
            (package_directory / "carla").mkdir(parents=True)
            (package_directory / "carla" / "__init__.py").write_text(
                "CLIENT_VERSION = '1.0'\n", encoding="utf-8"
            )
            self.assertTrue(has_importable_carla_module(package_directory))

            empty_directory = root / "empty-install"
            empty_directory.mkdir()
            self.assertFalse(has_importable_carla_module(empty_directory))

    def test_prepends_agent_and_pythonapi_paths_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_root(Path(temporary))
            old_path = list(sys.path)
            try:
                client_directory = Path(temporary) / "client"
                prepend_carla_paths(root, client_directory)
                self.assertEqual(sys.path[0], str(client_directory.resolve()))
                self.assertEqual(
                    sys.path[1],
                    str((root / "PythonAPI" / "carla").resolve()),
                )
                self.assertEqual(sys.path[2], str((root / "PythonAPI").resolve()))
                prepend_carla_paths(root, client_directory)
                self.assertEqual(sys.path.count(str((root / "PythonAPI").resolve())), 1)
            finally:
                sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
