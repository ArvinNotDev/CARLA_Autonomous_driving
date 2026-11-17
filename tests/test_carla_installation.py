from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from utils.carla_installation import (
    find_compatible_wheel,
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

    def test_prepends_agent_and_pythonapi_paths_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._make_root(Path(temporary))
            old_path = list(sys.path)
            try:
                prepend_carla_paths(root, Path(temporary) / "client")
                self.assertEqual(sys.path[0], str((Path(temporary) / "client").resolve()))
                self.assertEqual(
                    sys.path[1],
                    str((root / "PythonAPI" / "carla").resolve()),
                )
                self.assertEqual(sys.path[2], str((root / "PythonAPI").resolve()))
                prepend_carla_paths(root, Path(temporary) / "client")
                self.assertEqual(sys.path.count(str((root / "PythonAPI").resolve())), 1)
            finally:
                sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
