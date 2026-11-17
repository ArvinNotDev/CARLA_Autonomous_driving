from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Optional


ORGANIZATION_NAME = "ArvinNotDev"
APPLICATION_NAME = "CARLA_Autonomous_Driving"
CARLA_ROOT_SETTING = "carla/root"
_DLL_DIRECTORY_HANDLES = []


def normalize_carla_root(path: str | Path) -> Path:
    """Return the CARLA package root, accepting the root or its PythonAPI folder."""
    root = Path(path).expanduser()
    if root.name.lower() == "carla" and root.parent.name.lower() == "pythonapi":
        root = root.parent.parent
    elif root.name.lower() == "pythonapi":
        root = root.parent
    return root.resolve(strict=False)


def is_carla_root(path: str | Path) -> bool:
    """Check for the PythonAPI agent sources required by this application."""
    root = normalize_carla_root(path)
    agent_planner = (
        root / "PythonAPI" / "carla" / "agents" / "navigation"
        / "global_route_planner.py"
    )
    return (
        root.is_dir()
        and (root / "PythonAPI" / "carla").is_dir()
        and agent_planner.is_file()
    )


def _current_platform_tag() -> str:
    tag = sysconfig.get_platform().lower().replace("-", "_").replace(".", "_")
    if sys.platform == "win32":
        machine = platform.machine().lower()
        return "win_arm64" if machine in {"arm64", "aarch64"} else "win_amd64"
    return tag


def find_compatible_wheel(
    dist_directory: str | Path,
    python_tag: Optional[str] = None,
    platform_tag: Optional[str] = None,
) -> Optional[Path]:
    """Find a CARLA wheel matching this interpreter's CPython ABI and platform."""
    dist = Path(dist_directory)
    if python_tag is None:
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if platform_tag is None:
        platform_tag = _current_platform_tag()

    candidates = []
    if not dist.is_dir():
        return None
    for wheel in dist.glob("carla-*.whl"):
        parts = wheel.name[:-4].split("-")
        if len(parts) < 5:
            continue
        wheel_python, wheel_abi, wheel_platform = parts[-3:]
        if (
            wheel_python == python_tag
            and wheel_abi in {python_tag, "abi3", "none"}
            and wheel_platform == platform_tag
        ):
            candidates.append(wheel)

    return sorted(candidates, key=lambda item: item.name)[0] if candidates else None


def has_importable_carla_module(directory: str | Path) -> bool:
    """
    Check the wheel's actual import layout.

    Do not assume the wheel must install a 'carla' directory: Python extension
    modules can also be installed as a top-level carla*.pyd file.
    """
    root = Path(directory).resolve(strict=False)
    try:
        return importlib.machinery.PathFinder.find_spec("carla", [str(root)]) is not None
    except (ImportError, OSError, ValueError):
        return False


def _register_windows_dll_directory(path: Path) -> None:
    """Keep Windows DLL search handles alive for the native CARLA extension."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    if path.is_dir():
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


def _install_wheel(wheel: Path, app_data_directory: Path) -> Path:
    """Install the CARLA wheel into an application-local cache, not global Python."""
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()[:20]
    destination = app_data_directory / "carla_python_api" / digest
    marker = destination / ".carla-wheel-installed"

    if marker.is_file() and has_importable_carla_module(destination):
        return destination

    shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--no-warn-script-location",
        "--target",
        str(destination),
        str(wheel),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        shutil.rmtree(destination, ignore_errors=True)
        details = (result.stderr or result.stdout or "pip returned an error").strip()
        raise RuntimeError(
            "Could not install the CARLA Python wheel into the application cache.\n"
            + details[-2500:]
        )

    if not has_importable_carla_module(destination):
        installed_entries = sorted(
            item.name for item in destination.iterdir()
        ) if destination.is_dir() else []
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(
            "Pip reported that the CARLA wheel was installed, but Python cannot "
            "find an importable 'carla' module in the installation.\n\n"
            f"Wheel: {wheel.name}\n"
            f"Install target: {destination}\n"
            f"Top-level installed entries: {installed_entries or '(none)'}\n\n"
            "The loader now accepts both a 'carla' package directory and a "
            "top-level carla*.pyd extension module. Check that the selected wheel "
            "is the Windows wheel for this Python version."
        )

    marker.write_text(wheel.name, encoding="utf-8")
    return destination


def prepend_carla_paths(root: str | Path, client_directory: Optional[str | Path] = None) -> None:
    """Expose the selected wheel install and the bundled PythonAPI agent sources."""
    carla_root = normalize_carla_root(root)
    python_api = carla_root / "PythonAPI"
    carla_sources = python_api / "carla"

    paths = []
    if client_directory is not None:
        paths.append(Path(client_directory))
    paths.extend((carla_sources, python_api))

    for path in reversed(paths):
        resolved = str(path.resolve(strict=False))
        while resolved in sys.path:
            sys.path.remove(resolved)
        sys.path.insert(0, resolved)

    importlib.invalidate_caches()


def _prepare_selected_root(
    root: Path,
    app_data_directory: Path,
) -> tuple[bool, str]:
    dist_directory = root / "PythonAPI" / "carla" / "dist"
    wheel = find_compatible_wheel(dist_directory)
    if wheel is None:
        expected_python = f"cp{sys.version_info.major}{sys.version_info.minor}"
        expected_platform = _current_platform_tag()
        available = sorted(item.name for item in dist_directory.glob("*.whl")) if dist_directory.is_dir() else []
        raise RuntimeError(
            "No CARLA wheel matching this Python interpreter was found.\n\n"
            f"Expected: {expected_python} / {expected_platform}\n"
            f"Folder: {dist_directory}\n"
            f"Available wheels: {available or '(no .whl files found)'}\n\n"
            "The app will not fall back to a globally installed CARLA client, "
            "because that could silently load CARLA 0.9.16 against a 1.0 server."
        )

    client_directory = _install_wheel(wheel, app_data_directory)

    # CARLA's Windows wheel contains a native extension. Register both the
    # target root and the package directory (when present) for dependent DLLs.
    _register_windows_dll_directory(client_directory)
    _register_windows_dll_directory(client_directory / "carla")

    prepend_carla_paths(root, client_directory)
    module_spec = importlib.machinery.PathFinder.find_spec(
        "carla",
        [str(client_directory.resolve(strict=False))],
    )
    if module_spec is None:
        raise RuntimeError(
            "The selected CARLA wheel was installed, but its module is not "
            "importable from the application cache."
        )

    return True, wheel.name


def configure_carla_api(parent=None) -> bool:
    """
    Ask on every launch whether to reuse the saved CARLA installation or choose
    another one. The selected root is remembered in QSettings.

    The matching wheel is installed into an application-local cache so
    switching between CARLA versions does not overwrite the global Python
    environment.
    """
    from PySide6.QtCore import QSettings, QStandardPaths
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
    saved_value = str(settings.value(CARLA_ROOT_SETTING, "") or "").strip()
    saved_root = normalize_carla_root(saved_value) if saved_value else None

    app_data_text = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppLocalDataLocation
    )
    app_data = (
        Path(app_data_text)
        if app_data_text
        else Path.home() / ".carla_autonomous_driving"
    )

    use_saved = False
    if saved_root is not None and is_carla_root(saved_root):
        prompt = QMessageBox(parent)
        prompt.setIcon(QMessageBox.Icon.Question)
        prompt.setWindowTitle("CARLA installation")
        prompt.setText("Use the previously selected CARLA installation?")
        prompt.setInformativeText(
            f"Saved location:\n{saved_root}\n\n"
            "You can reuse this installation or browse for a different CARLA folder."
        )
        use_saved_button = prompt.addButton(
            "Use saved installation",
            QMessageBox.ButtonRole.AcceptRole,
        )
        choose_new_button = prompt.addButton(
            "Choose another folder",
            QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = prompt.addButton(QMessageBox.StandardButton.Cancel)
        prompt.setDefaultButton(use_saved_button)
        prompt.exec()
        clicked = prompt.clickedButton()

        if clicked == cancel_button or clicked is None:
            return False
        use_saved = clicked == use_saved_button

        if use_saved:
            try:
                ready, _ = _prepare_selected_root(saved_root, app_data)
                if ready:
                    return True
            except Exception as exc:
                QMessageBox.warning(
                    parent,
                    "Saved CARLA installation could not be loaded",
                    f"The saved CARLA folder could not be prepared:\n"
                    f"{saved_root}\n\n{exc}\n\n"
                    "Choose another folder to locate the correct installation.",
                )
            # Continue to the folder picker after a failed saved installation.
        elif clicked != choose_new_button:
            return False
    elif saved_value:
        QMessageBox.warning(
            parent,
            "Saved CARLA folder was not found",
            f"The previously selected CARLA folder is no longer valid:\n"
            f"{saved_value}\n\nPlease select the CARLA installation again.",
        )
    else:
        QMessageBox.information(
            parent,
            "Locate CARLA installation",
            "Select the root folder of your CARLA installation.\n\n"
            "Choose the folder that contains the PythonAPI directory "
            "(and normally CarlaUnreal.exe). The selected location will be "
            "remembered for future launches.",
        )

    initial_directory = (
        str(saved_root)
        if saved_root is not None and saved_root.exists()
        else str(Path.home())
    )

    while True:
        selected = QFileDialog.getExistingDirectory(
            parent,
            "Locate CARLA installation",
            initial_directory,
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return False

        root = normalize_carla_root(selected)
        if not is_carla_root(root):
            QMessageBox.warning(
                parent,
                "Invalid CARLA folder",
                "The selected folder does not contain the expected CARLA PythonAPI "
                "agent files:\n\n"
                "PythonAPI\\carla\\agents\\navigation\\global_route_planner.py\n\n"
                "Select the CARLA package root, not its CarlaUnreal, Engine, "
                "Binaries, or dist subfolder.",
            )
            initial_directory = selected
            continue

        try:
            ready, description = _prepare_selected_root(root, app_data)
            if not ready:
                initial_directory = str(root)
                continue

            settings.setValue(CARLA_ROOT_SETTING, str(root))
            settings.setValue("carla/client_distribution", description)
            settings.sync()
            return True
        except Exception as exc:
            QMessageBox.critical(
                parent,
                "Could not configure CARLA",
                f"Could not use the selected CARLA installation:\n"
                f"{root}\n\n{exc}",
            )
            initial_directory = str(root)
