from __future__ import annotations

import hashlib
import importlib.metadata
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


def _install_wheel(wheel: Path, app_data_directory: Path) -> Path:
    """Install the CARLA wheel into an application-local cache, not global Python."""
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()[:20]
    destination = app_data_directory / "carla_python_api" / digest
    marker = destination / ".carla-wheel-installed"

    if marker.is_file():
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

    if not (destination / "carla").is_dir():
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(
            "The selected CARLA wheel was installed, but its 'carla' package "
            "could not be found in the installation directory."
        )

    marker.write_text(wheel.name, encoding="utf-8")
    return destination


def _installed_carla_version() -> Optional[str]:
    try:
        return importlib.metadata.version("carla")
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _prepare_selected_root(root: Path, app_data_directory: Path, parent) -> tuple[bool, str]:
    from PySide6.QtWidgets import QMessageBox

    dist_directory = root / "PythonAPI" / "carla" / "dist"
    wheel = find_compatible_wheel(dist_directory)
    client_directory = None
    client_description = ""

    if wheel is not None:
        client_directory = _install_wheel(wheel, app_data_directory)
        client_description = wheel.name
    else:
        installed_version = _installed_carla_version()
        if not installed_version:
            python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            platform_tag = _current_platform_tag()
            raise RuntimeError(
                "No CARLA wheel matching this Python interpreter was found.\n\n"
                f"Expected a wheel for {python_tag} / {platform_tag} under:\n"
                f"{dist_directory}\n\n"
                "Install the Python version supported by this CARLA package, or "
                "select a package containing the matching wheel."
            )

        answer = QMessageBox.warning(
            parent,
            "CARLA Python API wheel not found",
            "The selected CARLA folder does not contain a wheel compatible with "
            "this Python interpreter.\n\n"
            f"The currently installed Python client is carla=={installed_version}.\n"
            "The client library must match the CARLA server version exactly. "
            "Continue using the installed client only if you have verified that "
            "it matches this CARLA installation.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False, ""

        client_description = f"installed carla=={installed_version} (user-verified)"

    prepend_carla_paths(root, client_directory)
    return True, client_description


def configure_carla_api(parent=None) -> bool:
    """
    First-run CARLA folder picker. The chosen root is remembered in QSettings.

    The package's matching wheel is installed into an application-local cache so
    switching between CARLA versions does not overwrite the user's global Python
    environment.
    """
    from PySide6.QtCore import QSettings, QStandardPaths
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
    saved_value = str(settings.value(CARLA_ROOT_SETTING, "") or "").strip()
    saved_root = normalize_carla_root(saved_value) if saved_value else None

    if saved_root is not None and is_carla_root(saved_root):
        try:
            app_data_text = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            app_data = (
                Path(app_data_text)
                if app_data_text
                else Path.home() / ".carla_autonomous_driving"
            )
            ready, description = _prepare_selected_root(saved_root, app_data, parent)
            if ready:
                return True
        except Exception as exc:
            QMessageBox.warning(
                parent,
                "Saved CARLA installation could not be loaded",
                f"The saved CARLA folder could not be prepared:\n{saved_root}\n\n"
                f"{exc}\n\nPlease select the CARLA installation again.",
            )
    elif saved_value:
        QMessageBox.warning(
            parent,
            "Saved CARLA folder was not found",
            f"The previously selected CARLA folder is no longer valid:\n{saved_value}\n\n"
            "Please select the CARLA installation again.",
        )
    else:
        QMessageBox.information(
            parent,
            "Locate CARLA installation",
            "Select the root folder of your CARLA installation.\n\n"
            "Choose the folder that contains the PythonAPI directory "
            "(and normally CarlaUnreal.exe or CarlaUE4.exe). "
            "The selected location will be remembered for future launches.",
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
                "Select the CARLA package root, not its WindowsNoEditor, Binaries, "
                "or dist subfolder.",
            )
            initial_directory = selected
            continue

        try:
            app_data_text = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            app_data = Path(app_data_text) if app_data_text else (
                Path.home() / ".carla_autonomous_driving"
            )
            ready, description = _prepare_selected_root(root, app_data, parent)
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
                f"Could not use the selected CARLA installation:\n{root}\n\n{exc}",
            )
            initial_directory = str(root)
