from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from runtime_settings import SETTING_SPECS, RuntimeSettings


class ControlPanel(QMainWindow):
    settings_changed = Signal(dict)
    reset_requested = Signal()

    def __init__(self, settings: RuntimeSettings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.widgets: dict[str, QWidget] = {}
        self.setWindowTitle("CARLA Runtime Control Panel")
        self.resize(470, 760)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)
        self.status = QLabel("Live settings apply immediately.")
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)

        groups = {
            "Driving": {"JUNCTION_CONTROL_MODE", "AUTO_MODE_DEFAULT", "FIXED_THROTTLE", "KP", "KI", "KD", "STEER_LIMIT", "MAX_STEER_STEP", "TARGET_SPEED_KMH"},
            "Trajectory": {"TRAJECTORY_INFERENCE_INTERVAL_SECONDS", "TRAJECTORY_STEER_GAIN", "TRAJECTORY_MAX_STEER"},
            "Vision / debug": {"LANE_PROB_THRESHOLD", "LANE_CENTER_SMOOTH_ALPHA", "VISION_DEBUG", "DEBUG_SHOW_ROIS", "DEBUG_SHOW_LANE_MASK", "DEBUG_SHOW_TRAJECTORY", "DEBUG_SHOW_FPS", "DEBUG_SHOW_GPS", "SHOW_OPENCV_WINDOW"},
            "Hardware (reset)": {"CAMERA_IMAGE_WIDTH", "CAMERA_IMAGE_HEIGHT", "MODEL_PATH"},
        }
        for title, keys in groups.items():
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for spec in SETTING_SPECS:
                if spec[0] in keys:
                    self._add_setting(form, spec)
            self.tabs.addTab(self._scroll(page), title)

        self._build_profiles(root_layout)
        self._load_values(self.settings.snapshot())

    def _scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _add_setting(self, form: QFormLayout, spec) -> None:
        key, label, kind, limits, _realtime = spec
        if kind == "bool":
            widget = QCheckBox()
            widget.stateChanged.connect(lambda _state, k=key: self._emit(k))
        elif kind == "choice":
            widget = QComboBox()
            widget.addItems(list(limits))
            widget.currentTextChanged.connect(lambda _text, k=key: self._emit(k))
        elif kind == "int":
            widget = QSpinBox()
            widget.setRange(int(limits[0]), int(limits[1]))
            widget.setSingleStep(int(limits[2]))
            widget.valueChanged.connect(lambda _value, k=key: self._emit(k))
        elif kind == "float":
            widget = QDoubleSpinBox()
            widget.setRange(float(limits[0]), float(limits[1]))
            widget.setSingleStep(float(limits[2]))
            widget.setDecimals(4)
            widget.valueChanged.connect(lambda _value, k=key: self._emit(k))
        else:
            widget = QLineEdit()
            widget.editingFinished.connect(lambda k=key: self._emit(k))
        self.widgets[key] = widget
        form.addRow(label, widget)

    def _build_profiles(self, root_layout: QVBoxLayout) -> None:
        box = QGroupBox("Profiles (persist between sessions)")
        layout = QHBoxLayout(box)
        self.profile_combo = QComboBox()
        names = sorted(name for name in self.settings.profiles if not name.startswith("_"))
        self.profile_combo.addItems(names)
        active = self.settings.profiles.get("_active_profile")
        if isinstance(active, str) and active in names:
            self.profile_combo.setCurrentText(active)
        self.profile_name = QLineEdit("default")
        load = QPushButton("Load")
        save = QPushButton("Save")
        load.clicked.connect(self._load_profile)
        save.clicked.connect(self._save_profile)
        layout.addWidget(self.profile_combo, 1)
        layout.addWidget(self.profile_name, 1)
        layout.addWidget(load)
        layout.addWidget(save)
        root_layout.addWidget(box)

    def _value(self, key: str) -> Any:
        widget = self.widgets[key]
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        return widget.text()

    def _load_values(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            widget = self.widgets.get(key)
            if widget is None:
                continue
            widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(float(value))
            else:
                widget.setText(str(value))
            widget.blockSignals(False)

    def _emit(self, key: str) -> None:
        value = self._value(key)
        self.settings.apply({key: value})
        spec = next(s for s in SETTING_SPECS if s[0] == key)
        if not spec[4]:
            self.status.setText(f"{spec[1]} changed — reset the car to apply it.")
            self.reset_requested.emit()
        else:
            self.status.setText(f"{spec[1]} updated live.")
        self.settings_changed.emit({key: value})

    def _load_profile(self) -> None:
        values = self.settings.profiles.get(self.profile_combo.currentText())
        if values:
            self.settings.apply(values)
            self._load_values(values)
            self.settings_changed.emit(values)
            self.status.setText(f"Loaded profile: {self.profile_combo.currentText()}")

    def _save_profile(self) -> None:
        name = self.profile_name.text().strip() or "default"
        values = {key: self._value(key) for key in self.widgets}
        self.settings.save_profile(name, values)
        self.settings.profiles["_active_profile"] = name
        self.settings.save()
        if self.profile_combo.findText(name) < 0:
            self.profile_combo.addItem(name)
        self.profile_combo.setCurrentText(name)
        self.status.setText(f"Saved profile: {name}")
