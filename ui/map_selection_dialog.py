from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)


class MapSelectionDialog(QDialog):
    """Choose an installed CARLA map before selecting spawn and goal points."""

    def __init__(self, maps: list[str], current_map: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Select CARLA Map")
        self.setMinimumSize(430, 170)
        self.setStyleSheet("""
            QDialog {
                background: #0d1929;
                color: #e7eef8;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 10pt;
            }
            QLabel { color: #dce8f5; }
            QComboBox {
                min-height: 32px;
                padding: 5px 9px;
                color: #e7eef8;
                background: #091524;
                border: 1px solid #30465f;
                border-radius: 7px;
            }
            QPushButton {
                min-width: 86px;
                min-height: 30px;
                padding: 5px 11px;
                color: #e7eef8;
                background: #172c43;
                border: 1px solid #35516d;
                border-radius: 7px;
            }
            QPushButton:hover { background: #21415d; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)
        layout.addWidget(QLabel("Choose the map to load:"))

        self.map_combo = QComboBox()
        for map_name in maps:
            display_name = map_name.rstrip("/").split("/")[-1]
            self.map_combo.addItem(display_name, map_name)
        layout.addWidget(self.map_combo)

        if current_map:
            current_short_name = current_map.rstrip("/").split("/")[-1]
            for index in range(self.map_combo.count()):
                if self.map_combo.itemText(index) == current_short_name:
                    self.map_combo.setCurrentIndex(index)
                    break

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_map(self) -> str | None:
        return self.map_combo.currentData()
