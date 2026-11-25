from __future__ import annotations

from unittest.mock import patch

import config_city as conf
import stream


def test_flask_dashboard_renders_grouped_settings_and_wheel_guard():
    response = stream.app.test_client().get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "CARLA Runtime Studio" in page
    assert "Live monitor" in page
    assert "Runtime configuration" in page
    assert "setting-group" in page
    assert "addEventListener('wheel'" in page
    assert "event.preventDefault()" in page
    assert "confirm_advanced" not in page


def test_all_runtime_specs_are_available_to_the_web_settings_renderer():
    response = stream.app.test_client().get("/")
    page = response.get_data(as_text=True)
    for key, *_ in stream.SETTING_SPECS:
        assert key in page
    assert "JUNCTION_MOVEMENT_SEQUENCES" in page
    assert "DEBUG_SHOW_VEHICLES" in page
    assert "CROSSWALK_THRESH_SPEND" in page


def test_invalid_setting_payload_is_rejected_without_partial_updates():
    previous_kp = conf.KP
    with (
        patch.object(stream, "_local_request_only", return_value=True),
        patch.object(stream, "_notify_settings") as notify,
    ):
        response = stream.app.test_client().post(
            "/set_settings",
            json={"KP": previous_kp + 0.01, "UNKNOWN_SETTING": 1},
        )

    assert response.status_code == 400
    assert conf.KP == previous_kp
    notify.assert_not_called()


def test_live_runtime_setting_is_applied_and_reported():
    previous_kp = conf.KP
    updated_kp = min(2.0, previous_kp + 0.01)
    try:
        with (
            patch.object(stream, "_local_request_only", return_value=True),
            patch.object(stream, "_notify_settings") as notify,
        ):
            response = stream.app.test_client().post(
                "/set_settings",
                json={"KP": updated_kp},
            )
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["success"] is True
        assert payload["reset_required"] is False
        assert payload["changed"] == ["KP"]
        assert conf.KP == updated_kp
        notify.assert_called_once_with({"KP": updated_kp})
    finally:
        conf.KP = previous_kp


def test_desktop_control_panel_is_not_part_of_main_runtime_entrypoint():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "ui.control_panel" not in source
    assert "ControlPanel" not in source
    assert "QTimer" not in source

def test_main_keeps_qt_alive_and_opens_the_flask_dashboard():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "main.py"
    ).read_text(encoding="utf-8")

    # Closing the map/spawn dialogs must not quit the process now that Flask
    # is the persistent runtime UI.
    assert source.count("setQuitOnLastWindowClosed(False)") >= 2
    assert "webbrowser.open(local_url, new=2)" in source

def test_runtime_settings_button_has_one_save_handler_and_clear_json_errors():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "stream.py").read_text(
        encoding="utf-8"
    )
    assert source.count("document.getElementById('apply_runtime').addEventListener") == 1
    assert "Invalid JSON sequence." not in source
    assert "Invalid JSON in '+key" in source


def test_junction_sequences_can_be_updated_from_the_web_dashboard():
    original = conf.JUNCTION_MOVEMENT_SEQUENCES
    updated = {
        "LEFT": [{"distance_m": 12.5, "steering_value": -0.35}],
        "RIGHT": [{"distance_m": 10.0, "steering_value": 0.4}],
        "STRAIGHT": [{"distance_m": 8.0, "steering_value": 0.0}],
    }
    try:
        with (
            patch.object(stream, "_local_request_only", return_value=True),
            patch.object(stream, "_notify_settings") as notify,
        ):
            response = stream.app.test_client().post(
                "/set_settings",
                json={"JUNCTION_MOVEMENT_SEQUENCES": updated},
            )
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["success"] is True
        assert conf.JUNCTION_MOVEMENT_SEQUENCES == updated
        notify.assert_called_once_with(
            {"JUNCTION_MOVEMENT_SEQUENCES": updated}
        )
    finally:
        conf.JUNCTION_MOVEMENT_SEQUENCES = original

def test_web_settings_apply_only_parses_sequence_when_it_was_edited():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "stream.py").read_text(
        encoding="utf-8"
    )
    assert "input.dataset.originalValue" in source
    assert "if(currentValue===input.dataset.originalValue)return;" in source

