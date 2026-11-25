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
    assert "self._input_timer.timeout.connect(self._poll_input_events)" in source

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

def test_speed_settings_reject_minimum_equal_to_or_above_maximum():
    previous_min = conf.MIN_SPEED_KMH
    previous_max = conf.MAX_SPEED_KMH
    try:
        with (
            patch.object(stream, "_local_request_only", return_value=True),
            patch.object(stream, "_notify_settings") as notify,
        ):
            response = stream.app.test_client().post(
                "/set_settings",
                json={"MIN_SPEED_KMH": previous_max},
            )
        assert response.status_code == 400
        assert response.get_json()["success"] is False
        assert conf.MIN_SPEED_KMH == previous_min
        assert conf.MAX_SPEED_KMH == previous_max
        notify.assert_not_called()
    finally:
        conf.MIN_SPEED_KMH = previous_min
        conf.MAX_SPEED_KMH = previous_max


def test_speed_controller_settings_are_live_runtime_settings():
    previous = (
        conf.ACCELERATION_MPS2,
        conf.MIN_SPEED_KMH,
        conf.MAX_SPEED_KMH,
    )
    updated = {
        "ACCELERATION_MPS2": min(8.0, previous[0] + 0.1),
        "MIN_SPEED_KMH": previous[1] + 0.5,
        "MAX_SPEED_KMH": max(previous[2], previous[1] + 1.0),
    }
    try:
        with (
            patch.object(stream, "_local_request_only", return_value=True),
            patch.object(stream, "_notify_settings") as notify,
        ):
            response = stream.app.test_client().post(
                "/set_settings",
                json=updated,
            )
        assert response.status_code == 200
        assert response.get_json()["success"] is True
        assert conf.ACCELERATION_MPS2 == updated["ACCELERATION_MPS2"]
        assert conf.MIN_SPEED_KMH == updated["MIN_SPEED_KMH"]
        assert conf.MAX_SPEED_KMH == updated["MAX_SPEED_KMH"]
        notify.assert_called_once_with(updated)
    finally:
        (
            conf.ACCELERATION_MPS2,
            conf.MIN_SPEED_KMH,
            conf.MAX_SPEED_KMH,
        ) = previous

def test_models_finish_initializing_before_dashboard_browser_startup():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(
        encoding="utf-8"
    )
    dashboard_start = source.index(
        "started = start_stream_background(conf.STREAM_HOST, conf.STREAM_PORT)"
    )
    model_load = source.index(
        "self.vision_processor = VisionProcessor("
    )
    worker_start = source.index("self.inference_worker.start()")
    assert model_load < worker_start < dashboard_start
    assert "QTimer.singleShot(0, open_dashboard)" in source
    assert "_start_vision_model_load" not in source
    assert 'self._vision_model_status = "warming_up"' in source


def test_qt_runtime_loop_reenters_when_qt_quits_unexpectedly():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(
        encoding="utf-8"
    )
    run_method = source.split("    def run(self) -> None:", 1)[1]
    assert "qt_app.setQuitOnLastWindowClosed(False)" in run_method
    assert "while self.running and not self._stop_event.is_set():" in run_method
    assert "restarting the event loop." in run_method


def test_pygame_input_is_polled_on_the_main_qt_thread():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(
        encoding="utf-8"
    )
    control_iteration = source.split("def _control_iteration", 1)[1]
    assert "self.input_manager.poll()" not in control_iteration
    assert "def _poll_input_events" in source
    assert "self._input_timer.timeout.connect(self._poll_input_events)" in source
    assert "restart_requested.emit" in source


def test_native_crash_diagnostics_are_enabled_for_all_threads():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(
        encoding="utf-8"
    )
    assert "faulthandler.enable(file=_CRASH_LOG_HANDLE, all_threads=True)" in source
    assert "carla_runtime_crash.log" in source

