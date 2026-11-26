from __future__ import annotations

import json

from avis_dashboard import AvisDashboardState, create_avis_dashboard_app


def _state(tmp_path):
    return AvisDashboardState(
        settings_path=tmp_path / "avis_settings.json",
        profile_path=tmp_path / "avis_profiles.json",
        capture_directory=tmp_path / "AVIS_RoadForge_Captures",
    )


def test_capture_endpoint_saves_latest_camera_and_manifest(tmp_path):
    state = _state(tmp_path)
    state.publish_images({"camera": b"\xff\xd8fake-jpeg\xff\xd9"})
    state.publish_telemetry({"speed_kmh": 12.5, "gps": [1.0, 2.0, 0.0]})
    state.set_mode("manual")
    client = create_avis_dashboard_app(state).test_client()

    response = client.post("/api/roadforge/capture")
    assert response.status_code == 200
    result = response.get_json()
    assert result["captured"] is True
    image_path = tmp_path / "AVIS_RoadForge_Captures" / result["image"]
    assert image_path.read_bytes() == b"\xff\xd8fake-jpeg\xff\xd9"

    manifest = tmp_path / "AVIS_RoadForge_Captures" / "manifest.jsonl"
    row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert row["session_id"] == result["session_id"]
    assert row["mode"] == "manual"
    assert row["telemetry"]["speed_kmh"] == 12.5
    assert row["camera_view"] == "front_rgb"


def test_capture_endpoint_reports_when_no_camera_frame_exists(tmp_path):
    client = create_avis_dashboard_app(_state(tmp_path)).test_client()
    response = client.post("/api/roadforge/capture")
    assert response.status_code == 409
    assert "No front-camera frame" in response.get_json()["error"]


def test_dashboard_exposes_roadforge_capture_control(tmp_path):
    html = create_avis_dashboard_app(_state(tmp_path)).test_client().get("/").get_data(as_text=True)
    assert "Capture frame for RoadForge" in html
    assert "/api/roadforge/capture" in html
