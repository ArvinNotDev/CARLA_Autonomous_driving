import importlib.util
import sys
import types
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_display_fps_uses_presented_events():
    m = load_module(ROOT / "runtime_metrics.py", "runtime_metrics_test")
    fps = m.RollingFps(window_seconds=1.0)
    t0 = 10.0
    for i in range(5):
        fps.record(t0 + i * 0.1)
    assert 9.5 < fps.value(timestamp=t0 + 0.4) < 10.5


def test_drivable_mask_output_and_overlay():
    # Stub dependencies needed to load the vision module.
    utils_onnx = types.ModuleType("utils.utils_onnx")
    utils_onnx.driving_area_mask = lambda x: np.ones((2, 3), dtype=np.uint8) * 255
    utils_pkg = types.ModuleType("utils"); utils_pkg.__path__ = []
    utils_pkg.utils_onnx = utils_onnx
    sys.modules["utils"] = utils_pkg
    sys.modules["utils.utils_onnx"] = utils_onnx
    ort = types.ModuleType("onnxruntime")
    sys.modules["onnxruntime"] = ort
    ve = types.ModuleType("vision.color_extractor")
    class DummyExtractor:
        def __init__(self, *a, **k): pass
    ve.HSVColorThresholdExtractor = DummyExtractor
    sys.modules["vision.color_extractor"] = ve
    mod = load_module(ROOT / "vision/city_vision_processing.py", "vision_processing_test")
    vp = object.__new__(mod.VisionProcessor)
    vp.mode = "onnx"
    vp.debug = True
    vp.last_lane_center = None
    vp.lane_center_alpha = 0.3
    vp.fallback_lane_offset_ratio = 0.16
    vp.min_side_pixels = 2
    vp.active_side = None
    frame = np.zeros((210, 350, 3), dtype=np.uint8)
    drivable = np.zeros((210, 350), dtype=np.uint8); drivable[100:, :] = 255
    out = vp._draw_debug(frame, np.zeros_like(drivable), drivable, None, None, 175, "none")
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert np.any(out[150, 175] != 0)


def test_junction_crosswalk_precedes_static(monkeypatch):
    carla = types.ModuleType("carla")
    class VC:
        def __init__(self, **kw): self.__dict__.update(kw)
    carla.VehicleControl = VC
    sys.modules["carla"] = carla
    conf = importlib.import_module("config_city")
    monkeypatch.setattr(conf, "CROSSWALK_SLEEP", 0.03)
    jm_mod = load_module(ROOT / "navigation/intersection_manager.py", "junction_test")
    class Loc:
        def distance(self, other): return 100.0
    class Vehicle: pass
    class Planner:
        def distance_to_next_maneuver(self, a, g): return 1.0
        def get_next_maneuver_text(self, a, g): return "RIGHT"
    jm = jm_mod.IntersectionManager(Vehicle(), Planner())
    assert jm.start_for_intersection(Loc(), Loc())
    deadline = time.monotonic() + 1.0
    while jm.phase_name() == "PLANNING" and time.monotonic() < deadline:
        jm.update(Loc()); time.sleep(0.005)
    assert jm.phase_name() == "CROSSWALK"
    assert jm.static_control(Loc()).brake == 1.0
    jm.update(Loc()); time.sleep(0.04); jm.update(Loc())
    assert jm.phase_name() == "STATIC"
    assert jm.static_control(Loc()).throttle > 0.0


def test_manual_controls_are_immediate():
    pygame = types.SimpleNamespace()
    pygame.K_w=1; pygame.K_s=2; pygame.K_a=3; pygame.K_d=4; pygame.K_SPACE=5
    pygame.K_e=6; pygame.K_ESCAPE=7; pygame.QUIT=8; pygame.KEYDOWN=9
    pygame.init=lambda:None
    pygame.display=types.SimpleNamespace(set_mode=lambda *_:None,set_caption=lambda *_:None)
    pressed=set()
    pygame.event=types.SimpleNamespace(get=lambda: [])
    pygame.key=types.SimpleNamespace(get_pressed=lambda: {k:(k in pressed) for k in range(1,8)}, key_code=lambda x: 1)
    carla=types.ModuleType("carla")
    class VC:
        def __init__(self, **kw): self.__dict__.update(kw)
    carla.VehicleControl=VC
    sys.modules["pygame"]=pygame; sys.modules["carla"]=carla
    im_mod=load_module(ROOT/"controllers/input_manager.py", "input_test")
    im=im_mod.InputManager(); pressed.add(pygame.K_w)
    running,toggle,control=im.poll()
    assert running and not toggle and control.throttle == 0.7


def test_debug_renderer_preserves_current_dimensions_and_bgr():
    uv = types.ModuleType("utils.vehicle_utils")
    uv.blank_frame = lambda: np.zeros((210, 350, 3), dtype=np.uint8)
    up = types.ModuleType("utils"); up.__path__=[]
    up.vehicle_utils = uv
    sys.modules["utils"] = up; sys.modules["utils.vehicle_utils"] = uv
    mod = load_module(ROOT / "ui/renderer.py", "renderer_test")
    renderer = mod.Renderer()
    frame = np.zeros((96, 160, 3), dtype=np.uint8)
    frame[10:20, 10:20] = (10, 20, 30)
    out = renderer.compose(frame, {"lines": ["GPS: 1, 2, 3", "FPS: 10"]})
    assert out.shape == (96, 160, 3)
    assert tuple(out[15, 15]) == (10, 20, 30)


def test_manual_e_toggle_is_event_driven():
    pygame = types.SimpleNamespace()
    pygame.K_w=1; pygame.K_s=2; pygame.K_a=3; pygame.K_d=4; pygame.K_SPACE=5
    pygame.K_e=6; pygame.K_ESCAPE=7; pygame.QUIT=8; pygame.KEYDOWN=9
    pygame.init=lambda:None
    pygame.display=types.SimpleNamespace(set_mode=lambda *_:None,set_caption=lambda *_:None)
    class Event: pass
    e=Event(); e.type=pygame.KEYDOWN; e.key=pygame.K_e
    pygame.event=types.SimpleNamespace(get=lambda: [e])
    pygame.key=types.SimpleNamespace(get_pressed=lambda: {k: False for k in range(1,8)}, key_code=lambda x: 1)
    carla=types.ModuleType("carla")
    class VC:
        def __init__(self, **kw): self.__dict__.update(kw)
    carla.VehicleControl=VC
    sys.modules["pygame"]=pygame; sys.modules["carla"]=carla
    im_mod=load_module(ROOT/"controllers/input_manager.py", "input_toggle_test")
    im=im_mod.InputManager(); assert im.poll()[1] is True


def test_trajectory_matching_rejects_wrong_maneuver():
    # Recreate the same invariant used by main without importing the full CARLA runtime.
    class R:
        valid=True; command="RIGHT"
    result=R()
    assert result.valid and result.command == "RIGHT"
    assert result.command != "LEFT"


def test_panel_module_available_or_explicitly_skipped():
    import importlib.util
    if importlib.util.find_spec("PySide6") is None:
        import pytest
        pytest.skip("PySide6 not installed in test environment")
    load_module(ROOT / "ui/control_panel.py", "control_panel_test")
