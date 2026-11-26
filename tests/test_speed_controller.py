from __future__ import annotations

import pytest

from controllers.speed_controller import SpeedPIDController


def test_positive_targets_are_clamped_to_minimum_and_maximum_speed():
    controller = SpeedPIDController(
        acceleration_mps2=2.5,
        min_speed_kmh=8.0,
        max_speed_kmh=25.0,
    )
    assert controller.normalize_target_speed(2.0) == pytest.approx(8.0)
    assert controller.normalize_target_speed(80.0) == pytest.approx(25.0)


def test_zero_target_is_an_explicit_stop_and_bypasses_minimum_speed():
    controller = SpeedPIDController(min_speed_kmh=8.0, max_speed_kmh=25.0)
    throttle, brake = controller.update(12.0, 0.0)
    assert throttle == 0.0
    assert brake == pytest.approx(1.0)


def test_standstill_launch_has_enough_throttle_to_start_moving():
    controller = SpeedPIDController(min_speed_kmh=8.0, max_speed_kmh=25.0)
    throttle, brake = controller.update(0.0, 8.0)
    assert throttle >= controller.LAUNCH_THROTTLE
    assert brake == 0.0


def test_lower_positive_target_cannot_cancel_moving_request():
    controller = SpeedPIDController(min_speed_kmh=8.0, max_speed_kmh=25.0)
    throttle, brake = controller.update(0.0, 3.0)
    assert controller.normalize_target_speed(3.0) == pytest.approx(8.0)
    assert throttle >= controller.LAUNCH_THROTTLE
    assert brake == 0.0


def test_per_update_minimum_speed_override_allows_slower_lane_change_without_changing_normal_floor():
    controller = SpeedPIDController(min_speed_kmh=8.0, max_speed_kmh=25.0)

    assert controller.normalize_target_speed(4.5) == pytest.approx(8.0)
    assert controller.normalize_target_speed(
        4.5, minimum_speed_override_kmh=4.5
    ) == pytest.approx(4.5)

    throttle, brake = controller.update(
        8.0, 4.5, minimum_speed_override_kmh=4.5
    )
    assert throttle == 0.0
    assert brake > 0.0
    # The temporary override does not mutate the normal driving limit.
    assert controller.normalize_target_speed(4.5) == pytest.approx(8.0)


def test_minimum_speed_override_must_be_positive_and_finite():
    controller = SpeedPIDController(min_speed_kmh=8.0, max_speed_kmh=25.0)
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="override"):
            controller.normalize_target_speed(
                4.5, minimum_speed_override_kmh=value
            )


def test_overspeed_commands_braking_instead_of_positive_throttle():
    controller = SpeedPIDController(min_speed_kmh=8.0, max_speed_kmh=25.0)
    throttle, brake = controller.update(18.0, 10.0)
    assert throttle == 0.0
    assert brake > 0.0


def test_acceleration_parameter_limits_requested_acceleration():
    gentle = SpeedPIDController(acceleration_mps2=0.8, min_speed_kmh=8.0, max_speed_kmh=25.0)
    responsive = SpeedPIDController(acceleration_mps2=3.5, min_speed_kmh=8.0, max_speed_kmh=25.0)
    gentle_throttle, _ = gentle.update(0.0, 25.0)
    responsive_throttle, _ = responsive.update(0.0, 25.0)
    assert responsive_throttle > gentle_throttle
    assert responsive_throttle <= responsive.MAX_THROTTLE
