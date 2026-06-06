import carla
import math


def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def get_turn_options(waypoint, lookahead=10.0):
    """
    Returns available directions:
    {'left', 'straight', 'right'}
    """

    options = set()

    candidates = waypoint.next(lookahead)

    current_yaw = waypoint.transform.rotation.yaw

    for wp in candidates:
        delta = normalize_angle(
            wp.transform.rotation.yaw - current_yaw
        )

        if delta > 35:
            options.add("right")
        elif delta < -35:
            options.add("left")
        else:
            options.add("straight")

    return options


def classify_junction(junction_wp):
    options = get_turn_options(junction_wp)

    if options == {"left", "straight", "right"}:
        return "4_way"

    if len(options) == 2:
        return "t_junction"

    if len(options) == 1:
        return "straight"

    return "unknown"