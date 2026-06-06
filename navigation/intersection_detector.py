import carla


def get_junction_ahead(current_waypoint, distance=30.0, step=2.0):
    """
    Returns the first junction found ahead within 'distance' meters.
    """

    wp = current_waypoint
    travelled = 0.0

    while travelled < distance:
        next_wps = wp.next(step)

        if not next_wps:
            return None

        wp = next_wps[0]
        travelled += step

        if wp.is_junction:
            return wp.get_junction()

    return None


def is_intersection_ahead(current_waypoint, distance=30.0):
    return get_junction_ahead(current_waypoint, distance) is not None


def is_four_way_intersection_ahead(current_waypoint, distance=30.0):
    junction = get_junction_ahead(current_waypoint, distance)

    if junction is None:
        return False

    # Count unique road IDs inside the junction
    waypoints = junction.get_waypoints(carla.LaneType.Driving)

    road_ids = set()

    for start_wp, end_wp in waypoints:
        road_ids.add(start_wp.road_id)
        road_ids.add(end_wp.road_id)

    # Typical 4-way intersections have 4 or more roads
    return len(road_ids) >= 4