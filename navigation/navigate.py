import math
import carla
import threading
import time

def move_toward_target(vehicle, target_location, throttle=0.35, stop_distance=3.0):
    vehicle_tf = vehicle.get_transform()
    vehicle_loc = vehicle_tf.location
    vehicle_yaw = math.radians(vehicle_tf.rotation.yaw)

    dx = target_location.x - vehicle_loc.x
    dy = target_location.y - vehicle_loc.y

    distance = math.hypot(dx, dy)
    if distance < stop_distance:
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

    target_yaw = math.atan2(dy, dx)
    angle = target_yaw - vehicle_yaw

    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi

    steer = angle / math.radians(45.0)
    steer = max(-1.0, min(1.0, steer))

    return carla.VehicleControl(
        throttle=throttle,
        steer=steer,
        brake=0.0,
        hand_brake=False,
        reverse=False,
        manual_gear_shift=False,
    )

def _move_vehicle_for_distance_blocking(
    vehicle: carla.Vehicle,
    distance_m: float,
    steer: float = 0.0,
    forward: bool = True,
    throttle: float = 0.35,
    timeout: float = 20.0,
):
    distance_m = abs(float(distance_m))
    steer = max(-1.0, min(1.0, float(steer)))
    throttle = max(0.0, min(1.0, float(throttle)))

    world = vehicle.get_world()
    start_location = vehicle.get_location()
    start_time = time.time()

    control = carla.VehicleControl()
    control.throttle = throttle
    control.steer = steer
    control.brake = 0.0
    control.reverse = not forward

    while True:
        vehicle.apply_control(control)

        if vehicle.get_location().distance(start_location) >= distance_m:
            break

        if time.time() - start_time > timeout:
            break

        time.sleep(0.02)

    # stop = carla.VehicleControl()
    # stop.throttle = 0.0
    # stop.steer = 0.0
    # stop.brake = 1.0
    # stop.reverse = False
    # stop.hand_brake = False
    # stop.manual_gear_shift = False
    # vehicle.apply_control(stop)


def move_vehicle_for_distance(
    vehicle: carla.Vehicle,
    distance_m: float,
    steer: float = 0.0,
    forward: bool = True,
    throttle: float = 0.35,
    timeout: float = 20.0,
    *,
    blocking: bool = False,
):
    """
    Move the vehicle for a specific distance with a fixed steer.

    Args:
        vehicle: CARLA vehicle actor
        distance_m: target distance in meters
        steer: steering value in [-1.0, 1.0]
        forward: True = move forward, False = move backward
        throttle: throttle in [0.0, 1.0]
        timeout: safety timeout in seconds
    """
    if blocking:
        _move_vehicle_for_distance_blocking(
            vehicle=vehicle,
            distance_m=distance_m,
            steer=steer,
            forward=forward,
            throttle=throttle,
            timeout=timeout,
        )
        return None

    thread = threading.Thread(
        target=_move_vehicle_for_distance_blocking,
        kwargs={
            "vehicle": vehicle,
            "distance_m": distance_m,
            "steer": steer,
            "forward": forward,
            "throttle": throttle,
            "timeout": timeout,
        },
        daemon=True,
    )
    thread.start()
    return thread