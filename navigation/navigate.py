import math
import carla

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