import carla
import atexit
import time


class ParkingSensors:
    def __init__(self, world, ego_vehicle):
        self.world = world
        self.ego = ego_vehicle
        self.sensors = []

        self.sensor_data = {
            "front": {
                "detected": False,
                "distance": None,
                "other_actor": None,
                "last_seen": None,
            },
            "back": {
                "detected": False,
                "distance": None,
                "other_actor": None,
                "last_seen": None,
            },
            "left": {
                "detected": False,
                "distance": None,
                "other_actor": None,
                "last_seen": None,
            },
            "right": {
                "detected": False,
                "distance": None,
                "other_actor": None,
                "last_seen": None,
            },
            "collision": {
                "detected": False,
                "other_actor": None,
                "impulse": None,
                "last_seen": None,
            },
        }

        blueprint_library = self.world.get_blueprint_library()

        self.bp_obstacle = blueprint_library.find("sensor.other.obstacle")
        self.bp_collision = blueprint_library.find("sensor.other.collision")

        self.bp_obstacle.set_attribute("distance", "6.0")
        self.bp_obstacle.set_attribute("hit_radius", "0.5")
        self.bp_obstacle.set_attribute("only_dynamics", "false")
        self.bp_obstacle.set_attribute("debug_linetrace", "false")
        self.bp_obstacle.set_attribute("sensor_tick", "0.05")

    def _obstacle_callback(self, name):
        def callback(event):
            self.sensor_data[name] = {
                "detected": True,
                "distance": getattr(event, "distance", None),
                "other_actor": event.other_actor.type_id if event.other_actor else None,
                "last_seen": time.time(),
            }

        return callback

    def _collision_callback(self, event):
        self.sensor_data["collision"] = {
            "detected": True,
            "other_actor": event.other_actor.type_id if event.other_actor else None,
            "impulse": getattr(event, "normal_impulse", None),
            "last_seen": time.time(),
        }

    def get_sensors(self):
        return self.sensor_data

    def get_current_distance(self, name, stale_after=0.20):
        data = self.sensor_data.get(name, {})
        last_seen = data.get("last_seen")
        if last_seen is None:
            return None

        if time.time() - last_seen > stale_after:
            return None

        return data.get("distance")

    def is_detected(self, name, stale_after=0.20):
        return self.get_current_distance(name, stale_after=stale_after) is not None

    def spawn(self):
        self._spawn_obstacle(
            "front",
            carla.Transform(
                carla.Location(x=1.8, z=0.8),
                carla.Rotation(yaw=0.0),
            ),
        )

        self._spawn_obstacle(
            "back",
            carla.Transform(
                carla.Location(x=-1.8, z=0.8),
                carla.Rotation(yaw=180.0),
            ),
        )

        self._spawn_obstacle(
            "left",
            carla.Transform(
                carla.Location(y=-1.0, z=0.8),
                carla.Rotation(yaw=-90.0),
            ),
        )

        self._spawn_obstacle(
            "right",
            carla.Transform(
                carla.Location(y=1.0, z=0.8),
                carla.Rotation(yaw=90.0),
            ),
        )

        collision_sensor = self.world.spawn_actor(
            self.bp_collision,
            carla.Transform(),
            attach_to=self.ego,
        )
        collision_sensor.listen(self._collision_callback)
        self.sensors.append(collision_sensor)

        return self.sensors

    def _spawn_obstacle(self, name, transform):
        sensor = self.world.spawn_actor(
            self.bp_obstacle,
            transform,
            attach_to=self.ego,
        )
        sensor.listen(self._obstacle_callback(name))
        self.sensors.append(sensor)
        return sensor

    def destroy(self):
        for sensor in self.sensors:
            try:
                sensor.stop()
            except Exception:
                pass

            try:
                sensor.destroy()
            except Exception:
                pass

        self.sensors.clear()


atexit.register