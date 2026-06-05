import random
from typing import List, Optional

import carla
import config_city as conf


class CarlaManager:
    def __init__(self, host: str = None, port: int = None, timeout: float = None):
        self.host = host if host is not None else conf.HOST
        self.port = port if port is not None else conf.PORT
        self.timeout = timeout if timeout is not None else conf.TIMEOUT

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.original_settings = None

        self.vehicle: Optional[carla.Vehicle] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self.traffic_vehicles: List[carla.Actor] = []

    def connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(240.0)

        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(False)

        return self.world

    def spawn_vehicle(self):
        if self.world is None:
            raise RuntimeError("World is not connected. Call connect() first.")

        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find(conf.VEHICLE_BLUEPRINT)

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available")

        import random
        random.shuffle(spawn_points)

        vehicle = None

        for sp in spawn_points:
            vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break

        if vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle on any spawn point")

        self.vehicle = vehicle
        self.vehicle.set_autopilot(False)
        self.vehicle.set_simulate_physics(True)

        return self.vehicle

        return self.vehicle

    def spawn_traffic(
        self,
        num_vehicles: int = 20,
        speed_difference: float = 20.0,
        distance_to_lead: float = 2.5,
    ):
        """
        Spawn background traffic using CARLA Traffic Manager.
        This is simulated traffic, not live real-world traffic data.
        """
        if self.world is None or self.client is None:
            raise RuntimeError("World is not connected. Call connect() first.")

        if self.traffic_manager is None:
            self.traffic_manager = self.client.get_trafficmanager(8000)

        self.traffic_manager.set_synchronous_mode(False)
        self.traffic_manager.global_percentage_speed_difference(speed_difference)
        self.traffic_manager.set_global_distance_to_leading_vehicle(distance_to_lead)

        bp_lib = self.world.get_blueprint_library()
        vehicle_blueprints = bp_lib.filter("vehicle.*")

        # Optional: filter out weird vehicles and keep common road cars/trucks.
        filtered = []
        for bp in vehicle_blueprints:
            # Keep vehicles with 4 wheels when possible for more realistic traffic.
            if bp.has_attribute("number_of_wheels"):
                try:
                    if int(bp.get_attribute("number_of_wheels").as_int()) != 4:
                        continue
                except Exception:
                    pass
            filtered.append(bp)

        if not filtered:
            filtered = vehicle_blueprints

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found on this map.")

        random.shuffle(spawn_points)

        spawned = 0
        max_to_spawn = min(num_vehicles, len(spawn_points))

        for spawn_point in spawn_points:
            if spawned >= max_to_spawn:
                break

            # Avoid spawning too close to ego spawn point if it exists.
            if self.vehicle is not None:
                ego_loc = self.vehicle.get_transform().location
                spawn_loc = spawn_point.location
                if ego_loc.distance(spawn_loc) < 8.0:
                    continue

            vehicle_bp = random.choice(filtered)

            if vehicle_bp.has_attribute("color"):
                color = random.choice(vehicle_bp.get_attribute("color").recommended_values)
                vehicle_bp.set_attribute("color", color)

            if vehicle_bp.has_attribute("driver_id"):
                driver_id = random.choice(vehicle_bp.get_attribute("driver_id").recommended_values)
                vehicle_bp.set_attribute("driver_id", driver_id)

            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is None:
                continue

            vehicle.set_autopilot(True, self.traffic_manager.get_port())
            vehicle.set_simulate_physics(True)
            self.traffic_vehicles.append(vehicle)
            spawned += 1

        return self.traffic_vehicles

    def destroy_traffic(self):
        for actor in self.traffic_vehicles:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass
        self.traffic_vehicles = []

    def cleanup(self):
        self.destroy_traffic()

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None

        if self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                pass