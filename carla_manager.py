import json
from pathlib import Path
from typing import Optional

import carla
import config_city as conf

MAP_CONFIG_PATH = Path("map_config.json")


def _load_map_config(path: Path = MAP_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _spawn_transform_from_config(cfg: dict) -> carla.Transform:
    sp = cfg.get("spawn_point", {})
    return carla.Transform(
        carla.Location(
            x=float(sp.get("x", 0.0)),
            y=float(sp.get("y", 0.0)),
            z=float(sp.get("z", 0.0)),
        ),
        carla.Rotation(
            roll=float(sp.get("roll", 0.0)),
            pitch=float(sp.get("pitch", 0.0)),
            yaw=float(sp.get("yaw", 0.0)),
        ),
    )


class CarlaManager:
    def __init__(self, host: str = None, port: int = None, timeout: float = None):
        self.host = host if host is not None else conf.HOST
        self.port = port if port is not None else conf.PORT
        self.timeout = timeout if timeout is not None else conf.TIMEOUT

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.original_settings = None

        self.vehicle: Optional[carla.Vehicle] = None

    def connect(self) -> carla.World:
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)

        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        return self.world

    def spawn_vehicle(self, spawn_transform=None, config_path: Path = MAP_CONFIG_PATH) -> carla.Vehicle:
        if self.world is None:
            raise RuntimeError("World is not connected. Call connect() first.")

        if spawn_transform is None:
            cfg = _load_map_config(config_path)
            spawn_transform = _spawn_transform_from_config(cfg)

        carla_map = self.world.get_map()
        if carla_map is not None:
            wp = carla_map.get_waypoint(
                spawn_transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is not None:
                spawn_transform = wp.transform
                spawn_transform.location.z += 0.8

        bp_lib = self.world.get_blueprint_library()

        try:
            vehicle_bp = bp_lib.find(conf.VEHICLE_BLUEPRINT)
        except Exception:
            vehicle_bp = bp_lib.filter("vehicle.*")[0]

        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if vehicle is None:
            raise RuntimeError(
                f"Failed to spawn ego vehicle at x={spawn_transform.location.x:.2f}, "
                f"y={spawn_transform.location.y:.2f}, z={spawn_transform.location.z:.2f}, "
                f"yaw={spawn_transform.rotation.yaw:.2f}"
            )

        self.vehicle = vehicle
        self.vehicle.set_autopilot(False)
        self.vehicle.set_simulate_physics(True)
        return self.vehicle

    def cleanup(self):
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
