# collect_lane_side_dataset_carla.py
# Controls:
#   W / Up    -> throttle
#   S / Down  -> brake
#   A / Left  -> steer left
#   D / Right -> steer right
#   Space     -> handbrake
#   E         -> save current frame with label 1 (right lane)
#   Q         -> save current frame with label 0 (left lane)
#   C         -> save current frame with label 2 (out from right)
#   Z         -> save current frame with label 3 (out from left)
#   ESC       -> quit

import csv
import random
import time
from pathlib import Path
from typing import Optional

import carla
import cv2
import numpy as np
import pygame


STEER_VALUE = 0.6
PREFERRED_VEHICLE_BLUEPRINT = "vehicle.tesla.model3"

LABEL_LEFT = 0
LABEL_RIGHT = 1
LABEL_OUT_RIGHT = 2
LABEL_OUT_LEFT = 3

LABEL_NAMES = {
    LABEL_LEFT: "left_lane",
    LABEL_RIGHT: "right_lane",
    LABEL_OUT_RIGHT: "out_from_right",
    LABEL_OUT_LEFT: "out_from_left",
}


class CarlaDatasetCollector:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        out_dir: str = "dataset_lane",
        width: int = 640,
        height: int = 480,
        fov: float = 90.0,
        camera_x: float = 1.5,
        camera_y: float = 0.0,
        camera_z: float = 2.2,
    ):
        self.host = host
        self.port = port
        self.out_dir = Path(out_dir)
        self.width = width
        self.height = height
        self.fov = fov
        self.camera_x = camera_x
        self.camera_y = camera_y
        self.camera_z = camera_z

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.vehicle: Optional[carla.Vehicle] = None
        self.camera: Optional[carla.Sensor] = None

        self.latest_image: Optional[np.ndarray] = None
        self.latest_bgr: Optional[np.ndarray] = None
        self.running = True

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for label in LABEL_NAMES:
            (self.out_dir / str(label)).mkdir(parents=True, exist_ok=True)

        self.csv_path = self.out_dir / "labels.csv"
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "filename",
                        "label",
                        "label_name",
                        "x",
                        "y",
                        "z",
                        "yaw",
                        "timestamp",
                    ]
                )

    def connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

    def _choose_vehicle_blueprint(self, blueprint_library):
        tesla_bp = blueprint_library.filter(PREFERRED_VEHICLE_BLUEPRINT)
        if tesla_bp:
            return random.choice(tesla_bp)

        vehicle_blueprints = blueprint_library.filter("vehicle.*")
        if not vehicle_blueprints:
            raise RuntimeError("No vehicle blueprints found in CARLA.")
        return random.choice(vehicle_blueprints)

    def spawn_vehicle(self):
        assert self.world is not None

        blueprint_library = self.world.get_blueprint_library()
        bp = self._choose_vehicle_blueprint(blueprint_library)

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in this map.")

        self.cleanup_vehicle_only()

        random.shuffle(spawn_points)
        self.vehicle = None
        for spawn_point in spawn_points:
            self.vehicle = self.world.try_spawn_actor(bp, spawn_point)
            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError(f"Failed to spawn vehicle using blueprint: {bp.id}")

        self.vehicle.set_autopilot(False)

    def spawn_camera(self):
        assert self.world is not None
        assert self.vehicle is not None

        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", str(self.fov))

        cam_transform = carla.Transform(
            carla.Location(x=self.camera_x, y=self.camera_y, z=self.camera_z),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        self.camera = self.world.spawn_actor(
            camera_bp,
            cam_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.camera.listen(self._on_camera_image)

    def _on_camera_image(self, image: carla.Image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        self.latest_bgr = bgr
        self.latest_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def save_sample(self, label: int):
        if self.latest_bgr is None or self.vehicle is None:
            return False

        label = int(label)
        if label not in LABEL_NAMES:
            return False

        loc = self.vehicle.get_location()
        rot = self.vehicle.get_transform().rotation

        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000)
        filename = f"{ts}_{ms}.png"
        save_path = self.out_dir / str(label) / filename

        cv2.imwrite(str(save_path), self.latest_bgr)

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    str(save_path.relative_to(self.out_dir)),
                    label,
                    LABEL_NAMES[label],
                    f"{loc.x:.3f}",
                    f"{loc.y:.3f}",
                    f"{loc.z:.3f}",
                    f"{rot.yaw:.3f}",
                    ts,
                ]
            )

        print(
            f"[SAVED] {save_path} | label={label} ({LABEL_NAMES[label]}) | "
            f"pos=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})"
        )
        return True

    def control_from_keyboard(self) -> carla.VehicleControl:
        keys = pygame.key.get_pressed()

        control = carla.VehicleControl()
        control.throttle = 1.0 if keys[pygame.K_w] or keys[pygame.K_UP] else 0.0
        control.brake = 1.0 if keys[pygame.K_s] or keys[pygame.K_DOWN] else 0.0

        steer_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        steer_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        if steer_left and not steer_right:
            control.steer = -STEER_VALUE
        elif steer_right and not steer_left:
            control.steer = STEER_VALUE
        else:
            control.steer = 0.0

        control.hand_brake = keys[pygame.K_SPACE]
        control.reverse = False
        control.manual_gear_shift = False
        return control

    def draw_status(self, surface, font):
        assert self.vehicle is not None

        loc = self.vehicle.get_location()
        vel = self.vehicle.get_velocity()
        speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6

        lines = [
            "Vehicle: Tesla Model 3 preferred",
            f"Speed: {speed:.1f} km/h",
            f"Location: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})",
            "WASD / arrows drive | E right lane | Q left lane | C out-right | Z out-left | ESC quit",
        ]

        y = 10
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            surface.blit(text, (10, y))
            y += 24

    def run(self):
        self.connect()
        self.spawn_vehicle()
        self.spawn_camera()

        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CARLA Lane Side Dataset Collector")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 20)

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_e:
                            self.save_sample(LABEL_RIGHT)
                        elif event.key == pygame.K_q:
                            self.save_sample(LABEL_LEFT)
                        elif event.key == pygame.K_c:
                            self.save_sample(LABEL_OUT_RIGHT)
                        elif event.key == pygame.K_z:
                            self.save_sample(LABEL_OUT_LEFT)

                if self.vehicle is not None:
                    self.vehicle.apply_control(self.control_from_keyboard())

                if self.latest_image is not None:
                    frame = np.copy(self.latest_image)
                    surface = pygame.surfarray.make_surface(np.rot90(frame))
                    screen.blit(surface, (0, 0))
                    self.draw_status(screen, font)
                    pygame.display.flip()

                clock.tick(30)

        finally:
            self.cleanup_all()
            pygame.quit()

    def cleanup_vehicle_only(self):
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None

    def cleanup_all(self):
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None


def main():
    collector = CarlaDatasetCollector(
        host="127.0.0.1",
        port=2000,
        out_dir="dataset_lane",
        width=1280,
        height=720,
        fov=90.0,
        camera_x=1.4,
        camera_y=0.0,
        camera_z=1.8,
    )
    collector.run()


if __name__ == "__main__":
    main()