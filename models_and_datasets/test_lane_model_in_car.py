# live_lane_side_viewer.py
# Controls:
#   W / Up    -> throttle
#   S / Down  -> brake
#   A / Left  -> steer left
#   D / Right -> steer right
#   R         -> toggle reverse
#   Space     -> handbrake
#   ESC       -> quit
#
# Shows live model prediction on screen:
#   Left lane / Right lane / Out from right / Out from left
#   Model FPS

import random
import time
from pathlib import Path
from typing import Optional, Tuple

import carla
import cv2
import numpy as np
import pygame
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


PREFERRED_VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
STEER_VALUE = 0.6


class LiveLaneSideViewer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        checkpoint_path: str = "lane_side_model_resnet18.pt",
        width: int = 1280,
        height: int = 720,
        fov: float = 90.0,
        camera_x: float = 1.4,
        camera_y: float = 0.0,
        camera_z: float = 1.8,
    ):
        self.host = host
        self.port = port
        self.checkpoint_path = Path(checkpoint_path)

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

        self.latest_bgr: Optional[np.ndarray] = None
        self.running = True
        self.reverse_mode = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.image_size, self.class_names = self._load_model()
        self.transform = self._build_transform()

        self.model_fps_ema = 0.0
        self.last_pred = 0
        self.last_conf = 0.0

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        image_size = int(ckpt.get("image_size", 224))
        class_names = ckpt.get(
            "class_names",
            {
                0: "left_lane",
                1: "right_lane",
                2: "out_from_right",
                3: "out_from_left",
            },
        )

        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 4)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model, image_size, class_names

    def _build_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
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

        spawn_point = random.choice(spawn_points)

        self.cleanup_vehicle_only()

        self.vehicle = self.world.try_spawn_actor(bp, spawn_point)
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

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray) -> Tuple[int, float]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
        elapsed = time.perf_counter() - start

        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        if self.model_fps_ema <= 0.0:
            self.model_fps_ema = fps
        else:
            self.model_fps_ema = 0.9 * self.model_fps_ema + 0.1 * fps

        self.last_pred = pred
        self.last_conf = conf
        return pred, conf

    def control_from_keyboard(self) -> carla.VehicleControl:
        keys = pygame.key.get_pressed()

        control = carla.VehicleControl()

        control.throttle = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
        control.brake = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0

        steer_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        steer_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        if steer_left and not steer_right:
            control.steer = STEER_VALUE
        elif steer_right and not steer_left:
            control.steer = -STEER_VALUE
        else:
            control.steer = 0.0

        control.hand_brake = keys[pygame.K_SPACE]
        control.reverse = self.reverse_mode
        return control

    def _prediction_text(self):
        return self.class_names.get(self.last_pred, str(self.last_pred))

    def draw_overlay(self, frame_bgr: np.ndarray):
        pred_name = self._prediction_text()

        color_text = pred_name.replace("_", " ").upper()

        cv2.putText(
            frame_bgr,
            f"Prediction: {color_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Class id: {self.last_pred}   Conf: {self.last_conf:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Model FPS: {self.model_fps_ema:.1f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"Reverse: {'ON' if self.reverse_mode else 'OFF'}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            "W/A/S/D drive | R reverse | SPACE brake | ESC quit",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    def run(self):
        self.connect()
        self.spawn_vehicle()
        self.spawn_camera()

        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CARLA Live Lane Side Classifier")
        clock = pygame.time.Clock()

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_r:
                            self.reverse_mode = not self.reverse_mode

                if self.vehicle is not None:
                    self.vehicle.apply_control(self.control_from_keyboard())

                if self.latest_bgr is not None:
                    frame = self.latest_bgr.copy()
                    self.predict_frame(frame)
                    frame = self.draw_overlay(frame)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
                    screen.blit(surface, (0, 0))
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
    app = LiveLaneSideViewer(
        host="127.0.0.1",
        port=2000,
        checkpoint_path="models/lane_side_model_resnet18.pt",
        width=1280,
        height=720,
        fov=90.0,
        camera_x=1.4,
        camera_y=0.0,
        camera_z=1.8,
    )
    app.run()


if __name__ == "__main__":
    main()