from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from config_city import COMMANDS, COMMAND_TO_IDX, PipelineConfig
from utils.carla_utils import clamp
from trajectory.model import ResNet18TrajectoryRegressor


class TrajectorySteeringAgent:
    def __init__(
        self,
        checkpoint_path: str | Path,
        cfg: PipelineConfig,
        command_name: str = "LANE_FOLLOW",
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        self.command_name = command_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = ResNet18TrajectoryRegressor(
            n_commands=len(COMMANDS),
            n_waypoints=cfg.n_waypoints,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.prev_steer: float = 0.0

        self._tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(cfg.model_image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @staticmethod
    def carla_image_to_rgb(image) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        return arr[:, :, :3][:, :, ::-1].copy()

    def preprocess_for_model(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Convert the camera's OpenCV BGR frame to RGB exactly once."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty camera frame")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._tf(rgb)

    @torch.inference_mode()
    def predict_waypoints(self, frame_bgr: np.ndarray, command_name: Optional[str] = None) -> np.ndarray:
        if command_name is None:
            command_name = self.command_name

        if command_name not in COMMAND_TO_IDX:
            raise ValueError(f"Unknown command: {command_name}. Available: {list(COMMAND_TO_IDX.keys())}")

        image_t = self.preprocess_for_model(frame_bgr).unsqueeze(0).to(self.device)
        cmd_idx = torch.tensor([COMMAND_TO_IDX[command_name]], dtype=torch.long, device=self.device)

        pred = self.model(image_t, cmd_idx).squeeze(0).detach().cpu().numpy()

        pred *= 1.1
        return pred

    def steering_from_waypoint(
        self,
        pred_ego: np.ndarray,
        prev_steer: float = 0.0,
        speed_kmh: float = 0.0,
        max_steer: float = 1.0,
        steer_gain: Optional[float] = None,
    ) -> float:
        if pred_ego is None or len(pred_ego) == 0:
            return prev_steer

        x_ref = 8
        idx = int(np.argmin(np.abs(pred_ego[:, 0] - x_ref)))
        target = pred_ego[idx].copy()

        x = max(1e-3, float(target[0]))
        y = float(target[1])

        if steer_gain is None:
            steer_gain = float(getattr(self.cfg, "TRAJECTORY_STEER_GAIN", 0.9))
        desired = steer_gain * (y / x_ref)

        desired = float(clamp(desired, -0.25, 0.25))

        max_delta = 1.0
        delta = clamp(desired - prev_steer, -max_delta, max_delta)
        steer = prev_steer + delta

        steer = 0.85 * prev_steer + 0.15 * steer

        return float(clamp(steer, -max_steer, max_steer))

    def throttle_brake_from_speed(
        self,
        speed_kmh: float,
        target_kmh: Optional[float] = None,
    ) -> tuple[float, float]:
        if target_kmh is None:
            target_kmh = getattr(
                self.cfg,
                "target_speed_kmh",
                getattr(self.cfg, "TARGET_SPEED_KMH", 25.0),
            )

        error = float(target_kmh - speed_kmh)

        if error > 1.5:
            throttle = clamp(error / 20.0, 0.15, 0.75)
            brake = 0.0
        elif error < -1.5:
            throttle = 0.0
            brake = clamp((-error) / 15.0, 0.10, 0.45)
        else:
            throttle = 0.0
            brake = 0.0

        return float(throttle), float(brake)

    def get_steering_angle(
        self,
        frame_bgr: np.ndarray,
        speed_kmh: float = 0.0,
        command_name: Optional[str] = None,
        max_steer: float = 1.0,
    ) -> float:
        pred = self.predict_waypoints(frame_bgr, command_name=command_name)
        steer = self.steering_from_waypoint(
            pred_ego=pred,
            prev_steer=self.prev_steer,
            speed_kmh=speed_kmh,
            max_steer=max_steer,
        )
        self.prev_steer = steer
        return steer

    def get_steering_and_pred(
        self,
        frame_bgr: np.ndarray,
        speed_kmh: float = 0.0,
        command_name: Optional[str] = None,
        max_steer: Optional[float] = None,
    ) -> tuple[float, np.ndarray]:
        pred = self.predict_waypoints(frame_bgr, command_name=command_name)
        steer = self.steering_from_waypoint(
            pred_ego=pred,
            prev_steer=self.prev_steer,
            speed_kmh=speed_kmh,
            max_steer=float(max_steer if max_steer is not None else getattr(self.cfg, "TRAJECTORY_MAX_STEER", 0.45)),
        )
        self.prev_steer = steer
        return steer, pred

    def get_control_from_frame(
        self,
        frame_bgr: np.ndarray,
        speed_kmh: float = 0.0,
        command_name: Optional[str] = None,
        max_steer: float = 0.45,
    ) -> tuple[float, float, float, np.ndarray]:
        pred = self.predict_waypoints(frame_bgr, command_name=command_name)
        steer = self.steering_from_waypoint(
            pred_ego=pred,
            prev_steer=self.prev_steer,
            speed_kmh=speed_kmh,
            max_steer=max_steer,
        )
        self.prev_steer = steer
        throttle, brake = self.throttle_brake_from_speed(speed_kmh)
        return steer, throttle, brake, pred
