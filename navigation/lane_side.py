from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class LaneSideModel:
    def __init__(self, checkpoint_path: str = "lane_side_model_resnet18.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.image_size, self.class_names = self._load_model()
        self.transform = self._build_transform()

    def _load_model(self):
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

    def _prepare_frame(self, frame: Union[np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array or PIL Image")

        if frame.ndim != 3:
            raise ValueError("frame must be HxWxC")

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return Image.fromarray(frame)

    @torch.inference_mode()
    def predict(self, frame: Union[np.ndarray, Image.Image]) -> Tuple[int, float]:
        """
        Returns:
            pred_class: int
            confidence: float
        Class ids:
            0 -> left_lane
            1 -> right_lane
            2 -> out_from_right
            3 -> out_from_left
        """
        img = self._prepare_frame(frame)
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

        return pred, conf

    def predict_name(self, frame: Union[np.ndarray, Image.Image]) -> Tuple[str, float]:
        pred, conf = self.predict(frame)
        return self.class_names.get(pred, str(pred)), conf

    def is_left_lane(self, frame: Union[np.ndarray, Image.Image]) -> bool:
        pred, _ = self.predict(frame)
        return pred == 0

    def is_right_lane(self, frame: Union[np.ndarray, Image.Image]) -> bool:
        pred, _ = self.predict(frame)
        return pred == 1

    def is_out_from_right(self, frame: Union[np.ndarray, Image.Image]) -> bool:
        pred, _ = self.predict(frame)
        return pred == 2

    def is_out_from_left(self, frame: Union[np.ndarray, Image.Image]) -> bool:
        pred, _ = self.predict(frame)
        return pred == 3


# if __name__ == "__main__":
#     model = LaneSideModel("lane_side_model_resnet18.pt")
#
#     img = cv2.imread("test.jpg")
#     if img is None:
#         raise FileNotFoundError("test.jpg not found")
#
#     pred, conf = model.predict(img)
#     print("pred =", pred)
#     print("class_name =", model.class_names.get(pred))
#     print("confidence =", conf)
