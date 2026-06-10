from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class IntersectionModel:
    def __init__(self, checkpoint_path: str = "junction_model_resnet18.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.image_size, self.class_names = self._load_model()
        self.transform = self._build_transform()

    def _load_model(self):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        image_size = int(ckpt.get("image_size", 224))
        class_names = ckpt.get("class_names", {0: "not_junction", 1: "junction"})

        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
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

    @torch.no_grad()
    def predict(self, frame: Union[np.ndarray, Image.Image]) -> Tuple[bool, float]:
        """
        Returns:
            is_intersection_ahead: bool
            confidence: float
        """
        img = self._prepare_frame(frame)
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

        is_intersection_ahead = (pred == 1)
        return is_intersection_ahead, conf

    def is_intersection_ahead(self, frame: Union[np.ndarray, Image.Image]) -> bool:
        """
        True  -> intersection / junction
        False -> no intersection
        """
        pred, _ = self.predict(frame)
        return pred


# if __name__ == "__main__":
#     model = IntersectionModel("junction_model_resnet18.pt")

#     img = cv2.imread("test.jpg")
#     if img is None:
#         raise FileNotFoundError("test.jpg not found")

#     result, conf = model.predict(img)
#     print("is_intersection_ahead =", result)
#     print("confidence =", conf)