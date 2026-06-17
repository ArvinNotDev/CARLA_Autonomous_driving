from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18TrajectoryRegressor(nn.Module):
    def __init__(self, n_commands: int = 4, n_waypoints: int = 5, cmd_emb_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.command_embed = nn.Embedding(n_commands, cmd_emb_dim)
        self.head = nn.Sequential(
            nn.Linear(512 + cmd_emb_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_waypoints * 2),
        )
        self.n_waypoints = n_waypoints

    def forward(self, image: torch.Tensor, command_idx: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)
        cmd = self.command_embed(command_idx)
        x = torch.cat([feat, cmd], dim=1)
        out = self.head(x)
        return out.view(-1, self.n_waypoints, 2)