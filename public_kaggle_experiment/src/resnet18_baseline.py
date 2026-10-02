"""Frozen ResNet18 ImageNet feature extractor for the controlled baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FrozenResNet18Baseline(nn.Module):
    """L2-normalized native 512-D features with no classifier or learned head."""

    def __init__(self, weights: ResNet18_Weights | None) -> None:
        super().__init__()
        self.backbone = resnet18(weights=weights)
        self.original_resnet18_parameters = sum(
            parameter.numel() for parameter in self.backbone.parameters()
        )
        self.embedding_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.feature_extractor_parameters = sum(
            parameter.numel() for parameter in self.parameters()
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenResNet18Baseline":
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(images), p=2, dim=1)