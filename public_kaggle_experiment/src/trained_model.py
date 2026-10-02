"""Trainable ResNet18 embedding model for the controlled experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ColorInvariantEmbeddingModel(nn.Module):
    """ImageNet ResNet18 with a single trainable Linear(512, 128) projection."""

    def __init__(
        self,
        weights: ResNet18_Weights | None,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.backbone = resnet18(weights=weights)
        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(feature_count, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return F.normalize(self.projection(features), p=2, dim=1)