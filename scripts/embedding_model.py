"""ResNet18 embeddings and a simple contrastive metric-learning objective."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Embedding(nn.Module):
    """Pretrained ResNet18 backbone with a normalized projection head."""

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(feature_count, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable
        if self.freeze_backbone:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNet18Embedding":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=1)


class GrayscaleResNet18Baseline(nn.Module):
    """Pretrained ResNet18 features for grayscale-replicated RGB inputs."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(images), p=2, dim=1)


def contrastive_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    same_design: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pull same-design pairs together and push different designs past margin."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    distance = F.pairwise_distance(embedding_a, embedding_b, p=2)
    target = same_design.to(dtype=distance.dtype).view(-1)
    positive_loss = target * distance.square()
    negative_loss = (1.0 - target) * F.relu(margin - distance).square()
    return (positive_loss + negative_loss).mean()


def cosine_similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between normalized or unnormalized vectors."""
    return F.cosine_similarity(embedding_a, embedding_b, dim=-1)