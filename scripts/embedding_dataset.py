"""PyTorch datasets and conservative transforms for saree retrieval."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

try:
    from .dataset_utils import resolve_image_path
except ImportError:
    from dataset_utils import resolve_image_path


COLOR_AUGMENTATION = {
    "brightness": 0.35,
    "contrast": 0.35,
    "saturation": 0.75,
    "hue": 0.08,
}


def build_transform(
    mode: str = "color_aug",
    train: bool = False,
    image_size: int = 224,
    horizontal_flip: bool = False,
) -> transforms.Compose:
    """Create ImageNet-normalized grayscale or color-jitter transforms.

    Geometry is deliberately limited to a near-full-frame crop. Horizontal
    flipping is opt-in because some textile layouts can be directional.
    """
    if mode not in {"grayscale", "color_aug"}:
        raise ValueError("mode must be 'grayscale' or 'color_aug'")
    if image_size < 32:
        raise ValueError("image_size must be at least 32")

    weights_transform = ResNet18_Weights.DEFAULT.transforms()
    operations: list[object] = []
    if mode == "grayscale":
        operations.extend(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
            ]
        )
    elif train:
        operations.append(
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
            )
        )
        if horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip(p=0.5))
        operations.append(transforms.ColorJitter(**COLOR_AUGMENTATION))
    else:
        operations.extend(
            [transforms.Resize(image_size + 32), transforms.CenterCrop(image_size)]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=weights_transform.mean, std=weights_transform.std),
        ]
    )
    return transforms.Compose(operations)


class SareeImageDataset(Dataset):
    """Load images listed in a prepared split CSV from external data roots."""

    def __init__(
        self,
        csv_path: Path | pd.DataFrame,
        transform: transforms.Compose,
        data_root: Path | None = None,
        deeplure_root: Path | None = None,
        kaggle_root: Path | None = None,
        require_design_ids: bool = True,
    ) -> None:
        if isinstance(csv_path, pd.DataFrame):
            self.frame = csv_path.copy().fillna("").astype(str)
        else:
            self.frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        required = {"image_id", "relative_path", "source_dataset"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
        if require_design_ids and "design_id" not in self.frame.columns:
            raise ValueError("Split CSV must contain human-confirmed design_id values.")
        if require_design_ids and self.frame["design_id"].str.strip().eq("").any():
            raise ValueError("Split CSV contains blank design IDs.")
        self.transform = transform
        self.data_root = data_root
        self.deeplure_root = deeplure_root
        self.kaggle_root = kaggle_root

    def __len__(self) -> int:
        return len(self.frame)

    def path_for_index(self, index: int) -> Path:
        row = self.frame.iloc[index].to_dict()
        return resolve_image_path(
            row, self.data_root, self.deeplure_root, self.kaggle_root
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index].to_dict()
        image_path = resolve_image_path(
            row, self.data_root, self.deeplure_root, self.kaggle_root
        )
        try:
            with Image.open(image_path) as source:
                image = self.transform(source.convert("RGB"))
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Could not load image {row['image_id']} at {image_path}: {error}"
            ) from error
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image transform must return a torch.Tensor.")
        return {
            "image": image,
            "image_id": row["image_id"],
            "design_id": row.get("design_id", ""),
            "colorway": row.get("colorway", ""),
            "source_dataset": row["source_dataset"],
        }


class ContrastivePairDataset(Dataset):
    """Sample balanced image pairs using only one prepared split's labels."""

    def __init__(
        self,
        images: SareeImageDataset,
        pairs_per_epoch: int = 256,
        seed: int = 42,
    ) -> None:
        if pairs_per_epoch < 2:
            raise ValueError("pairs_per_epoch must be at least 2")
        self.images = images
        self.pairs_per_epoch = pairs_per_epoch + pairs_per_epoch % 2
        self.seed = seed
        self.epoch = 0
        self.by_design: dict[str, list[int]] = defaultdict(list)
        self.by_design_color: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, row in images.frame.iterrows():
            design_id = row["design_id"]
            colorway = row.get("colorway", "").strip().casefold()
            self.by_design[design_id].append(index)
            if colorway:
                self.by_design_color[design_id][colorway].append(index)
        self.eligible_designs = [
            design_id for design_id, indexes in self.by_design.items() if len(indexes) >= 2
        ]
        if not self.eligible_designs:
            raise ValueError("Training split has no design with at least two images.")
        self.design_ids = sorted(self.by_design)
        if len(self.design_ids) < 2:
            raise ValueError("Training split needs at least two distinct design IDs.")
        self.cross_color_designs = [
            design_id
            for design_id, colors in self.by_design_color.items()
            if len(colors) >= 2
        ]

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _sample_positive(self, rng: random.Random) -> tuple[int, int]:
        candidates = self.cross_color_designs or self.eligible_designs
        design_id = rng.choice(candidates)
        if design_id in self.cross_color_designs:
            colors = list(self.by_design_color[design_id])
            first_color, second_color = rng.sample(colors, 2)
            return (
                rng.choice(self.by_design_color[design_id][first_color]),
                rng.choice(self.by_design_color[design_id][second_color]),
            )
        return tuple(rng.sample(self.by_design[design_id], 2))

    def _sample_negative(self, rng: random.Random) -> tuple[int, int]:
        first_design, second_design = rng.sample(self.design_ids, 2)
        shared_colors = set(self.by_design_color[first_design]) & set(
            self.by_design_color[second_design]
        )
        if shared_colors:
            color = rng.choice(sorted(shared_colors))
            first = rng.choice(self.by_design_color[first_design][color])
            second = rng.choice(self.by_design_color[second_design][color])
            return first, second
        return (
            rng.choice(self.by_design[first_design]),
            rng.choice(self.by_design[second_design]),
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        same_design = index % 2 == 0
        first, second = (
            self._sample_positive(rng) if same_design else self._sample_negative(rng)
        )
        first_item = self.images[first]
        second_item = self.images[second]
        label = torch.tensor(float(same_design), dtype=torch.float32)
        return first_item["image"], second_item["image"], label