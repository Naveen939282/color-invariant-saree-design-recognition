"""Optional pretrained-image-embedding suggestions; never creates design labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, default=Path("dataset_audit_output/image_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("candidate_pairs.csv"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    try:
        import torch
        from PIL import Image
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise SystemExit(
            "Optional dependencies are missing. Install them with: "
            "python -m pip install torch torchvision"
        ) from error

    metadata = pd.read_csv(args.metadata, dtype=str, keep_default_na=False)
    metadata = metadata[metadata["readable"].str.casefold().eq("true")].reset_index(drop=True)
    if len(metadata) < 2:
        raise SystemExit("At least two readable images are needed.")
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    tensors = []
    for row in metadata.to_dict("records"):
        path = args.data_root / Path(row["relative_path"])
        if not path.is_file():
            raise SystemExit(f"Image file not found: {path}")
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    batch = torch.stack(tensors)
    with torch.inference_mode():
        embeddings = torch.nn.functional.normalize(model(batch), dim=1)
        similarities = embeddings @ embeddings.T

    candidate_indices: set[tuple[int, int]] = set()
    for index in range(len(metadata)):
        scores = similarities[index].clone()
        scores[index] = -1
        count = min(args.top_k, len(metadata) - 1)
        for other in torch.topk(scores, count).indices.tolist():
            candidate_indices.add(tuple(sorted((index, other))))
    rows = [
        {
            "image_id_1": metadata.iloc[first]["image_id"],
            "image_id_2": metadata.iloc[second]["image_id"],
            "similarity_score": round(float(similarities[first, second]), 6),
            "review_status": "NEEDS_HUMAN_REVIEW",
        }
        for first, second in sorted(candidate_indices)
    ]
    pd.DataFrame(
        rows,
        columns=["image_id_1", "image_id_2", "similarity_score", "review_status"],
    ).to_csv(args.output, index=False)
    print(
        f"Wrote {len(rows)} candidate pairs to {args.output}. "
        "Embedding similarity is only a review suggestion, never a ground-truth design label."
    )


if __name__ == "__main__":
    main()