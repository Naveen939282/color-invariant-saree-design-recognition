"""Suggest visually similar image pairs for human review only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .dataset_utils import resolve_image_path
except ImportError:
    from dataset_utils import resolve_image_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("metadata/image_metadata.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--deeplure-root", type=Path)
    parser.add_argument("--kaggle-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("local_outputs/candidate_pairs.csv"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        parser.error("--top-k and --batch-size must be positive")

    try:
        import torch
        from PIL import Image
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:
        raise SystemExit(
            "Optional packages are missing. Install with: "
            "python -m pip install -r requirements-similarity.txt"
        ) from error

    metadata = pd.read_csv(args.metadata, dtype=str, keep_default_na=False)
    metadata = metadata[metadata["readable"].str.casefold().eq("true")].reset_index(drop=True)
    if len(metadata) < 2:
        raise SystemExit("At least two readable images are required.")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    embedding_batches = []
    rows = metadata.to_dict("records")
    for start in range(0, len(rows), args.batch_size):
        batch_tensors = []
        for row in rows[start : start + args.batch_size]:
            path = resolve_image_path(
                row, args.data_root, args.deeplure_root, args.kaggle_root
            )
            if not path.is_file():
                raise SystemExit(f"Image not found: {path}")
            with Image.open(path) as image:
                batch_tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(batch_tensors)
        with torch.inference_mode():
            features = torch.nn.functional.normalize(model(batch), dim=1)
        embedding_batches.append(features.cpu())
    embeddings = torch.cat(embedding_batches, dim=0)

    candidate_pairs: set[tuple[int, int]] = set()
    for start in range(0, len(rows), args.batch_size):
        similarities = embeddings[start : start + args.batch_size] @ embeddings.T
        for offset, scores in enumerate(similarities):
            index = start + offset
            scores[index] = -1
            count = min(args.top_k, len(rows) - 1)
            for other in torch.topk(scores, count).indices.tolist():
                candidate_pairs.add(tuple(sorted((index, other))))

    output_rows = [
        {
            "image_id_1": metadata.iloc[first]["image_id"],
            "image_id_2": metadata.iloc[second]["image_id"],
            "similarity_score": round(float(torch.dot(embeddings[first], embeddings[second])), 6),
            "review_status": "NEEDS_HUMAN_REVIEW",
        }
        for first, second in sorted(candidate_pairs)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        output_rows,
        columns=["image_id_1", "image_id_2", "similarity_score", "review_status"],
    ).to_csv(args.output, index=False)
    print(
        f"Wrote {len(output_rows)} candidate pairs to {args.output}. "
        "Embedding similarity is a suggestion only; it never assigns design IDs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())