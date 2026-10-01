"""Create local PCA plots colored by human design IDs and/or colorway."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True, help="NPZ emitted by evaluate_retrieval.py")
    parser.add_argument("--labels", type=Path, default=Path("metadata/design_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_outputs/figures"))
    parser.add_argument("--color-by", choices=["design_id", "colorway", "both"], default="both")
    args = parser.parse_args(argv)

    saved = np.load(args.embeddings, allow_pickle=False)
    image_ids = saved["image_ids"].astype(str)
    vectors = saved["embeddings"]
    if len(image_ids) < 2:
        parser.error("At least two embeddings are required for PCA.")
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    selected = labels.set_index("image_id").reindex(image_ids).reset_index()
    if selected["design_id"].eq("").any():
        parser.error("Embedding visualization requires human-confirmed design IDs.")
    coordinates = PCA(n_components=2, random_state=42).fit_transform(vectors)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dimensions = PCA(n_components=2, random_state=42).fit(vectors).explained_variance_ratio_
    for column in (["design_id", "colorway"] if args.color_by == "both" else [args.color_by]):
        values = selected[column].replace("", "(blank)")
        classes = sorted(values.unique())
        palette = plt.get_cmap("tab20", max(len(classes), 1))
        figure, axis = plt.subplots(figsize=(11, 8))
        for index, category in enumerate(classes):
            mask = values.eq(category).to_numpy()
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                label=category,
                color=palette(index),
                s=44,
                alpha=0.8,
            )
        axis.set_title(f"Embedding PCA colored by human {column}")
        axis.set_xlabel(f"PC1 ({dimensions[0] * 100:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({dimensions[1] * 100:.1f}% variance)")
        axis.legend(title=column, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
        figure.tight_layout()
        output = args.output_dir / f"pca_by_{column}.png"
        figure.savefig(output, dpi=160)
        plt.close(figure)
        print(f"Wrote local plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())