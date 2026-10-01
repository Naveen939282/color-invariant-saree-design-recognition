"""Generate local, paginated HTML contact sheets for human image review."""

from __future__ import annotations

import argparse
import html
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
    parser.add_argument("--output-dir", type=Path, default=Path("local_outputs/contact_sheets"))
    parser.add_argument("--page-size", type=int, choices=[25, 50], default=25)
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata, dtype=str, keep_default_na=False)
    output_dir = args.output_dir.resolve()
    if any(
        root is not None and (output_dir == root.resolve() or root.resolve() in output_dir.parents)
        for root in (args.data_root, args.deeplure_root, args.kaggle_root)
    ):
        parser.error("Contact sheet output must be outside all dataset roots.")
    page_dir = output_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    pages = [metadata.iloc[start : start + args.page_size] for start in range(0, len(metadata), args.page_size)]
    page_links = [
        f'<a href="page_{number:03d}.html">Page {number}</a>'
        for number in range(1, len(pages) + 1)
    ]
    index_links = [
        f'<a href="pages/page_{number:03d}.html">Page {number}</a>'
        for number in range(1, len(pages) + 1)
    ]
    for page_number, page in enumerate(pages, start=1):
        figures = []
        for row in page.to_dict("records"):
            image_path = resolve_image_path(
                row, args.data_root, args.deeplure_root, args.kaggle_root
            )
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image_href = image_path.resolve().as_uri()
            caption = " | ".join(
                [
                    row["image_id"],
                    row["filename"],
                    row.get("source_dataset", "unknown"),
                    row.get("source_category", "") or "(no category)",
                ]
            )
            figures.append(
                "<figure>"
                f'<a href="{html.escape(image_href, quote=True)}"><img loading="lazy" '
                f'src="{html.escape(image_href, quote=True)}" alt="{html.escape(row["filename"], quote=True)}"></a>'
                f'<figcaption>{html.escape(caption)}</figcaption></figure>'
            )
        content = (
            "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Image review</title>"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><style>"
            "body{font:14px system-ui,sans-serif;margin:24px;color:#242424}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}"
            "figure{margin:0;border:1px solid #bbb;padding:8px}"
            "img{width:100%;height:210px;object-fit:contain;background:#f2f2f2}"
            "figcaption{overflow-wrap:anywhere;margin-top:8px}</style>"
            f"<h1>Image review, page {page_number}</h1><nav>{' | '.join(page_links)}</nav>"
            f"<main class=\"grid\">{''.join(figures)}</main></html>"
        )
        (page_dir / f"page_{page_number:03d}.html").write_text(content, encoding="utf-8")
    index = (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Image review</title>"
        "<h1>Local image review</h1><p>Visual suggestions only. Design identity requires human confirmation.</p>"
        f"<nav>{' | '.join(index_links) if index_links else 'No images found.'}</nav></html>"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"Wrote {len(metadata)} image references across {len(pages)} pages to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())