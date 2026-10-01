"""Build a local HTML report of query images and their top gallery matches."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

try:
    from .dataset_utils import resolve_image_path
except ImportError:
    from dataset_utils import resolve_image_path


def _local_image_uri(row: dict, roots: dict[str, Path | None]) -> str:
    return resolve_image_path(
        row,
        roots["data_root"],
        roots["deeplure_root"],
        roots["kaggle_root"],
    ).as_uri()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-csv", type=Path, required=True)
    parser.add_argument("--query-csv", type=Path, default=Path("metadata/query.csv"))
    parser.add_argument("--gallery-csv", type=Path, default=Path("metadata/gallery.csv"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--deeplure-root", type=Path)
    parser.add_argument("--kaggle-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("local_outputs/retrieval_reports/index.html"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    roots = {
        "data_root": args.data_root,
        "deeplure_root": args.deeplure_root,
        "kaggle_root": args.kaggle_root,
    }
    output = args.output.resolve()
    for root in roots.values():
        if root and (root.resolve() == output or root.resolve() in output.parents):
            parser.error("Retrieval report output must be outside dataset roots.")

    ranking = pd.read_csv(args.ranking_csv)
    queries = pd.read_csv(args.query_csv, dtype=str, keep_default_na=False)
    gallery = pd.read_csv(args.gallery_csv, dtype=str, keep_default_na=False)
    query_by_id = queries.set_index("image_id").to_dict("index")
    gallery_by_id = gallery.set_index("image_id").to_dict("index")
    sections = []
    for query_id, ranked in ranking.groupby("query_image_id", sort=True):
        if query_id not in query_by_id:
            raise ValueError(f"Ranking refers to unknown query image ID: {query_id}")
        query_row = query_by_id[query_id]
        try:
            query_uri = _local_image_uri(query_row, roots)
        except (ValueError, FileNotFoundError) as error:
            raise ValueError(f"Cannot resolve query image {query_id}: {error}") from error
        matches = []
        for item in ranked.sort_values("rank").head(args.top_k).to_dict("records"):
            gallery_id = item["gallery_image_id"]
            if gallery_id not in gallery_by_id:
                raise ValueError(f"Ranking refers to unknown gallery image ID: {gallery_id}")
            gallery_row = gallery_by_id[gallery_id]
            gallery_uri = _local_image_uri(gallery_row, roots)
            design_status = "same design" if str(item["same_design"]) == "1" else "different design"
            color_status = str(item["color_relationship"])
            matches.append(
                "<figure>"
                f'<img loading="lazy" src="{html.escape(gallery_uri, quote=True)}" '
                f'alt="{html.escape(gallery_row.get("filename", gallery_id), quote=True)}">'
                f"<figcaption>Rank {int(item['rank'])} | cosine {float(item['similarity']):.4f}<br>"
                f"{html.escape(design_status)} | colorway {html.escape(color_status)}<br>"
                f"{html.escape(gallery_row.get('filename', gallery_id))}</figcaption></figure>"
            )
        sections.append(
            "<section><h2>Query "
            f"{html.escape(query_id)} | design {html.escape(query_row['design_id'])} | "
            f"colorway {html.escape(query_row.get('colorway', ''))}</h2>"
            "<div class=\"query\"><figure>"
            f'<img src="{html.escape(query_uri, quote=True)}" alt="query image">'
            f"<figcaption>Query | {html.escape(query_row.get('filename', query_id))}</figcaption></figure>"
            f"<div class=\"matches\">{''.join(matches)}</div></div></section>"
        )
    document = (
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" "
        "content=\"width=device-width,initial-scale=1\"><title>Local retrieval report</title><style>"
        "body{font:14px system-ui,sans-serif;margin:24px;color:#202020}section{border-top:1px solid #aaa;padding:16px 0}"
        ".query{display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:16px}"
        ".matches{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}"
        "figure{margin:0;border:1px solid #bbb;padding:8px}img{width:100%;height:190px;object-fit:contain;background:#f3f3f3}"
        "figcaption{margin-top:8px;overflow-wrap:anywhere}@media(max-width:650px){.query{grid-template-columns:1fr}}"
        "</style><h1>Local retrieval report</h1><p>Contains local proprietary image references; do not publish or commit.</p>"
        f"{''.join(sections)}</html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"Wrote local retrieval report: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error