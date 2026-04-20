from __future__ import annotations

import csv
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .models import ManifestRow, ManifestSummary


def resolve_default_manifest_paths(data_root: Path) -> dict[str, Path]:
    """Find train/val/test manifest files in a data root."""

    candidates = [
        data_root / "list",
        data_root,
    ]
    manifests: dict[str, Path] = {}
    for directory in candidates:
        for split in ("train", "val", "test"):
            candidate = directory / f"{split}_data_list.txt"
            if candidate.exists():
                manifests[split] = candidate
        if manifests:
            break
    if not manifests:
        raise FileNotFoundError(f"No CreativeRanking-style manifests found under {data_root}")
    return manifests


def iter_manifest_rows(path: Path) -> Iterator[CreativeRankingRow]:
    """Yield manifest rows from a TSV file."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 5:
                raise ValueError(f"{path}:{line_number} expected 5 columns, found {len(row)}")
            yield ManifestRow(
                item_id=row[0],
                image_name=row[1],
                ds=int(row[2]),
                pv=int(row[3]),
                clk=int(row[4]),
            )


def aggregate_rows_by_item(rows: Iterable[ManifestRow]) -> dict[str, list[ManifestRow]]:
    """Group manifest rows by item id."""

    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[row.item_id].append(row)
    return dict(grouped)


def summarize_manifest(path: Path) -> ManifestSummary:
    """Compute aggregate stats for a manifest file."""

    item_to_images: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    total_pv = 0
    total_clk = 0
    unique_images: set[str] = set()

    for row in iter_manifest_rows(path):
        row_count += 1
        total_pv += row.pv
        total_clk += row.clk
        item_to_images[row.item_id].add(row.image_name)
        unique_images.add(row.image_name)

    creatives_per_item = [len(images) for images in item_to_images.values()]
    return ManifestSummary(
        path=str(path),
        row_count=row_count,
        item_count=len(item_to_images),
        unique_image_count=len(unique_images),
        total_pv=total_pv,
        total_clk=total_clk,
        mean_creatives_per_item=sum(creatives_per_item) / max(1, len(creatives_per_item)),
        min_creatives_per_item=min(creatives_per_item) if creatives_per_item else 0,
        max_creatives_per_item=max(creatives_per_item) if creatives_per_item else 0,
    )


def export_manifest_to_parquet(path: Path, output_path: Path) -> None:
    """Export a manifest TSV to parquet."""

    try:
        pandas = importlib.import_module("pandas")
        importlib.import_module("pyarrow")
    except ImportError as exc:  # pragma: no cover - import error exercised in runtime only
        raise RuntimeError("Install pandas and pyarrow to export parquet manifests") from exc

    rows = [row.as_dict() for row in iter_manifest_rows(path)]
    frame = pandas.DataFrame.from_records(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
