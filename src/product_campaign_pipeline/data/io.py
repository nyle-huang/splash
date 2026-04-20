from __future__ import annotations

import csv
import importlib
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .models import (
    CreativeRankingCorpus,
    ItemGroup,
    ManifestRow,
    SplitSummary,
    SyntheticBootstrapRecord,
)

MANIFEST_COLUMNS = ("item_id", "image_name", "ds", "pv", "clk")
DEFAULT_SPLIT_FILES: Mapping[str, str] = OrderedDict(
    (("train", "train_data_list.txt"), ("val", "val_data_list.txt"), ("test", "test_data_list.txt"))
)


class OptionalDependencyError(ImportError):
    """Raised when an optional runtime dependency is not installed."""


def _parse_manifest_fields(
    fields: Sequence[str],
    *,
    line_number: int | None = None,
    source: Path | None = None,
) -> ManifestRow:
    if len(fields) != len(MANIFEST_COLUMNS):
        location = ""
        if source is not None and line_number is not None:
            location = f" in {source}:{line_number}"
        elif line_number is not None:
            location = f" at line {line_number}"

        raise ValueError(
            f"Expected {len(MANIFEST_COLUMNS)} tab-separated columns{location}; got {len(fields)}"
        )

    item_id, image_name, ds, pv, clk = (field.strip() for field in fields)
    try:
        return ManifestRow(item_id=item_id, image_name=image_name, ds=int(ds), pv=int(pv), clk=int(clk))
    except ValueError as exc:
        location = ""
        if source is not None and line_number is not None:
            location = f" in {source}:{line_number}"
        elif line_number is not None:
            location = f" at line {line_number}"
        raise ValueError(f"Invalid manifest row{location}: {fields}") from exc


def parse_manifest_line(line: str, *, line_number: int | None = None) -> ManifestRow:
    return _parse_manifest_fields(line.rstrip("\n").split("\t"), line_number=line_number)


def load_manifest(path: str | Path) -> tuple[ManifestRow, ...]:
    manifest_path = Path(path)
    rows: list[ManifestRow] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, fields in enumerate(reader, start=1):
            if not fields or not any(field.strip() for field in fields):
                continue
            if line_number == 1 and tuple(field.strip() for field in fields) == MANIFEST_COLUMNS:
                continue
            rows.append(_parse_manifest_fields(fields, line_number=line_number, source=manifest_path))

    return tuple(rows)


def _resolve_manifest_path(root: Path, file_name: str, *, list_dir_name: str) -> Path:
    candidates = (root / list_dir_name / file_name, root / file_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find split manifest {file_name!r} under {root}")


def load_split_manifests(
    root: str | Path,
    *,
    split_files: Mapping[str, str] | None = None,
    list_dir_name: str = "list",
) -> tuple[dict[str, Path], dict[str, tuple[ManifestRow, ...]]]:
    root_path = Path(root)
    split_file_map = DEFAULT_SPLIT_FILES if split_files is None else split_files
    manifest_paths: dict[str, Path] = {}
    splits: dict[str, tuple[ManifestRow, ...]] = {}

    for split_name, file_name in split_file_map.items():
        manifest_path = _resolve_manifest_path(root_path, file_name, list_dir_name=list_dir_name)
        manifest_paths[split_name] = manifest_path
        splits[split_name] = load_manifest(manifest_path)

    return manifest_paths, splits


def load_creative_ranking_corpus(
    root: str | Path,
    *,
    split_files: Mapping[str, str] | None = None,
    image_dir_name: str = "images",
    list_dir_name: str = "list",
) -> CreativeRankingCorpus:
    root_path = Path(root)
    manifest_paths, splits = load_split_manifests(
        root_path, split_files=split_files, list_dir_name=list_dir_name
    )
    return CreativeRankingCorpus(
        root=root_path,
        image_root=root_path / image_dir_name,
        manifest_paths=manifest_paths,
        splits=splits,
    )


def group_rows_by_item(rows: Iterable[ManifestRow]) -> tuple[ItemGroup, ...]:
    grouped: dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.item_id, []).append(row)
    return tuple(ItemGroup(item_id=item_id, rows=tuple(item_rows)) for item_id, item_rows in grouped.items())


def summarize_split(rows: Sequence[ManifestRow], split_name: str) -> SplitSummary:
    groups = group_rows_by_item(rows)
    row_count = len(rows)
    item_count = len(groups)
    image_count = len({row.image_name for row in rows})
    ds_count = len({row.ds for row in rows})
    total_page_views = sum(row.pv for row in rows)
    total_clicks = sum(row.clk for row in rows)
    mean_ctr = 0.0 if row_count == 0 else sum(row.ctr for row in rows) / row_count
    weighted_ctr = 0.0 if total_page_views == 0 else total_clicks / total_page_views
    avg_rows_per_item = 0.0 if item_count == 0 else row_count / item_count
    avg_images_per_item = (
        0.0 if item_count == 0 else sum(group.unique_image_count for group in groups) / item_count
    )

    return SplitSummary(
        split_name=split_name,
        row_count=row_count,
        item_count=item_count,
        image_count=image_count,
        ds_count=ds_count,
        total_page_views=total_page_views,
        total_clicks=total_clicks,
        mean_ctr=mean_ctr,
        weighted_ctr=weighted_ctr,
        avg_rows_per_item=avg_rows_per_item,
        avg_images_per_item=avg_images_per_item,
    )


def summarize_splits(splits: Mapping[str, Sequence[ManifestRow]]) -> dict[str, SplitSummary]:
    return {split_name: summarize_split(rows, split_name) for split_name, rows in splits.items()}


def _require_parquet_dependencies() -> object:
    try:
        pandas = importlib.import_module("pandas")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Parquet export requires optional dependency 'pandas'. Install pandas and pyarrow."
        ) from exc

    try:
        importlib.import_module("pyarrow")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Parquet export requires optional dependency 'pyarrow'. Install pandas and pyarrow."
        ) from exc

    return pandas


def export_manifest_rows_to_parquet(
    rows: Sequence[ManifestRow],
    destination: str | Path,
    *,
    image_root: Path | None = None,
) -> Path:
    pandas = _require_parquet_dependencies()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pandas.DataFrame.from_records([row.as_dict(image_root=image_root) for row in rows])
    frame.to_parquet(destination_path, index=False, engine="pyarrow")
    return destination_path


def export_split_summaries_to_parquet(
    summaries: Mapping[str, SplitSummary] | Sequence[SplitSummary],
    destination: str | Path,
) -> Path:
    pandas = _require_parquet_dependencies()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(summaries, Mapping):
        records = [summary.as_dict() for summary in summaries.values()]
    else:
        records = [summary.as_dict() for summary in summaries]

    frame = pandas.DataFrame.from_records(records)
    frame.to_parquet(destination_path, index=False, engine="pyarrow")
    return destination_path


def export_synthetic_bootstrap_to_parquet(
    records: Sequence[SyntheticBootstrapRecord],
    destination: str | Path,
) -> Path:
    pandas = _require_parquet_dependencies()
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pandas.DataFrame.from_records([record.as_dict() for record in records])
    frame.to_parquet(destination_path, index=False, engine="pyarrow")
    return destination_path

