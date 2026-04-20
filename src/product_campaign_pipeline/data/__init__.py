"""Coherent package-level data API for manifests and synthetic bootstrap helpers."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .bootstrap import (
    PseudoHandheldBootstrapper,
    SyntheticBootstrapConfig,
    generate_bootstrap_examples,
)
from .io import (
    DEFAULT_SPLIT_FILES,
    MANIFEST_COLUMNS,
    OptionalDependencyError,
    export_manifest_rows_to_parquet,
    export_split_summaries_to_parquet,
    export_synthetic_bootstrap_to_parquet,
    group_rows_by_item,
    load_creative_ranking_corpus,
    load_manifest,
    load_split_manifests,
    parse_manifest_line,
    summarize_split,
    summarize_splits,
)
from .models import (
    CreativeRankingCorpus,
    ItemGroup,
    ManifestRow,
    ManifestSummary,
    SplitSummary,
    SyntheticBootstrapRecord,
)


def iter_manifest_rows(path: str | Path) -> Iterator[ManifestRow]:
    """Yield ManifestRow instances from a CreativeRanking-style manifest file."""

    yield from load_manifest(path)


def resolve_default_manifest_paths(
    data_root: str | Path,
    *,
    split_files: Mapping[str, str] | None = None,
    list_dir_name: str = "list",
    strict: bool = False,
) -> dict[str, Path]:
    """Find manifests under either `data_root/list` or `data_root`.

    By default missing splits are ignored so callers can operate on partial fixtures.
    Set `strict=True` to require all declared split files.
    """

    root = Path(data_root)
    resolved: dict[str, Path] = OrderedDict()
    split_map = DEFAULT_SPLIT_FILES if split_files is None else split_files
    for split_name, file_name in split_map.items():
        for candidate in (root / list_dir_name / file_name, root / file_name):
            if candidate.exists():
                resolved[split_name] = candidate
                break
        else:
            if strict:
                raise FileNotFoundError(f"Could not find split manifest {file_name!r} under {root}")
            continue
    if strict and len(resolved) != len(split_map):
        missing = sorted(set(split_map) - set(resolved))
        raise FileNotFoundError(
            f"Could not find manifest files for splits {', '.join(missing)} under {root}"
        )
    return resolved


def summarize_manifest(path: str | Path, split_name: str | None = None) -> ManifestSummary:
    """Summarize a single manifest file using the dataclass-backed row loader."""

    manifest_path = Path(path)
    rows = load_manifest(manifest_path)
    summary = summarize_split(rows, split_name or manifest_path.stem.replace("_data_list", ""))
    return ManifestSummary(
        path=str(manifest_path),
        row_count=summary.row_count,
        item_count=summary.item_count,
        unique_image_count=summary.image_count,
        total_pv=summary.total_page_views,
        total_clk=summary.total_clicks,
        mean_creatives_per_item=summary.avg_images_per_item,
        min_creatives_per_item=min((group.unique_image_count for group in group_rows_by_item(rows)), default=0),
        max_creatives_per_item=max((group.unique_image_count for group in group_rows_by_item(rows)), default=0),
    )


def export_manifest_to_parquet(path: str | Path, destination: str | Path) -> Path:
    """Load a manifest file and export it as parquet via the package-level exporter."""

    manifest_path = Path(path)
    return export_manifest_rows_to_parquet(load_manifest(manifest_path), destination)


__all__ = [
    "CreativeRankingCorpus",
    "DEFAULT_SPLIT_FILES",
    "ItemGroup",
    "MANIFEST_COLUMNS",
    "ManifestRow",
    "ManifestSummary",
    "OptionalDependencyError",
    "PseudoHandheldBootstrapper",
    "SplitSummary",
    "SyntheticBootstrapConfig",
    "SyntheticBootstrapRecord",
    "export_manifest_rows_to_parquet",
    "export_manifest_to_parquet",
    "export_split_summaries_to_parquet",
    "export_synthetic_bootstrap_to_parquet",
    "generate_bootstrap_examples",
    "group_rows_by_item",
    "iter_manifest_rows",
    "load_creative_ranking_corpus",
    "load_manifest",
    "load_split_manifests",
    "parse_manifest_line",
    "resolve_default_manifest_paths",
    "summarize_manifest",
    "summarize_split",
    "summarize_splits",
]
