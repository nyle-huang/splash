from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Single CreativeRanking-style manifest row."""

    item_id: str
    image_name: str
    ds: int
    pv: int
    clk: int

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("item_id must be non-empty")
        if not self.image_name.strip():
            raise ValueError("image_name must be non-empty")

        for field_name, value in (("ds", self.ds), ("pv", self.pv), ("clk", self.clk)):
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

    @property
    def ctr(self) -> float:
        return 0.0 if self.pv == 0 else self.clk / self.pv

    def image_path(self, image_root: Path) -> Path:
        return image_root / self.image_name

    def as_dict(self, *, image_root: Path | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "item_id": self.item_id,
            "image_name": self.image_name,
            "ds": self.ds,
            "pv": self.pv,
            "clk": self.clk,
            "ctr": self.ctr,
        }
        if image_root is not None:
            record["image_path"] = str(self.image_path(image_root))
        return record


@dataclass(frozen=True, slots=True)
class ItemGroup:
    """Manifest rows grouped by product item."""

    item_id: str
    rows: tuple[ManifestRow, ...]

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("rows must be non-empty")

        mismatched = [row.item_id for row in self.rows if row.item_id != self.item_id]
        if mismatched:
            raise ValueError("all rows in a group must share the same item_id")

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(row.image_name for row in self.rows))

    @property
    def ds_values(self) -> tuple[int, ...]:
        return tuple(sorted({row.ds for row in self.rows}))

    @property
    def unique_image_count(self) -> int:
        return len(self.image_names)

    @property
    def total_page_views(self) -> int:
        return sum(row.pv for row in self.rows)

    @property
    def total_clicks(self) -> int:
        return sum(row.clk for row in self.rows)

    @property
    def mean_ctr(self) -> float:
        return sum(row.ctr for row in self.rows) / len(self.rows)

    @property
    def weighted_ctr(self) -> float:
        return 0.0 if self.total_page_views == 0 else self.total_clicks / self.total_page_views

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "row_count": self.row_count,
            "unique_image_count": self.unique_image_count,
            "ds_count": len(self.ds_values),
            "total_page_views": self.total_page_views,
            "total_clicks": self.total_clicks,
            "mean_ctr": self.mean_ctr,
            "weighted_ctr": self.weighted_ctr,
        }


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Aggregate statistics for a data split."""

    split_name: str
    row_count: int
    item_count: int
    image_count: int
    ds_count: int
    total_page_views: int
    total_clicks: int
    mean_ctr: float
    weighted_ctr: float
    avg_rows_per_item: float
    avg_images_per_item: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "row_count": self.row_count,
            "item_count": self.item_count,
            "image_count": self.image_count,
            "ds_count": self.ds_count,
            "total_page_views": self.total_page_views,
            "total_clicks": self.total_clicks,
            "mean_ctr": self.mean_ctr,
            "weighted_ctr": self.weighted_ctr,
            "avg_rows_per_item": self.avg_rows_per_item,
            "avg_images_per_item": self.avg_images_per_item,
        }


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    """Aggregate statistics for a single manifest file."""

    path: str
    row_count: int
    item_count: int
    unique_image_count: int
    total_pv: int
    total_clk: int
    mean_creatives_per_item: float
    min_creatives_per_item: int
    max_creatives_per_item: int

    @property
    def global_ctr(self) -> float:
        return 0.0 if self.total_pv == 0 else self.total_clk / self.total_pv

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "row_count": self.row_count,
            "item_count": self.item_count,
            "unique_image_count": self.unique_image_count,
            "total_pv": self.total_pv,
            "total_clk": self.total_clk,
            "mean_creatives_per_item": self.mean_creatives_per_item,
            "min_creatives_per_item": self.min_creatives_per_item,
            "max_creatives_per_item": self.max_creatives_per_item,
            "global_ctr": self.global_ctr,
        }


@dataclass(frozen=True, slots=True)
class SyntheticBootstrapRecord:
    """Metadata emitted for a synthetic pseudo-handheld image."""

    source_item_id: str
    source_image_name: str
    synthetic_image_name: str
    synthetic_image_path: Path
    split_name: str
    variant_index: int
    seed: int
    output_width: int
    output_height: int
    rotation_degrees: float
    scale: float
    perspective_jitter: float
    blur_radius: float
    brightness: float
    contrast: float
    background_r: int
    background_g: int
    background_b: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_item_id": self.source_item_id,
            "source_image_name": self.source_image_name,
            "synthetic_image_name": self.synthetic_image_name,
            "synthetic_image_path": str(self.synthetic_image_path),
            "split_name": self.split_name,
            "variant_index": self.variant_index,
            "seed": self.seed,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "rotation_degrees": self.rotation_degrees,
            "scale": self.scale,
            "perspective_jitter": self.perspective_jitter,
            "blur_radius": self.blur_radius,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "background_r": self.background_r,
            "background_g": self.background_g,
            "background_b": self.background_b,
        }


@dataclass(frozen=True, slots=True)
class CreativeRankingCorpus:
    """Loaded corpus rooted at a data directory."""

    root: Path
    image_root: Path
    manifest_paths: Mapping[str, Path]
    splits: Mapping[str, tuple[ManifestRow, ...]]
