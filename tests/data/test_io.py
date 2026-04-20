from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import product_campaign_pipeline.data.io as data_io
from product_campaign_pipeline.data import (
    ManifestRow,
    OptionalDependencyError,
    export_manifest_rows_to_parquet,
    export_split_summaries_to_parquet,
    group_rows_by_item,
    load_creative_ranking_corpus,
    parse_manifest_line,
    summarize_split,
    summarize_splits,
)


def _write_manifest(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_manifest_line_parses_expected_columns() -> None:
    row = parse_manifest_line("item-1\tcreative.png\t3\t20\t5")

    assert row == ManifestRow(item_id="item-1", image_name="creative.png", ds=3, pv=20, clk=5)
    assert row.ctr == pytest.approx(0.25)


def test_load_creative_ranking_corpus_groups_and_summarizes(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "images").mkdir(parents=True)
    for image_name in ("a.png", "b.png", "c.png"):
        (root / "images" / image_name).write_bytes(b"")

    _write_manifest(
        root / "list" / "train_data_list.txt",
        [
            "sku-1\ta.png\t0\t10\t1",
            "sku-1\tb.png\t1\t20\t5",
            "sku-2\tc.png\t0\t5\t0",
        ],
    )
    _write_manifest(root / "list" / "val_data_list.txt", ["sku-3\ta.png\t0\t3\t1"])
    _write_manifest(root / "test_data_list.txt", ["sku-4\tb.png\t0\t8\t2"])

    corpus = load_creative_ranking_corpus(root)

    assert corpus.image_root == root / "images"
    assert tuple(corpus.splits) == ("train", "val", "test")

    groups = group_rows_by_item(corpus.splits["train"])
    assert len(groups) == 2
    assert groups[0].item_id == "sku-1"
    assert groups[0].unique_image_count == 2
    assert groups[0].weighted_ctr == pytest.approx(6 / 30)

    train_summary = summarize_split(corpus.splits["train"], "train")
    assert train_summary.row_count == 3
    assert train_summary.item_count == 2
    assert train_summary.image_count == 3
    assert train_summary.total_page_views == 35
    assert train_summary.total_clicks == 6
    assert train_summary.weighted_ctr == pytest.approx(6 / 35)

    all_summaries = summarize_splits(corpus.splits)
    assert all_summaries["test"].weighted_ctr == pytest.approx(0.25)


def test_export_manifest_rows_to_parquet_raises_clear_error_without_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name in {"pandas", "pyarrow"}:
            raise ImportError(f"missing {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(data_io.importlib, "import_module", fake_import_module)

    with pytest.raises(OptionalDependencyError, match="Install pandas and pyarrow"):
        export_manifest_rows_to_parquet(
            [ManifestRow(item_id="sku-1", image_name="creative.png", ds=0, pv=1, clk=0)],
            tmp_path / "rows.parquet",
        )


def test_export_split_summaries_to_parquet_writes_file_when_dependencies_exist(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    summary = summarize_split(
        [
            ManifestRow(item_id="sku-1", image_name="creative.png", ds=0, pv=4, clk=1),
            ManifestRow(item_id="sku-1", image_name="detail.png", ds=1, pv=6, clk=2),
        ],
        "train",
    )

    parquet_path = export_split_summaries_to_parquet([summary], tmp_path / "summaries.parquet")

    frame = pandas.read_parquet(parquet_path)
    assert parquet_path.exists()
    assert frame.loc[0, "split_name"] == "train"
    assert frame.loc[0, "total_clicks"] == 3
