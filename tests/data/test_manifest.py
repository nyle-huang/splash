from __future__ import annotations

from pathlib import Path

from product_campaign_pipeline.data import (
    ManifestRow,
    export_manifest_to_parquet,
    iter_manifest_rows,
    parse_manifest_line,
    resolve_default_manifest_paths,
    summarize_manifest,
)


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "item-1\timg-a.png\t0\t10\t2\n"
        "item-1\timg-b.png\t1\t7\t1\n"
        "item-2\timg-c.png\t0\t5\t0\n",
        encoding="utf-8",
    )


def test_package_level_manifest_exports_are_coherent(tmp_path: Path) -> None:
    data_root = tmp_path / "corpus"
    manifest_path = data_root / "list" / "train_data_list.txt"
    _write_manifest(manifest_path)

    paths = resolve_default_manifest_paths(data_root)
    rows = list(iter_manifest_rows(paths["train"]))
    summary = summarize_manifest(paths["train"])

    assert paths["train"] == manifest_path
    assert rows[0] == ManifestRow(item_id="item-1", image_name="img-a.png", ds=0, pv=10, clk=2)
    assert summary.row_count == 3
    assert summary.item_count == 2
    assert summary.total_pv == 22
    assert summary.total_clk == 3


def test_parse_manifest_line_round_trips_to_manifest_row() -> None:
    row = parse_manifest_line("sku-1\tcreative.png\t3\t20\t5")
    assert row == ManifestRow(item_id="sku-1", image_name="creative.png", ds=3, pv=20, clk=5)


def test_export_manifest_to_parquet_requires_optional_dependencies(tmp_path: Path) -> None:
    pandas = __import__("importlib").import_module
    if False:  # pragma: no cover - keeps linters from folding the import away
        pandas

    manifest_path = tmp_path / "train_data_list.txt"
    _write_manifest(manifest_path)

    try:
        __import__("pandas")
        __import__("pyarrow")
    except ImportError:
        try:
            export_manifest_to_parquet(manifest_path, tmp_path / "rows.parquet")
        except Exception as exc:  # noqa: BLE001
            assert "pandas and pyarrow" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected parquet export to fail without optional dependencies")
