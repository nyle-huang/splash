from pathlib import Path

from product_campaign_pipeline.data.manifests import aggregate_rows_by_item, iter_manifest_rows, summarize_manifest


def test_iter_manifest_rows_and_summary(tmp_path: Path) -> None:
    manifest = tmp_path / "train_data_list.txt"
    manifest.write_text(
        "item-1\timg-a.png\t0\t10\t2\n"
        "item-1\timg-b.png\t1\t7\t1\n"
        "item-2\timg-c.png\t0\t5\t0\n",
        encoding="utf-8",
    )

    rows = list(iter_manifest_rows(manifest))
    assert len(rows) == 3
    grouped = aggregate_rows_by_item(rows)
    assert sorted(grouped) == ["item-1", "item-2"]

    summary = summarize_manifest(manifest)
    assert summary.row_count == 3
    assert summary.item_count == 2
    assert summary.total_pv == 22
    assert summary.total_clk == 3
