from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module():
    script_path = Path("/workspace/product_campaign_pipeline/scripts/build_final_benchmark_bundle.py")
    spec = importlib.util.spec_from_file_location("build_final_benchmark_bundle", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_generation_report(bundle_dir: Path, *, seed_id: str, marker: str) -> None:
    reports_dir = bundle_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": seed_id,
            "line": "baseline",
            "expected_category": "bag",
            "source_image_path": f"{marker}_source.png",
            "crop_path": f"{marker}_crop.png",
            "output_path": f"{marker}_baseline.png",
        },
        {
            "id": seed_id,
            "line": "business_prior",
            "expected_category": "bag",
            "source_image_path": f"{marker}_source.png",
            "crop_path": f"{marker}_crop.png",
            "output_path": f"{marker}_business_prior.png",
        },
    ]
    (reports_dir / "generation_report.json").write_text(json.dumps(rows), encoding="utf-8")


def test_build_bundle_keeps_first_bundle_precedence(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    output_root = tmp_path / "outputs"
    earlier = output_root / "earlier_bundle"
    later = output_root / "later_bundle"
    _write_generation_report(earlier, seed_id="shared_id", marker="earlier")
    _write_generation_report(later, seed_id="shared_id", marker="later")

    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(module, "DEFAULT_EXCLUSION_SOURCES", [])
    monkeypatch.setattr(module, "sanitize_review_rows_for_bundle", lambda rows, output_dir: rows)

    def _render(rows, board_path: Path) -> Path:
        board_path.write_text("<html></html>", encoding="utf-8")
        return board_path

    monkeypatch.setattr(module, "render_review_board", _render)

    summary = module.build_bundle(
        precedence=["earlier_bundle", "later_bundle"],
        output_dir=tmp_path / "merged",
    )

    assert summary["resolved_source_bundle_by_id"]["shared_id"] == "earlier_bundle"

    merged_rows = json.loads((tmp_path / "merged" / "reports" / "generation_report.json").read_text(encoding="utf-8"))
    assert {row["benchmark_source_bundle"] for row in merged_rows} == {"earlier_bundle"}
    assert {Path(row["output_path"]).name for row in merged_rows} == {"earlier_baseline.png", "earlier_business_prior.png"}


def test_build_bundle_applies_extra_exclusion_reports(tmp_path, monkeypatch) -> None:
    module = _load_script_module()
    output_root = tmp_path / "outputs"
    bundle = output_root / "bundle"
    _write_generation_report(bundle, seed_id="excluded_id", marker="bundle")

    extra_exclusion = tmp_path / "extra_invalid_sources.json"
    extra_exclusion.write_text(
        json.dumps(
            [
                {
                    "id": "excluded_id",
                    "source_validity": "invalid",
                    "source_validity_issues": ["localized_crop_visual_type_conflict"],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(module, "DEFAULT_EXCLUSION_SOURCES", [])
    monkeypatch.setattr(module, "sanitize_review_rows_for_bundle", lambda rows, output_dir: rows)

    def _render(rows, board_path: Path) -> Path:
        board_path.write_text("<html></html>", encoding="utf-8")
        return board_path

    monkeypatch.setattr(module, "render_review_board", _render)

    summary = module.build_bundle(
        precedence=["bundle"],
        output_dir=tmp_path / "merged",
        extra_exclusion_reports=[extra_exclusion],
    )

    assert summary["id_count"] == 0
    assert "excluded_id" in summary["excluded_or_held_ids"]

    merged_rows = json.loads((tmp_path / "merged" / "reports" / "generation_report.json").read_text(encoding="utf-8"))
    assert merged_rows == []
