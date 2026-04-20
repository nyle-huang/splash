#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from product_campaign_pipeline.review_batch import render_review_board, sanitize_review_rows_for_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs"

DEFAULT_PRECEDENCE = [
    "human_review_batch_v33_final_bundle",
    "generalization_diverse_v6_approved_bundle",
    "generalization_diverse_v2_new_categories_final_bundle",
    "generalization_diverse_v2_targeted_rootfix_v30_curated_final_bundle",
    "generalization_diverse_v3_targeted_repair_v5_final_bundle",
    "generalization_diverse_v4_approved_bundle",
    "generalization_diverse_v5_approved_bundle",
    "generalization_diverse_v5_targeted_rootfix_v10_curated",
    "generalization_diverse_v7_curated_v7",
    "generalization_diverse_v8_curated_v2",
]

DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "final_benchmark_candidate_v1"
DEFAULT_EXCLUSION_SOURCES = [
    OUTPUT_ROOT / "generalization_diverse_v7_curated_v7" / "reports" / "invalid_sources.json",
    OUTPUT_ROOT / "generalization_diverse_v8_curated_v2" / "reports" / "invalid_sources.json",
    OUTPUT_ROOT / "generalization_diverse_v8_curated_v2" / "reports" / "held_rows.json",
    OUTPUT_ROOT / "generalization_diverse_v9_source_screen_upstream_v1" / "reports" / "invalid_sources.json",
]


def _load_rows(bundle_name: str) -> list[dict[str, Any]]:
    path = OUTPUT_ROOT / bundle_name / "reports" / "generation_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _find_board_asset(bundle_dir: Path, asset_group: str, asset_stem: str) -> str | None:
    asset_dir = bundle_dir / "board_assets" / asset_group
    if not asset_dir.exists():
        return None
    candidates = sorted(asset_dir.glob(f"{asset_stem}.*"))
    if not candidates:
        return None
    return str(candidates[0])


def _repo_remap_if_possible(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    if path.exists():
        return str(path)
    needle = "/product_campaign_pipeline/"
    text = str(value)
    if needle in text:
        suffix = text.split(needle, 1)[1]
        candidate = REPO_ROOT / suffix
        if candidate.exists():
            return str(candidate)
    candidate = REPO_ROOT / text
    if candidate.exists():
        return str(candidate)
    return value


def _normalize_row(bundle_name: str, row: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = OUTPUT_ROOT / bundle_name
    seed_id = str(row["id"])
    line_name = str(row["line"])
    normalized = dict(row)

    source_asset = _find_board_asset(bundle_dir, "source", f"{seed_id}.source")
    crop_asset = _find_board_asset(bundle_dir, "crop", f"{seed_id}.crop")
    generated_asset = _find_board_asset(bundle_dir, "generated", f"{seed_id}.{line_name}")

    normalized["source_image_path"] = source_asset or _repo_remap_if_possible(row.get("source_image_path"))
    normalized["crop_path"] = crop_asset or _repo_remap_if_possible(row.get("crop_path"))
    normalized["output_path"] = generated_asset or _repo_remap_if_possible(row.get("output_path"))
    normalized["mask_path"] = _repo_remap_if_possible(row.get("mask_path"))
    normalized["benchmark_source_bundle"] = bundle_name

    observed_evidence = row.get("observed_evidence")
    if isinstance(observed_evidence, dict):
        normalized_observed = dict(observed_evidence)
        for field_name, asset_group, asset_stem in (
            ("reference_crop_path", "reference", f"{seed_id}.evidence_crop"),
            ("reference_cutout_path", "reference", f"{seed_id}.evidence_cutout"),
            ("reference_silhouette_path", "reference", f"{seed_id}.evidence_silhouette"),
            ("reference_mask_path", "reference", f"{seed_id}.evidence_mask"),
        ):
            normalized_observed[field_name] = _find_board_asset(bundle_dir, asset_group, asset_stem) or _repo_remap_if_possible(
                observed_evidence.get(field_name)
            )
        normalized["observed_evidence"] = normalized_observed

    candidate_prompts = row.get("candidate_prompts")
    if isinstance(candidate_prompts, list):
        normalized_candidates: list[dict[str, Any]] = []
        for candidate in candidate_prompts:
            normalized_candidate = dict(candidate)
            prompt_payload = dict(candidate.get("prompt", {}))
            normalized_references: list[dict[str, Any]] = []
            for index, reference in enumerate(prompt_payload.get("reference_images", ())):
                normalized_reference = dict(reference)
                candidate_asset = _find_board_asset(
                    bundle_dir,
                    "reference",
                    f"{seed_id}.{line_name}.{candidate.get('mode', 'candidate')}.reference_{index}",
                )
                normalized_reference["path"] = candidate_asset or _repo_remap_if_possible(reference.get("path"))
                normalized_references.append(normalized_reference)
            prompt_payload["reference_images"] = normalized_references
            normalized_candidate["prompt"] = prompt_payload
            normalized_candidates.append(normalized_candidate)
        normalized["candidate_prompts"] = normalized_candidates

    return normalized


def _load_exclusion_payload(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized_rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                normalized = dict(item)
                normalized["source_report"] = str(path)
                normalized_rows.append(normalized)
    return normalized_rows


def _collect_exclusion_registry(
    precedence: list[str],
    *,
    extra_reports: list[Path] | None = None,
) -> list[dict[str, Any]]:
    registry: list[dict[str, Any]] = []
    seen_reports: set[Path] = set()
    for path in DEFAULT_EXCLUSION_SOURCES:
        if path in seen_reports:
            continue
        seen_reports.add(path)
        registry.extend(_load_exclusion_payload(path))
    for path in extra_reports or []:
        if path in seen_reports:
            continue
        seen_reports.add(path)
        registry.extend(_load_exclusion_payload(path))
    for bundle_name in precedence:
        reports_dir = OUTPUT_ROOT / bundle_name / "reports"
        for filename in ("invalid_sources.json", "held_rows.json"):
            candidate = reports_dir / filename
            if candidate in seen_reports:
                continue
            seen_reports.add(candidate)
            registry.extend(_load_exclusion_payload(candidate))
    return registry


def _category_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    category = str(row.get("expected_category") or row.get("category") or "product")
    return (category, str(row["id"]))


def build_bundle(
    *,
    precedence: list[str],
    output_dir: Path,
    extra_exclusion_reports: list[Path] | None = None,
) -> dict[str, Any]:
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    source_bundle_by_id: dict[str, str] = {}
    excluded_registry = _collect_exclusion_registry(precedence, extra_reports=extra_exclusion_reports)
    excluded_ids = {
        str(item.get("id"))
        for item in excluded_registry
        if str(item.get("id", "")).strip()
    }

    for bundle_name in precedence:
        rows = _load_rows(bundle_name)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["id"]), []).append(_normalize_row(bundle_name, row))
        for seed_id, seed_rows in grouped.items():
            if seed_id in rows_by_id:
                continue
            rows_by_id[seed_id] = seed_rows
            source_bundle_by_id[seed_id] = bundle_name

    selected_rows: list[dict[str, Any]] = []
    for seed_id in sorted(rows_by_id):
        if seed_id in excluded_ids:
            continue
        seed_rows = rows_by_id[seed_id]
        line_names = {str(row["line"]) for row in seed_rows}
        if line_names != {"baseline", "business_prior"}:
            raise ValueError(f"{seed_id} does not contain both lines in selected source bundle.")
        selected_rows.extend(sorted(seed_rows, key=lambda row: str(row["line"])))

    selected_rows.sort(key=_category_sort_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_rows = sanitize_review_rows_for_bundle(selected_rows, output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    board_path = render_review_board(sanitized_rows, output_dir / "human_review_board.html")

    source_bundle_counts = Counter(
        bundle_name for seed_id, bundle_name in source_bundle_by_id.items() if seed_id not in excluded_ids
    )
    category_counts = Counter(
        str((rows_by_id[seed_id][0].get("expected_category") or rows_by_id[seed_id][0].get("category") or "product"))
        for seed_id in rows_by_id
        if seed_id not in excluded_ids
    )
    summary = {
        "id_count": len([seed_id for seed_id in rows_by_id if seed_id not in excluded_ids]),
        "row_count": len(sanitized_rows),
        "source_bundles_precedence": precedence,
        "resolved_source_bundle_by_id": dict(
            sorted((seed_id, bundle_name) for seed_id, bundle_name in source_bundle_by_id.items() if seed_id not in excluded_ids)
        ),
        "source_bundle_counts": dict(sorted(source_bundle_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "excluded_or_held_count": len(excluded_registry),
        "excluded_or_held_ids": sorted(excluded_ids),
        "board_path": str(board_path),
    }

    (reports_dir / "generation_report.json").write_text(
        json.dumps(sanitized_rows, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (reports_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (reports_dir / "excluded_or_held_sources.json").write_text(
        json.dumps(excluded_registry, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bundle", action="append", dest="bundles", default=[])
    parser.add_argument("--exclusion-report", action="append", dest="exclusion_reports", default=[])
    args = parser.parse_args()

    precedence = args.bundles or list(DEFAULT_PRECEDENCE)
    summary = build_bundle(
        precedence=precedence,
        output_dir=Path(args.output_dir),
        extra_exclusion_reports=[Path(path) for path in args.exclusion_reports],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
