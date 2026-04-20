#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from product_campaign_pipeline.localization import (
    build_model_backed_localization_pipeline,
    save_localization_artifacts,
    select_primary_mask,
)
from product_campaign_pipeline.localization.models import ProductPhoto


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "human_review_seed" / "review_seed_manifest.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "human_review_seed_localization"
REPORT_PATH = OUTPUT_DIR / "localization_report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_model_backed_localization_pipeline(device=str(args.device))
    report: list[dict[str, object]] = []

    for record in manifest:
        photo = ProductPhoto(
            image_path=REPO_ROOT / str(record["local_image_path"]),
            product_id=str(record["id"]),
            title=str(record["product_title"]),
            hint_phrases=tuple(str(item) for item in record.get("hint_phrases", ())),
        )
        result = pipeline.localize(photo)
        selected = select_primary_mask(result)
        artifacts = save_localization_artifacts(
            result,
            output_dir,
            selected_mask=selected,
        )

        report.append(
            {
                "id": record["id"],
                "product_title": record["product_title"],
                "source_page_url": record["source_page_url"],
                "source_image_url": record["source_image_url"],
                "local_image_path": str(photo.image_path),
                "selected_phrase": None if selected is None else selected.phrase.text,
                "selected_confidence": None if selected is None else selected.confidence,
                "selected_box": None
                if selected is None
                else {
                    "x0": selected.box.x0,
                    "y0": selected.box.y0,
                    "x1": selected.box.x1,
                    "y1": selected.box.y1,
                },
                "overlay_path": None if artifacts is None else artifacts.overlay_path,
                "crop_path": None if artifacts is None else artifacts.crop_path,
                "mask_path": None if artifacts is None else artifacts.mask_path,
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"localized": len(report), "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
