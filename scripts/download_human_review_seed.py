#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "human_review_seed" / "review_seed_manifest.json"
REPORT_PATH = REPO_ROOT / "data" / "human_review_seed" / "download_report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    report_path = Path(args.report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report: list[dict[str, object]] = []

    for record in manifest:
        target_path = REPO_ROOT / str(record["local_image_path"])
        target_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(str(record["source_image_url"]), timeout=60)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(target_path)

        report.append(
            {
                "id": record["id"],
                "local_image_path": str(target_path),
                "width": image.width,
                "height": image.height,
                "platform": record["platform"],
                "source_page_url": record["source_page_url"],
                "source_image_url": record["source_image_url"],
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"downloaded": len(report), "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
