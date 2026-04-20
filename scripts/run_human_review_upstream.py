#!/usr/bin/env python3
from __future__ import annotations

import argparse

from product_campaign_pipeline.review_batch import generate_upstream_review_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--localization-report", required=True)
    parser.add_argument("--retrieval-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id", action="append", dest="include_ids", default=[])
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    args = parser.parse_args()

    board_path = generate_upstream_review_batch(
        args.review_manifest,
        args.localization_report,
        args.retrieval_index,
        output_dir=args.output_dir,
        include_ids=args.include_ids or None,
        device=args.device,
    )
    print(board_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
