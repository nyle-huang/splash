#!/usr/bin/env python3
from __future__ import annotations

import argparse

from product_campaign_pipeline.review_batch import build_retrieval_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--min-page-views", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    args = parser.parse_args()

    output_path = build_retrieval_index(
        args.manifest,
        args.image_root,
        output_path=args.output,
        top_k=args.top_k,
        pool_size=args.pool_size,
        min_page_views=args.min_page_views,
        device=args.device,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
