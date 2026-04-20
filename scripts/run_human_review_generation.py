#!/usr/bin/env python3
from __future__ import annotations

import argparse

from product_campaign_pipeline.review_batch import generate_review_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--localization-report", required=True)
    parser.add_argument("--retrieval-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--id", action="append", dest="include_ids", default=[])
    parser.add_argument("--line", action="append", dest="include_lines", default=[])
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--analysis-device", default="cpu", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--candidate-mode", action="append", dest="candidate_modes", default=[])
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    board_path = generate_review_batch(
        args.review_manifest,
        args.localization_report,
        args.retrieval_index,
        output_dir=args.output_dir,
        include_ids=args.include_ids or None,
        include_lines=args.include_lines or None,
        model_id=args.model,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        analysis_device=args.analysis_device,
        candidate_modes_override=args.candidate_modes or None,
        skip_analysis=args.skip_analysis,
    )
    print(board_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
