#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from product_campaign_pipeline.review_batch import load_review_seed_manifest, render_review_board


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
    parser.add_argument("--num-inference-steps", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--analysis-device", default="cpu", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--candidate-mode", action="append", dest="candidate_modes", default=[])
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    manifest = load_review_seed_manifest(args.review_manifest)
    include_ids = [str(value) for value in args.include_ids] or [seed.id for seed in manifest]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_runs_dir = output_dir / "seed_runs"
    seed_runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    generation_script = Path(__file__).with_name("run_human_review_generation.py")
    combined_rows: list[dict] = []
    candidate_modes = list(args.candidate_modes)
    include_lines = args.include_lines or ["baseline", "business_prior"]

    for seed_id in include_ids:
        for line_name in include_lines:
            seed_output_dir = seed_runs_dir / seed_id / line_name
            seed_report_path = seed_output_dir / "reports" / "generation_report.json"

            if not (args.reuse_existing and seed_report_path.exists()):
                cmd = [
                    sys.executable,
                    str(generation_script),
                    "--review-manifest",
                    args.review_manifest,
                    "--localization-report",
                    args.localization_report,
                    "--retrieval-index",
                    args.retrieval_index,
                    "--output-dir",
                    str(seed_output_dir),
                    "--model",
                    args.model,
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                    "--num-inference-steps",
                    str(args.num_inference_steps),
                    "--guidance-scale",
                    str(args.guidance_scale),
                    "--device",
                    args.device,
                    "--analysis-device",
                    args.analysis_device,
                    "--id",
                    seed_id,
                    "--line",
                    line_name,
                ]
                for mode in candidate_modes:
                    cmd.extend(["--candidate-mode", mode])
                if args.skip_analysis:
                    cmd.append("--skip-analysis")
                subprocess.run(cmd, check=True)

            combined_rows.extend(json.loads(seed_report_path.read_text(encoding="utf-8")))

    combined_rows.sort(key=lambda row: (str(row.get("id", "")), str(row.get("line", ""))))
    combined_report_path = reports_dir / "generation_report.json"
    combined_report_path.write_text(json.dumps(combined_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    board_path = render_review_board(combined_rows, output_dir / "human_review_board.html")
    print(board_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
