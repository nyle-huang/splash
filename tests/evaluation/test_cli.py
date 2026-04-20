"""CLI tests that exercise the offline wiring surface."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from product_campaign_pipeline.cli import app
from product_campaign_pipeline.production import BusinessPriorInferenceResult


class CLITests(unittest.TestCase):
    def test_generate_campaign_prints_business_prior_prompt_without_credentials(self) -> None:
        manifest_body = textwrap.dedent(
            """\
            item-1\thero.png\t0\t100\t40
            """
        )
        metadata = {
            "by_image": {
                "hero.png": {
                    "style_hints": ["sparkling condensation highlights"],
                    "context_hints": ["summer picnic table"],
                    "embedding": [1.0, 0.0],
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.tsv"
            metadata_path = Path(tmpdir) / "metadata.json"
            manifest_path.write_text(manifest_body, encoding="utf-8")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = app(
                    [
                        "generate",
                        "campaign",
                        "--line",
                        "business-prior",
                        "--manifest",
                        str(manifest_path),
                        "--creative-metadata",
                        str(metadata_path),
                        "--product-name",
                        "OceanMist Bottle",
                        "--query-embedding",
                        "1.0,0.0",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["line"], "business-prior")
        self.assertIn("sparkling condensation highlights", payload["prompt"]["style"])
        self.assertNotIn("task", payload)

    def test_localize_photo_uses_placeholder_backend_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "bottle.png"
            output_dir = Path(tmpdir) / "localized"
            from PIL import Image

            Image.new("RGB", (320, 240), (240, 240, 240)).save(image_path)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = app(
                    [
                        "localize",
                        "--image",
                        str(image_path),
                        "--title",
                        "Sparkling Water Bottle",
                        "--hint-phrase",
                        "label",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["backend"], "placeholder")
            self.assertIn("artifacts", payload)
            self.assertTrue(Path(payload["artifacts"]["crop_path"]).exists())
            self.assertTrue(Path(payload["artifacts"]["mask_path"]).exists())
            self.assertTrue(Path(payload["artifacts"]["overlay_path"]).exists())

    def test_generate_business_prior_photo_prints_structured_runtime_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "wallet.png"
            output_dir = Path(tmpdir) / "runtime"
            from PIL import Image

            Image.new("RGB", (256, 256), (240, 240, 240)).save(image_path)

            fake_result = BusinessPriorInferenceResult(
                status="ok",
                request_id="wallet-request",
                request_output_dir=str(output_dir / "wallet-request"),
                source_image_path=str(image_path),
                localization={"selected_phrase": "wallet"},
                source_validity="valid",
                output_path=str(output_dir / "wallet-request" / "images" / "wallet-request.business_prior.png"),
                selected_candidate_mode="hero",
                candidate_count=2,
                prompt={"subject": "a floral wallet"},
                retrieval_metadata={"retrieval_mode": "retrieval"},
            )

            stdout = io.StringIO()
            with patch("product_campaign_pipeline.cli.run_business_prior_inference", return_value=fake_result):
                with contextlib.redirect_stdout(stdout):
                    exit_code = app(
                        [
                            "generate",
                            "business-prior-photo",
                            "--image",
                            str(image_path),
                            "--product-title",
                            "Floral Wallet",
                            "--retrieval-index",
                            str(Path(tmpdir) / "retrieval.json"),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["request_id"], "wallet-request")
            self.assertEqual(payload["selected_candidate_mode"], "hero")
            self.assertEqual(payload["retrieval_metadata"]["retrieval_mode"], "retrieval")


if __name__ == "__main__":
    unittest.main()
