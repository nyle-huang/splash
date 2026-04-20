"""Tests for the local FLUX.2 Klein client boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from product_campaign_pipeline.composer import FluxPrompt
from product_campaign_pipeline.flux import DEFAULT_MODEL_ID, Flux2KleinClient


class FluxClientTests(unittest.TestCase):
    def test_request_builds_local_generation_config_with_reference_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            reference_path = Path(tmpdir) / "reference.png"
            input_path.write_bytes(b"input-bytes")
            reference_path.write_bytes(b"reference-bytes")

            request = Flux2KleinClient().build_request(
                prompt=FluxPrompt(
                    subject="Exact bottle hero shot",
                    action="Place it in a premium campaign scene.",
                    style="Commercial lifestyle photography.",
                    context="Believable kitchen setting.",
                    preservation_constraints=("Preserve branding.",),
                ),
                input_image=input_path,
                reference_images=(reference_path,),
                width=1024,
                height=1024,
                seed=11,
            )

        self.assertEqual(request.model_id, DEFAULT_MODEL_ID)
        self.assertEqual(request.num_inference_steps, 4)
        self.assertEqual(request.guidance_scale, 1.0)
        self.assertTrue(request.cpu_offload)
        self.assertEqual(request.width, 1024)
        self.assertEqual(request.height, 1024)
        self.assertEqual(request.seed, 11)
        self.assertEqual(request.input_images, (str(input_path.resolve()), str(reference_path.resolve())))
        self.assertIn('"subject":"Exact bottle hero shot"', request.prompt)

    def test_missing_local_reference_image_is_rejected_during_request_build(self) -> None:
        missing = Path("/tmp/does-not-exist.png")
        client = Flux2KleinClient()
        with self.assertRaises(FileNotFoundError):
            client.build_request(prompt="hello", input_image=missing)

    def test_small_reference_images_are_upscaled_to_flux_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            small_path = Path(tmpdir) / "small.png"
            from PIL import Image

            Image.new("RGB", (59, 145), (32, 32, 32)).save(small_path)
            client = Flux2KleinClient()

            images = client._load_reference_images((str(small_path),))

        self.assertEqual(len(images), 1)
        self.assertGreaterEqual(images[0].width, 64)
        self.assertGreaterEqual(images[0].height, 64)


if __name__ == "__main__":
    unittest.main()
