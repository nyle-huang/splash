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
        self.assertEqual(
            request.input_images,
            (str(input_path.resolve()), str(reference_path.resolve())),
        )
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

    def test_runpod_cached_model_snapshot_is_preferred_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "huggingface-cache" / "hub"
            snapshot = (
                root
                / "models--black-forest-labs--FLUX.2-klein-9B"
                / "snapshots"
                / "abc123"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
            refs = root / "models--black-forest-labs--FLUX.2-klein-9B" / "refs"
            refs.mkdir()
            (refs / "main").write_text("abc123\n", encoding="utf-8")

            client = Flux2KleinClient(cached_model_root=root)

            self.assertEqual(
                client._resolve_model_load_source(DEFAULT_MODEL_ID),
                str(snapshot.resolve()),
            )

    def test_runpod_cached_model_snapshot_matching_is_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "huggingface-cache" / "hub"
            snapshot = (
                root
                / "models--black-forest-labs--flux.2-klein-9b"
                / "snapshots"
                / "92196c8e11f7b6cf2b7493e037d8c5345c559216"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
            client = Flux2KleinClient(cached_model_root=root)

            self.assertEqual(
                client._resolve_model_load_source(DEFAULT_MODEL_ID),
                str(snapshot.resolve()),
            )

    def test_runpod_cached_model_snapshot_accepts_huggingface_url_with_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "huggingface-cache" / "hub"
            snapshot = (
                root
                / "models--black-forest-labs--flux.2-klein-9b"
                / "snapshots"
                / "92196c8e11f7b6cf2b7493e037d8c5345c559216"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
            client = Flux2KleinClient(
                model_id=(
                    "https://huggingface.co/black-forest-labs/"
                    "flux.2-klein-9b:92196c8e11f7b6cf2b7493e037d8c5345c559216"
                ),
                cached_model_root=root,
            )

            self.assertEqual(
                client._resolve_model_load_source(client.model_id),
                str(snapshot.resolve()),
            )

    def test_missing_runpod_cached_model_keeps_default_model_id_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = Flux2KleinClient(cached_model_root=Path(tmpdir) / "missing")

            self.assertEqual(client._resolve_model_load_source(DEFAULT_MODEL_ID), DEFAULT_MODEL_ID)

    def test_explicit_model_load_path_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = Path(tmpdir) / "snapshot"
            snapshot.mkdir()
            (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
            client = Flux2KleinClient(model_load_path=snapshot)
            self.assertEqual(
                client._resolve_model_load_source(DEFAULT_MODEL_ID),
                str(snapshot.resolve()),
            )

            missing = Flux2KleinClient(model_load_path=Path(tmpdir) / "missing")
            with self.assertRaises(RuntimeError):
                missing._resolve_model_load_source(DEFAULT_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
