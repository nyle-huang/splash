from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from product_campaign_pipeline.production import BusinessPriorInferenceResult
from product_campaign_pipeline.runpod_worker import handle_public_generation_job
from product_campaign_pipeline.runtime import (
    BusinessPriorRuntimeSettings,
    RuntimeCache,
    WarmupStatus,
)

SMALL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZV0kAAAAASUVORK5CYII="
)


def _runtime_cache(tmp_path: Path) -> RuntimeCache:
    settings = BusinessPriorRuntimeSettings(
        output_root=str(tmp_path / "runtime"),
        retrieval_index_path=str(tmp_path / "retrieval.json"),
        device="cpu",
        analysis_device="cpu",
        localization_device="cpu",
    )
    return RuntimeCache(settings)


def _public_job_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "job-123",
        "input": {
            "image_base64": SMALL_PNG_BASE64,
            "mime_type": "image/png",
            "product_title": "Floral Wallet",
            "hint_phrases": ["wallet", "floral"],
            "request_id": "demo-request",
        },
    }
    payload.update(overrides)
    return payload


def test_handle_public_generation_job_returns_public_success_result(tmp_path: Path) -> None:
    cache = _runtime_cache(tmp_path)
    output_path = (
        Path(cache.settings.output_root)
        / "demo-request"
        / "images"
        / "demo-request.business_prior.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (220, 120, 90)).save(output_path, format="PNG")
    fake_result = BusinessPriorInferenceResult(
        status="ok",
        request_id="demo-request",
        request_output_dir=str(output_path.parent.parent),
        source_image_path=str(tmp_path / "source.png"),
        localization={"selected_phrase": "wallet"},
        source_validity="valid",
        output_path=str(output_path),
        selected_candidate_mode="hero",
        candidate_count=2,
        retrieval_metadata={"retrieval_mode": "retrieval"},
    )

    with (
        patch.object(cache, "ensure_localization_pipeline", return_value=object()),
        patch.object(cache, "ensure_retrieval_index", return_value=[]),
        patch.object(cache, "ensure_backbone", return_value=object()),
        patch.object(cache, "ensure_client", return_value=object()),
        patch.object(cache, "ensure_generated_localizer", return_value=object()),
        patch(
            "product_campaign_pipeline.runpod_worker.run_business_prior_inference",
            return_value=fake_result,
        ) as run_mock,
    ):
        payload = handle_public_generation_job(_public_job_payload(), cache=cache)

    assert payload["status"] == "succeeded"
    assert payload["request_id"] == "demo-request"
    assert payload["selected_candidate_mode"] == "hero"
    assert payload["final_image_base64"]
    assert payload["final_image_mime_type"] == "image/jpeg"

    runtime_request = run_mock.call_args.args[0]
    assert runtime_request.product_title == "Floral Wallet"
    assert runtime_request.request_id == "demo-request"
    assert Path(runtime_request.image_path).exists()


def test_handle_public_generation_job_returns_invalid_source_details(tmp_path: Path) -> None:
    cache = _runtime_cache(tmp_path)
    fake_result = BusinessPriorInferenceResult(
        status="invalid_source",
        request_id="demo-request",
        request_output_dir=str(Path(cache.settings.output_root) / "demo-request"),
        source_image_path=str(tmp_path / "source.png"),
        localization={"selected_phrase": "wallet"},
        source_validity="invalid",
        source_validity_score=0.18,
        source_validity_issues=["localization_failed", "low_subject_confidence"],
        observed_evidence={"source_validity": "invalid"},
        invalid_reason="invalid_source_photo",
    )

    with (
        patch.object(cache, "ensure_localization_pipeline", return_value=object()),
        patch.object(cache, "ensure_retrieval_index", return_value=[]),
        patch.object(cache, "ensure_backbone", return_value=object()),
        patch.object(cache, "ensure_client", return_value=object()),
        patch.object(cache, "ensure_generated_localizer", return_value=object()),
        patch(
            "product_campaign_pipeline.runpod_worker.run_business_prior_inference",
            return_value=fake_result,
        ),
    ):
        payload = handle_public_generation_job(_public_job_payload(), cache=cache)

    assert payload["status"] == "invalid_source"
    assert payload["request_id"] == "demo-request"
    assert payload["invalid_source"]["reason"] == "invalid_source_photo"
    assert payload["invalid_source"]["issues"] == ["localization_failed", "low_subject_confidence"]
    assert "input-quality checks" in payload["summary"]


def test_handle_public_generation_job_runs_internal_warmup(tmp_path: Path) -> None:
    cache = _runtime_cache(tmp_path)
    warmup_status = WarmupStatus(
        status="ready",
        retrieval_index_loaded=True,
        localization_pipeline_loaded=True,
        analysis_backbone_loaded=True,
        generation_client_initialized=True,
        generation_pipeline_loaded=True,
    )

    with patch.object(cache, "warmup", return_value=warmup_status) as warmup_mock:
        payload = handle_public_generation_job(
            {
                "id": "warmup-job",
                "input": {
                    "_internal_warmup": True,
                    "include_generation": True,
                },
            },
            cache=cache,
        )

    assert payload["status"] == "succeeded"
    assert payload["request_id"] == "warmup-job"
    assert payload["warmup"]["generation_pipeline_loaded"] is True
    warmup_mock.assert_called_once_with(include_generation=True)
