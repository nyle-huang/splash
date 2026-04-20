from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from product_campaign_pipeline.production import BusinessPriorInferenceResult
from product_campaign_pipeline.service import BusinessPriorServiceSettings, create_app


def test_healthz_reports_unwarmed_but_ready_service(tmp_path: Path) -> None:
    app = create_app(
        BusinessPriorServiceSettings(
            output_root=str(tmp_path / "runtime"),
            retrieval_index_path=str(tmp_path / "retrieval.json"),
        )
    )
    client = TestClient(app)

    response = client.get("/healthz")
    response.raise_for_status()
    payload = response.json()

    assert payload["status"] == "ready"
    assert payload["retrieval_index_loaded"] is False
    assert payload["generation_pipeline_loaded"] is False


def test_generate_business_prior_endpoint_returns_structured_result(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    app = create_app(
        BusinessPriorServiceSettings(
            output_root=str(runtime_dir),
            retrieval_index_path=str(tmp_path / "retrieval.json"),
        )
    )
    fake_result = BusinessPriorInferenceResult(
        status="ok",
        request_id="demo-request",
        request_output_dir=str(runtime_dir / "demo-request"),
        source_image_path=str(tmp_path / "uploaded.png"),
        localization={"selected_phrase": "wallet"},
        source_validity="valid",
        output_path=str(
            runtime_dir / "demo-request" / "images" / "demo-request.business_prior.png"
        ),
        selected_candidate_mode="hero",
        candidate_count=2,
        retrieval_metadata={"retrieval_mode": "retrieval"},
    )

    with (
        patch(
            "product_campaign_pipeline.service.RuntimeCache.ensure_localization_pipeline",
            return_value=object(),
        ),
        patch(
            "product_campaign_pipeline.service.RuntimeCache.ensure_retrieval_index",
            return_value=[],
        ),
        patch(
            "product_campaign_pipeline.service.RuntimeCache.ensure_backbone",
            return_value=object(),
        ),
        patch(
            "product_campaign_pipeline.service.RuntimeCache.ensure_client",
            return_value=object(),
        ),
        patch(
            "product_campaign_pipeline.service.RuntimeCache.ensure_generated_localizer",
            return_value=object(),
        ),
        patch(
            "product_campaign_pipeline.service.run_business_prior_inference",
            return_value=fake_result,
        ),
    ):
        client = TestClient(app)
        response = client.post(
            "/generate/business-prior",
            data={
                "product_title": "Floral Wallet",
                "hint_phrases": "wallet|floral",
            },
            files={"image": ("wallet.png", b"fake-image-bytes", "image/png")},
        )

    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["request_id"] == "demo-request"
    assert payload["selected_candidate_mode"] == "hero"
    assert payload["retrieval_metadata"]["retrieval_mode"] == "retrieval"
