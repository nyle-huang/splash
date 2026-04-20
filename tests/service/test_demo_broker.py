from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from product_campaign_pipeline.demo_broker import (
    DemoBrokerSettings,
    RunpodJobClient,
    create_demo_broker_app,
)

SMALL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZV0kAAAAASUVORK5CYII="
)


def _settings(**overrides: object) -> DemoBrokerSettings:
    payload = {
        "demo_token": "demo-token",
        "runpod_api_key": "runpod-key",
        "runpod_endpoint_id": "endpoint-abc",
        "allowed_origin": "https://demo.example",
        "max_image_bytes": 4_500_000,
        "max_request_payload_bytes": 9_500_000,
    }
    payload.update(overrides)
    return DemoBrokerSettings(**payload)


def _headers(token: str = "demo-token") -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "image_base64": SMALL_PNG_BASE64,
        "mime_type": "image/png",
        "product_title": "Floral Wallet",
        "hint_phrases": ["wallet", "floral"],
        "request_id": "demo-request",
    }
    payload.update(overrides)
    return payload


def _client_with_transport(
    settings: DemoBrokerSettings,
    handler: httpx.MockTransport,
) -> TestClient:
    app = create_demo_broker_app(
        settings,
        job_client=RunpodJobClient(settings, transport=handler),
    )
    return TestClient(app)


def test_create_job_rejects_missing_or_bad_token() -> None:
    settings = _settings()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"id": "unused"}))
    client = _client_with_transport(settings, transport)

    response = client.post("/api/jobs", json=_payload())

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid demo token."


def test_create_job_rejects_oversized_image_before_runpod_submission() -> None:
    settings = _settings(max_image_bytes=10)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Runpod should not be called when broker-side validation fails.")

    client = _client_with_transport(settings, httpx.MockTransport(handler))
    response = client.post("/api/jobs", headers=_headers(), json=_payload())

    assert response.status_code == 400
    assert "decoded image is too large" in response.json()["detail"]


def test_create_job_rejects_unsupported_mime_type() -> None:
    settings = _settings()
    client = _client_with_transport(
        settings,
        httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    response = client.post(
        "/api/jobs",
        headers=_headers(),
        json=_payload(mime_type="image/gif"),
    )

    assert response.status_code == 422


def test_create_job_submits_runpod_job_and_returns_queue_status() -> None:
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v2/endpoint-abc/run"
        assert request.headers["authorization"] == "Bearer runpod-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["input"]["product_title"] == "Floral Wallet"
        return httpx.Response(200, json={"id": "job-123"})

    client = _client_with_transport(settings, httpx.MockTransport(handler))
    response = client.post("/api/jobs", headers=_headers(), json=_payload())

    response.raise_for_status()
    assert response.json() == {
        "status": "queued",
        "job_id": "job-123",
        "summary": "The request was accepted and queued for GPU generation.",
        "selected_candidate_mode": None,
        "final_image_base64": None,
        "final_image_mime_type": None,
        "invalid_source": None,
        "error_code": None,
    }


def test_get_job_maps_runpod_failure_to_public_failed_status() -> None:
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v2/endpoint-abc/status/job-123"
        return httpx.Response(200, json={"status": "FAILED", "error": "worker crashed"})

    client = _client_with_transport(settings, httpx.MockTransport(handler))
    response = client.get("/api/jobs/job-123", headers=_headers())

    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error_code"] == "runpod_job_failed"
    assert "worker crashed" in payload["summary"]


def test_get_job_maps_running_status_for_polling() -> None:
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "IN_PROGRESS"})

    client = _client_with_transport(settings, httpx.MockTransport(handler))
    response = client.get("/api/jobs/job-123", headers=_headers())

    response.raise_for_status()
    assert response.json()["status"] == "running"


def test_get_job_maps_completed_result_payload() -> None:
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "COMPLETED",
                "output": {
                    "status": "succeeded",
                    "request_id": "demo-request",
                    "summary": "Generation completed successfully.",
                    "selected_candidate_mode": "hero",
                    "final_image_base64": "ZmFrZS1pbWFnZQ==",
                    "final_image_mime_type": "image/jpeg",
                    "invalid_source": None,
                    "error_code": None,
                },
            },
        )

    client = _client_with_transport(settings, httpx.MockTransport(handler))
    response = client.get("/api/jobs/job-123", headers=_headers())

    response.raise_for_status()
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["selected_candidate_mode"] == "hero"
    assert payload["final_image_mime_type"] == "image/jpeg"
