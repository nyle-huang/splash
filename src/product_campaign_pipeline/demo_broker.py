"""Thin public broker that fronts Runpod Serverless for the demo website."""

from __future__ import annotations

import hmac
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from product_campaign_pipeline.public_api import (
    DEFAULT_BROKER_MAX_IMAGE_BYTES,
    DEFAULT_RUNPOD_PAYLOAD_LIMIT_BYTES,
    PublicInferenceRequest,
    PublicJobResponse,
    WorkerGenerationResult,
    validate_public_request_budget,
)


class DemoBrokerSettings(BaseModel):
    """Settings for the public browser-facing broker."""

    model_config = ConfigDict(extra="ignore")

    demo_token: str = "change-me"
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""
    allowed_origin: str = "*"
    runpod_base_url: str = "https://api.runpod.ai/v2"
    max_image_bytes: int = DEFAULT_BROKER_MAX_IMAGE_BYTES
    max_request_payload_bytes: int = DEFAULT_RUNPOD_PAYLOAD_LIMIT_BYTES
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> DemoBrokerSettings:
        import os

        return cls(
            demo_token=os.getenv("PCP_DEMO_TOKEN", cls.model_fields["demo_token"].default),
            runpod_api_key=os.getenv(
                "PCP_RUNPOD_API_KEY",
                cls.model_fields["runpod_api_key"].default,
            ),
            runpod_endpoint_id=os.getenv(
                "PCP_RUNPOD_ENDPOINT_ID",
                cls.model_fields["runpod_endpoint_id"].default,
            ),
            allowed_origin=os.getenv(
                "PCP_ALLOWED_ORIGIN",
                cls.model_fields["allowed_origin"].default,
            ),
            runpod_base_url=os.getenv(
                "PCP_RUNPOD_BASE_URL",
                cls.model_fields["runpod_base_url"].default,
            ),
            max_image_bytes=int(
                os.getenv(
                    "PCP_BROKER_MAX_IMAGE_BYTES",
                    str(cls.model_fields["max_image_bytes"].default),
                )
            ),
            max_request_payload_bytes=int(
                os.getenv(
                    "PCP_BROKER_MAX_PAYLOAD_BYTES",
                    str(cls.model_fields["max_request_payload_bytes"].default),
                )
            ),
            request_timeout_seconds=float(
                os.getenv(
                    "PCP_BROKER_REQUEST_TIMEOUT_SECONDS",
                    str(cls.model_fields["request_timeout_seconds"].default),
                )
            ),
        )


class BrokerUpstreamError(RuntimeError):
    """Raised when Runpod cannot be reached or returns an invalid response."""


class RunpodJobClient:
    """Small HTTP client for the subset of Runpod operations used by the demo broker."""

    def __init__(
        self,
        settings: DemoBrokerSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def submit_job(self, payload: PublicInferenceRequest) -> str:
        validate_public_request_budget(
            payload,
            max_image_bytes=self.settings.max_image_bytes,
            max_payload_bytes=self.settings.max_request_payload_bytes,
        )
        body = {"input": payload.model_dump(mode="json")}
        data = await self._request("POST", f"/{self.settings.runpod_endpoint_id}/run", json=body)
        job_id = data.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise BrokerUpstreamError("Runpod did not return a job id.")
        return job_id

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/{self.settings.runpod_endpoint_id}/status/{job_id}")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.settings.runpod_api_key:
            raise BrokerUpstreamError("Runpod API key is not configured.")
        if not self.settings.runpod_endpoint_id:
            raise BrokerUpstreamError("Runpod endpoint id is not configured.")

        headers = {
            "authorization": f"Bearer {self.settings.runpod_api_key}",
            "accept": "application/json",
        }
        if method.upper() == "POST":
            headers["content-type"] = "application/json"
        async with httpx.AsyncClient(
            base_url=self.settings.runpod_base_url.rstrip("/"),
            timeout=self.settings.request_timeout_seconds,
            transport=self.transport,
        ) as client:
            response = await client.request(method.upper(), path, headers=headers, json=json)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BrokerUpstreamError("Runpod returned a non-JSON response.") from exc
        if response.status_code >= 400:
            detail = payload.get("error") if isinstance(payload, dict) else None
            raise BrokerUpstreamError(
                detail or f"Runpod request failed with {response.status_code}."
            )
        if not isinstance(payload, dict):
            raise BrokerUpstreamError("Runpod returned an unexpected response payload.")
        return payload


def create_demo_broker_app(
    settings: DemoBrokerSettings | None = None,
    *,
    job_client: RunpodJobClient | None = None,
) -> FastAPI:
    runtime_settings = settings or DemoBrokerSettings.from_env()
    client = job_client or RunpodJobClient(runtime_settings)
    app = FastAPI(title="Product Campaign Pipeline Demo Broker", version="0.1.0")
    allowed_origins = (
        ["*"] if runtime_settings.allowed_origin == "*" else [runtime_settings.allowed_origin]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["authorization", "content-type"],
    )

    @app.post("/api/jobs", response_model=PublicJobResponse)
    async def create_job(payload: PublicInferenceRequest, request: Request) -> PublicJobResponse:
        _authorize_request(request, runtime_settings)
        try:
            job_id = await client.submit_job(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except BrokerUpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return PublicJobResponse(
            status="queued",
            job_id=job_id,
            summary="The request was accepted and queued for GPU generation.",
        )

    @app.get("/api/jobs/{job_id}", response_model=PublicJobResponse)
    async def get_job(job_id: str, request: Request) -> PublicJobResponse:
        _authorize_request(request, runtime_settings)
        try:
            payload = await client.get_job(job_id)
        except BrokerUpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return _map_runpod_status_to_public_response(job_id, payload)

    return app


def _authorize_request(request: Request, settings: DemoBrokerSettings) -> None:
    header_value = request.headers.get("authorization", "")
    expected = f"Bearer {settings.demo_token}"
    if not hmac.compare_digest(header_value, expected):
        raise HTTPException(status_code=401, detail="Invalid demo token.")


def _map_runpod_status_to_public_response(
    job_id: str,
    payload: dict[str, Any],
) -> PublicJobResponse:
    status_value = str(payload.get("status", "")).upper()
    if status_value in {"IN_QUEUE", "QUEUED"}:
        return PublicJobResponse(
            status="queued",
            job_id=job_id,
            summary="The request is still waiting for a worker.",
        )
    if status_value in {"IN_PROGRESS", "PROCESSING"}:
        return PublicJobResponse(
            status="running",
            job_id=job_id,
            summary="The request is currently running on the GPU worker.",
        )
    if status_value == "COMPLETED":
        worker_payload = payload.get("output", {})
        try:
            worker_result = WorkerGenerationResult.model_validate(worker_payload)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Runpod returned an invalid worker result.",
            ) from exc
        return PublicJobResponse(
            status=worker_result.status,
            job_id=job_id,
            summary=worker_result.summary,
            selected_candidate_mode=worker_result.selected_candidate_mode,
            final_image_base64=worker_result.final_image_base64,
            final_image_mime_type=worker_result.final_image_mime_type,
            invalid_source=worker_result.invalid_source,
            error_code=worker_result.error_code,
        )
    if status_value in {"FAILED", "TIMED_OUT", "CANCELLED"}:
        error_detail = payload.get("error")
        summary = "The request failed before a final image was produced."
        if isinstance(error_detail, str) and error_detail.strip():
            summary = f"{summary} {error_detail.strip()}"
        return PublicJobResponse(
            status="failed",
            job_id=job_id,
            summary=summary,
            error_code="runpod_job_failed",
        )
    raise HTTPException(
        status_code=502,
        detail=f"Unexpected Runpod job status: {status_value or 'unknown'}",
    )


app = create_demo_broker_app()
