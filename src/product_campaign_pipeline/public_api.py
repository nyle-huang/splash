"""Public browser-facing contract models and helpers for the demo stack."""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUPPORTED_PUBLIC_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
DEFAULT_BROKER_MAX_IMAGE_BYTES = 4_500_000
DEFAULT_RUNPOD_PAYLOAD_LIMIT_BYTES = 9_500_000
MAX_PRODUCT_TITLE_LENGTH = 160
MAX_HINT_PHRASES = 8
MAX_HINT_PHRASE_LENGTH = 48


class PublicInferenceRequest(BaseModel):
    """Narrow browser-facing request payload."""

    model_config = ConfigDict(extra="forbid")

    image_base64: str = Field(min_length=8)
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    product_title: str = Field(min_length=1, max_length=MAX_PRODUCT_TITLE_LENGTH)
    hint_phrases: list[str] = Field(default_factory=list)
    request_id: str | None = None

    @field_validator("image_base64")
    @classmethod
    def _validate_image_base64(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.startswith("data:"):
            raise ValueError("image_base64 must not include a data URL prefix")
        try:
            base64.b64decode(normalized, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image_base64 is not valid base64 data") from exc
        return normalized

    @field_validator("product_title")
    @classmethod
    def _normalize_title(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("product_title must not be empty")
        return normalized

    @field_validator("hint_phrases", mode="before")
    @classmethod
    def _normalize_hint_phrases(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [part.strip() for part in value.split("|")]
        elif isinstance(value, list | tuple):
            raw_items = [str(item).strip() for item in value]
        else:
            raise ValueError("hint_phrases must be a list of strings")

        hints = [hint for hint in raw_items if hint]
        if len(hints) > MAX_HINT_PHRASES:
            raise ValueError(f"hint_phrases must contain at most {MAX_HINT_PHRASES} items")
        for hint in hints:
            if len(hint) > MAX_HINT_PHRASE_LENGTH:
                raise ValueError(
                    f"hint phrases must be at most {MAX_HINT_PHRASE_LENGTH} characters each"
                )
        return hints

    @field_validator("request_id")
    @classmethod
    def _normalize_request_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_public_request_id(value)
        if normalized is None:
            raise ValueError("request_id must contain at least one alphanumeric character")
        return normalized


class PublicInvalidSourceDetails(BaseModel):
    """User-visible invalid-source details."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    issues: list[str] = Field(default_factory=list)
    score: float | None = None


class WorkerGenerationResult(BaseModel):
    """Structured output returned by the Runpod worker."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "invalid_source", "failed"]
    request_id: str
    summary: str
    selected_candidate_mode: str | None = None
    final_image_base64: str | None = None
    final_image_mime_type: str | None = None
    invalid_source: PublicInvalidSourceDetails | None = None
    error_code: str | None = None


class PublicJobResponse(BaseModel):
    """User-visible response returned by the public broker."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["queued", "running", "succeeded", "invalid_source", "failed"]
    job_id: str
    summary: str
    selected_candidate_mode: str | None = None
    final_image_base64: str | None = None
    final_image_mime_type: str | None = None
    invalid_source: PublicInvalidSourceDetails | None = None
    error_code: str | None = None


def decode_public_image_base64(value: str) -> bytes:
    """Decode validated browser-upload image data."""

    return base64.b64decode(value, validate=True)


def normalize_public_request_id(value: str | None) -> str | None:
    """Normalize user-provided request ids into filesystem-safe slugs."""

    if value is None:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        return None
    return slug[:64]


def public_request_file_extension(mime_type: str) -> str:
    """Map supported browser MIME types to file extensions."""

    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[mime_type]


def estimate_public_request_payload_bytes(request: PublicInferenceRequest) -> int:
    """Estimate the serialized JSON payload size sent to Runpod."""

    payload = {"input": request.model_dump(mode="json")}
    return len(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))


def validate_public_request_budget(
    request: PublicInferenceRequest,
    *,
    max_image_bytes: int = DEFAULT_BROKER_MAX_IMAGE_BYTES,
    max_payload_bytes: int = DEFAULT_RUNPOD_PAYLOAD_LIMIT_BYTES,
) -> bytes:
    """Validate image and payload size budgets before broker submission."""

    image_bytes = decode_public_image_base64(request.image_base64)
    if len(image_bytes) > max_image_bytes:
        raise ValueError(
            f"decoded image is too large: {len(image_bytes)} bytes exceeds {max_image_bytes}"
        )
    payload_bytes = estimate_public_request_payload_bytes(request)
    if payload_bytes > max_payload_bytes:
        raise ValueError(
            f"request payload is too large: {payload_bytes} bytes exceeds {max_payload_bytes}"
        )
    return image_bytes


def build_public_success_summary(selected_candidate_mode: str | None) -> str:
    """Generate a concise success summary for the browser."""

    if selected_candidate_mode:
        return (
            "Generation completed successfully. "
            f"The selected campaign variant used the `{selected_candidate_mode}` mode."
        )
    return "Generation completed successfully."


def build_public_invalid_source_summary(details: PublicInvalidSourceDetails) -> str:
    """Generate a concise invalid-source summary for the browser."""

    if details.issues:
        top_issues = ", ".join(details.issues[:2])
        return (
            "The upload could not be used for generation because the source photo "
            f"failed the input-quality checks: {top_issues}."
        )
    return (
        "The upload could not be used for generation because the source photo "
        "failed the input-quality checks."
    )
