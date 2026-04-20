"""Runpod Serverless worker for the public demo contract."""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from product_campaign_pipeline.flux import (
    MissingCredentialsError,
    MissingDependencyError,
    MissingModelAccessError,
)
from product_campaign_pipeline.production import (
    BusinessPriorInferenceRequest,
    BusinessPriorInferenceResult,
    run_business_prior_inference,
)
from product_campaign_pipeline.public_api import (
    PublicInferenceRequest,
    PublicInvalidSourceDetails,
    WorkerGenerationResult,
    build_public_invalid_source_summary,
    build_public_success_summary,
    decode_public_image_base64,
    normalize_public_request_id,
    public_request_file_extension,
)
from product_campaign_pipeline.runtime import BusinessPriorRuntimeSettings, RuntimeCache

DEFAULT_RUNTIME_CACHE = RuntimeCache(BusinessPriorRuntimeSettings.from_env())


def handle_public_generation_job(
    job: dict[str, Any],
    *,
    cache: RuntimeCache | None = None,
) -> dict[str, Any]:
    """Handle a single public browser-originated generation job."""

    runtime_cache = cache or DEFAULT_RUNTIME_CACHE
    fallback_request_id = _fallback_request_id(job)
    try:
        request = PublicInferenceRequest.model_validate(job.get("input", {}))
        request_id = request.request_id or fallback_request_id
        source_image_path = _save_public_source_image(
            request,
            output_root=runtime_cache.settings.output_root,
            request_id=request_id,
        )
        runtime_request = BusinessPriorInferenceRequest(
            image_path=str(source_image_path),
            product_title=request.product_title,
            retrieval_index_path=runtime_cache.settings.retrieval_index_path,
            output_dir=runtime_cache.settings.output_root,
            hint_phrases=list(request.hint_phrases),
            request_id=request_id,
            source_page_url="demo://github-pages",
            source_image_url="demo://browser-upload",
            model_id=runtime_cache.settings.model_id,
            width=runtime_cache.settings.width,
            height=runtime_cache.settings.height,
            num_inference_steps=runtime_cache.settings.num_inference_steps,
            guidance_scale=runtime_cache.settings.guidance_scale,
            device=runtime_cache.settings.device,
            analysis_device=runtime_cache.settings.analysis_device,
            localization_device=runtime_cache.settings.localization_device,
            top_k=runtime_cache.settings.top_k,
        )
        result = run_business_prior_inference(
            runtime_request,
            localization_pipeline=runtime_cache.ensure_localization_pipeline(),
            retrieval_index=runtime_cache.ensure_retrieval_index(),
            backbone=runtime_cache.ensure_backbone(),
            client=runtime_cache.ensure_client(),
            generated_localizer=runtime_cache.ensure_generated_localizer(),
        )
        worker_result = _worker_result_from_inference(result)
    except ValidationError:
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=fallback_request_id,
            summary="The request payload was invalid and could not be processed.",
            error_code="invalid_request",
        )
    except ValueError:
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=fallback_request_id,
            summary="The request payload failed validation and could not be processed.",
            error_code="invalid_request",
        )
    except (MissingDependencyError, MissingCredentialsError, MissingModelAccessError):
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=fallback_request_id,
            summary="The generation runtime is currently unavailable.",
            error_code="runtime_unavailable",
        )
    except FileNotFoundError:
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=fallback_request_id,
            summary="The uploaded image could not be staged for generation.",
            error_code="input_staging_failed",
        )
    except Exception:
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=fallback_request_id,
            summary="The generation job failed before a final image could be produced.",
            error_code="inference_failed",
        )
    return worker_result.model_dump(mode="json")


def _worker_result_from_inference(result: BusinessPriorInferenceResult) -> WorkerGenerationResult:
    request_id = result.request_id
    if result.status == "invalid_source":
        invalid_source = PublicInvalidSourceDetails(
            reason=result.invalid_reason or "invalid_source_photo",
            issues=list(result.source_validity_issues),
            score=result.source_validity_score,
        )
        return WorkerGenerationResult(
            status="invalid_source",
            request_id=request_id,
            summary=build_public_invalid_source_summary(invalid_source),
            invalid_source=invalid_source,
        )

    output_path = Path(result.output_path or "")
    if not output_path.exists():
        return WorkerGenerationResult(
            status="failed",
            request_id=request_id,
            summary="Generation completed without producing a readable image artifact.",
            error_code="missing_output_image",
        )
    image_base64, mime_type = _encode_public_output_image(output_path)
    return WorkerGenerationResult(
        status="succeeded",
        request_id=request_id,
        summary=build_public_success_summary(result.selected_candidate_mode),
        selected_candidate_mode=result.selected_candidate_mode,
        final_image_base64=image_base64,
        final_image_mime_type=mime_type,
    )


def _save_public_source_image(
    request: PublicInferenceRequest,
    *,
    output_root: str,
    request_id: str,
) -> Path:
    uploads_dir = Path(output_root) / "_serverless_inputs"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        uploads_dir
        / f"{request_id}-{uuid.uuid4().hex}{public_request_file_extension(request.mime_type)}"
    )
    destination.write_bytes(decode_public_image_base64(request.image_base64))
    return destination


def _encode_public_output_image(path: Path) -> tuple[str, str]:
    buffer = io.BytesIO()
    with Image.open(path) as image:
        if image.mode not in {"RGB", "L"}:
            rgba = image.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, (255, 255, 255))
            flattened.paste(rgba, mask=rgba.split()[-1])
            image = flattened
        elif image.mode == "L":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, "image/jpeg"


def _fallback_request_id(job: dict[str, Any]) -> str:
    for candidate in (
        job.get("id"),
        job.get("input", {}).get("request_id") if isinstance(job.get("input"), dict) else None,
        uuid.uuid4().hex,
    ):
        normalized = normalize_public_request_id(str(candidate)) if candidate is not None else None
        if normalized:
            return normalized
    return f"public-request-{uuid.uuid4().hex[:8]}"


def start_runpod_worker() -> None:
    """Start the worker in a real Runpod Serverless runtime."""

    try:
        import runpod
    except ImportError as exc:  # pragma: no cover - deployment-only code path
        raise MissingDependencyError(
            "Runpod serverless dependencies are missing. "
            "Install the worker image requirements first."
        ) from exc

    runpod.serverless.start({"handler": handle_public_generation_job})


if __name__ == "__main__":  # pragma: no cover - deployment-only entrypoint
    start_runpod_worker()
