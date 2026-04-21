"""Runpod Serverless worker for the public demo contract."""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

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

LOGGER = logging.getLogger(__name__)
_DEFAULT_RUNTIME_CACHE: Any | None = None
DEFAULT_WORKER_LOG_PATH = "/runpod-volume/logs/runpod_worker.log"


def handle_public_generation_job(
    job: dict[str, Any],
    *,
    cache: Any | None = None,
) -> dict[str, Any]:
    """Handle a single public browser-originated generation job."""

    started_at = time.perf_counter()
    fallback_request_id = _fallback_request_id(job)
    LOGGER.info(
        "Runpod job received. request_id=%s input_keys=%s",
        fallback_request_id,
        _job_input_keys(job),
    )
    if _is_internal_ping_job(job):
        return _handle_internal_ping_job(request_id=fallback_request_id)

    if _is_internal_warmup_job(job):
        result = _handle_internal_warmup_job(
            job,
            cache=cache,
            request_id=fallback_request_id,
        )
        LOGGER.info(
            "Runpod warmup job completed. request_id=%s status=%s elapsed_seconds=%.3f",
            fallback_request_id,
            result.get("status"),
            time.perf_counter() - started_at,
        )
        return result

    debug_errors = _debug_errors_enabled(job)
    try:
        request = PublicInferenceRequest.model_validate(_public_job_input(job))
    except ValidationError:
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=fallback_request_id,
            summary="The request payload was invalid and could not be processed.",
            error_code="invalid_request",
        )
        payload = worker_result.model_dump(mode="json")
        LOGGER.info(
            "Runpod generation job completed. request_id=%s status=%s elapsed_seconds=%.3f",
            fallback_request_id,
            payload.get("status"),
            time.perf_counter() - started_at,
        )
        return payload
    except ValueError:
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=fallback_request_id,
            summary="The request payload failed validation and could not be processed.",
            error_code="invalid_request",
        )
        payload = worker_result.model_dump(mode="json")
        LOGGER.info(
            "Runpod generation job completed. request_id=%s status=%s elapsed_seconds=%.3f",
            fallback_request_id,
            payload.get("status"),
            time.perf_counter() - started_at,
        )
        return payload

    request_id = request.request_id or fallback_request_id
    try:
        runtime_cache = cache or _get_default_runtime_cache()
        source_image_path = _save_public_source_image(
            request,
            output_root=runtime_cache.settings.output_root,
            request_id=request_id,
        )
        from product_campaign_pipeline.production import BusinessPriorInferenceRequest

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
    except ValueError as exc:
        LOGGER.exception("Runpod generation job failed with ValueError after request validation.")
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=request_id,
            summary=_failure_summary(
                "The generation job failed before a final image could be produced.",
                exc,
                debug_errors=debug_errors,
            ),
            error_code="inference_failed",
        )
    except FileNotFoundError:
        worker_result = WorkerGenerationResult(
            status="failed",
            request_id=request_id,
            summary="The uploaded image could not be staged for generation.",
            error_code="input_staging_failed",
        )
    except Exception as exc:
        LOGGER.exception("Runpod generation job failed before producing a final image.")
        if _is_runtime_unavailable_exception(exc):
            worker_result = WorkerGenerationResult(
                status="failed",
                request_id=request_id,
                summary=_failure_summary(
                    "The generation runtime is currently unavailable.",
                    exc,
                    debug_errors=debug_errors,
                ),
                error_code="runtime_unavailable",
            )
        else:
            worker_result = WorkerGenerationResult(
                status="failed",
                request_id=request_id,
                summary=_failure_summary(
                    "The generation job failed before a final image could be produced.",
                    exc,
                    debug_errors=debug_errors,
                ),
                error_code="inference_failed",
            )
    payload = worker_result.model_dump(mode="json")
    LOGGER.info(
        "Runpod generation job completed. request_id=%s status=%s elapsed_seconds=%.3f",
        fallback_request_id,
        payload.get("status"),
        time.perf_counter() - started_at,
    )
    return payload


def _get_default_runtime_cache() -> Any:
    global _DEFAULT_RUNTIME_CACHE
    if _DEFAULT_RUNTIME_CACHE is None:
        from product_campaign_pipeline.runtime import BusinessPriorRuntimeSettings, RuntimeCache

        _DEFAULT_RUNTIME_CACHE = RuntimeCache(BusinessPriorRuntimeSettings.from_env())
    return _DEFAULT_RUNTIME_CACHE


def _handle_internal_warmup_job(
    job: dict[str, Any],
    *,
    cache: Any | None,
    request_id: str,
) -> dict[str, Any]:
    """Populate model/runtime caches from a direct Runpod API job."""

    runtime_cache = cache or _get_default_runtime_cache()
    status = runtime_cache.warmup(include_generation=_warmup_include_generation(job))
    succeeded = status.status == "ready"
    return {
        "status": "succeeded" if succeeded else "failed",
        "request_id": request_id,
        "summary": (
            "Runtime warmup completed."
            if succeeded
            else "Runtime warmup completed with errors; inspect warmup details."
        ),
        "warmup": status.model_dump(mode="json"),
    }


def _handle_internal_ping_job(*, request_id: str) -> dict[str, Any]:
    """Return immediately so Runpod dispatch can be tested without loading models."""

    versions = {
        "python": sys.version.split()[0],
        "accelerate": _package_version("accelerate"),
        "diffusers": _package_version("diffusers"),
        "runpod": _package_version("runpod"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
    }
    LOGGER.info("Runpod ping job completed. request_id=%s versions=%s", request_id, versions)
    return {
        "status": "succeeded",
        "request_id": request_id,
        "summary": "Runpod worker ping completed.",
        "versions": versions,
    }


def _is_internal_ping_job(job: dict[str, Any]) -> bool:
    job_input = job.get("input")
    return isinstance(job_input, dict) and bool(job_input.get("_internal_ping"))


def _is_internal_warmup_job(job: dict[str, Any]) -> bool:
    job_input = job.get("input")
    return isinstance(job_input, dict) and bool(
        job_input.get("_internal_warmup") or job_input.get("warmup")
    )


def _debug_errors_enabled(job: dict[str, Any]) -> bool:
    job_input = job.get("input")
    return isinstance(job_input, dict) and bool(job_input.get("_internal_debug"))


def _public_job_input(job: dict[str, Any]) -> object:
    job_input = job.get("input", {})
    if not isinstance(job_input, dict):
        return job_input
    if "_internal_debug" not in job_input:
        return job_input
    public_input = dict(job_input)
    public_input.pop("_internal_debug", None)
    return public_input


def _warmup_include_generation(job: dict[str, Any]) -> bool:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return False
    raw_value = job_input.get("include_generation", True)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw_value)


def _failure_summary(base_summary: str, exc: Exception, *, debug_errors: bool) -> str:
    if not debug_errors:
        return base_summary
    detail = " ".join(str(exc).split())
    if len(detail) > 240:
        detail = f"{detail[:237]}..."
    return f"{base_summary} Debug detail: {type(exc).__name__}: {detail}"


def _warmup_default_runtime_on_start_if_configured() -> None:
    runtime_cache = _get_default_runtime_cache()
    settings = runtime_cache.settings
    if not (settings.warmup_on_start or settings.warmup_generation_on_start):
        return

    LOGGER.info(
        "Starting Runpod worker runtime warmup. include_generation=%s",
        settings.warmup_generation_on_start,
    )
    status = runtime_cache.warmup(include_generation=settings.warmup_generation_on_start)
    if status.status == "ready":
        LOGGER.info("Runpod worker runtime warmup completed.")
    else:
        LOGGER.warning("Runpod worker runtime warmup degraded: %s", status.warmup_error)


def _configure_runpod_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    log_path = Path(os.getenv("PCP_WORKER_LOG_PATH", DEFAULT_WORKER_LOG_PATH))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
            for handler in root_logger.handlers
        ):
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            root_logger.addHandler(file_handler)
        LOGGER.info("Runpod worker persistent log configured at %s", log_path)
    except OSError:
        LOGGER.exception("Failed to configure Runpod worker persistent log at %s", log_path)


def _job_input_keys(job: dict[str, Any]) -> list[str]:
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return []
    return sorted(str(key) for key in job_input)


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def run_business_prior_inference(request: Any, **kwargs: Any) -> Any:
    """Lazy proxy kept patchable for tests while avoiding startup imports."""

    from product_campaign_pipeline.production import run_business_prior_inference as _run

    return _run(request, **kwargs)


def _is_runtime_unavailable_exception(exc: Exception) -> bool:
    try:
        from product_campaign_pipeline.flux import (
            MissingCredentialsError,
            MissingDependencyError,
            MissingModelAccessError,
        )
    except Exception:
        return False
    return isinstance(
        exc,
        (MissingDependencyError, MissingCredentialsError, MissingModelAccessError),
    )


def _worker_result_from_inference(result: Any) -> WorkerGenerationResult:
    request_id = result.request_id
    if result.status == "invalid_source":
        invalid_source = PublicInvalidSourceDetails(
            reason=result.invalid_reason or "invalid_source_photo",
            issues=list(result.source_validity_issues),
            score=result.source_validity_score,
        )
        worker_result = WorkerGenerationResult(
            status="invalid_source",
            request_id=request_id,
            summary=build_public_invalid_source_summary(invalid_source),
            invalid_source=invalid_source,
        )
        return worker_result

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

    _configure_runpod_logging()
    LOGGER.info(
        "Starting Runpod worker. python=%s accelerate=%s diffusers=%s runpod=%s torch=%s",
        sys.version.split()[0],
        _package_version("accelerate"),
        _package_version("diffusers"),
        _package_version("runpod"),
        _package_version("torch"),
    )

    try:
        import runpod
    except ImportError as exc:  # pragma: no cover - deployment-only code path
        raise RuntimeError(
            "Runpod serverless dependencies are missing. "
            "Install the worker image requirements first."
        ) from exc

    _warmup_default_runtime_on_start_if_configured()
    runpod.serverless.start({"handler": handle_public_generation_job})


if __name__ == "__main__":  # pragma: no cover - deployment-only entrypoint
    start_runpod_worker()
