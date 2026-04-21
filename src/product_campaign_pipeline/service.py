"""HTTP service wrapper for production business-prior inference."""

from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

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
from product_campaign_pipeline.runtime import (
    BusinessPriorRuntimeSettings,
    RuntimeCache,
    WarmupStatus,
)

# Keep the original service-facing name for backwards compatibility with tests and scripts.
BusinessPriorServiceSettings = BusinessPriorRuntimeSettings


def create_app(settings: BusinessPriorServiceSettings | None = None) -> FastAPI:
    runtime_settings = settings or BusinessPriorServiceSettings.from_env()
    cache = RuntimeCache(runtime_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = runtime_settings
        app.state.runtime_cache = cache
        if runtime_settings.warmup_on_start:
            cache.warmup(include_generation=runtime_settings.warmup_generation_on_start)
        yield

    app = FastAPI(
        title="Product Campaign Pipeline Business Prior Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=WarmupStatus)
    async def healthz() -> WarmupStatus:
        return cache.status()

    @app.post("/warmup", response_model=WarmupStatus)
    async def warmup(include_generation: bool = False) -> WarmupStatus:
        return cache.warmup(include_generation=include_generation)

    @app.post("/generate/business-prior", response_model=BusinessPriorInferenceResult)
    async def generate_business_prior(
        image: UploadFile = File(...),
        product_title: str = Form(...),
        hint_phrases: str = Form(""),
        request_id: str | None = Form(default=None),
        product_id: str | None = Form(default=None),
        source_page_url: str = Form(default="uploaded://local"),
        source_image_url: str = Form(default="uploaded://local"),
        width: int | None = Form(default=None),
        height: int | None = Form(default=None),
        num_inference_steps: int | None = Form(default=None),
        guidance_scale: float | None = Form(default=None),
        top_k: int | None = Form(default=None),
        seed: int | None = Form(default=None),
        skip_analysis: bool = Form(default=False),
    ) -> BusinessPriorInferenceResult:
        try:
            upload_path = _save_upload(image, runtime_settings.output_root)
            request_payload = BusinessPriorInferenceRequest(
                image_path=str(upload_path),
                product_title=product_title,
                retrieval_index_path=runtime_settings.retrieval_index_path,
                output_dir=runtime_settings.output_root,
                hint_phrases=_parse_hint_phrases(hint_phrases),
                request_id=request_id,
                product_id=product_id,
                source_page_url=source_page_url,
                source_image_url=source_image_url,
                model_id=runtime_settings.model_id,
                width=width or runtime_settings.width,
                height=height or runtime_settings.height,
                num_inference_steps=num_inference_steps or runtime_settings.num_inference_steps,
                guidance_scale=guidance_scale or runtime_settings.guidance_scale,
                device=runtime_settings.device,
                analysis_device=runtime_settings.analysis_device,
                localization_device=runtime_settings.localization_device,
                top_k=top_k or runtime_settings.top_k,
                seed=seed,
                skip_analysis=skip_analysis,
                cpu_offload=runtime_settings.cpu_offload,
                sequential_cpu_offload=runtime_settings.sequential_cpu_offload,
                attention_slicing=runtime_settings.attention_slicing,
            )
            return run_business_prior_inference(
                request_payload,
                localization_pipeline=cache.ensure_localization_pipeline(),
                retrieval_index=cache.ensure_retrieval_index(),
                backbone=cache.ensure_backbone(),
                client=cache.ensure_client(),
                generated_localizer=None if skip_analysis else cache.ensure_generated_localizer(),
            )
        except (MissingDependencyError, MissingCredentialsError, MissingModelAccessError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _save_upload(upload: UploadFile, output_root: str) -> Path:
    uploads_dir = Path(output_root) / "_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "upload.png").suffix or ".png"
    destination = uploads_dir / f"{uuid.uuid4().hex}{suffix}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def _parse_hint_phrases(value: str) -> list[str]:
    if not value.strip():
        return []
    return [part.strip() for part in value.split("|") if part.strip()]
app = create_app()
