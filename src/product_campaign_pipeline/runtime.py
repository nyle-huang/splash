"""Shared runtime settings and lazy dependency cache for deployment surfaces."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from product_campaign_pipeline.flux import Flux2KleinClient
    from product_campaign_pipeline.review_batch import VisionBackbone

DEFAULT_RUNTIME_MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"


class BusinessPriorRuntimeSettings(BaseModel):
    """Runtime settings shared by pod-mode and serverless deployment surfaces."""

    model_config = ConfigDict(extra="ignore")

    output_root: str = "/workspace/runtime_outputs"
    retrieval_index_path: str = (
        "/workspace/product_campaign_pipeline/data/creative_ranking/retrieval_index.train_top1024.json"
    )
    model_id: str = DEFAULT_RUNTIME_MODEL_ID
    device: str = "cuda"
    analysis_device: str = "cpu"
    localization_device: str = "cuda"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    top_k: int = 5
    cpu_offload: bool = True
    sequential_cpu_offload: bool = False
    attention_slicing: bool = True
    warmup_on_start: bool = False
    warmup_generation_on_start: bool = False

    @classmethod
    def from_env(cls) -> BusinessPriorRuntimeSettings:
        return cls(
            output_root=os.getenv("PCP_OUTPUT_ROOT", cls.model_fields["output_root"].default),
            retrieval_index_path=os.getenv(
                "PCP_RETRIEVAL_INDEX_PATH",
                cls.model_fields["retrieval_index_path"].default,
            ),
            model_id=os.getenv("PCP_MODEL_ID", cls.model_fields["model_id"].default),
            device=os.getenv("PCP_DEVICE", cls.model_fields["device"].default),
            analysis_device=os.getenv(
                "PCP_ANALYSIS_DEVICE",
                cls.model_fields["analysis_device"].default,
            ),
            localization_device=os.getenv(
                "PCP_LOCALIZATION_DEVICE",
                cls.model_fields["localization_device"].default,
            ),
            width=int(os.getenv("PCP_WIDTH", str(cls.model_fields["width"].default))),
            height=int(os.getenv("PCP_HEIGHT", str(cls.model_fields["height"].default))),
            num_inference_steps=int(
                os.getenv(
                    "PCP_NUM_INFERENCE_STEPS",
                    str(cls.model_fields["num_inference_steps"].default),
                )
            ),
            guidance_scale=float(
                os.getenv(
                    "PCP_GUIDANCE_SCALE",
                    str(cls.model_fields["guidance_scale"].default),
                )
            ),
            top_k=int(os.getenv("PCP_TOP_K", str(cls.model_fields["top_k"].default))),
            cpu_offload=_env_flag("PCP_CPU_OFFLOAD", default=True),
            sequential_cpu_offload=_env_flag("PCP_SEQUENTIAL_CPU_OFFLOAD", default=False),
            attention_slicing=_env_flag("PCP_ATTENTION_SLICING", default=True),
            warmup_on_start=_env_flag("PCP_WARMUP_ON_START", default=False),
            warmup_generation_on_start=_env_flag("PCP_WARMUP_GENERATION_ON_START", default=False),
        )


class WarmupStatus(BaseModel):
    """Deployment-readable runtime readiness status."""

    model_config = ConfigDict(extra="forbid")

    status: str
    retrieval_index_loaded: bool
    localization_pipeline_loaded: bool
    analysis_backbone_loaded: bool
    generation_client_initialized: bool
    generation_pipeline_loaded: bool
    warmup_error: str | None = None


class RuntimeCache:
    """Lazy-loaded runtime dependencies for deployment-facing execution."""

    def __init__(self, settings: BusinessPriorRuntimeSettings) -> None:
        self.settings = settings
        self.retrieval_index: list[Any] | None = None
        self.localization_pipeline: Any | None = None
        self.generated_localizer: Any | None = None
        self.backbone: VisionBackbone | None = None
        self.client: Flux2KleinClient | None = None
        self.warmup_error: str | None = None

    def ensure_retrieval_index(self) -> list[Any]:
        if self.retrieval_index is None:
            from product_campaign_pipeline.review_batch import load_retrieval_index

            self.retrieval_index = load_retrieval_index(self.settings.retrieval_index_path)
        return self.retrieval_index

    def ensure_localization_pipeline(self) -> Any:
        if self.localization_pipeline is None:
            from product_campaign_pipeline.localization import (
                build_model_backed_localization_pipeline,
            )

            self.localization_pipeline = build_model_backed_localization_pipeline(
                device=self.settings.localization_device
            )
        return self.localization_pipeline

    def ensure_generated_localizer(self) -> Any:
        if self.generated_localizer is None:
            from product_campaign_pipeline.localization import (
                build_model_backed_localization_pipeline,
            )

            device = "cpu" if self.settings.device != "cpu" else self.settings.device
            self.generated_localizer = build_model_backed_localization_pipeline(device=device)
        return self.generated_localizer

    def ensure_backbone(self) -> VisionBackbone:
        if self.backbone is None:
            from product_campaign_pipeline.review_batch import VisionBackbone

            self.backbone = VisionBackbone(device=self.settings.analysis_device)
        return self.backbone

    def ensure_client(self) -> Flux2KleinClient:
        if self.client is None:
            from product_campaign_pipeline.flux import Flux2KleinClient

            self.client = Flux2KleinClient(
                model_id=self.settings.model_id,
                device=self.settings.device,
                dtype="bfloat16",
                cpu_offload=self.settings.cpu_offload,
                sequential_cpu_offload=self.settings.sequential_cpu_offload,
                attention_slicing=self.settings.attention_slicing,
            )
        return self.client

    def warmup(self, *, include_generation: bool = False) -> WarmupStatus:
        try:
            self.ensure_retrieval_index()
            self.ensure_localization_pipeline()
            self.ensure_backbone()
            self.ensure_client()
            if include_generation:
                self.ensure_client().warmup()
            self.warmup_error = None
        except Exception as exc:  # pragma: no cover - exercised in live deployment
            self.warmup_error = str(exc)
        return self.status()

    def status(self) -> WarmupStatus:
        return WarmupStatus(
            status="ready" if self.warmup_error is None else "degraded",
            retrieval_index_loaded=self.retrieval_index is not None,
            localization_pipeline_loaded=self.localization_pipeline is not None,
            analysis_backbone_loaded=self.backbone is not None,
            generation_client_initialized=self.client is not None,
            generation_pipeline_loaded=bool(self.client and self.client.pipeline_loaded),
            warmup_error=self.warmup_error,
        )


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
