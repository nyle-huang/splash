"""Production-oriented single-request business-prior inference surface."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from product_campaign_pipeline.composer.prompts import PromptComposer
from product_campaign_pipeline.flux import DEFAULT_MODEL_ID, Flux2KleinClient
from product_campaign_pipeline.localization import (
    ProductPhoto,
    build_model_backed_localization_pipeline,
    save_localization_artifacts,
    select_primary_mask,
)
from product_campaign_pipeline.review_batch import (
    SCENE_FAMILY_DEFAULTS_BY_SUPPORT,
    LocalizationArtifactRecord,
    ReviewSeedRecord,
    VisionBackbone,
    _conditioning_reference_images,
    _line_seed_offset,
    _primary_generation_input_image,
    assess_category_consistency,
    assess_evidence_consistency,
    assess_prompt_readiness,
    assess_semantic_plausibility,
    build_business_prior,
    build_generation_request,
    build_localized_product,
    compact_focus_alignment_threshold,
    default_support_relation_for_identity,
    extract_generated_focus_artifacts,
    load_retrieval_index,
    maybe_repair_generated_dominant_body_color,
    score_generation_candidate,
    select_reinvention_candidate_modes_for_line,
    should_strengthen_dominant_body_color_guidance,
)


class BusinessPriorInferenceRequest(BaseModel):
    """Single-request runtime contract for future service deployment."""

    model_config = ConfigDict(extra="forbid")

    image_path: str
    product_title: str
    retrieval_index_path: str
    output_dir: str
    hint_phrases: list[str] = Field(default_factory=list)
    request_id: str | None = None
    product_id: str | None = None
    source_page_url: str = "uploaded://local"
    source_image_url: str = "uploaded://local"
    model_id: str = DEFAULT_MODEL_ID
    width: int = Field(default=512, ge=64)
    height: int = Field(default=512, ge=64)
    num_inference_steps: int = Field(default=4, ge=1)
    guidance_scale: float = Field(default=1.0, ge=0.0)
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    analysis_device: Literal["cuda", "cpu", "auto"] = "cpu"
    localization_device: Literal["cuda", "cpu", "auto"] = "cuda"
    candidate_modes: list[str] = Field(default_factory=list)
    skip_analysis: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    seed: int | None = None
    cpu_offload: bool = True
    sequential_cpu_offload: bool = False
    attention_slicing: bool = True


class BusinessPriorCandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    index: int
    score: float
    evidence_score: float
    semantic_score: float = 0.0
    category_consistent: bool = True
    evidence_consistent: bool = True
    semantic_plausible: bool = True


class BusinessPriorInferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "invalid_source"]
    request_id: str
    line: Literal["business_prior"] = "business_prior"
    request_output_dir: str
    source_image_path: str
    localization: dict[str, Any] = Field(default_factory=dict)
    source_validity: str
    source_validity_score: float | None = None
    source_validity_issues: list[str] = Field(default_factory=list)
    output_path: str | None = None
    selected_candidate_mode: str | None = None
    candidate_count: int = 0
    candidate_scores: list[BusinessPriorCandidateScore] = Field(default_factory=list)
    prompt: dict[str, Any] = Field(default_factory=dict)
    prompt_readiness: dict[str, Any] = Field(default_factory=dict)
    observed_evidence: dict[str, Any] = Field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    category_consistency: dict[str, Any] = Field(default_factory=dict)
    semantic_plausibility: dict[str, Any] = Field(default_factory=dict)
    evidence_consistency: dict[str, Any] = Field(default_factory=dict)
    invalid_reason: str | None = None


def run_business_prior_inference(
    request: BusinessPriorInferenceRequest | dict[str, Any],
    *,
    localization_pipeline: Any | None = None,
    retrieval_index: list[Any] | None = None,
    backbone: VisionBackbone | None = None,
    client: Flux2KleinClient | None = None,
    generated_localizer: Any | None = None,
) -> BusinessPriorInferenceResult:
    req = (
        request
        if isinstance(request, BusinessPriorInferenceRequest)
        else BusinessPriorInferenceRequest.model_validate(request)
    )
    request_id = _resolve_request_id(req)
    request_dir = Path(req.output_dir) / request_id
    localization_dir = request_dir / "localization"
    images_dir = request_dir / "images"
    candidates_dir = request_dir / "candidates"
    request_dir.mkdir(parents=True, exist_ok=True)
    localization_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    source_image = Path(req.image_path)
    if not source_image.exists():
        raise FileNotFoundError(f"source image does not exist: {source_image}")

    localizer = localization_pipeline or build_model_backed_localization_pipeline(
        device=req.localization_device
    )
    photo = ProductPhoto(
        image_path=source_image,
        product_id=req.product_id or request_id,
        title=req.product_title,
        hint_phrases=tuple(req.hint_phrases),
    )
    localization_result = localizer.localize(photo)
    selected_mask = select_primary_mask(localization_result)
    artifacts = save_localization_artifacts(
        localization_result,
        localization_dir,
        selected_mask=selected_mask,
    )
    if selected_mask is None or artifacts is None:
        return BusinessPriorInferenceResult(
            status="invalid_source",
            request_id=request_id,
            request_output_dir=str(request_dir),
            source_image_path=str(source_image),
            localization={},
            source_validity="invalid",
            source_validity_issues=["localization_failed"],
            invalid_reason="localization_failed",
        )

    seed = ReviewSeedRecord(
        id=request_id,
        platform="runtime",
        source_page_url=req.source_page_url,
        source_image_url=req.source_image_url,
        product_title=req.product_title,
        hint_phrases=tuple(req.hint_phrases),
        capture_date="runtime",
        local_image_path=source_image,
    )
    localization_record = LocalizationArtifactRecord(
        id=request_id,
        product_title=req.product_title,
        source_page_url=req.source_page_url,
        source_image_url=req.source_image_url,
        local_image_path=source_image,
        selected_phrase=artifacts.phrase,
        selected_confidence=artifacts.confidence,
        selected_box=artifacts.box,
        overlay_path=Path(artifacts.overlay_path),
        crop_path=Path(artifacts.crop_path),
        mask_path=Path(artifacts.mask_path),
    )
    retrieval_items = retrieval_index or load_retrieval_index(req.retrieval_index_path)
    analysis_device = req.analysis_device if req.analysis_device != "auto" else (
        "cpu" if req.device != "cpu" else req.device
    )
    analysis_backbone = backbone or VisionBackbone(device=analysis_device)
    localized = build_localized_product(seed, localization_record, backbone=analysis_backbone)

    localization_payload = {
        "selected_phrase": artifacts.phrase,
        "selected_confidence": round(float(artifacts.confidence), 4),
        "crop_path": artifacts.crop_path,
        "mask_path": artifacts.mask_path,
        "overlay_path": artifacts.overlay_path,
    }

    if localized.identity.observed_evidence.source_validity != "valid":
        return BusinessPriorInferenceResult(
            status="invalid_source",
            request_id=request_id,
            request_output_dir=str(request_dir),
            source_image_path=str(source_image),
            localization=localization_payload,
            source_validity=str(localized.identity.observed_evidence.source_validity),
            source_validity_score=localized.identity.observed_evidence.source_validity_score,
            source_validity_issues=list(localized.identity.observed_evidence.source_validity_issues),
            observed_evidence=localized.identity.observed_evidence.model_dump(),
            invalid_reason="invalid_source_photo",
        )

    prior = build_business_prior(
        seed,
        localized,
        localization_record,
        retrieval_items,
        analysis_backbone,
        top_k=req.top_k,
    )
    scene_family = prior.scene_family or (
        localized.identity.default_scene_family
        or SCENE_FAMILY_DEFAULTS_BY_SUPPORT.get(
            prior.support_relation or default_support_relation_for_identity(localized.identity),
            "editorial_interior",
        )
    )
    support_relation = prior.support_relation or default_support_relation_for_identity(localized.identity)
    candidate_modes = tuple(req.candidate_modes) or select_reinvention_candidate_modes_for_line(
        localized.identity,
        line_name="business_prior",
    )
    candidate_steps = max(req.num_inference_steps, 6) if len(candidate_modes) > 1 else req.num_inference_steps
    candidate_guidance_scale = req.guidance_scale
    if localized.identity.requires_human_model or localized.identity.interaction_mode in {
        "worn",
        "worn_or_carried",
        "held_in_hand",
        "carried_or_resting",
    }:
        candidate_steps = max(candidate_steps, 8)
    if should_strengthen_dominant_body_color_guidance(localized.identity):
        candidate_steps = max(candidate_steps, 8)
        candidate_guidance_scale = max(candidate_guidance_scale, 1.15)

    generation_client = client or Flux2KleinClient(
        model_id=req.model_id,
        device=req.device,
        dtype="bfloat16",
        cpu_offload=req.cpu_offload,
        sequential_cpu_offload=req.sequential_cpu_offload,
        attention_slicing=req.attention_slicing,
    )
    generated_focus_localizer = generated_localizer
    if not req.skip_analysis and generated_focus_localizer is None:
        generated_focus_localizer = build_model_backed_localization_pipeline(
            device="cpu" if req.device != "cpu" else req.device
        )
    composer = PromptComposer()
    candidate_rows: list[dict[str, Any]] = []
    base_seed = req.seed if req.seed is not None else int(prior.metadata.get("creative_seed", 1000))

    for candidate_index, reinvention_mode in enumerate(candidate_modes):
        candidate_seed = base_seed + _line_seed_offset("business_prior") + candidate_index * 101
        prompt_spec = composer.compose_business_prior(
            localized,
            prior,
            seed=candidate_seed,
            reinvention_mode=reinvention_mode,
        )
        candidate_output_path = (
            images_dir / f"{request_id}.business_prior.png"
            if len(candidate_modes) == 1
            else candidates_dir / f"{request_id}.business_prior.{candidate_index:02d}.{reinvention_mode}.png"
        )
        generation_request = build_generation_request(
            generation_client,
            prompt_spec,
            source_image=Path(localized.source_image),
            reference_images=_conditioning_reference_images(localized),
            primary_input_image=_primary_generation_input_image(localized),
            allow_reference_only=False,
            output_path=candidate_output_path,
            width=req.width,
            height=req.height,
            num_inference_steps=candidate_steps,
            guidance_scale=candidate_guidance_scale,
        )
        generation = generation_client.generate(generation_request)
        maybe_repair_generated_dominant_body_color(
            Path(generation.output_path),
            localized,
            generated_localizer=generated_focus_localizer,
            product_photo_factory=ProductPhoto,
            save_artifacts=save_localization_artifacts,
            select_mask=select_primary_mask,
        )
        prompt_readiness = assess_prompt_readiness(
            localized,
            prompt_spec,
            scene_family=scene_family,
            support_relation=support_relation,
        )
        if req.skip_analysis:
            category_consistency: dict[str, Any] = {}
            semantic_plausibility: dict[str, Any] = {}
            evidence_consistency: dict[str, Any] = {}
            candidate_score = float(prompt_readiness.get("score", 0.0))
            if reinvention_mode == "balanced":
                candidate_score += 0.02
        else:
            focus_artifacts = extract_generated_focus_artifacts(
                Path(generation.output_path),
                localized,
                generated_localizer=generated_focus_localizer,
                product_photo_factory=ProductPhoto,
                save_artifacts=save_localization_artifacts,
                select_mask=select_primary_mask,
            )
            category_consistency = assess_category_consistency(
                generation.output_path,
                expected_category=localized.identity.category,
                expected_product_type=localized.identity.canonical_product_type or localized.identity.category,
                backbone=analysis_backbone,
                focus_image_path=None if focus_artifacts is None else focus_artifacts.get("crop_path"),
            )
            semantic_plausibility = assess_semantic_plausibility(
                generation.output_path,
                localized.identity,
                prompt_spec=prompt_spec,
                scene_family=scene_family,
                support_relation=support_relation,
                backbone=analysis_backbone,
                generated_localizer=generated_focus_localizer,
                product_photo_factory=ProductPhoto,
            )
            evidence_consistency = assess_evidence_consistency(
                generation.output_path,
                localized,
                backbone=analysis_backbone,
                generated_localizer=generated_focus_localizer,
                product_photo_factory=ProductPhoto,
                save_artifacts=save_localization_artifacts,
                select_mask=select_primary_mask,
                focus_artifacts=focus_artifacts,
            )
            candidate_score = score_generation_candidate(
                category_consistency=category_consistency,
                semantic_plausibility=semantic_plausibility,
                evidence_consistency=evidence_consistency,
            )
        candidate_rows.append(
            {
                "id": request_id,
                "line": "business_prior",
                "product_title": req.product_title,
                "expected_category": localized.identity.category,
                "canonical_product_type": localized.identity.canonical_product_type,
                "scene_family": scene_family,
                "support_relation": support_relation,
                "source_image_path": str(source_image),
                "crop_path": localized.crop_path,
                "mask_path": localized.mask_path,
                "output_path": generation.output_path,
                "prompt": prompt_spec.model_dump(),
                "observed_evidence": localized.identity.observed_evidence.model_dump(),
                "retrieval_metadata": prior.metadata,
                "category_consistency": category_consistency,
                "semantic_plausibility": semantic_plausibility,
                "evidence_consistency": evidence_consistency,
                "prompt_readiness": prompt_readiness,
                "candidate_mode": reinvention_mode,
                "candidate_index": candidate_index,
                "candidate_score": round(float(candidate_score), 4),
            }
        )
        if req.skip_analysis:
            generation_client.reset_pipeline()

    selected_row = _select_candidate_row(candidate_rows, localized)
    final_output_path = images_dir / f"{request_id}.business_prior.png"
    if Path(selected_row["output_path"]).resolve() != final_output_path.resolve():
        shutil.copy2(selected_row["output_path"], final_output_path)
        selected_row["output_path"] = str(final_output_path)

    candidate_scores = [
        BusinessPriorCandidateScore(
            mode=str(row["candidate_mode"]),
            index=int(row["candidate_index"]),
            score=float(row["candidate_score"]),
            evidence_score=float(row["evidence_consistency"].get("score", 0.0)),
            semantic_score=float(row["semantic_plausibility"].get("score", 0.0)),
            category_consistent=bool(row["category_consistency"].get("is_consistent", True)),
            evidence_consistent=bool(row["evidence_consistency"].get("is_consistent", True)),
            semantic_plausible=bool(row["semantic_plausibility"].get("is_plausible", True)),
        )
        for row in candidate_rows
    ]

    return BusinessPriorInferenceResult(
        status="ok",
        request_id=request_id,
        request_output_dir=str(request_dir),
        source_image_path=str(source_image),
        localization=localization_payload,
        source_validity=str(localized.identity.observed_evidence.source_validity),
        source_validity_score=localized.identity.observed_evidence.source_validity_score,
        source_validity_issues=list(localized.identity.observed_evidence.source_validity_issues),
        output_path=str(selected_row["output_path"]),
        selected_candidate_mode=str(selected_row["candidate_mode"]),
        candidate_count=len(candidate_rows),
        candidate_scores=candidate_scores,
        prompt=dict(selected_row["prompt"]),
        prompt_readiness=dict(selected_row["prompt_readiness"]),
        observed_evidence=dict(selected_row["observed_evidence"]),
        retrieval_metadata=dict(selected_row["retrieval_metadata"]),
        category_consistency=dict(selected_row["category_consistency"]),
        semantic_plausibility=dict(selected_row["semantic_plausibility"]),
        evidence_consistency=dict(selected_row["evidence_consistency"]),
    )


def _resolve_request_id(request: BusinessPriorInferenceRequest) -> str:
    for candidate in (request.request_id, request.product_id, Path(request.image_path).stem, request.product_title):
        if candidate:
            slug = re.sub(r"[^a-z0-9]+", "-", str(candidate).lower()).strip("-")
            if slug:
                return slug
    return "business-prior-request"


def _select_candidate_row(candidate_rows: list[dict[str, Any]], localized: Any) -> dict[str, Any]:
    selection_pool = list(candidate_rows)
    evidence_consistent_rows = [
        row for row in selection_pool if row["evidence_consistency"].get("is_consistent", True)
    ]
    if evidence_consistent_rows:
        selection_pool = evidence_consistent_rows
    category_consistent_rows = [
        row for row in selection_pool if row["category_consistency"].get("is_consistent", True)
    ]
    if category_consistent_rows:
        selection_pool = category_consistent_rows
    ghost_free_rows = [
        row for row in selection_pool if not row["semantic_plausibility"].get("ghost_composite_flag", False)
    ]
    if ghost_free_rows:
        selection_pool = ghost_free_rows
    background_resolved_rows = [
        row for row in selection_pool if not row["semantic_plausibility"].get("background_collapse_flag", False)
    ]
    if background_resolved_rows:
        selection_pool = background_resolved_rows
    product_only_rows = [
        row
        for row in selection_pool
        if not (
            row["semantic_plausibility"].get("people_out_of_frame_required", False)
            and row["semantic_plausibility"].get("person_presence_flag", False)
        )
    ]
    if product_only_rows:
        selection_pool = product_only_rows
    casting_aligned_rows = [
        row
        for row in selection_pool
        if (
            not row["semantic_plausibility"].get("human_supported", False)
            or float(row["semantic_plausibility"].get("casting_margin", 0.0)) >= 0.01
        )
    ]
    if casting_aligned_rows:
        selection_pool = casting_aligned_rows
    dress_layering_margin_values = [
        float(row["semantic_plausibility"].get("dress_layering_margin", 0.0))
        for row in selection_pool
    ]
    if dress_layering_margin_values:
        best_dress_layering_margin = max(dress_layering_margin_values)
        worst_dress_layering_margin = min(dress_layering_margin_values)
        if best_dress_layering_margin - worst_dress_layering_margin >= 0.01:
            dress_layering_preferred_rows = [
                row
                for row in selection_pool
                if float(row["semantic_plausibility"].get("dress_layering_margin", 0.0))
                >= best_dress_layering_margin - 0.004
            ]
            if dress_layering_preferred_rows:
                selection_pool = dress_layering_preferred_rows
    single_model_margin_values = [
        float(row["semantic_plausibility"].get("single_model_margin", 0.0)) for row in selection_pool
    ]
    if single_model_margin_values:
        best_single_model_margin = max(single_model_margin_values)
        worst_single_model_margin = min(single_model_margin_values)
        if best_single_model_margin - worst_single_model_margin >= 0.01:
            single_model_preferred_rows = [
                row
                for row in selection_pool
                if float(row["semantic_plausibility"].get("single_model_margin", 0.0))
                >= best_single_model_margin - 0.004
            ]
            if single_model_preferred_rows:
                selection_pool = single_model_preferred_rows
    compact_focus_rows = [
        row
        for row in selection_pool
        if (
            not row["evidence_consistency"].get("compact_product_focus_required", False)
            or float(row["evidence_consistency"].get("product_prominence_alignment", 0.5))
            >= compact_focus_alignment_threshold(row["evidence_consistency"])
        )
    ]
    if compact_focus_rows:
        selection_pool = compact_focus_rows
    semantic_plausible_rows = [
        row for row in selection_pool if row["semantic_plausibility"].get("is_plausible", True)
    ]
    if semantic_plausible_rows:
        selection_pool = semantic_plausible_rows
    selected_row = max(
        selection_pool,
        key=lambda row: (
            float(row["candidate_score"]),
            float(row["evidence_consistency"].get("score", 0.0)),
            float(row["semantic_plausibility"].get("score", 0.0)),
        ),
    )
    if localized.identity.canonical_product_type == "dress" and float(
        selected_row["semantic_plausibility"].get("dress_layering_margin", 0.0)
    ) < 0.0:
        dress_layering_clean_rows = [
            row
            for row in candidate_rows
            if float(row["semantic_plausibility"].get("dress_layering_margin", 0.0)) >= 0.0
            and row["category_consistency"].get("is_consistent", True)
            and not row["semantic_plausibility"].get("ghost_composite_flag", False)
            and not row["semantic_plausibility"].get("background_collapse_flag", False)
        ]
        if dress_layering_clean_rows:
            selected_row = max(
                dress_layering_clean_rows,
                key=lambda row: (
                    float(row["candidate_score"]),
                    float(row["evidence_consistency"].get("score", 0.0)),
                    float(row["semantic_plausibility"].get("score", 0.0)),
                ),
            )
    return selected_row


def write_business_prior_inference_result(
    result: BusinessPriorInferenceResult,
    destination: str | Path,
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=True), encoding="utf-8")
    return path
