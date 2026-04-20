"""Shared data structures for the product campaign pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel space."""

    x0: int
    y0: int
    x1: int
    y1: int

    @field_validator("x1")
    @classmethod
    def _validate_x1(cls, value: int, info: Any) -> int:
        x0 = info.data.get("x0", 0)
        if value <= x0:
            raise ValueError("x1 must be greater than x0")
        return value

    @field_validator("y1")
    @classmethod
    def _validate_y1(cls, value: int, info: Any) -> int:
        y0 = info.data.get("y0", 0)
        if value <= y0:
            raise ValueError("y1 must be greater than y0")
        return value

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


class CreativeRankingRow(BaseModel):
    """Single row from the CreativeRanking-style manifests."""

    item_id: str
    image_name: str
    ds: int
    pv: int
    clk: int

    @property
    def ctr(self) -> float:
        if self.pv <= 0:
            return 0.0
        return self.clk / float(self.pv)


class ManifestSummary(BaseModel):
    """Summary information for a manifest file."""

    path: str
    row_count: int
    item_count: int
    unique_image_count: int
    total_pv: int
    total_clk: int
    mean_creatives_per_item: float
    min_creatives_per_item: int
    max_creatives_per_item: int

    @property
    def global_ctr(self) -> float:
        if self.total_pv <= 0:
            return 0.0
        return self.total_clk / float(self.total_pv)


class ObservedEvidenceSpec(BaseModel):
    """Observed source evidence split from unobserved degrees of freedom."""

    surface_scope: str = "single_photo_limited"
    uncertainty_level: str = "medium"
    palette: list[str] = Field(default_factory=list)
    structural_palette: list[str] = Field(default_factory=list)
    accent_palette: list[str] = Field(default_factory=list)
    hard_facts: list[str] = Field(default_factory=list)
    soft_hypotheses: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    color_note: str | None = None
    color_confidence: float | None = None
    pattern_note: str | None = None
    coverage_class: str | None = None
    coverage_note: str | None = None
    coverage_ratio: float | None = None
    value_relation_note: str | None = None
    trim_note: str | None = None
    trim_confidence: float | None = None
    boundary_color: str | None = None
    interior_color: str | None = None
    silhouette_note: str | None = None
    aspect_ratio: float | None = None
    top_width_ratio: float | None = None
    form_factor_note: str | None = None
    upper_region_note: str | None = None
    upper_region_confidence: float | None = None
    upper_component_state: str | None = None
    upper_region_color: str | None = None
    body_region_color: str | None = None
    upper_component_count: int | None = None
    lower_region_note: str | None = None
    lower_region_confidence: float | None = None
    lower_component_state: str | None = None
    lower_region_color: str | None = None
    edge_profile_note: str | None = None
    edge_profile_confidence: float | None = None
    edge_thickness_class: str | None = None
    edge_inner_ratio: float | None = None
    soft_structure_note: str | None = None
    soft_structure_confidence: float | None = None
    soft_structure_class: str | None = None
    material_note: str | None = None
    surface_relief_note: str | None = None
    evidence_tags: list[str] = Field(default_factory=list)
    artifact_flags: list[str] = Field(default_factory=list)
    source_validity: str = "valid"
    source_validity_score: float | None = None
    source_validity_issues: list[str] = Field(default_factory=list)
    raw_evidence_caption: str | None = None
    evidence_caption: str | None = None
    reference_crop_path: str | None = None
    reference_cutout_path: str | None = None
    reference_silhouette_path: str | None = None
    reference_mask_path: str | None = None


class ProductIdentitySpec(BaseModel):
    """Identity features extracted or inferred from the product image."""

    phrase: str = "featured product"
    category: str = "product"
    canonical_product_type: str | None = None
    subtype_hint: str | None = None
    source_title: str | None = None
    support_mode: str | None = None
    default_scene_family: str | None = None
    interaction_mode: str | None = None
    style_persona: str | None = None
    casting_note: str | None = None
    stable_base: bool | None = None
    colors: list[str] = Field(default_factory=list)
    material_class: str | None = None
    brand_notes: list[str] = Field(default_factory=list)
    rigid_vs_soft: str | None = None
    requires_human_model: bool = False
    weak_shape_evidence: bool = False
    observed_evidence: ObservedEvidenceSpec = Field(default_factory=ObservedEvidenceSpec)


class LocalizedProduct(BaseModel):
    """Result of localizing the product in an arbitrary source image."""

    source_image: str
    phrase: str
    bbox: BoundingBox
    confidence: float
    crop_path: str | None = None
    mask_path: str | None = None
    identity: ProductIdentitySpec = Field(default_factory=ProductIdentitySpec)


class CampaignPriorSpec(BaseModel):
    """Planner output used to enrich the final prompt."""

    neighbor_item_ids: list[str] = Field(default_factory=list)
    style_atoms: list[str] = Field(default_factory=list)
    scenario_slots: list[str] = Field(default_factory=list)
    scene_family: str | None = None
    support_relation: str | None = None
    semantic_constraints: list[str] = Field(default_factory=list)
    banned_identity_edits: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FluxReferenceImage(BaseModel):
    """Reference image specification for FLUX editing requests."""

    role: str
    path: str
    description: str

    def validate_exists(self) -> None:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {self.path}")


class FluxPromptSpec(BaseModel):
    """Structured representation of the final prompt and request metadata."""

    subject: str
    action: str
    style: str
    context: str
    preservation_constraints: list[str] = Field(default_factory=list)
    reference_images: list[FluxReferenceImage] = Field(default_factory=list)
    prompt_upsampling: bool = False
    seed: int | None = None
    output_format: str = "png"
    width: int | None = None
    height: int | None = None
    safety_tolerance: int = 2
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    max_sequence_length: int = 256
    model: str = "black-forest-labs/FLUX.2-klein-9B"

    def to_prompt_text(self) -> str:
        sections = [
            f"Subject: {self.subject.strip()}",
            f"Action: {self.action.strip()}",
            f"Style: {self.style.strip()}",
            f"Context: {self.context.strip()}",
        ]
        if self.preservation_constraints:
            constraints = "; ".join(self.preservation_constraints)
            sections.append(f"Preservation constraints: {constraints}")
        return " ".join(section for section in sections if section.strip())

    def to_bfl_json(self) -> dict[str, Any]:
        return {
            "prompt": self.to_prompt_text(),
            "seed": self.seed,
            "output_format": self.output_format,
            "width": self.width,
            "height": self.height,
            "safety_tolerance": self.safety_tolerance,
            "prompt_upsampling": self.prompt_upsampling,
        }


class PromptEvaluation(BaseModel):
    """Text-only evaluation of a composed prompt."""

    guideline_alignment: float
    goal_alignment: float
    clarity: float
    preservation_conformity: float
    notes: list[str] = Field(default_factory=list)


class OutputEvaluation(BaseModel):
    """Image-level evaluation output."""

    product_preservation: float
    semantic_correctness: float
    creative_diversity: float
    aesthetic_performance: float
    notes: list[str] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    """Full evaluation bundle."""

    prompt: PromptEvaluation
    output: OutputEvaluation | None = None
