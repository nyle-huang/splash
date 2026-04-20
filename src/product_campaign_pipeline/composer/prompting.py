"""Prompt composition utilities for baseline and business-prior generation lines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from product_campaign_pipeline.planner import RetrievalPlan


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split())


def _dedupe_phrases(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = _clean_text(str(value))
        key = candidate.lower()
        if not candidate or key in seen:
            continue
        normalized.append(candidate)
        seen.add(key)
    return tuple(normalized)


def _join_fragments(values: Sequence[str]) -> str:
    return ", ".join(value for value in values if value)


def _sentence(*parts: str) -> str:
    text = " ".join(part.strip().rstrip(".") for part in parts if part.strip())
    return f"{text}." if text else ""


def _is_fashion_category(category: str | None) -> bool:
    candidate = (category or "").lower()
    fashion_keywords = ("fashion", "apparel", "clothing", "garment", "shoe", "bag")
    return any(keyword in candidate for keyword in fashion_keywords)


@dataclass(frozen=True)
class ProductBrief:
    """Minimal product and campaign context shared by both composers."""

    product_name: str
    category: str | None = None
    product_description: str | None = None
    key_attributes: tuple[str, ...] = ()
    preservation_constraints: tuple[str, ...] = ()
    target_audience: str | None = None
    campaign_goal: str | None = None
    brand_style: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "product_name", _clean_text(self.product_name))
        object.__setattr__(self, "category", _clean_text(self.category) or None)
        object.__setattr__(self, "product_description", _clean_text(self.product_description) or None)
        object.__setattr__(self, "target_audience", _clean_text(self.target_audience) or None)
        object.__setattr__(self, "campaign_goal", _clean_text(self.campaign_goal) or None)
        object.__setattr__(self, "brand_style", _clean_text(self.brand_style) or None)
        object.__setattr__(self, "key_attributes", _dedupe_phrases(self.key_attributes))
        object.__setattr__(
            self,
            "preservation_constraints",
            _dedupe_phrases(self.preservation_constraints),
        )


@dataclass(frozen=True)
class FluxPrompt:
    """Ordered FLUX prompt structure encoded as JSON for local FLUX.2 Klein runs."""

    subject: str
    action: str
    style: str
    context: str
    preservation_constraints: tuple[str, ...] = ()
    negative_constraints: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _clean_text(self.subject))
        object.__setattr__(self, "action", _clean_text(self.action))
        object.__setattr__(self, "style", _clean_text(self.style))
        object.__setattr__(self, "context", _clean_text(self.context))
        object.__setattr__(
            self,
            "preservation_constraints",
            _dedupe_phrases(self.preservation_constraints),
        )
        object.__setattr__(
            self,
            "negative_constraints",
            _dedupe_phrases(self.negative_constraints),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subject": self.subject,
            "action": self.action,
            "style": self.style,
            "context": self.context,
        }
        if self.preservation_constraints:
            payload["preservation_constraints"] = list(self.preservation_constraints)
        if self.negative_constraints:
            payload["negative_constraints"] = list(self.negative_constraints)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=True)

    def to_bfl_prompt(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FluxPrompt":
        return cls(
            subject=str(value.get("subject", "")),
            action=str(value.get("action", "")),
            style=str(value.get("style", "")),
            context=str(value.get("context", "")),
            preservation_constraints=tuple(value.get("preservation_constraints", ())),
            negative_constraints=tuple(value.get("negative_constraints", ())),
            metadata=value.get("metadata", {}),
        )


class BaselineComposer:
    """Compose a FLUX prompt directly from the product brief."""

    def compose(self, brief: ProductBrief) -> FluxPrompt:
        product_label = _join_fragments(
            part
            for part in (
                brief.product_name,
                brief.category,
                brief.product_description,
            )
            if part
        )
        subject_parts = [
            f"The exact {product_label}" if product_label else "The exact product from the reference image",
        ]
        if brief.key_attributes:
            subject_parts.append(
                f"with { _join_fragments(brief.key_attributes[:4]) } kept visually identical to the reference photo"
            )

        action_parts = [
            "Stage the product as the clear hero element in a campaign-grade scene",
            "replace the original background with a richer environment that feels intentionally art directed",
        ]
        if _is_fashion_category(brief.category):
            action_parts.append(
                "if a person is present, allow a different model and pose while keeping the garment design unchanged"
            )
        if brief.campaign_goal:
            action_parts.append(f"optimize the scene for {brief.campaign_goal}")

        style_parts = [
            "High-end commercial photography",
            "premium lighting",
            "realistic materials",
            "credible shadows",
            "professional retouching",
        ]
        if brief.brand_style:
            style_parts.append(brief.brand_style)
        if brief.target_audience:
            style_parts.append(f"tailored to {brief.target_audience}")

        context_parts = [
            "Use a believable lifestyle or editorial setting with depth, props, and atmosphere",
            "avoid a plain catalog look or a monochrome seamless background",
            "make the background, framing, and supporting details feel intentionally creative",
        ]

        preservation_constraints = list(self._baseline_preservation_constraints(brief))
        negative_constraints = [
            "Do not alter the product logo, proportions, materials, or core geometry",
            "Do not introduce duplicate hero products or contradictory product details",
            "Do not generate a plain white catalog background",
        ]

        return FluxPrompt(
            subject=_sentence(*subject_parts),
            action=_sentence(*action_parts),
            style=_sentence(_join_fragments(style_parts)),
            context=_sentence(*context_parts),
            preservation_constraints=tuple(preservation_constraints),
            negative_constraints=tuple(negative_constraints),
            metadata={"composer": self.__class__.__name__},
        )

    def _baseline_preservation_constraints(self, brief: ProductBrief) -> tuple[str, ...]:
        constraints = [
            "Preserve the exact product identity from the reference image",
            "Keep branding, silhouette, material cues, and dominant colors unchanged",
            "Maintain realistic product physics and semantic correctness",
        ]
        if _is_fashion_category(brief.category):
            constraints.append(
                "Keep the exact garment pattern, cut, trims, and fabric behavior while the model and pose may change"
            )
        constraints.extend(brief.preservation_constraints)
        return _dedupe_phrases(constraints)


class BusinessPriorComposer(BaselineComposer):
    """Compose a FLUX prompt using retrieved high-CTR creative priors."""

    def compose(self, brief: ProductBrief, retrieval_plan: RetrievalPlan) -> FluxPrompt:
        baseline = super().compose(brief)
        subject = baseline.subject
        if retrieval_plan.aggregated_subject_hints:
            subject = _sentence(
                baseline.subject,
                f"Borrow supporting hero-framing cues such as {_join_fragments(retrieval_plan.aggregated_subject_hints)}",
            )

        action = baseline.action
        if retrieval_plan.aggregated_action_hints:
            action = _sentence(
                baseline.action,
                f"Integrate high-CTR action cues such as {_join_fragments(retrieval_plan.aggregated_action_hints)}",
            )

        style = baseline.style
        if retrieval_plan.aggregated_style_hints:
            style = _sentence(
                baseline.style,
                f"Use styling cues inspired by retrieved winners such as {_join_fragments(retrieval_plan.aggregated_style_hints)}",
            )

        context = baseline.context
        if retrieval_plan.aggregated_context_hints:
            context = _sentence(
                baseline.context,
                f"Shape the environment with retrieved contextual cues such as {_join_fragments(retrieval_plan.aggregated_context_hints)}",
            )

        preservation_constraints = _dedupe_phrases(
            (
                *baseline.preservation_constraints,
                *retrieval_plan.aggregated_preservation_hints,
            )
        )
        negative_constraints = _dedupe_phrases(
            (
                *baseline.negative_constraints,
                *retrieval_plan.aggregated_negative_hints,
            )
        )

        return FluxPrompt(
            subject=subject,
            action=action,
            style=style,
            context=context,
            preservation_constraints=preservation_constraints,
            negative_constraints=negative_constraints,
            metadata={
                "composer": self.__class__.__name__,
                "retrieved_examples": len(retrieval_plan.selected),
            },
        )
