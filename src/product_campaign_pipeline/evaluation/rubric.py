"""Prompt rubric checks plus generic output scoring interfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from product_campaign_pipeline.composer import FluxPrompt


@dataclass(frozen=True)
class PromptCriterionResult:
    """Single rubric criterion for a composed prompt."""

    name: str
    score: float
    passed: bool
    feedback: str


@dataclass(frozen=True)
class PromptEvaluationResult:
    """Evaluation bundle for a composed prompt."""

    prompt: FluxPrompt
    criteria: tuple[PromptCriterionResult, ...]

    @property
    def average_score(self) -> float:
        if not self.criteria:
            return 0.0
        return sum(criterion.score for criterion in self.criteria) / len(self.criteria)

    @property
    def passed(self) -> bool:
        return all(criterion.passed for criterion in self.criteria)

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt.as_dict(),
            "average_score": self.average_score,
            "passed": self.passed,
            "criteria": [
                {
                    "name": criterion.name,
                    "score": criterion.score,
                    "passed": criterion.passed,
                    "feedback": criterion.feedback,
                }
                for criterion in self.criteria
            ],
        }


@dataclass(frozen=True)
class OutputCriterionScore:
    """Single score emitted by an output evaluator."""

    name: str
    score: float
    feedback: str = ""


@dataclass(frozen=True)
class OutputScore:
    """Scored output candidate from a concrete evaluator."""

    evaluator_name: str
    criteria: tuple[OutputCriterionScore, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def average_score(self) -> float:
        if not self.criteria:
            return 0.0
        return sum(criterion.score for criterion in self.criteria) / len(self.criteria)


@dataclass(frozen=True)
class GeneratedOutputCandidate:
    """Interface object passed to pluggable output scorers."""

    image_path: str | None = None
    prompt: FluxPrompt | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class OutputScorer(Protocol):
    """Protocol for model-backed or heuristic output evaluators."""

    name: str

    def score(self, candidate: GeneratedOutputCandidate) -> OutputScore:
        """Return scores for a generated output candidate."""


class CompositeOutputScorer:
    """Combine multiple scorer implementations behind one interface."""

    def __init__(self, scorers: Sequence[OutputScorer]) -> None:
        self._scorers = tuple(scorers)

    def score(self, candidate: GeneratedOutputCandidate) -> tuple[OutputScore, ...]:
        return tuple(scorer.score(candidate) for scorer in self._scorers)


class PromptRubric:
    """Rule-based prompt rubric aligned with the project brief."""

    ordered_sections = ("subject", "action", "style", "context")

    def __init__(self, *, pass_threshold: float = 0.7) -> None:
        self.pass_threshold = float(pass_threshold)

    def evaluate(self, prompt: FluxPrompt | Mapping[str, Any] | str) -> PromptEvaluationResult:
        normalized = self._normalize_prompt(prompt)
        criteria = (
            self._evaluate_guideline_alignment(normalized),
            self._evaluate_project_goal_alignment(normalized),
            self._evaluate_prompt_clarity(normalized),
            self._evaluate_preservation_conformity(normalized),
        )
        return PromptEvaluationResult(prompt=normalized, criteria=criteria)

    def _normalize_prompt(self, prompt: FluxPrompt | Mapping[str, Any] | str) -> FluxPrompt:
        if isinstance(prompt, FluxPrompt):
            return prompt
        if isinstance(prompt, Mapping):
            return FluxPrompt.from_mapping(prompt)
        return FluxPrompt.from_mapping(json.loads(prompt))

    def _evaluate_guideline_alignment(self, prompt: FluxPrompt) -> PromptCriterionResult:
        payload = prompt.as_dict()
        keys = tuple(payload.keys())
        ordered = tuple(key for key in keys if key in self.ordered_sections)
        score = 1.0
        feedback: list[str] = []
        if ordered != self.ordered_sections:
            score -= 0.4
            feedback.append("Prompt sections are not ordered as Subject, Action, Style, Context.")
        missing = [key for key in self.ordered_sections if not payload.get(key)]
        if missing:
            score -= 0.4
            feedback.append(f"Missing required sections: {', '.join(missing)}.")
        if not prompt.preservation_constraints:
            score -= 0.2
            feedback.append("Prompt should include explicit preservation constraints.")
        score = max(score, 0.0)
        return PromptCriterionResult(
            name="official_guideline_alignment",
            score=score,
            passed=score >= self.pass_threshold,
            feedback=" ".join(feedback) or "Prompt follows the expected FLUX structured layout.",
        )

    def _evaluate_project_goal_alignment(self, prompt: FluxPrompt) -> PromptCriterionResult:
        text = " ".join(
            (
                prompt.subject,
                prompt.action,
                prompt.style,
                prompt.context,
                " ".join(prompt.negative_constraints),
            )
        ).lower()
        score = 1.0
        feedback: list[str] = []
        campaign_cues = ("campaign", "editorial", "lifestyle", "commercial", "hero")
        if not any(cue in text for cue in campaign_cues):
            score -= 0.4
            feedback.append("Prompt does not clearly frame the output as a campaign image.")
        if "background" not in text and "environment" not in text and "setting" not in text:
            score -= 0.2
            feedback.append("Prompt should clearly direct a changed background or environment.")
        blocked_catalog_phrases = ("plain catalog", "white background", "monochrome")
        if any(phrase in text for phrase in blocked_catalog_phrases):
            score -= 0.1
        if "seamless white" in text or "plain white" in text:
            score -= 0.1
        score = max(score, 0.0)
        return PromptCriterionResult(
            name="project_goal_alignment",
            score=score,
            passed=score >= self.pass_threshold,
            feedback=" ".join(feedback) or "Prompt describes a campaign-oriented output instead of a catalog shot.",
        )

    def _evaluate_prompt_clarity(self, prompt: FluxPrompt) -> PromptCriterionResult:
        score = 1.0
        feedback: list[str] = []
        combined_sections = [prompt.subject, prompt.action, prompt.style, prompt.context]
        if any(len(section.split()) < 4 for section in combined_sections):
            score -= 0.25
            feedback.append("One or more sections are too short to guide generation reliably.")
        placeholders = ("tbd", "todo", "???", "<", ">")
        serialized = prompt.to_json(indent=None).lower()
        if any(placeholder in serialized for placeholder in placeholders):
            score -= 0.5
            feedback.append("Prompt still contains placeholders or unresolved tokens.")
        if "  " in serialized:
            score -= 0.1
        score = max(score, 0.0)
        return PromptCriterionResult(
            name="prompt_clarity",
            score=score,
            passed=score >= self.pass_threshold,
            feedback=" ".join(feedback) or "Prompt is structured and specific enough for deterministic automation.",
        )

    def _evaluate_preservation_conformity(self, prompt: FluxPrompt) -> PromptCriterionResult:
        preservation_text = " ".join(prompt.preservation_constraints).lower()
        full_text = " ".join(
            (
                prompt.subject,
                prompt.action,
                prompt.style,
                prompt.context,
                " ".join(prompt.negative_constraints),
            )
        ).lower()
        score = 1.0
        feedback: list[str] = []
        required_cues = ("preserve", "exact", "unchanged", "identity", "branding")
        if not any(cue in preservation_text for cue in required_cues):
            score -= 0.5
            feedback.append("Preservation constraints do not clearly enforce exact product identity.")
        risky_phrases = ("change the logo", "new colorway", "different material", "distort")
        if any(phrase in full_text for phrase in risky_phrases):
            score -= 0.5
            feedback.append("Prompt appears to request a product-identity change.")
        score = max(score, 0.0)
        return PromptCriterionResult(
            name="preservation_conformity",
            score=score,
            passed=score >= self.pass_threshold,
            feedback=" ".join(feedback) or "Prompt explicitly preserves the product while allowing scene creativity.",
        )
