"""Lightweight rubric-based prompt and image scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

from product_campaign_pipeline.types import FluxPromptSpec, OutputEvaluation, PromptEvaluation


def _clamp_score(value: float) -> float:
    return max(1.0, min(5.0, round(value, 2)))


def evaluate_prompt(spec: FluxPromptSpec) -> PromptEvaluation:
    """Score a prompt specification with simple rubric heuristics."""

    notes: list[str] = []
    guideline_alignment = 1.0
    if all([spec.subject.strip(), spec.action.strip(), spec.style.strip(), spec.context.strip()]):
        guideline_alignment += 3.0
    if not spec.prompt_upsampling:
        guideline_alignment += 0.5
    if spec.reference_images:
        guideline_alignment += 0.3

    goal_alignment = 2.0
    prompt_text = spec.to_prompt_text().lower()
    if "preserve" in prompt_text:
        goal_alignment += 1.0
    if "background" in prompt_text or "scene" in prompt_text:
        goal_alignment += 1.0
    if "catalog" in prompt_text:
        notes.append("Prompt explicitly avoids plain catalog treatment.")

    clarity = 2.5
    if len(prompt_text.split()) >= 30:
        clarity += 1.0
    if "todo" not in prompt_text and "placeholder" not in prompt_text:
        clarity += 1.0

    preservation_conformity = 2.0
    if len(spec.preservation_constraints) >= 3:
        preservation_conformity += 1.5
    if any("logo" in item.lower() for item in spec.preservation_constraints):
        preservation_conformity += 0.5
    if any("do not" in item.lower() for item in spec.preservation_constraints):
        preservation_conformity += 0.5

    return PromptEvaluation(
        guideline_alignment=_clamp_score(guideline_alignment),
        goal_alignment=_clamp_score(goal_alignment),
        clarity=_clamp_score(clarity),
        preservation_conformity=_clamp_score(preservation_conformity),
        notes=notes,
    )


def _histogram_similarity(image_a: Image.Image, image_b: Image.Image) -> float:
    hist_a = np.asarray(image_a.histogram(), dtype=np.float32)
    hist_b = np.asarray(image_b.histogram(), dtype=np.float32)
    hist_a /= max(1.0, hist_a.sum())
    hist_b /= max(1.0, hist_b.sum())
    return float(1.0 - np.abs(hist_a - hist_b).sum() / 2.0)


def evaluate_output(source_image_path: Path, generated_image_path: Path) -> OutputEvaluation:
    """Score a generated image using simple visual heuristics."""

    source = Image.open(source_image_path).convert("RGB").resize((256, 256))
    generated = Image.open(generated_image_path).convert("RGB").resize((256, 256))

    preservation = 1.0 + 4.0 * _histogram_similarity(source, generated)
    stat = ImageStat.Stat(generated)
    brightness = sum(stat.mean) / len(stat.mean)
    contrast = sum(stat.stddev) / len(stat.stddev)

    semantic = 3.0
    if 40 <= brightness <= 220:
        semantic += 0.7
    if contrast >= 35:
        semantic += 0.8

    diversity = 3.2 if contrast >= 45 else 2.8
    aesthetics = 2.8
    if 55 <= brightness <= 190:
        aesthetics += 0.8
    if contrast >= 40:
        aesthetics += 0.8

    notes: list[str] = []
    if preservation < 3.0:
        notes.append("Low color-histogram similarity to source product image.")

    return OutputEvaluation(
        product_preservation=_clamp_score(preservation),
        semantic_correctness=_clamp_score(semantic),
        creative_diversity=_clamp_score(diversity),
        aesthetic_performance=_clamp_score(aesthetics),
        notes=notes,
    )
