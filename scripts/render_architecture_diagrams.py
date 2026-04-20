#!/usr/bin/env python3
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "presentation" / "figures"


@dataclass(frozen=True)
class BoxSpec:
    name: str
    x: float
    y: float
    w: float
    h: float
    title: str
    body: tuple[str, ...]
    face: str
    edge: str
    title_color: str = "#1f2f52"
    text_color: str = "#233548"


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.facecolor": "#fcfbf8",
            "figure.facecolor": "#fcfbf8",
            "savefig.facecolor": "#fcfbf8",
            "savefig.bbox": "tight",
        }
    )


def _wrap_lines(lines: tuple[str, ...], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        pieces = textwrap.wrap(
            line,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not pieces:
            wrapped.append("")
            continue
        for idx, piece in enumerate(pieces):
            prefix = "• " if idx == 0 else "  "
            wrapped.append(f"{prefix}{piece}")
    return wrapped


def _draw_box(ax, spec: BoxSpec, *, body_width: int = 28, title_size: int = 12, body_size: float = 8.2) -> None:
    patch = FancyBboxPatch(
        (spec.x, spec.y),
        spec.w,
        spec.h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=spec.edge,
        facecolor=spec.face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        spec.x + 0.03 * spec.w,
        spec.y + spec.h - 0.08 * spec.h,
        spec.title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=spec.title_color,
        zorder=6,
    )
    ax.text(
        spec.x + 0.03 * spec.w,
        spec.y + spec.h - 0.23 * spec.h,
        "\n".join(_wrap_lines(spec.body, width=body_width)),
        ha="left",
        va="top",
        fontsize=body_size,
        color=spec.text_color,
        linespacing=1.45,
        zorder=6,
    )


def _point(spec: BoxSpec, side: str, frac: float = 0.5, pad: float = 0.0) -> tuple[float, float]:
    if side == "left":
        return (spec.x - pad, spec.y + spec.h * frac)
    if side == "right":
        return (spec.x + spec.w + pad, spec.y + spec.h * frac)
    if side == "top":
        return (spec.x + spec.w * frac, spec.y + spec.h + pad)
    if side == "bottom":
        return (spec.x + spec.w * frac, spec.y - pad)
    raise ValueError(f"Unsupported side: {side}")


def _line(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "#475569", width: float = 2.0) -> None:
    ax.add_line(
        Line2D(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=width,
            solid_capstyle="round",
            zorder=4,
        )
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#475569",
    width: float = 2.0,
    style: str = "-|>",
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=16,
        linewidth=width,
        color=color,
        connectionstyle="arc3,rad=0.0",
        zorder=5,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(patch)


def _poly_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = "#475569",
    width: float = 2.0,
) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1], strict=False):
        _line(ax, start, end, color=color, width=width)
    if len(points) == 2:
        _arrow(ax, points[0], points[1], color=color, width=width)
    else:
        _arrow(ax, points[-2], points[-1], color=color, width=width)


def _label(ax, x: float, y: float, text: str, *, size: float = 9.0, color: str = "#526176", weight: str = "normal") -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight, zorder=7)


def _panel(ax, x: float, y: float, w: float, h: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0.9,
            edgecolor="#ddd6ce",
            facecolor="#fafaf9",
            zorder=0,
        )
    )
    ax.text(x + 0.018, y + h - 0.04, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color="#374151", zorder=1)


def _save_figure(fig, stem: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{stem}.png"
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path


def render_system_overview() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, "Figure 1. Dual-Line Product-to-Campaign System Overview", fontsize=18, fontweight="bold", color="#14213d")
    ax.text(
        0.03,
        0.932,
        "Shared upstream inference feeds two generation lines: a direct baseline composer and a business-prior line "
        "enriched by evidence-compatible CTR-aware retrieval from CreativeRanking.",
        fontsize=10.5,
        color="#526176",
    )

    boxes = {
        "input": BoxSpec(
            "input", 0.05, 0.64, 0.18, 0.18,
            "User Input",
            (
                "Single everyday product photo",
                "Often cluttered, partial-view, or weakly framed",
                "Source of identity evidence and reference images",
            ),
            "#e8f1ff", "#4b7bec",
        ),
        "corpus": BoxSpec(
            "corpus", 0.05, 0.34, 0.18, 0.18,
            "CreativeRanking Corpus",
            (
                "Creative images plus CTR logs",
                "Offline retrieval index and prior source",
                "Scene and style supervision for business-prior",
            ),
            "#fff3db", "#d97706",
        ),
        "upstream": BoxSpec(
            "upstream", 0.31, 0.47, 0.22, 0.34,
            "Shared Upstream Inference",
            (
                "ProductLocalizationPipeline",
                "Crop, mask, overlay, cutout, and silhouette artifacts",
                "build_localized_product()",
                "ObservedEvidenceSpec and ProductIdentitySpec",
            ),
            "#e7f8f2", "#0f766e",
        ),
        "baseline": BoxSpec(
            "baseline", 0.60, 0.67, 0.16, 0.15,
            "Baseline Line",
            (
                "compose_baseline()",
                "Identity-led scene and support planning",
                "Evidence-preserving FluxPromptSpec",
            ),
            "#eef2ff", "#6366f1",
        ),
        "business": BoxSpec(
            "business", 0.60, 0.36, 0.16, 0.20,
            "Business-Prior Line",
            (
                "Visual retrieval over CreativeRanking",
                "Evidence compatibility filter",
                "CTR-aware style and scene planning",
                "CampaignPriorSpec -> compose_business_prior()",
            ),
            "#fff1f2", "#e11d48",
        ),
        "generator": BoxSpec(
            "generator", 0.81, 0.47, 0.15, 0.24,
            "Shared Local Generator",
            (
                "Flux2KleinClient",
                "black-forest-labs/FLUX.2-klein-9B",
                "Diffusers runtime on GCE L4",
                "Reference-image conditioning and CPU offload",
            ),
            "#f3f4f6", "#334155",
        ),
        "review": BoxSpec(
            "review", 0.81, 0.13, 0.15, 0.20,
            "Evaluation and Review",
            (
                "Upstream review board",
                "Candidate reranking: category, semantics, evidence",
                "Portable HTML bundles and human checkpoints",
            ),
            "#edf7ed", "#2f855a",
        ),
    }

    for box in boxes.values():
        _draw_box(ax, box, body_width=27 if box.w < 0.17 else 29)

    _poly_arrow(ax, [_point(boxes["input"], "right", 0.48), _point(boxes["upstream"], "left", 0.64)], color="#4b5563")
    _poly_arrow(ax, [_point(boxes["corpus"], "right", 0.50), _point(boxes["upstream"], "left", 0.36)], color="#4b5563")
    _poly_arrow(ax, [_point(boxes["upstream"], "right", 0.72), _point(boxes["baseline"], "left", 0.50)], color="#4b5563")
    _poly_arrow(ax, [_point(boxes["upstream"], "right", 0.30), _point(boxes["business"], "left", 0.52)], color="#4b5563")
    _poly_arrow(ax, [_point(boxes["baseline"], "right", 0.48), _point(boxes["generator"], "left", 0.70)], color="#4b5563")
    _poly_arrow(ax, [_point(boxes["business"], "right", 0.52), _point(boxes["generator"], "left", 0.35)], color="#4b5563")
    _poly_arrow(ax, [_point(boxes["generator"], "bottom", 0.5), _point(boxes["review"], "top", 0.5)], color="#4b5563")

    _label(ax, 0.52, 0.84, "localized product + evidence graph", size=9.5, color="#0f766e")
    _label(ax, 0.52, 0.31, "retrieval candidates + CTR priors", size=9.5, color="#be123c")
    _label(ax, 0.84, 0.75, "shared prompt -> image generation", size=9.5, color="#334155")
    _label(ax, 0.885, 0.405, "selected outputs and reports", size=9.5, color="#2f855a")

    footer = (
        "Implemented architecture in /src/product_campaign_pipeline: shared localization and evidence modeling, baseline prompt "
        "composition, business-prior retrieval planning, local FLUX generation, and review-board export."
    )
    ax.text(0.03, 0.045, footer, fontsize=9, color="#64748b")
    return _save_figure(fig, "figure_1_system_overview")


def render_component_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, "Figure 2. Runtime Component and Artifact Architecture", fontsize=18, fontweight="bold", color="#14213d")
    ax.text(
        0.03,
        0.932,
        "Component-level view of the implemented runtime, including the artifacts that move through the baseline and business-prior lines.",
        fontsize=10.5,
        color="#526176",
    )

    _panel(ax, 0.03, 0.10, 0.20, 0.80, "Inputs and Localization")
    _panel(ax, 0.27, 0.10, 0.20, 0.80, "Identity and Evidence")
    _panel(ax, 0.51, 0.10, 0.20, 0.80, "Baseline and Business-Prior")
    _panel(ax, 0.75, 0.10, 0.20, 0.80, "Generation and Review")

    boxes = {
        "record": BoxSpec(
            "record", 0.06, 0.72, 0.15, 0.10,
            "Input Record",
            (
                "source photo or seeded review image",
                "title, hint phrases, and metadata",
            ),
            "#e8f1ff", "#4b7bec",
        ),
        "localization": BoxSpec(
            "localization", 0.06, 0.49, 0.15, 0.11,
            "Localization Runtime",
            (
                "phrase grounding and mask selection",
                "crop, mask, and overlay export",
            ),
            "#e7f8f2", "#0f766e",
        ),
        "artifacts": BoxSpec(
            "artifacts", 0.06, 0.25, 0.15, 0.11,
            "Localization Artifacts",
            (
                "selected phrase and confidence",
                "crop, mask, and overlay paths",
            ),
            "#f3f4f6", "#475569",
        ),
        "builder": BoxSpec(
            "builder", 0.30, 0.69, 0.15, 0.12,
            "Identity Builder",
            (
                "build_localized_product()",
                "category, subtype, affordance, persona inference",
            ),
            "#eef2ff", "#6366f1",
        ),
        "identity": BoxSpec(
            "identity", 0.29, 0.45, 0.17, 0.18,
            "Evidence and Identity",
            (
                "ObservedEvidenceSpec: palette, coverage, trim, form factor",
                "ProductIdentitySpec: support, scene, interaction, persona",
                "hard facts, soft hypotheses, and unknowns",
            ),
            "#ecfeff", "#0891b2",
        ),
        "localized": BoxSpec(
            "localized", 0.30, 0.22, 0.15, 0.12,
            "Localized Product",
            (
                "bbox, crop and mask paths",
                "identity bundle and reference images",
            ),
            "#f5f3ff", "#7c3aed",
        ),
        "baseline": BoxSpec(
            "baseline", 0.54, 0.69, 0.15, 0.12,
            "Baseline Path",
            (
                "identity-led prompt composition",
                "evidence-preserving FluxPromptSpec",
            ),
            "#eef2ff", "#4f46e5",
        ),
        "planner": BoxSpec(
            "planner", 0.53, 0.46, 0.17, 0.14,
            "Business-Prior Planner",
            (
                "visual retrieval and evidence filter",
                "CTR-aware CampaignPriorSpec",
                "scene, support, and style atoms",
            ),
            "#fff1f2", "#e11d48",
        ),
        "prompt": BoxSpec(
            "prompt", 0.53, 0.22, 0.17, 0.13,
            "Prompt and Request Artifacts",
            (
                "FluxPromptSpec and reference bundle",
                "candidate modes: balanced, reveal, hero",
            ),
            "#f3f4f6", "#334155",
        ),
        "flux": BoxSpec(
            "flux", 0.78, 0.69, 0.15, 0.12,
            "Local FLUX Runtime",
            (
                "build_request() and generate()",
                "local FLUX.2-klein-9B on L4",
            ),
            "#f3f4f6", "#334155",
        ),
        "scoring": BoxSpec(
            "scoring", 0.78, 0.45, 0.15, 0.14,
            "Scoring and Selection",
            (
                "category, semantics, and evidence checks",
                "score_generation_candidate()",
            ),
            "#edf7ed", "#2f855a",
        ),
        "outputs": BoxSpec(
            "outputs", 0.78, 0.22, 0.15, 0.12,
            "Review Outputs",
            (
                "portable review board",
                "generation report and logs",
            ),
            "#fff7ed", "#c2410c",
        ),
    }

    for box in boxes.values():
        _draw_box(ax, box, body_width=24, title_size=11, body_size=7.2)

    _poly_arrow(ax, [_point(boxes["record"], "bottom", 0.50), _point(boxes["localization"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["localization"], "bottom", 0.50), _point(boxes["artifacts"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["record"], "right", 0.55), _point(boxes["builder"], "left", 0.55)])
    _poly_arrow(ax, [_point(boxes["artifacts"], "right", 0.55), _point(boxes["builder"], "left", 0.25)])
    _poly_arrow(ax, [_point(boxes["builder"], "bottom", 0.50), _point(boxes["identity"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["identity"], "bottom", 0.50), _point(boxes["localized"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["localized"], "right", 0.78), _point(boxes["baseline"], "left", 0.50)])
    _poly_arrow(ax, [_point(boxes["localized"], "right", 0.38), _point(boxes["planner"], "left", 0.52)])
    _poly_arrow(ax, [_point(boxes["baseline"], "bottom", 0.50), _point(boxes["prompt"], "top", 0.30)])
    _poly_arrow(ax, [_point(boxes["planner"], "bottom", 0.50), _point(boxes["prompt"], "top", 0.70)])
    _poly_arrow(ax, [_point(boxes["prompt"], "right", 0.52), _point(boxes["flux"], "left", 0.52)])
    _poly_arrow(ax, [_point(boxes["flux"], "bottom", 0.50), _point(boxes["scoring"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["scoring"], "bottom", 0.50), _point(boxes["outputs"], "top", 0.50)])

    _label(ax, 0.18, 0.66, "crop, mask, and overlay assets", size=8.4)
    _label(ax, 0.38, 0.38, "shared identity bundle passed to both lines", size=8.4, color="#7c3aed")
    _label(ax, 0.62, 0.38, "prompt assembly remains shared", size=8.4, color="#334155")
    _label(ax, 0.855, 0.39, "generation, scoring, and review are shared", size=8.4, color="#2f855a")

    footer = (
        "The business-prior line differs from the baseline only after LocalizedProduct formation: it adds evidence-compatible "
        "retrieval and CTR-aware scene/style planning before prompt composition, while generation and review remain shared."
    )
    ax.text(0.03, 0.01, footer, fontsize=9, color="#64748b")
    return _save_figure(fig, "figure_2_runtime_components")


def render_planner_detail() -> Path:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, "Figure 3. Business-Prior Planner", fontsize=18, fontweight="bold", color="#14213d")
    ax.text(
        0.03,
        0.932,
        "The planner combines an offline CreativeRanking retrieval index with runtime evidence compatibility to produce a CampaignPriorSpec.",
        fontsize=10.5,
        color="#526176",
    )

    _panel(ax, 0.03, 0.60, 0.94, 0.28, "Offline Index Construction")
    _panel(ax, 0.03, 0.15, 0.94, 0.37, "Runtime Planning for One Localized Product")

    boxes = {
        "manifest": BoxSpec(
            "manifest", 0.06, 0.66, 0.17, 0.12,
            "CreativeRanking Manifests",
            ("item_id, image_name, pv, clk; creative image paths; CTR supervision",),
            "#fff3db", "#d97706",
        ),
        "select": BoxSpec(
            "select", 0.29, 0.66, 0.17, 0.12,
            "Top-Creative Selection",
            ("select_top_creatives() keeps high-signal candidate creatives before indexing",),
            "#fef3c7", "#ca8a04",
        ),
        "backbone": BoxSpec(
            "backbone", 0.52, 0.64, 0.19, 0.16,
            "VisionBackbone",
            ("caption_image()", "encode_image()", "encode_texts()", "category and evidence inference"),
            "#e7f8f2", "#0f766e",
        ),
        "index": BoxSpec(
            "index", 0.77, 0.65, 0.16, 0.14,
            "Retrieval Index",
            ("RetrievalCandidate records with caption, embedding, scene, support, and observed evidence",),
            "#f3f4f6", "#475569",
        ),
        "localized": BoxSpec(
            "localized", 0.06, 0.25, 0.18, 0.14,
            "LocalizedProduct",
            ("source image, crop/mask, ProductIdentitySpec, ObservedEvidenceSpec",),
            "#f5f3ff", "#7c3aed",
        ),
        "filter": BoxSpec(
            "filter", 0.30, 0.24, 0.18, 0.18,
            "Candidate Retrieval",
            (
                "filter_retrieval_candidates()",
                "allowed support relations",
                "image similarity + evidence compatibility",
                "hard-conflict rejection",
            ),
            "#eef2ff", "#4f46e5",
        ),
        "gate": BoxSpec(
            "gate", 0.54, 0.25, 0.16, 0.16,
            "Compatibility Gate",
            (
                "evaluate_retrieval_candidate()",
                "should_fallback_to_category_prior()",
                "top_matches or fallback decision",
            ),
            "#fff1f2", "#e11d48",
        ),
        "scene": BoxSpec(
            "scene", 0.76, 0.25, 0.18, 0.15,
            "Scene and Style Plan",
            (
                "choose_support_relation()",
                "choose_scene_family()",
                "build_style_plan()",
            ),
            "#ecfeff", "#0891b2",
        ),
        "fallback": BoxSpec(
            "fallback", 0.54, 0.05, 0.16, 0.12,
            "Fallback Prior",
            ("build_category_fallback_prior() with neutral scene-safe defaults",),
            "#fff7ed", "#c2410c",
        ),
        "output": BoxSpec(
            "output", 0.76, 0.04, 0.18, 0.14,
            "CampaignPriorSpec",
            (
                "neighbor_item_ids",
                "style_atoms, scenario_slots",
                "scene_family, support_relation",
                "semantic_constraints + metadata",
            ),
            "#edf7ed", "#2f855a",
        ),
    }

    for box in boxes.values():
        _draw_box(ax, box, body_width=24, title_size=11, body_size=7.4)

    _poly_arrow(ax, [_point(boxes["manifest"], "right", 0.5), _point(boxes["select"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["select"], "right", 0.5), _point(boxes["backbone"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["backbone"], "right", 0.5), _point(boxes["index"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["localized"], "right", 0.5), _point(boxes["filter"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["index"], "bottom", 0.50), (0.85, 0.58), (0.85, 0.44), (0.39, 0.44), _point(boxes["filter"], "top", 0.78)])
    _poly_arrow(ax, [_point(boxes["filter"], "right", 0.5), _point(boxes["gate"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["gate"], "right", 0.5), _point(boxes["scene"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["gate"], "bottom", 0.5), _point(boxes["fallback"], "top", 0.5)])
    _poly_arrow(ax, [_point(boxes["scene"], "bottom", 0.55), _point(boxes["output"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["fallback"], "right", 0.5), _point(boxes["output"], "left", 0.35)])

    _label(ax, 0.84, 0.58, "offline RetrievalCandidate index", size=8.6)
    _label(ax, 0.41, 0.21, "visual-first candidate shortlist", size=8.6)
    _label(ax, 0.65, 0.22, "fallback only when compatibility is weak", size=8.6, color="#be123c")
    _label(ax, 0.85, 0.22, "business-prior output passed to composer", size=8.6, color="#2f855a")

    return _save_figure(fig, "figure_3_planner_detail")


def render_composer_detail() -> Path:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, "Figure 4. Prompt Composer", fontsize=18, fontweight="bold", color="#14213d")
    ax.text(
        0.03,
        0.932,
        "The composer converts localized evidence and optional business priors into a structured FluxPromptSpec with explicit guardrails.",
        fontsize=10.5,
        color="#526176",
    )

    _panel(ax, 0.03, 0.64, 0.94, 0.24, "Shared Composer Core")
    _panel(ax, 0.03, 0.32, 0.94, 0.24, "Baseline and Business-Prior Branches")
    _panel(ax, 0.03, 0.07, 0.94, 0.18, "Prompt Artifact Output")

    boxes = {
        "localized": BoxSpec(
            "localized", 0.06, 0.69, 0.17, 0.12,
            "LocalizedProduct",
            ("identity phrase, evidence graph, persona, support mode, reference image paths",),
            "#f5f3ff", "#7c3aed",
        ),
        "prior": BoxSpec(
            "prior", 0.06, 0.39, 0.17, 0.11,
            "CampaignPriorSpec",
            ("style_atoms, scene_family, support_relation, semantic constraints, banned edits",),
            "#fff1f2", "#e11d48",
        ),
        "core": BoxSpec(
            "core", 0.30, 0.68, 0.24, 0.15,
            "Shared Prompt Core",
            (
                "_references()",
                "_subject_phrase(), _observed_evidence_summary()",
                "_persona_clause(), _support_clause(), _scene_clause()",
            ),
            "#e7f8f2", "#0f766e",
        ),
        "guardrails": BoxSpec(
            "guardrails", 0.61, 0.68, 0.28, 0.15,
            "Evidence Guardrails",
            (
                "_evidence_completion_guardrails()",
                "_type_guardrails()",
                "hard facts preserved before unseen-surface invention",
            ),
            "#ecfeff", "#0891b2",
        ),
        "baseline": BoxSpec(
            "baseline", 0.31, 0.39, 0.24, 0.12,
            "compose_baseline()",
            ("subject + action + style + context from localized evidence only",),
            "#eef2ff", "#4f46e5",
        ),
        "business": BoxSpec(
            "business", 0.62, 0.36, 0.27, 0.17,
            "compose_business_prior()",
            (
                "starts from baseline prompt",
                "adds style atoms and extra context",
                "keeps retrieval cues scene-only when evidence conflicts",
            ),
            "#fff1f2", "#e11d48",
        ),
        "refs": BoxSpec(
            "refs", 0.08, 0.11, 0.22, 0.11,
            "Reference Image Bundle",
            ("base image, crop, and evidence artifacts passed to local FLUX editing",),
            "#fff7ed", "#c2410c",
        ),
        "output": BoxSpec(
            "output", 0.40, 0.11, 0.24, 0.11,
            "FluxPromptSpec",
            ("subject, action, style, context, preservation_constraints",),
            "#f3f4f6", "#334155",
        ),
    }

    for box in boxes.values():
        _draw_box(ax, box, body_width=24, title_size=11, body_size=7.4)

    _poly_arrow(ax, [_point(boxes["localized"], "right", 0.55), _point(boxes["core"], "left", 0.55)])
    _poly_arrow(ax, [_point(boxes["core"], "right", 0.55), _point(boxes["guardrails"], "left", 0.55)])
    _poly_arrow(ax, [_point(boxes["core"], "bottom", 0.50), _point(boxes["baseline"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["guardrails"], "bottom", 0.50), _point(boxes["business"], "top", 0.65)])
    _poly_arrow(ax, [_point(boxes["prior"], "right", 0.50), _point(boxes["business"], "left", 0.50)])
    _poly_arrow(ax, [_point(boxes["baseline"], "bottom", 0.50), _point(boxes["output"], "top", 0.35)])
    _poly_arrow(ax, [_point(boxes["business"], "bottom", 0.40), _point(boxes["output"], "top", 0.78)])
    _poly_arrow(ax, [_point(boxes["localized"], "bottom", 0.32), _point(boxes["refs"], "top", 0.50)])

    _label(ax, 0.50, 0.61, "shared evidence-to-language synthesis", size=8.6)
    _label(ax, 0.76, 0.59, "business prior augments but does not override evidence", size=8.4, color="#be123c")
    _label(ax, 0.21, 0.24, "reference-image path stays separate from prompt text", size=8.4, color="#c2410c")

    return _save_figure(fig, "figure_4_composer_detail")


def render_reranker_detail() -> Path:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.965, "Figure 5. Candidate Reranker and Review Loop", fontsize=18, fontweight="bold", color="#14213d")
    ax.text(
        0.03,
        0.932,
        "For each seed and line, multiple reinvention candidates are generated, scored, and collapsed into one selected row for the review board.",
        fontsize=10.5,
        color="#526176",
    )

    _panel(ax, 0.03, 0.64, 0.94, 0.24, "Candidate Generation")
    _panel(ax, 0.03, 0.31, 0.94, 0.25, "Scoring Modules")
    _panel(ax, 0.03, 0.07, 0.94, 0.17, "Selection and Reporting")

    boxes = {
        "inputs": BoxSpec(
            "inputs", 0.06, 0.70, 0.18, 0.11,
            "Line Inputs",
            ("LocalizedProduct plus CampaignPriorSpec when the line is business-prior",),
            "#f5f3ff", "#7c3aed",
        ),
        "modes": BoxSpec(
            "modes", 0.30, 0.68, 0.17, 0.15,
            "Candidate Modes",
            ("select_reinvention_candidate_modes(): balanced, reveal, hero",),
            "#eef2ff", "#4f46e5",
        ),
        "request": BoxSpec(
            "request", 0.53, 0.68, 0.17, 0.15,
            "Prompt + Generation Request",
            ("compose_*() -> build_generation_request()",),
            "#e7f8f2", "#0f766e",
        ),
        "generate": BoxSpec(
            "generate", 0.76, 0.68, 0.16, 0.15,
            "Generated Candidate",
            ("Flux2KleinClient.generate() writes one image per mode and line",),
            "#f3f4f6", "#334155",
        ),
        "category": BoxSpec(
            "category", 0.07, 0.37, 0.23, 0.12,
            "Category Consistency",
            ("assess_category_consistency() compares image embeddings to category text prompts",),
            "#edf7ed", "#2f855a",
        ),
        "semantic": BoxSpec(
            "semantic", 0.38, 0.37, 0.23, 0.12,
            "Semantic Plausibility",
            (
                "assess_semantic_plausibility() scores support relation and scene-family compatibility",
                "evaluate_prompt_scene_conflicts() penalizes contradictory prompts",
            ),
            "#ecfeff", "#0891b2",
        ),
        "evidence": BoxSpec(
            "evidence", 0.69, 0.35, 0.24, 0.16,
            "Evidence Consistency",
            (
                "assess_evidence_consistency() compares generated appearance to observed source evidence",
                "uses generated localization, palette, parts, and contradiction checks",
            ),
            "#fff1f2", "#e11d48",
        ),
        "score": BoxSpec(
            "score", 0.18, 0.11, 0.23, 0.10,
            "score_generation_candidate()",
            ("0.65 evidence + 0.18 semantics + consistency bonuses",),
            "#fff7ed", "#c2410c",
        ),
        "select": BoxSpec(
            "select", 0.47, 0.10, 0.20, 0.12,
            "Winner Selection",
            ("max(candidate_score) selects one output per seed and line",),
            "#fef3c7", "#ca8a04",
        ),
        "report": BoxSpec(
            "report", 0.73, 0.10, 0.20, 0.12,
            "Review Bundle",
            ("generation_report.json, copied final image, render_review_board()",),
            "#f3f4f6", "#475569",
        ),
    }

    for box in boxes.values():
        _draw_box(ax, box, body_width=24, title_size=11, body_size=7.4)

    _poly_arrow(ax, [_point(boxes["inputs"], "right", 0.5), _point(boxes["modes"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["modes"], "right", 0.5), _point(boxes["request"], "left", 0.5)])
    _poly_arrow(ax, [_point(boxes["request"], "right", 0.5), _point(boxes["generate"], "left", 0.5)])
    _line(ax, (0.20, 0.605), (0.85, 0.605), color="#64748b", width=2.0)
    _poly_arrow(ax, [_point(boxes["generate"], "bottom", 0.50), (0.85, 0.605)])
    _poly_arrow(ax, [(0.20, 0.605), _point(boxes["category"], "top", 0.52)])
    _poly_arrow(ax, [(0.50, 0.605), _point(boxes["semantic"], "top", 0.50)])
    _poly_arrow(ax, [(0.80, 0.605), _point(boxes["evidence"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["category"], "bottom", 0.70), (0.22, 0.30), _point(boxes["score"], "top", 0.18)])
    _poly_arrow(ax, [_point(boxes["semantic"], "bottom", 0.50), (0.50, 0.30), _point(boxes["score"], "top", 0.50)])
    _poly_arrow(ax, [_point(boxes["evidence"], "bottom", 0.35), (0.77, 0.28), (0.38, 0.28), _point(boxes["score"], "top", 0.82)])
    _poly_arrow(ax, [_point(boxes["score"], "right", 0.50), _point(boxes["select"], "left", 0.50)])
    _poly_arrow(ax, [_point(boxes["select"], "right", 0.50), _point(boxes["report"], "left", 0.50)])

    _label(ax, 0.51, 0.63, "one candidate image per mode", size=8.6)
    _label(ax, 0.80, 0.54, "strongest signal: evidence consistency", size=8.6, color="#be123c")
    _label(ax, 0.30, 0.26, "scoring happens after generation", size=8.6)
    _label(ax, 0.82, 0.25, "selected row becomes the human-review artifact", size=8.6, color="#475569")

    return _save_figure(fig, "figure_5_reranker_detail")


def main() -> int:
    _setup_style()
    outputs = [
        render_system_overview(),
        render_component_architecture(),
        render_planner_detail(),
        render_composer_detail(),
        render_reranker_detail(),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
