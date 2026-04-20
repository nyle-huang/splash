"""Prompt composition aligned to the FLUX.2 prompt structure."""

from __future__ import annotations

import random
import re
from pathlib import Path

from product_campaign_pipeline.taxonomy import (
    BEDDING_CANONICAL_TYPES,
    KITCHEN_APPLIANCE_CANONICAL_TYPES,
)
from product_campaign_pipeline.types import (
    CampaignPriorSpec,
    FluxPromptSpec,
    FluxReferenceImage,
    LocalizedProduct,
)

COLOR_TOKEN_ALIASES: dict[str, str] = {
    "grey": "gray",
    "scarlet": "red",
    "crimson": "red",
    "burgundy": "red",
    "maroon": "red",
    "navy": "blue",
    "cobalt": "blue",
    "azure": "blue",
    "turquoise": "teal",
    "aqua": "teal",
    "mint": "green",
    "olive": "green",
    "ivory": "white",
    "cream": "beige",
    "tan": "beige",
    "taupe": "beige",
    "khaki": "beige",
    "charcoal": "gray",
    "slate": "gray",
    "graphite": "gray",
    "silver": "gray",
}


class PromptComposer:
    """Compose baseline and business-prior FLUX prompts."""

    scene_palettes = {
        "tabletop_display": {
            "location": [
                "a brushed-stone kitchen island vignette",
                "a matte lacquer vanity counter set",
                "a walnut cafe table editorial corner",
                "a polished pantry shelf merchandising bay",
                "a terrazzo plinth studio tabletop",
                "a brushed-metal hospitality counter scene",
            ],
            "support": [
                "with a clearly visible support plane and crisp contact shadow",
                "with a grounded resting surface and controlled reflective falloff",
                "with stable countertop placement and believable weight",
                "with an anchored display plane and clean depth separation",
                "with a premium shelf edge and realistic support contact",
                "with a refined merchandising surface and believable reflections",
            ],
            "props": [
                "with restrained utility props kept secondary to the product",
                "with minimal editorial accents and generous breathing room",
                "with clean functional styling and no clutter near the product",
                "with a sparse premium prop palette in compatible neutrals",
                "with subtle material contrast from nearby display objects",
                "with tightly controlled accessory placement that preserves hero focus",
            ],
            "atmosphere": [
                "with crisp commercial clarity and high value separation",
                "with premium merchandising polish and calm negative space",
                "with a bright utility-forward campaign mood",
                "with controlled specular detail and readable surfaces",
                "with a composed studio-retail blend rather than a casual room snapshot",
                "with clean editorial discipline and grounded physical realism",
            ],
        },
        "furnished_interior": {
            "location": [
                "a reading nook with architectural shelving",
                "a bedroom bench or upholstered daybed corner",
                "a sunlit breakfast banquette or breakfast-room alcove",
                "an entry-console styling vignette",
                "a nightstand and lounge-chair interior composition",
                "a warm studio apartment corner with layered furnishings",
            ],
            "support": [
                "with believable support from furniture, bedding, or another interior surface",
                "with natural contact, weight, and realistic placement against the room elements",
                "with grounded support from upholstery, wood, or a coherent interior plane",
                "with visible resting contact and believable compression where needed",
                "with realistic furniture interaction and stable physical placement",
                "with support cues that feel intentional rather than accidental",
            ],
            "props": [
                "with restrained home accents that do not compete with the product",
                "with a carefully edited prop palette of books, ceramics, or textiles",
                "with layered but controlled decor that stays secondary to the product",
                "with subtle lifestyle cues instead of generic living-room clutter",
                "with compatible interior materials and disciplined negative space",
                "with a premium home-styling vocabulary rather than catalog staging",
            ],
            "atmosphere": [
                "with warm depth and calm domestic polish",
                "with inviting interior mood and soft but controlled structure",
                "with a refined residential campaign feel instead of a repetitive sofa scene",
                "with believable home comfort and clear product prominence",
                "with a composed hospitality-like interior tone",
                "with editorial home styling and clean visual hierarchy",
            ],
        },
        "fashion_lifestyle": {
            "location": [
                "a gallery corridor or architectural arcade",
                "a boutique hotel hallway or lobby edge",
                "a cafe terrace or storefront threshold",
                "a daylight artist-studio threshold with textured plaster and architectural depth",
                "a city stoop or courtyard passage",
                "a market-lane or neighborhood promenade backdrop",
            ],
            "support": [
                "with natural human interaction, believable product support, and coherent body mechanics",
                "with visible body support, realistic gravity, and no awkward staging",
                "with grounded human movement and clear product handling or wear",
                "with a clean fashion-campaign relationship between body, product, and space",
                "with believable contact between the product and the model or hand",
                "with physically plausible carry, wear, or drape rather than casual snapshot posture",
            ],
            "props": [
                "with restrained styling props and open circulation space",
                "with minimal urban or interior accents that keep the product dominant",
                "with premium lifestyle cues and controlled secondary objects",
                "with casual editorial set dressing instead of office or business props",
                "with clean architectural details and limited accessory clutter",
                "with intentionally sparse campaign styling around the subject",
            ],
            "atmosphere": [
                "with relaxed premium styling and approachable energy",
                "with campaign-grade movement and clean spatial rhythm",
                "with a polished but non-corporate lifestyle mood",
                "with strong depth cues and flattering human-product interaction",
                "with expressive everyday luxury rather than formal businesswear",
                "with contemporary editorial pacing and confident negative space",
            ],
        },
        "outdoor_lifestyle": {
            "location": [
                "a riverside promenade or boardwalk",
                "a trailhead or park path campaign setup",
                "a stadium concourse or athletic venue edge",
                "a quiet street-corner or crosswalk editorial frame",
                "a plaza, courtyard, or urban greenway backdrop",
                "a coastal walkway or sunlit overlook scene",
            ],
            "support": [
                "with grounded contact against the outdoor surface and believable movement",
                "with natural human support and realistic environmental footing",
                "with clear spatial anchoring in the outdoor setting",
                "with stable placement or carrying behavior against the location",
                "with believable wind, weight, and body mechanics where relevant",
                "with confident location-based support instead of floating product placement",
            ],
            "props": [
                "with restrained location props and no noisy background crowding",
                "with environment cues that reinforce the scene without distracting from the product",
                "with edited outdoor details rather than generic park clutter",
                "with premium lifestyle accents that stay secondary to the hero object",
                "with clean location styling and controlled supporting objects",
                "with subtle sport or travel cues kept deliberately minimal",
            ],
            "atmosphere": [
                "with airy depth, directional perspective, and clear campaign energy",
                "with bright active readability rather than moody underexposure",
                "with open-space confidence and grounded realism",
                "with dynamic but disciplined outdoor storytelling",
                "with clean environmental depth and strong subject separation",
                "with an elevated location-shoot feel instead of a casual phone snapshot",
            ],
        },
        "retail_display": {
            "location": [
                "a boutique window display or display niche",
                "an open-shelving merchandising wall",
                "a premium pedestal-and-rack retail set",
                "a fitting-room anteroom or boutique bench display",
                "a modular department-store display bay",
                "a gallery-like merchandising plinth arrangement",
            ],
            "support": [
                "with structured presentation and visible support contact",
                "with coherent merchandising placement and stable physical anchoring",
                "with display hardware or shelving that clearly supports the product",
                "with intentional retail presentation and realistic contact points",
                "with stable boutique merchandising rather than a casual shelf snapshot",
                "with grounded placement and disciplined display geometry",
            ],
            "props": [
                "with restrained store-fixture accents and clean merchandising rhythm",
                "with edited retail props that stay secondary to the product",
                "with compatible display materials and minimal visual clutter",
                "with curated shelf, plinth, or rack elements only where useful",
                "with premium retail fixtures and generous negative space",
                "with subtle commercial styling details and no noisy signage",
            ],
            "atmosphere": [
                "with crisp retail polish and commercial clarity",
                "with premium store presentation and clean depth separation",
                "with a high-end merchandising tone rather than a stock catalog look",
                "with a composed showroom feel and stable geometry",
                "with controlled commercial lighting behavior and refined material contrast",
                "with visible retail intent and strong hero-product emphasis",
            ],
        },
        "editorial_interior": {
            "location": [
                "a textured plaster studio set",
                "a modular plinth and color-field interior stage",
                "a draped textile editorial room set",
                "a shadow-slit architectural alcove",
                "a brushed metal and stone studio interior",
                "a gallery-like campaign room with textured walls, warm stone, and controlled shadow depth",
            ],
            "support": [
                "with a coherent support surface and strong depth separation",
                "with believable grounding against the set architecture",
                "with stable physical placement and clear object-to-set contact",
                "with disciplined studio support cues and controlled geometry",
                "with an intentional campaign-stage relationship between product and set",
                "with believable support from the set rather than floating presentation",
            ],
            "props": [
                "with minimal editorial props and clean spatial editing",
                "with abstract set elements kept secondary to the product",
                "with controlled sculptural accents and no visual noise",
                "with a restrained art-direction palette around the hero object",
                "with subtle architectural or plinth accents only where useful",
                "with premium set dressing that avoids repetitive living-room cues",
            ],
            "atmosphere": [
                "with a high-end studio campaign mood and precise separation",
                "with polished interior art direction rather than generic room decor",
                "with sculptural depth and deliberate negative space",
                "with strong editorial control and premium material contrast",
                "with an elevated campaign-set feel rather than a casual interior snapshot",
                "with sophisticated studio composition and grounded realism",
            ],
        },
    }
    lighting = [
        "softbox hero lighting with subtle rim light",
        "window-lit editorial mood with clean highlight rolloff",
        "high-end studio key light with grounded shadows",
        "warm natural light with controlled contrast",
    ]
    camera = [
        "premium hero angle chosen to flatter the product while staying compatible with the source evidence",
        "designer-grade three-quarter or frontal campaign framing with believable perspective",
        "high-end editorial product angle with natural depth and evidence-consistent proportions",
        "commercial hero framing that may reveal new surfaces without contradicting the source evidence",
    ]
    fashion_poses = [
        "a different model in a new pose that showcases the product naturally",
        "a new model with dynamic editorial body language",
        "a different model captured mid-motion in a premium campaign pose",
    ]
    persona_scene_accents = {
        ("fashion_lifestyle", "playful_casual"): [
            "use friendly color accents, relaxed everyday styling, and zero formal business cues",
            "lean into playful approachable casting and casual wardrobe instead of officewear",
            "keep the scene youthful, bright, and casually expressive rather than corporate",
        ],
        ("fashion_lifestyle", "refined_neutral"): [
            "use polished casual styling through denim, understated accessories, or relaxed separates instead of suits or officewear",
            "keep the cast refined but informal, with modern everyday wardrobe and no blazers",
            "favor calm neutral styling and easy confidence rather than sharp tailoring",
        ],
        ("tabletop_display", "sport_utility"): [
            "favor bright functional clarity, crisp label readability, and active everyday utility cues",
            "keep the tabletop styling energetic, practical, and clean rather than moody",
            "use active-lifestyle value separation with restrained sport-oriented accents",
        ],
        ("furnished_interior", "cozy_home"): [
            "keep the interior relaxed, warm, and inviting with soft home cues",
            "favor tactile domestic warmth instead of formal editorial stiffness",
            "lean into calm residential comfort while preserving a premium campaign look",
        ],
        ("outdoor_lifestyle", "sport_utility"): [
            "favor active everyday movement, bright readability, and clean sport utility cues",
            "keep the outdoor scene practical, energetic, and performance-adjacent rather than fashion-formal",
            "use an athletic or travel-ready lifestyle tone with disciplined styling",
        ],
        ("retail_display", "playful_casual"): [
            "use approachable merchandising, bright value separation, and cheerful retail energy",
            "keep the retail styling colorful and inviting without turning childish or cluttered",
            "favor playful commercial polish over neutral luxury minimalism",
        ],
    }
    support_clauses = {
        "standing_on_surface": "Show the product standing securely on a stable surface with visible contact and grounded shadow.",
        "resting_with_back_support": "Show the product resting naturally with visible support from furniture or another object, with believable contact and compression.",
        "resting_on_surface": "Show the product resting naturally on a surface with visible contact, support, and grounded weight.",
        "carried_by_hand": "Show the product carried or held with visible support from a person, or hanging naturally under gravity.",
        "worn_on_body": "Show the product worn naturally on a person so its shape follows the body and gravity.",
        "mounted_or_hanging": "Show the product attached to a visible support point with believable tension, contact, and gravity.",
    }
    support_mode_defaults = {
        "self_supporting_display": "standing_on_surface",
        "externally_supported_soft": "resting_with_back_support",
        "portable_flexible": "carried_by_hand",
        "wearable": "worn_on_body",
        "supported_display": "resting_on_surface",
    }
    scene_labels = {
        "tabletop_display": "a coherent tabletop or support-surface display scene",
        "furnished_interior": "a furnished interior scene with believable support context",
        "fashion_lifestyle": "a fashion-oriented lifestyle scene with human interaction",
        "outdoor_lifestyle": "an outdoor lifestyle scene with grounded contact and support",
        "retail_display": "a structured retail display scene with stable placement",
        "editorial_interior": "an editorial interior scene with believable support and placement",
    }
    reinvention_clauses = {
        "balanced": (
            "You may improve viewpoint, pose, support, and framing as long as all invented details remain compatible "
            "with the observed source evidence."
        ),
        "clarity": (
            "Prefer a clean, stable campaign presentation with simple readable geometry, grounded support, and a composition "
            "that keeps the product easy to parse. If a person is required, keep the pose simple enough that hands, arms, "
            "and body alignment remain unmistakably plausible."
        ),
        "reveal": (
            "You may reveal additional surfaces or a cleaner front or three-quarter presentation, but treat unseen "
            "surfaces as design hypotheses that must stay compatible with the source evidence."
        ),
        "hero": (
            "Choose the most flattering professional campaign presentation you can, but never invent patterns, trims, "
            "panels, folds, or proportions that contradict the observed source evidence."
        ),
    }

    def _rng(self, seed: int | None) -> random.Random:
        return random.Random(0 if seed is None else seed)

    def _references(self, localized: LocalizedProduct) -> list[FluxReferenceImage]:
        evidence = localized.identity.observed_evidence
        refs: list[FluxReferenceImage] = []
        prioritize_cutout_reference = bool(
            localized.identity.category == "apparel"
            and localized.identity.requires_human_model
            and self._has_high_uncertainty_evidence(evidence)
            and evidence.reference_cutout_path
        )
        suppress_cutout_reference = bool(
            not localized.identity.requires_human_model
            and localized.identity.category in {"bedding", "pet home", "home decor"}
            and (
                "border_foreground_intrusion" in evidence.artifact_flags
                or (evidence.raw_evidence_caption and not evidence.evidence_caption)
            )
        )
        dominant_body_color = self._dominant_body_color_hint(localized)
        crop_only_color_lock = bool(
            not self._has_high_uncertainty_evidence(evidence)
            and localized.identity.rigid_vs_soft != "rigid"
            and dominant_body_color
            and evidence.reference_crop_path
            and localized.identity.category not in {"bedding", "pet home"}
            and self._canonical_product_type(localized) not in (BEDDING_CANONICAL_TYPES | {"pet bed"})
            and evidence.edge_profile_note is None
            and evidence.soft_structure_note is None
            and evidence.lower_region_note is None
            and evidence.coverage_class in {
                "low_variation_surface",
                "full_visible_surface_pattern",
                "broad_visible_surface_pattern",
            }
        )
        if crop_only_color_lock:
            reference_specs: list[tuple[str, str | None, str]] = [
                (
                    "product crop",
                    evidence.reference_crop_path or localized.crop_path,
                    "Product-focused crop with surrounding human or room context minimized.",
                )
            ]
        elif suppress_cutout_reference:
            reference_specs = [
                (
                    "product crop",
                    evidence.reference_crop_path or localized.crop_path,
                    "Product-focused crop with surrounding human or room context minimized.",
                ),
                (
                    "product silhouette",
                    evidence.reference_silhouette_path,
                    "Observed silhouette reference for contour and proportion control.",
                ),
            ]
        elif prioritize_cutout_reference:
            reference_specs = [
                (
                    "product cutout",
                    evidence.reference_cutout_path,
                    "Observed product cutout with non-product context removed.",
                ),
                (
                    "product crop",
                    evidence.reference_crop_path or localized.crop_path,
                    "Product-focused crop with surrounding human or room context minimized.",
                ),
                (
                    "product silhouette",
                    evidence.reference_silhouette_path,
                    "Observed silhouette reference for contour and proportion control.",
                ),
            ]
        else:
            reference_specs = [
                (
                    "product crop",
                    evidence.reference_crop_path or localized.crop_path,
                    "Product-focused crop with surrounding human or room context minimized.",
                ),
                (
                    "product cutout",
                    evidence.reference_cutout_path,
                    "Observed product cutout with non-product context removed.",
                ),
                (
                    "product silhouette",
                    evidence.reference_silhouette_path,
                    "Observed silhouette reference for contour and proportion control.",
                ),
            ]
        if self._requires_source_context_reference(localized):
            reference_specs.append(
                (
                    "base image",
                    localized.source_image,
                    "Primary source photo preserving functional subtype context that may extend beyond the localized crop.",
                )
            )
        if not localized.identity.requires_human_model and not any(path for _, path, _ in reference_specs):
            reference_specs.append(
                (
                    "base image",
                    localized.source_image,
                    "Primary source photo of the featured product.",
                )
            )
        for role, path, description in reference_specs:
            if not path:
                continue
            if any(existing.path == path for existing in refs):
                continue
            refs.append(FluxReferenceImage(role=role, path=path, description=description))
        if not refs:
            refs.append(
                FluxReferenceImage(
                    role="base image",
                    path=localized.source_image,
                    description="Primary source photo of the featured product.",
                )
            )
        return refs

    def _canonical_product_type(self, localized: LocalizedProduct) -> str:
        canonical = localized.identity.canonical_product_type or localized.identity.category or "product"
        return " ".join(str(canonical).split()).strip()

    def _scene_family(self, localized: LocalizedProduct, scene_family: str | None = None) -> str:
        selected = scene_family or localized.identity.default_scene_family or "editorial_interior"
        return " ".join(str(selected).split()).strip()

    def _support_relation(self, localized: LocalizedProduct, support_relation: str | None = None) -> str:
        if support_relation:
            return " ".join(str(support_relation).split()).strip()
        support_mode = localized.identity.support_mode or "supported_display"
        return self.support_mode_defaults.get(support_mode, "resting_on_surface")

    def _background(self, localized: LocalizedProduct, rng: random.Random, scene_family: str | None = None) -> str:
        selected_scene = self._scene_family(localized, scene_family)
        palette = self.scene_palettes.get(selected_scene, self.scene_palettes["editorial_interior"])
        canonical_type = self._canonical_product_type(localized)
        location_options = list(palette["location"])
        props_options = list(palette["props"])
        if selected_scene == "furnished_interior" and (
            localized.identity.category == "bedding" or canonical_type in BEDDING_CANONICAL_TYPES
        ):
            location_options = [
                "a layered bedroom bedscape with visible mattress support",
                "a boutique-bedroom bed or daybed corner with clear bedding context",
                "a guest-room bed scene with a visible headboard and bed surface",
                "a styled bedroom composition centered on a bed rather than standalone furniture",
                "a premium bed-and-bench bedroom vignette with the product still resting across the bed surface",
                "a bedroom alcove with the quilt or comforter spread across a clearly visible bed",
            ]
            props_options = [
                "with restrained bedside accents and no competing decorative throw styling",
                "with edited bedroom props that stay secondary to the bedding hero object",
                "with compatible bedroom materials and disciplined negative space around the bed",
                "with premium residential styling that keeps the bedding read clearly as bedding",
                "with subtle bedside or bench accents that never turn the product into a furniture throw",
                "with controlled bedroom set dressing instead of generic living-room decor",
            ]
        if self._expects_human_subject(localized):
            if selected_scene == "fashion_lifestyle":
                location_options = [
                    option
                    for option in location_options
                    if all(token not in option for token in ("market-lane", "promenade"))
                ]
                props_options = [
                    option
                    for option in props_options
                    if "open circulation space" not in option
                ]
            elif selected_scene == "outdoor_lifestyle":
                location_options = [
                    option
                    for option in location_options
                    if all(token not in option for token in ("boardwalk", "plaza", "greenway"))
                ]
        descriptor = ", ".join(
            rng.choice(
                {
                    "location": location_options or list(palette["location"]),
                    "props": props_options or list(palette["props"]),
                }.get(key, list(palette[key]))
            ).rstrip(".")
            for key in ("location", "support", "props", "atmosphere")
        )
        persona = localized.identity.style_persona
        persona_options = self.persona_scene_accents.get((selected_scene, persona or ""), ())
        if localized.identity.category == "apparel" and self._canonical_product_type(localized) == "dress":
            persona_options = tuple(
                option
                for option in persona_options
                if "denim" not in option.lower() and "relaxed separates" not in option.lower()
            )
        if persona_options:
            descriptor = f"{descriptor}, {rng.choice(persona_options).rstrip('.')}"
        if self._expects_human_subject(localized):
            descriptor = (
                f"{descriptor}, with a private or uncrowded setting and no other people visible anywhere in the background"
            )
        return descriptor

    def _requires_direct_grip(self, localized: LocalizedProduct) -> bool:
        identity = localized.identity
        if identity.interaction_mode != "held_in_hand":
            return False
        evidence = identity.observed_evidence
        observed_notes = " ".join(
            filter(
                None,
                [
                    evidence.form_factor_note,
                    evidence.upper_region_note,
                    *evidence.hard_facts,
                ],
            )
        ).lower()
        if "no visible handles" in observed_notes or "no visible straps" in observed_notes:
            return True
        return evidence.upper_component_state == "absent"

    def _prefers_compact_hand_focus(self, localized: LocalizedProduct) -> bool:
        if not self._requires_direct_grip(localized):
            return False
        evidence = localized.identity.observed_evidence
        canonical_type = self._canonical_product_type(localized)
        if canonical_type in {"wallet", "clutch", "wristlet", "card holder"}:
            return True
        observed_notes = " ".join(
            filter(
                None,
                [
                    evidence.form_factor_note,
                    evidence.silhouette_note,
                    *evidence.hard_facts,
                ],
            )
        ).lower()
        if "compact" in observed_notes or "hand-held" in observed_notes:
            return True
        aspect_ratio = evidence.aspect_ratio
        if aspect_ratio is not None and aspect_ratio <= 1.7:
            return True
        return False

    def _expects_human_subject(self, localized: LocalizedProduct) -> bool:
        return localized.identity.requires_human_model or localized.identity.interaction_mode in {
            "worn",
            "worn_or_carried",
            "held_in_hand",
            "carried_or_resting",
        }

    def _persona_clause(self, localized: LocalizedProduct) -> str:
        persona = localized.identity.style_persona
        clauses: list[str] = []
        if localized.identity.casting_note:
            clauses.append(localized.identity.casting_note)
        if persona == "playful_casual":
            clauses.append("If a person appears, keep the wardrobe casual, approachable, and product-compatible instead of formal businesswear.")
            if localized.identity.category == "bag" and self._requires_direct_grip(localized):
                clauses.append(
                    "If only a hand, wrist, or small arm fragment is visible, keep it soft, casual, and product-compatible. Avoid masculine-coded hands, business-formal sleeves, watches, or sharp suiting cues that fight the product's visual language."
                )
        elif persona == "sport_utility":
            clauses.append("Keep the styling bright, functional, and active rather than moody or luxury-editorial.")
        elif persona == "cozy_home":
            clauses.append("Keep the environment relaxed, warm, and home-oriented rather than formal editorial.")
        elif persona == "refined_neutral":
            if localized.identity.category == "apparel":
                if self._canonical_product_type(localized) == "dress":
                    clauses.append(
                        "If a person appears, keep the styling polished and casual through understated accessories and simple footwear rather than layered separates. Avoid jeans, trousers, leggings, visible underlayers, suits, blazers, sharp tailoring, officewear, and alternate hero garments that compete with the dress."
                    )
                else:
                    clauses.append(
                        "If a person appears, keep the styling polished and casual through denim, understated accessories, or relaxed separates around the featured garment. Avoid suits, blazers, sharp tailoring, officewear, and alternate hero tops that compete with the product."
                    )
                clauses.append(
                    "Use a private or tightly controlled corner of the chosen location so no background pedestrians, companions, or extra visible figures plausibly appear anywhere in frame."
                )
            else:
                clauses.append(
                    "If a person appears, use polished casual wardrobe through denim, understated accessories, or relaxed separates. Avoid suits, blazers, sharp tailoring, and officewear."
                )
        return " ".join(str(clause).strip() for clause in clauses if clause).strip()

    def _scene_clause(self, localized: LocalizedProduct, scene_family: str | None = None) -> str:
        selected_scene = self._scene_family(localized, scene_family)
        return self.scene_labels.get(selected_scene, "a coherent campaign scene with believable support")

    def _dominant_body_color_hint(self, localized: LocalizedProduct) -> str | None:
        evidence = localized.identity.observed_evidence
        if self._has_high_uncertainty_evidence(evidence):
            return None
        if evidence.color_note and (evidence.color_confidence is None or evidence.color_confidence >= 0.64):
            lowered = evidence.color_note.lower()
            for token in re.findall(r"[a-z0-9']+", lowered):
                normalized = "gray" if token == "grey" else token
                if normalized in {
                    "black",
                    "white",
                    "gray",
                    "blue",
                    "teal",
                    "green",
                    "red",
                    "pink",
                    "purple",
                    "yellow",
                    "gold",
                    "orange",
                    "brown",
                    "beige",
                }:
                    return normalized
        if evidence.body_region_color:
            return evidence.body_region_color
        if evidence.palette and evidence.color_confidence is not None and evidence.color_confidence >= 0.76:
            return evidence.palette[0]
        return None

    def _has_high_uncertainty_evidence(self, evidence: ObservedEvidenceSpec) -> bool:
        return evidence.uncertainty_level == "high" or evidence.surface_scope == "partial_or_occluded"

    def _hard_evidence_prompt_facts(self, evidence: ObservedEvidenceSpec) -> list[str]:
        facts: list[str] = []
        high_uncertainty = self._has_high_uncertainty_evidence(evidence)

        if high_uncertainty:
            candidates = (
                evidence.material_note,
                evidence.surface_relief_note,
                evidence.lower_region_note if (evidence.lower_region_confidence or 0.0) >= 0.7 else None,
                evidence.soft_structure_note if (evidence.soft_structure_confidence or 0.0) >= 0.75 else None,
                evidence.edge_profile_note if (evidence.edge_profile_confidence or 0.0) >= 0.75 else None,
                evidence.trim_note if (evidence.trim_confidence or 0.0) >= 0.85 else None,
            )
        else:
            candidates = (
                evidence.color_note,
                evidence.coverage_note,
                evidence.value_relation_note,
                evidence.material_note,
                None
                if (
                    not evidence.palette
                    or evidence.color_confidence is None
                    or evidence.color_confidence < 0.72
                    or evidence.coverage_class == "localized_visible_pattern"
                )
                else f"observed palette includes {', '.join(evidence.palette[:3])}",
                evidence.pattern_note,
                evidence.trim_note,
                evidence.surface_relief_note,
                evidence.lower_region_note,
                evidence.soft_structure_note,
                evidence.edge_profile_note,
                evidence.form_factor_note,
                evidence.upper_region_note,
                evidence.silhouette_note,
            )

        for candidate in candidates:
            if candidate and candidate not in facts:
                facts.append(candidate)
        for fact in evidence.hard_facts:
            if fact not in facts:
                facts.append(fact)
        return facts

    def _soft_evidence_prompt_facts(self, evidence: ObservedEvidenceSpec) -> list[str]:
        if not self._has_high_uncertainty_evidence(evidence):
            return []
        facts: list[str] = []
        for candidate in (
            evidence.evidence_caption,
            evidence.color_note,
            evidence.coverage_note,
            evidence.value_relation_note,
            evidence.pattern_note,
            evidence.trim_note,
            evidence.form_factor_note,
            evidence.upper_region_note,
            evidence.silhouette_note,
        ):
            if candidate and candidate not in facts:
                facts.append(candidate)
        for fact in evidence.hard_facts:
            if fact.startswith("the product remains a "):
                continue
            if fact not in facts:
                facts.append(fact)
        return facts

    def _has_chromatic_soft_textile_lock(self, localized: LocalizedProduct) -> bool:
        evidence = localized.identity.observed_evidence
        category = str(localized.identity.category or "").strip().lower()
        canonical_type = str(localized.identity.canonical_product_type or "").strip().lower()
        if category not in {"bedding", "pet home", "home decor", "apparel"} and canonical_type not in {
            *BEDDING_CANONICAL_TYPES,
            "pet bed",
            "decorative pillow",
            "shirt",
            "dress",
        }:
            return False
        dominant = self._dominant_body_color_hint(localized)
        if dominant not in {"green", "blue", "teal", "purple", "red", "pink", "yellow", "orange"}:
            return False
        if evidence.color_confidence is not None and evidence.color_confidence < 0.68:
            return False
        return evidence.coverage_class in {
            "low_variation_surface",
            "localized_visible_pattern",
            "full_visible_surface_pattern",
            "broad_visible_surface_pattern",
        }

    def _prefers_surface_dominant_soft_goods_view(self, localized: LocalizedProduct) -> bool:
        evidence = localized.identity.observed_evidence
        category = str(localized.identity.category or "").strip().lower()
        canonical_type = str(localized.identity.canonical_product_type or "").strip().lower()
        if localized.identity.rigid_vs_soft != "soft" and category not in {"bedding", "pet home", "home decor"}:
            return False
        if evidence.soft_structure_class in {"flat_surface", "low_perimeter_relief"}:
            return True
        return canonical_type in BEDDING_CANONICAL_TYPES and self._has_chromatic_soft_textile_lock(localized)

    def _should_lock_business_prior_view_control(self, localized: LocalizedProduct) -> bool:
        evidence = localized.identity.observed_evidence
        return bool(
            self._prefers_surface_dominant_soft_goods_view(localized)
            and (evidence.artifact_flags or evidence.uncertainty_level == "high")
        )

    def _compose_action(self, localized: LocalizedProduct, reinvention_mode: str, human_clause: str = "") -> str:
        canonical_type = self._canonical_product_type(localized)
        presentation_label = self._presentation_product_label(localized)
        reinvention_clause = self.reinvention_clauses.get(reinvention_mode, self.reinvention_clauses["balanced"])
        form_factor_clause = ""
        if presentation_label != canonical_type:
            form_factor_clause = (
                f" Preserve the observed {presentation_label} form factor while keeping it unmistakably a {canonical_type}."
            )
        surface_clause = ""
        if self._prefers_surface_dominant_soft_goods_view(localized):
            surface_clause = (
                " Prefer a broad surface-dominant presentation that keeps the main resting plane and perimeter height readable instead of an oblique hero angle that exaggerates loft."
            )
        display_isolation_clause = ""
        if self._requires_people_out_of_frame(localized):
            display_isolation_clause = (
                " Keep the frame product-only, with no people, hands, sleeves, or body fragments visible anywhere in the image. "
                "Present the product alone, fully supported by furniture or another nonhuman surface, and do not show it worn, "
                "wrapped around, touched by, or draped across any person."
            )
        return (
            "Reinvent the product into a new professional campaign image while preserving the "
            f"exact product identity, preserving all observed source evidence, and keeping it unmistakably a {canonical_type}. "
            f"{reinvention_clause}{form_factor_clause}{surface_clause}{display_isolation_clause} "
            f"{human_clause}".strip()
        )

    def _compose_context_core(
        self,
        localized: LocalizedProduct,
        *,
        background: str,
        scene_family: str,
        support_relation: str,
    ) -> str:
        framing_clause = ""
        if self._prefers_compact_hand_focus(localized):
            framing_clause = (
                " Favor a waist-up, half-body, or close hand-focused composition where the product stays large, clear, "
                "and visually dominant in frame. Avoid distant full-body staging, wide body overlap, or compositions "
                "where the product becomes too small relative to the model."
            )
        elif self._prefers_surface_dominant_soft_goods_view(localized):
            framing_clause = (
                " Favor a broad surface-led composition with a readable top or three-quarter view so the product's main field, perimeter relief, "
                "and overall thickness remain legible instead of being exaggerated by an oblique hero crop."
            )
        if self._requires_people_out_of_frame(localized):
            framing_clause += (
                " Keep all people, hands, sleeves, and body fragments completely out of frame so the product is presented alone. "
                "Do not stage the product as something being worn, held, wrapped around a body, or touched by a seated or standing model."
            )
        background_guardrail = (
            "Keep the background materially specific and spatially readable. "
            "Do not collapse the scene into a flat gray seamless, blank gray wall, monochrome void, "
            "or a partially rendered gray-and-white geometric backdrop."
        )
        return (
            f"Change the background to {background}. Use {self._scene_clause(localized, scene_family)}. "
            f"{self._support_clause(localized, support_relation)} Keep the product believable in scale and material. "
            f"{self._persona_clause(localized)} "
            f"{self._observed_evidence_summary(localized)} "
            f"{framing_clause}"
            f"{background_guardrail} "
            "You may present the product from a cleaner or more flattering angle than the source image, but unseen faces, folds, panels, or surfaces must remain compatible with the observed evidence instead of contradicting it. "
            "Do not assume that a locally observed pattern, trim, or wrinkle treatment necessarily appears identically on every unseen surface. "
            "Avoid plain catalog treatment and create a designer-grade campaign composition."
        )

    def _requires_people_out_of_frame(self, localized: LocalizedProduct) -> bool:
        canonical_type = self._canonical_product_type(localized)
        return bool(
            not localized.identity.requires_human_model
            and localized.identity.interaction_mode == "placed"
            and (
                canonical_type
                in (
                    KITCHEN_APPLIANCE_CANONICAL_TYPES
                    | BEDDING_CANONICAL_TYPES
                    | {"pet bed", "decorative pillow", "table lamp", "office chair", "folding chair", "chair"}
                )
                or localized.identity.category in {"bedding", "pet home", "home decor"}
            )
        )

    def _human_clause(
        self,
        localized: LocalizedProduct,
        rng: random.Random,
        *,
        reinvention_mode: str,
        override: str | None = None,
    ) -> str:
        single_model_clause = (
            "Show exactly one human model only. Do not include background people, companions, passersby, crowd silhouettes, "
            "reflections of other people, or extra human figures anywhere in frame."
        )
        if not self._expects_human_subject(localized):
            return ""
        if override:
            clauses: list[str] = []
            if localized.identity.casting_note:
                clauses.append(localized.identity.casting_note)
            clauses.append(" ".join(str(override).split()).strip())
            clauses.append(
                "Use exactly one clearly different model with natural anatomy, readable hands, and no duplicated or impossible limbs."
            )
            clauses.append(single_model_clause)
            return " ".join(dict.fromkeys(clause for clause in clauses if clause)).strip()
        if self._prefers_compact_hand_focus(localized):
            return " ".join(
                [
                    (
                "Use exactly one different model in a clean waist-up or hand-focused pose with simple readable body mechanics. "
                "Keep the product large in frame, avoid distant full-body staging, ensure natural anatomy with one clear set of arms and hands, "
                "and keep any visible wardrobe muted, neutral, and low-saturation so the product remains the only strong color or print near the hand. "
                "Avoid blazers, suit jackets, office shirts, ties, tailored suiting, or corporate styling."
                    ),
                    single_model_clause,
                ]
            )
        if reinvention_mode == "clarity":
            return " ".join(
                [
                    (
                "Use exactly one different model in a calm stable pose with clear natural anatomy, readable hands, "
                "and no extreme limb overlap or duplicated body parts."
                    ),
                    single_model_clause,
                ]
            )
        return " ".join([rng.choice(self.fashion_poses), single_model_clause])

    def _support_clause(self, localized: LocalizedProduct, support_relation: str | None = None) -> str:
        selected_relation = self._support_relation(localized, support_relation)
        canonical_type = self._canonical_product_type(localized)
        if canonical_type in BEDDING_CANONICAL_TYPES:
            return (
                "Show the bedding product spread across a clearly visible bed or daybed surface with grounded contact, "
                "readable coverage, and support from the mattress or bedding beneath it. Do not stage it as a throw blanket "
                "on a bench, chair, sofa, or occupied seat."
            )
        if canonical_type == "backpack":
            if selected_relation == "worn_on_body":
                return (
                    "Show the backpack worn naturally on the back or shoulder with visible strap support, believable weight, "
                    "and coherent attachment between the pack body and carry straps. Show exactly one coherent strap system only, "
                    "and do not duplicate front-facing and back-facing shoulder straps in the same presentation."
                )
            return (
                "Show the backpack carried with visible hand or shoulder support while preserving coherent strap attachment, "
                "believable weight, and a stable pack body. Keep the strap system coherent and do not duplicate or mirror extra harness straps."
            )
        if canonical_type == "table lamp":
            return (
                "Show the table lamp upright on a stable surface with grounded contact and a coherent relationship between "
                "the base, stem, and shade. Do not imply soft deformation or floating placement."
            )
        if canonical_type == "shoe" and selected_relation == "worn_on_body":
            return (
                "Show the shoe worn naturally on a foot with believable sole contact, upper-to-sole alignment, and realistic gait or stance."
            )
        if selected_relation == "carried_by_hand" and self._requires_direct_grip(localized):
            clause = (
                "Show the product directly gripped in one hand with visible hand contact on the product body itself. "
                "Do not add handles, straps, wrist loops, shoulder attachments, crossbody carry hardware, or dangling carry components."
            )
            if localized.identity.category == "bag":
                clause += " Show exactly one coherent bag or accessory instance and do not duplicate the product, add a second bag, or introduce a second carried object that reads as another copy of the same product."
            if self._prefers_compact_hand_focus(localized):
                clause += (
                    " Keep the interaction close and readable, with the hand and product clearly visible instead of staging them as a distant full-body action. "
                    "Do not style the accessory as a shoulder bag, crossbody bag, or hanging pouch."
                )
            return clause
        return self.support_clauses.get(
            selected_relation,
            "Show the product with visible, physically plausible support and grounded contact with its environment.",
        )

    def _subject_phrase(self, localized: LocalizedProduct) -> str:
        canonical_type = self._canonical_product_type(localized)
        presentation_label = self._presentation_product_label(localized)
        use_concise_identity = not self._has_visible_brand_or_text_signal(localized)
        if presentation_label != canonical_type:
            preferred_label = localized.identity.phrase or localized.phrase or presentation_label
        else:
            preferred_label = (
                localized.identity.phrase
                if (localized.identity.subtype_hint and localized.identity.phrase) or use_concise_identity
                else (localized.identity.source_title or localized.identity.phrase or localized.phrase)
            )
        title = self._sanitize_subject_title(
            localized,
            preferred_label,
            prefer_concise=use_concise_identity or presentation_label != canonical_type,
        )
        lowered_title = title.lower()
        if presentation_label == canonical_type and canonical_type.lower() not in lowered_title:
            title = f"{title} {canonical_type}".strip()
        if presentation_label != canonical_type and title.lower() == presentation_label.lower():
            return f"The featured {presentation_label} from image 1, which must remain a {canonical_type}"
        if presentation_label != canonical_type and len(title.split()) > 6:
            return f"The featured {presentation_label} from image 1, which must remain a {canonical_type}"
        if presentation_label != canonical_type:
            return (
                f"The featured {presentation_label} from image 1, which must remain a {canonical_type}: {title}"
            )
        return f"The featured {canonical_type} from image 1: {title}"

    def _has_visible_brand_or_text_signal(self, localized: LocalizedProduct) -> bool:
        evidence = localized.identity.observed_evidence
        text_blob = " ".join(
            part
            for part in (
                evidence.value_relation_note or "",
                evidence.coverage_note or "",
                evidence.pattern_note or "",
                evidence.evidence_caption or "",
                " ".join(evidence.hard_facts),
                " ".join(localized.identity.brand_notes),
            )
            if part
        ).lower()
        return bool(
            any(
                token in text_blob
                for token in ("logo", "brand", "wordmark", "watermark", "label", "letter", "text", "printed text")
            )
        )

    def _presentation_product_label(self, localized: LocalizedProduct) -> str:
        subtype_hint = " ".join(str(localized.identity.subtype_hint or "").split()).strip()
        if subtype_hint:
            return subtype_hint
        canonical_type = self._canonical_product_type(localized)
        evidence = localized.identity.observed_evidence
        if canonical_type == "pet bed" and evidence.soft_structure_class in {"flat_surface", "low_perimeter_relief"}:
            return "low-profile plush pet resting pad"
        return canonical_type

    def _requires_source_context_reference(self, localized: LocalizedProduct) -> bool:
        subtype_hint = str(localized.identity.subtype_hint or "").lower()
        evidence = localized.identity.observed_evidence
        if localized.identity.requires_human_model:
            return False
        if not subtype_hint:
            return False
        if "border_human_fragment" in evidence.artifact_flags or "source_contains_border_human_fragment" in evidence.source_validity_issues:
            return False
        return subtype_hint in {"backpack cooler"}

    def _sanitize_subject_title(
        self,
        localized: LocalizedProduct,
        raw_title: str,
        *,
        prefer_concise: bool,
    ) -> str:
        title = " ".join(str(raw_title).split()).strip()
        if not title:
            return self._presentation_product_label(localized)
        title = re.sub(
            r"\b\d+(?:[./]\d+)?\s*(?:oz|ounce|ounces|inch|inches|in|cm|mm|ft|lb|lbs|kg|g|qt|quart|quarts|w|v|xl|xxl)\b",
            "",
            title,
            flags=re.IGNORECASE,
        )
        color_tokens = {
            "black",
            "white",
            "gray",
            "grey",
            "blue",
            "teal",
            "green",
            "red",
            "pink",
            "purple",
            "yellow",
            "gold",
            "orange",
            "brown",
            "beige",
        }
        allowed_colors = self._evidence_color_tokens(localized)
        canonical_tokens = set(re.findall(r"[a-z0-9']+", self._canonical_product_type(localized).lower()))
        presentation_tokens = set(re.findall(r"[a-z0-9']+", self._presentation_product_label(localized).lower()))
        unsupported_pattern_tokens = self._unsupported_title_pattern_tokens(localized)
        raw_tokens = re.findall(r"[A-Za-z0-9']+", title)
        trimmed_brand_like_prefix = False
        if prefer_concise:
            while raw_tokens:
                lowered = COLOR_TOKEN_ALIASES.get(raw_tokens[0].lower(), raw_tokens[0].lower())
                if lowered in canonical_tokens or lowered in presentation_tokens or lowered in allowed_colors:
                    break
                raw_tokens.pop(0)
                trimmed_brand_like_prefix = True
            if trimmed_brand_like_prefix and not raw_tokens:
                return self._presentation_product_label(localized)
        kept: list[str] = []
        seen: dict[str, int] = {}
        for token in raw_tokens:
            lowered = token.lower()
            normalized_color = COLOR_TOKEN_ALIASES.get(lowered, lowered)
            if normalized_color == "gray" and "gray" in allowed_colors and lowered == "grey":
                lowered = "gray"
                token = "gray"
            else:
                lowered = normalized_color
            if lowered in color_tokens and allowed_colors and lowered not in allowed_colors:
                continue
            if lowered in unsupported_pattern_tokens:
                continue
            if lowered in canonical_tokens or lowered in presentation_tokens:
                if seen.get(lowered, 0) >= 1:
                    continue
            elif seen.get(lowered, 0) >= 2:
                continue
            kept.append(token)
            seen[lowered] = seen.get(lowered, 0) + 1
        sanitized = " ".join(kept).strip(" ,-/")
        if prefer_concise and trimmed_brand_like_prefix:
            return self._presentation_product_label(localized)
        if prefer_concise and len(sanitized.split()) > 8:
            return self._presentation_product_label(localized)
        return sanitized or self._presentation_product_label(localized)

    def _unsupported_title_pattern_tokens(self, localized: LocalizedProduct) -> set[str]:
        evidence = localized.identity.observed_evidence
        coverage_class = str(evidence.coverage_class or "")
        explicit_pattern_text = " ".join(
            part
            for part in (
                evidence.pattern_note or "",
                evidence.coverage_note or "",
                evidence.evidence_caption or "",
                " ".join(evidence.hard_facts),
            )
            if part
        ).lower()
        supported_tokens = set(re.findall(r"[a-z0-9']+", explicit_pattern_text))
        unsupported: set[str] = set()
        if coverage_class == "low_variation_surface":
            for token in (
                "floral",
                "flower",
                "flowers",
                "graphic",
                "graphics",
                "print",
                "printed",
                "pattern",
                "patterned",
                "stripe",
                "striped",
                "plaid",
                "paisley",
                "polka",
                "dot",
                "dots",
                "cartoon",
                "character",
                "motif",
            ):
                if token not in supported_tokens:
                    unsupported.add(token)
        return unsupported

    def _evidence_color_tokens(self, localized: LocalizedProduct) -> set[str]:
        evidence = localized.identity.observed_evidence
        dominant = self._dominant_body_color_hint(localized)
        low_variation_surface = evidence.coverage_class == "low_variation_surface"
        structured_color_lock = bool(
            dominant
            and evidence.color_confidence is not None
            and evidence.color_confidence >= 0.72
            and (
                low_variation_surface
                or localized.identity.category in {"drinkware", "furniture", "kitchen appliance", "home lighting"}
                or (localized.identity.canonical_product_type or "") in {"office chair", "table lamp", "water bottle", "mug", *KITCHEN_APPLIANCE_CANONICAL_TYPES}
            )
        )
        if structured_color_lock:
            colors = {str(dominant).lower()}
            if "gray" in colors:
                colors.add("grey")
            if "grey" in colors:
                colors.add("gray")
            return colors
        prioritized_colors = [
            evidence.body_region_color,
            evidence.upper_region_color,
            evidence.lower_region_color,
            dominant,
        ]
        if not any(prioritized_colors):
            prioritized_colors.extend([*evidence.palette[:2], *evidence.structural_palette[:2]])
        colors = {str(color).lower() for color in prioritized_colors if color}
        if "gray" in colors:
            colors.add("grey")
        if "grey" in colors:
            colors.add("gray")
        return colors

    def _observed_evidence_summary(self, localized: LocalizedProduct) -> str:
        evidence = localized.identity.observed_evidence
        summary_bits: list[str] = []
        prioritized_hard_facts = [
            fact
            for fact in self._hard_evidence_prompt_facts(evidence)
            if not fact.startswith("the product remains a ")
        ]
        for fact in prioritized_hard_facts[:2]:
            if fact not in summary_bits:
                summary_bits.append(fact)
        evidence_caption_summary = None
        if evidence.evidence_caption:
            evidence_caption_summary = f"source reads as {evidence.evidence_caption}"
            if any(evidence.evidence_caption in fact for fact in prioritized_hard_facts):
                evidence_caption_summary = None
        candidate_facts = list(self._hard_evidence_prompt_facts(evidence))
        if self._has_high_uncertainty_evidence(evidence):
            candidate_facts.extend(self._soft_evidence_prompt_facts(evidence))
        if evidence_caption_summary:
            candidate_facts.insert(0, evidence_caption_summary)
        for candidate in candidate_facts:
            if candidate and candidate not in summary_bits:
                summary_bits.append(candidate)
        if not summary_bits:
            return "Preserve the observed product evidence from image 1 and infer unseen areas conservatively."
        summary = "; ".join(summary_bits[:7])
        if self._has_high_uncertainty_evidence(evidence):
            return (
                f"Use these source cues from image 1 as compatibility guidance: {summary}. "
                "The source view is partial or uncertain, so preserve compatibility without over-committing speculative details."
            )
        return f"Preserve these observed source facts from image 1: {summary}."

    def _evidence_completion_guardrails(self, localized: LocalizedProduct) -> list[str]:
        evidence = localized.identity.observed_evidence
        clauses = [
            "do not invent new material zones, transparent inserts, or structural panels with edge alignments unsupported by the observed evidence",
            "do not introduce any new logos, brand names, letters, labels, stamps, watermarks, or readable text unless they are clearly present in the source evidence",
        ]
        if evidence.artifact_flags:
            clauses.append(
                "treat incidental source overlays, camera watermarks, interface chrome, and stray human fragments as non-product noise rather than product details"
            )
        if "border_human_fragment" in evidence.artifact_flags:
            clauses.append(
                "do not reinterpret incidental source hands, fingers, sleeves, or body fragments as product material, trim, structure, or surrounding objects"
            )
        if "border_text_overlay" in evidence.artifact_flags:
            clauses.append(
                "do not preserve or recreate incidental camera stamps, interface text, or watermark-like markings from the source photo"
            )
        dominant_body_color = self._dominant_body_color_hint(localized)
        if dominant_body_color and (
            evidence.color_confidence is None or evidence.color_confidence >= 0.64
        ):
            clauses.append(
                f"keep the dominant visible body color {dominant_body_color} across the main product body instead of shifting it into a different or lighter color family"
            )
            if localized.identity.category in {"apparel", "bedding", "pet home", "home decor"} or (
                localized.identity.canonical_product_type or ""
            ) in (BEDDING_CANONICAL_TYPES | {"pet bed", "decorative pillow", "shirt", "dress"}):
                clauses.append(
                    "preserve the observed overall lightness and tonal value of the main body instead of darkening it into a deeper or moodier colorway"
                )
        if evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
            clauses.append(
                "if a visible print or multicolor surface treatment spans most of the observed product body, preserve compatible coverage across newly revealed adjacent surfaces instead of collapsing it into a small patch or solid fill"
            )
            if not evidence.trim_note and evidence.upper_component_state != "present":
                clauses.append(
                    "if newly revealed structural zones, plain side panels, or support-facing surfaces are needed, keep them within the observed palette family and surface treatment instead of inventing unrelated solid accent colors"
                )
        elif evidence.coverage_class == "localized_visible_pattern":
            clauses.append(
                "keep localized printed, label-like, or contrast zones anchored to compatible regions instead of spreading them arbitrarily across every unseen surface or erasing them entirely"
            )
        if evidence.trim_note:
            clauses.append("preserve visible boundary or edging color relationships where they are observed")
        if evidence.value_relation_note:
            clauses.append("preserve observed light-dark relationships between localized panels, labels, and the main body")
        if evidence.upper_region_note:
            clauses.append(
                "preserve the visible upper component or attached support structure in compatible color, placement, and geometry"
            )
        elif evidence.upper_component_state == "absent":
            clauses.append(
                "do not invent handles, straps, lids, or attached upper structures that are not supported by the observed source evidence"
            )
        if self._requires_direct_grip(localized):
            clauses.append(
                "show direct hand contact on the product body itself instead of inventing a handle, strap, wrist loop, or suspension point"
            )
            clauses.append(
                "do not reinterpret a compact hand-held accessory as a shoulder bag, crossbody bag, hanging pouch, or strapped carry item"
            )
        if self._expects_human_subject(localized):
            clauses.append(
                "do not spread the product's dominant body color, print, or surface treatment onto large wardrobe panels, hair sections, or body-adjacent props; keep human styling secondary and visually distinct from the product"
            )
            if self._prefers_compact_hand_focus(localized):
                clauses.append(
                    "for compact hand-held accessories, keep visible wardrobe in muted neutral solids and keep the product larger and more visually prominent than any single clothing panel in frame"
                )
        if localized.identity.category == "apparel" and self._canonical_product_type(localized) in {"shirt", "dress"}:
            clauses.append(
                "do not invent contrasting sleeves, cuffs, collars, yokes, underlayers, or side panels unless those garment zones are clearly supported by the source evidence"
            )
            clauses.append(
                "keep the visible garment surface treatment continuous across attached sleeves and body panels unless the source explicitly shows a color-blocked or mixed-material construction"
            )
            if evidence.coverage_class == "low_variation_surface":
                clauses.append(
                    "keep the main garment body and attached sleeves within the same observed color family and tonal range instead of drifting into gray or contrasting sleeve panels"
                )
        if localized.identity.category == "drinkware" and self._canonical_product_type(localized) in {"water bottle", "mug"}:
            clauses.append(
                "if a label band, printed wrap, or localized graphic zone is visible, keep it flush to the vessel surface and consistent with the container curvature instead of inventing detached geometric patches or floating inserts"
            )
            if self._canonical_product_type(localized) == "mug":
                clauses.append(
                    "preserve the mug as one coherent handled vessel with compatible body curvature, rim profile, and handle attachment instead of switching to a different cup or vase silhouette"
                )
        if evidence.material_note:
            clauses.append(
                "preserve visible material and texture cues instead of swapping in a conflicting finish or construction"
            )
        if self._has_chromatic_soft_textile_lock(localized):
            clauses.append(
                "keep the product's chromatic body color clearly readable under neutral or daylight-balanced lighting instead of pushing it toward black, gray, or brown with warm low-key rendering"
            )
        if evidence.surface_relief_note:
            clauses.append(
                "preserve visible ridges, ribbing, fluting, or other structured relief where it is observed instead of flattening the surface into a smooth finish"
            )
        if evidence.lower_region_note:
            clauses.append(
                "preserve the visible lower support or base structure in compatible geometry, attachment, and proportion instead of dropping it or replacing it with a different lower assembly"
            )
        if evidence.edge_profile_note:
            clauses.append(
                "preserve the observed perimeter thickness and edge profile instead of inflating a low edge into bulky bolsters or flattening a raised edge into a thin sheet"
            )
        if evidence.soft_structure_note:
            clauses.append(
                "preserve the observed soft-surface structure and perimeter relief instead of inventing boxed sidewalls, nested inner pads, tray-like bolsters, or incompatible rim height changes"
            )
        if (
            localized.identity.category in {"bedding", "pet home", "home decor"}
            and evidence.coverage_class == "low_variation_surface"
        ):
            clauses.append(
                "do not invent patchwork blocks, framed panels, photo-print zones, medallions, graphic insets, or ink-like localized dark shapes on a tonal soft-surface product unless the source clearly shows those features"
            )
        return clauses

    def _type_guardrails(self, localized: LocalizedProduct) -> list[str]:
        canonical_type = self._canonical_product_type(localized)
        evidence = localized.identity.observed_evidence
        evidence_text = " ".join(evidence.hard_facts).lower()
        constraints = [
            f"the product must remain a {canonical_type} in the final image",
            "preserve the observed colors, prints, trims, materials, and signature details from image 1",
            "you may improve viewpoint, pose, drape, or support, but invented details must remain compatible with the observed source evidence",
            "treat visible source details as hard facts and unseen surfaces as design hypotheses",
            "do not mirror, tile, or project a locally observed panel, print, wrinkle pattern, or trim treatment onto every unseen surface unless the source supports it",
            "preserve the observed proportions and distinctive contour cues without freezing the exact daily-photo presentation",
            "retain material class and believable fabric behavior, but do not copy incidental daily-photo wrinkles, creases, or compression marks unless they define the product construction",
        ]
        if canonical_type == "backpack":
            constraints.extend(
                [
                    "retain the backpack structure as a carried pack body with coherent strap attachment rather than reinterpreting it as a tote or handbag",
                    "preserve visible strap, harness, or back-panel relationships when they are observed",
                ]
            )
            if "back of the backpack" in evidence_text or "back-panel" in evidence_text or "harness" in evidence_text:
                constraints.append(
                    "when the observed source shows the backpack's back panel or harness face, do not replace it with a contradictory front-face colorway or panel layout that overwhelms the observed back-side evidence"
                )
        elif canonical_type == "table lamp":
            constraints.extend(
                [
                    "retain the upright lamp structure with a coherent base-to-shade relationship",
                    "do not reinterpret structural shade-versus-base separation as a printed label or surface graphic",
                ]
            )
        if (
            canonical_type in (KITCHEN_APPLIANCE_CANONICAL_TYPES | {"table lamp", "office chair", "folding chair", "chair"})
            and not localized.identity.requires_human_model
            and localized.identity.interaction_mode == "placed"
        ):
            constraints.append(
                "do not add people, hands, sleeves, or partial body fragments around the product; keep the display product presented on its own unless human interaction is explicitly required"
            )
        if canonical_type == "shoe":
            constraints.append("retain the relationship between the shoe upper and sole without collapsing it into apparel or a soft accessory")
        elif canonical_type == "shirt":
            constraints.append("retain the product as an upper-body garment rather than a pillow, blanket, or other home textile")
            constraints.append(
                "do not reinterpret the featured shirt as knitwear, a sweater, a blouse, a hoodie, or a long-sleeve top unless the source evidence clearly supports that garment construction"
            )
        elif canonical_type == "dress":
            constraints.append("retain the product as a dress rather than shortening it into a shirt or top")
            constraints.append(
                "do not reinterpret the dress as a mixed-material outfit with separate gray knit sleeves, cardigan panels, or layered undergarments unless the source evidence clearly shows those attached garment zones"
            )
            constraints.append(
                "do not add visible jeans, trousers, leggings, or other lower-body garments beneath the dress unless the source evidence clearly supports layered styling"
            )
        elif canonical_type == "folding chair":
            constraints.append(
                "show exactly one coherent folding-chair structure with one seat plane and one backrest rather than duplicated seat slats, nested seat planes, or a second backrest-like panel"
            )
            constraints.append(
                "keep the chair in one physically coherent articulated state; do not mix folded and unfolded geometry across the seat, backrest, and support legs"
            )
            constraints.append(
                "keep the chair frame fully opaque and structurally continuous; do not render ghosted, semi-transparent, doubled, or partially overlapping slats, tubes, or support members"
            )
            constraints.append(
                "do not merge nearby plinths, tables, or background furniture planes into the chair silhouette or backrest geometry"
            )
        elif canonical_type == "pet bed" and evidence.soft_structure_class in {"flat_surface", "low_perimeter_relief"}:
            constraints.extend(
                [
                    "retain the product as a low-profile plush resting pad with at most a gentle perimeter rise instead of deep sidewalls or a nested inner tray",
                    "do not reinvent the pet bed as a boxed pet sofa, thick bolster bed, oversized perimeter bumper, or cushion tray with exaggerated edge height",
                    "do not merge the pet bed into a separate upholstered ottoman, pedestal, plinth, or furniture base that becomes part of the product silhouette",
                ]
            )
        if self._prefers_surface_dominant_soft_goods_view(localized):
            constraints.append(
                "prefer a broad surface-dominant framing that keeps the main resting plane and perimeter height readable instead of an oblique angle that exaggerates loft or edge thickness"
            )
        if canonical_type == "backpack" and (
            (evidence.upper_component_count or 0) >= 2
            or "multiple narrow segments" in str(evidence.upper_region_note or "").lower()
            or "harness" in evidence_text
            or "back-panel" in evidence_text
        ):
            constraints.append(
                "when the source shows the harness or back-panel side of the backpack, keep that evidence as one coherent strap system and do not add a second duplicated strap set or contradictory front-and-back harness presentation"
            )
        constraints.extend(self._evidence_completion_guardrails(localized))
        if localized.identity.requires_human_model:
            constraints.append(
                "replace any source human identity with a clearly different model; do not preserve the source face, hair, glasses, accessories, pose, or body styling"
            )
        prioritized_facts = self._hard_evidence_prompt_facts(evidence)
        if self._has_high_uncertainty_evidence(evidence):
            prioritized_facts.extend(
                fact for fact in self._soft_evidence_prompt_facts(evidence) if fact not in prioritized_facts
            )
            constraints.extend(f"stay compatible with source cue: {fact}" for fact in prioritized_facts[:6])
        else:
            constraints.extend(f"retain observed fact: {fact}" for fact in prioritized_facts[:7])
        if localized.identity.weak_shape_evidence:
            constraints.append(
                "the source view is close or pattern-heavy, so do not reinterpret the product as apparel, bedding, or a different product class"
            )
        return constraints

    def compose_baseline(
        self,
        localized: LocalizedProduct,
        seed: int | None = None,
        *,
        reinvention_mode: str = "balanced",
    ) -> FluxPromptSpec:
        rng = self._rng(seed)
        scene_family = self._scene_family(localized)
        support_relation = self._support_relation(localized)
        background = self._background(localized, rng, scene_family)
        lighting = rng.choice(self.lighting)
        camera = rng.choice(self.camera)
        human_clause = self._human_clause(localized, rng, reinvention_mode=reinvention_mode)

        subject = self._subject_phrase(localized)
        action = self._compose_action(localized, reinvention_mode, human_clause)
        style = f"{lighting}; {camera}; premium commercial photography"
        context = self._compose_context_core(
            localized,
            background=background,
            scene_family=scene_family,
            support_relation=support_relation,
        )
        constraints = [
            "preserve the exact product identity from image 1",
            "do not alter logos, printed marks, or signature materials",
            "maintain physically plausible geometry, shadows, and drapery",
            "use one coherent support relation and avoid conflicting placements or floating poses",
        ]
        constraints.extend(self._type_guardrails(localized))
        if self._expects_human_subject(localized):
            if localized.identity.requires_human_model:
                constraints.append("use a different human model and pose than the source image")
                constraints.append("do not reuse the source person, facial features, hair, glasses, or styling")
            else:
                constraints.append("if a person appears, use exactly one model with natural anatomy and product-compatible handling")
            constraints.append("show exactly one human model with natural anatomy and no duplicated or impossible limbs")
        else:
            constraints.append("change the background and overall scene composition")

        return FluxPromptSpec(
            subject=subject,
            action=action,
            style=style,
            context=context,
            preservation_constraints=constraints,
            reference_images=self._references(localized),
            seed=seed,
            prompt_upsampling=False,
            guidance_scale=1.0,
            num_inference_steps=4,
            max_sequence_length=512,
            model="black-forest-labs/FLUX.2-klein-9B",
        )

    def compose_business_prior(
        self,
        localized: LocalizedProduct,
        prior: CampaignPriorSpec,
        seed: int | None = None,
        *,
        reinvention_mode: str = "balanced",
    ) -> FluxPromptSpec:
        baseline = self.compose_baseline(localized, seed=seed, reinvention_mode=reinvention_mode)
        baseline_rng = self._rng(seed)
        baseline_style_parts = [part.strip() for part in baseline.style.split(";") if part.strip()]
        baseline_lighting = baseline_style_parts[0] if baseline_style_parts else baseline_rng.choice(self.lighting)
        baseline_camera = baseline_style_parts[1] if len(baseline_style_parts) > 1 else baseline_rng.choice(self.camera)
        scene_family = prior.scene_family or self._scene_family(localized)
        support_relation = prior.support_relation or self._support_relation(localized)
        scenario_slots = ", ".join(prior.scenario_slots) if prior.scenario_slots else scene_family
        retrieval_mode = str(prior.metadata.get("retrieval_mode", "retrieval"))
        retrieval_like = retrieval_mode in {"retrieval", "scene_retrieval_fallback"}
        creative_seed = int(
            prior.metadata.get(
                "creative_seed",
                (seed if seed is not None else 0) + (131 if not retrieval_like else 0),
            )
        )
        rng = self._rng(creative_seed)
        lock_view_control = self._should_lock_business_prior_view_control(localized)
        if retrieval_like and not lock_view_control:
            background = str(prior.metadata.get("background_hint") or self._background(localized, rng, scene_family))
            lighting = str(prior.metadata.get("lighting_hint") or rng.choice(self.lighting))
            camera = str(prior.metadata.get("camera_hint") or rng.choice(self.camera))
        elif lock_view_control:
            background = str(prior.metadata.get("background_hint") or self._background(localized, rng, scene_family))
            lighting = baseline_lighting
            camera = baseline_camera
        else:
            background = str(prior.metadata.get("background_hint") or self._background(localized, rng, scene_family))
            lighting = str(prior.metadata.get("lighting_hint") or baseline_lighting)
            camera = str(prior.metadata.get("camera_hint") or rng.choice(self.camera) or baseline_camera)
        creative_direction = str(prior.metadata.get("creative_direction") or "").strip()
        cast_hint = prior.metadata.get("cast_hint")
        human_clause = self._human_clause(
            localized,
            rng,
            reinvention_mode=reinvention_mode,
            override=None if cast_hint is None else str(cast_hint),
        )
        action = self._compose_action(localized, reinvention_mode, human_clause)
        context = self._compose_context_core(
            localized,
            background=background,
            scene_family=scene_family,
            support_relation=support_relation,
        )
        if retrieval_mode == "retrieval":
            style_parts = [lighting, camera, "premium commercial photography"]
            if creative_direction:
                style_parts.append(creative_direction)
            elif prior.style_atoms:
                style_parts.append(
                    "guided by evidence-compatible high-performing campaign traits: "
                    + ", ".join(str(atom) for atom in prior.style_atoms[:3])
                )
            style_phrase = "; ".join(style_parts)
            prior_clause = (
                "Use the selected business prior to materially influence environment choice, framing, casting, and "
                "pacing so this image does not collapse into a near-baseline treatment."
            )
        elif retrieval_mode == "scene_retrieval_fallback":
            style_parts = [lighting, camera, "premium commercial photography"]
            if creative_direction:
                style_parts.append(f"retrieval-informed scene direction: {creative_direction}")
            style_phrase = "; ".join(style_parts)
            prior_clause = (
                "Use the retrieved creative signals to materially influence environment choice, lighting, framing, casting, and pacing, "
                "but keep product completion governed by the source evidence instead of the retrieved product details."
            )
        else:
            style_parts = [lighting, camera, "premium commercial photography"]
            if creative_direction:
                style_parts.append(f"evidence-compatible fallback creative direction: {creative_direction}")
            style_phrase = "; ".join(style_parts)
            prior_clause = (
                "Use the fallback prior conservatively, but still give the business-prior line a distinct evidence-compatible "
                "environment and framing instead of collapsing into the baseline treatment."
            )
        compact_focus_clause = ""
        if self._prefers_compact_hand_focus(localized):
            compact_focus_clause = (
                "Keep wardrobe quiet and neutral around the product, avoid saturated overshirts or large color-blocked layers, "
                "and favor product-led close framing over torso-led styling. "
            )
        extra_context = (
            f"{context} "
            f"Follow one coherent scene plan: {self._scene_clause(localized, scene_family)}. "
            f"{self._support_clause(localized, support_relation)} "
            f"{prior_clause} "
            f"{compact_focus_clause}"
            f"{'' if not creative_direction or retrieval_mode not in {'retrieval', 'scene_retrieval_fallback'} else f'Prioritize this creative direction: {creative_direction}. '}"
            "If any retrieval-derived style cue conflicts with observed evidence about surface coverage, trim, attached components, material, or panel continuity, ignore the style cue and keep the source-compatible completion. "
            "Use retrieval cues to influence styling, composition, and scene only, not the invention of unseen product structure. "
            f"Prefer scenario patterns such as {scenario_slots}."
        )
        constraints = baseline.preservation_constraints + prior.semantic_constraints + prior.banned_identity_edits

        return FluxPromptSpec(
            subject=baseline.subject,
            action=action,
            style=style_phrase,
            context=extra_context,
            preservation_constraints=list(dict.fromkeys(constraints)),
            reference_images=baseline.reference_images,
            seed=seed,
            prompt_upsampling=False,
            guidance_scale=1.0,
            num_inference_steps=4,
            max_sequence_length=512,
            model="black-forest-labs/FLUX.2-klein-9B",
        )
