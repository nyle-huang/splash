import pytest

from product_campaign_pipeline.composer import PromptComposer
from product_campaign_pipeline.types import (
    BoundingBox,
    CampaignPriorSpec,
    LocalizedProduct,
    ObservedEvidenceSpec,
    ProductIdentitySpec,
)


def _localized_product() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/source.png",
        phrase="gold tote bag",
        bbox=BoundingBox(x0=10, y0=20, x1=110, y1=140),
        confidence=0.8,
        crop_path="/tmp/source.crop.png",
        identity=ProductIdentitySpec(
            phrase="gold tote bag",
            category="bag",
            canonical_product_type="tote bag",
            source_title="Gold Carryall Tote Bag",
            support_mode="portable_flexible",
            default_scene_family="fashion_lifestyle",
            interaction_mode="carried_or_resting",
            style_persona="playful_casual",
            stable_base=False,
            rigid_vs_soft="semi-rigid",
            requires_human_model=True,
            colors=["gold"],
            observed_evidence=ObservedEvidenceSpec(
                palette=["blue", "purple", "pink"],
                color_note="the main visible body reads as blue with compatible printed accents in purple, pink",
                coverage_class="broad_visible_surface_pattern",
                coverage_note="the visible print or color treatment spans a broad portion of the observed product surface on a blue base",
                pattern_note="the visible product body carries a multicolor or printed treatment on a blue base",
                upper_component_state="uncertain",
                upper_region_note="the visible upper component splits into multiple narrow segments in black above a blue main body",
                material_note="visible material cues suggest a woven or interlaced texture",
                hard_facts=[
                    "the product remains a tote bag",
                    "observed palette includes blue, purple, pink",
                    "the visible print or color treatment spans a broad portion of the observed product surface on a blue base",
                ],
                reference_cutout_path="/tmp/source.evidence_cutout.png",
                reference_silhouette_path="/tmp/source.evidence_silhouette.png",
            ),
        ),
    )


def _localized_wallet() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/wallet.png",
        phrase="blue floral wallet",
        bbox=BoundingBox(x0=12, y0=14, x1=120, y1=90),
        confidence=0.86,
        crop_path="/tmp/wallet.crop.png",
        identity=ProductIdentitySpec(
            phrase="blue floral wallet",
            category="bag",
            canonical_product_type="wallet",
            source_title="Blue Floral Wallet",
            support_mode="portable_flexible",
            default_scene_family="fashion_lifestyle",
            interaction_mode="held_in_hand",
            style_persona="refined_neutral",
            stable_base=False,
            rigid_vs_soft="semi-rigid",
            requires_human_model=True,
            colors=["blue"],
            observed_evidence=ObservedEvidenceSpec(
                palette=["blue", "beige"],
                coverage_class="broad_visible_surface_pattern",
                coverage_note="the visible print or color treatment covers most of the observed product surface on a blue base",
                pattern_note="the visible product body carries a multicolor or printed treatment on a blue base",
                upper_component_state="absent",
                form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
                hard_facts=[
                    "the product remains a wallet",
                    "the visible bag form is compact and hand-held with no visible handles or shoulder straps",
                ],
            ),
        ),
    )


def _localized_backpack() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/backpack.png",
        phrase="rawlings baseball backpack",
        bbox=BoundingBox(x0=18, y0=16, x1=124, y1=156),
        confidence=0.83,
        crop_path="/tmp/backpack.crop.png",
        identity=ProductIdentitySpec(
            phrase="rawlings baseball backpack",
            category="bag",
            canonical_product_type="backpack",
            source_title="Rawlings Baseball Backpack",
            support_mode="wearable",
            default_scene_family="outdoor_lifestyle",
            interaction_mode="worn_or_carried",
            style_persona="sport_utility",
            stable_base=False,
            rigid_vs_soft="semi-rigid",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                coverage_class="full_visible_surface_pattern",
                coverage_note="the visible print or color treatment covers most of the observed product surface",
                trim_note="visible outer trim or edging reads as black against a gray interior",
                form_factor_note="the visible bag form is a vertically oriented backpack body designed for shoulder or back carry",
                hard_facts=[
                    "the product remains a backpack",
                    "the product remains a backpack with a main body and visible carry straps",
                ],
            ),
        ),
    )


def _localized_backpack_cooler() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/backpack_cooler.png",
        phrase="titan backpack cooler",
        bbox=BoundingBox(x0=18, y0=16, x1=124, y1=156),
        confidence=0.83,
        crop_path="/tmp/backpack_cooler.crop.png",
        identity=ProductIdentitySpec(
            phrase="titan backpack cooler",
            category="bag",
            canonical_product_type="backpack",
            subtype_hint="backpack cooler",
            source_title="Titan 24 Can Backpack Cooler",
            support_mode="wearable",
            default_scene_family="outdoor_lifestyle",
            interaction_mode="worn_or_carried",
            style_persona="sport_utility",
            stable_base=False,
            rigid_vs_soft="semi-rigid",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                coverage_class="localized_visible_pattern",
                color_note="the main visible body reads as blue",
                trim_note="visible outer trim or edging reads as black against a blue interior",
                form_factor_note="the product reads as a backpack body rather than a handbag or tote",
                source_validity_issues=[],
                artifact_flags=[],
                hard_facts=[
                    "the product remains a backpack",
                    "the product remains a backpack cooler rather than a generic school or laptop backpack",
                    "keep the design compatible with an insulated cooler compartment and zipper opening rather than a plain daypack body",
                ],
                reference_crop_path="/tmp/backpack_cooler.evidence_crop.png",
                reference_cutout_path="/tmp/backpack_cooler.evidence_cutout.png",
                reference_silhouette_path="/tmp/backpack_cooler.evidence_silhouette.png",
            ),
        ),
    )


def _localized_lamp() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/lamp.png",
        phrase="pineapple table lamp",
        bbox=BoundingBox(x0=22, y0=14, x1=110, y1=190),
        confidence=0.88,
        crop_path="/tmp/lamp.crop.png",
        identity=ProductIdentitySpec(
            phrase="pineapple table lamp",
            category="home lighting",
            canonical_product_type="table lamp",
            source_title="Pineapple Table Lamp",
            support_mode="supported_display",
            default_scene_family="furnished_interior",
            interaction_mode="placed",
            style_persona="cozy_home",
            stable_base=True,
            rigid_vs_soft="rigid",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as black",
                form_factor_note="the visible lamp form is upright with a broad upper shade over a narrower lower support",
                hard_facts=[
                    "the product remains a table lamp",
                    "the product remains an upright table lamp with a stable base and upper light shade",
                ],
            ),
        ),
    )


def _localized_comforter() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/comforter.png",
        phrase="black comforter",
        bbox=BoundingBox(x0=14, y0=18, x1=150, y1=120),
        confidence=0.9,
        crop_path="/tmp/comforter.crop.png",
        identity=ProductIdentitySpec(
            phrase="black comforter",
            category="bedding",
            canonical_product_type="comforter",
            source_title="Black Comforter Set",
            support_mode="externally_supported_soft",
            default_scene_family="furnished_interior",
            interaction_mode="placed",
            style_persona="cozy_home",
            stable_base=False,
            rigid_vs_soft="soft",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as black with subtle tonal variation from textured fabric",
                color_confidence=0.78,
                coverage_class="full_visible_surface_pattern",
                hard_facts=["the product remains a comforter"],
            ),
        ),
    )


def _localized_noisy_quilt() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/quilt_source.png",
        phrase="blue quilt",
        bbox=BoundingBox(x0=10, y0=14, x1=150, y1=112),
        confidence=0.84,
        crop_path="/tmp/quilt.crop.png",
        identity=ProductIdentitySpec(
            phrase="blue quilt",
            category="bedding",
            canonical_product_type="quilt",
            source_title="Blue Floral Quilt",
            support_mode="externally_supported_soft",
            default_scene_family="furnished_interior",
            interaction_mode="placed",
            style_persona="cozy_home",
            stable_base=False,
            rigid_vs_soft="soft",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as blue with tonal variation from folds, pile, or soft-surface texture",
                coverage_class="low_variation_surface",
                hard_facts=["the product remains a quilt"],
                raw_evidence_caption="a white map with a brown outline",
                evidence_caption=None,
                artifact_flags=["border_foreground_intrusion"],
                reference_crop_path="/tmp/quilt.evidence_crop.png",
                reference_cutout_path="/tmp/quilt.evidence_cutout.png",
                reference_silhouette_path="/tmp/quilt.evidence_silhouette.png",
            ),
        ),
    )


def _localized_uncertain_tshirt() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/tshirt_source.png",
        phrase="white t-shirt",
        bbox=BoundingBox(x0=12, y0=16, x1=148, y1=118),
        confidence=0.78,
        crop_path="/tmp/tshirt.crop.png",
        identity=ProductIdentitySpec(
            phrase="white t-shirt",
            category="apparel",
            canonical_product_type="shirt",
            source_title="Perfect-T Women's T-Shirt",
            support_mode="wearable",
            default_scene_family="fashion_lifestyle",
            interaction_mode="worn",
            style_persona="refined_neutral",
            stable_base=False,
            rigid_vs_soft="soft",
            requires_human_model=True,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as a light neutral fabric with tonal variation from folds, drape, or soft-surface texture",
                color_confidence=0.54,
                coverage_class="low_variation_surface",
                coverage_note="the visible soft surface reads as tonal textile variation rather than a printed multicolor body",
                hard_facts=[
                    "the product remains a shirt",
                    "the product remains a shirt worn on the upper body",
                    "source evidence reads as the back of a shirt with a logo on it",
                ],
                surface_scope="partial_or_occluded",
                uncertainty_level="high",
                reference_crop_path="/tmp/tshirt.evidence_crop.png",
                reference_cutout_path="/tmp/tshirt.evidence_cutout.png",
                reference_silhouette_path="/tmp/tshirt.evidence_silhouette.png",
                silhouette_note="observed silhouette is horizontally spread",
            ),
        ),
    )


def _localized_dress() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/dress_source.png",
        phrase="floral dress",
        bbox=BoundingBox(x0=16, y0=12, x1=140, y1=180),
        confidence=0.84,
        crop_path="/tmp/dress.crop.png",
        identity=ProductIdentitySpec(
            phrase="floral dress",
            category="apparel",
            canonical_product_type="dress",
            source_title="Soft Floral Dress",
            support_mode="wearable",
            default_scene_family="fashion_lifestyle",
            interaction_mode="worn",
            style_persona="refined_neutral",
            stable_base=False,
            rigid_vs_soft="soft",
            requires_human_model=True,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as beige",
                color_confidence=0.72,
                coverage_class="broad_visible_surface_pattern",
                hard_facts=["the product remains a dress", "the product remains a dress worn on the body"],
            ),
        ),
    )


def _localized_folding_chair() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/folding_chair_source.png",
        phrase="black folding chair",
        bbox=BoundingBox(x0=18, y0=18, x1=132, y1=196),
        confidence=0.81,
        crop_path="/tmp/folding_chair.crop.png",
        identity=ProductIdentitySpec(
            phrase="black folding chair",
            category="furniture",
            canonical_product_type="folding chair",
            source_title="Commercial Black Folding Chair",
            support_mode="self_supporting_display",
            default_scene_family="editorial_interior",
            interaction_mode="placed",
            style_persona="refined_neutral",
            stable_base=True,
            rigid_vs_soft="rigid",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as black with reflective highlight variation",
                hard_facts=["the product remains a folding chair"],
                lower_region_note="the visible lower support frame continues below the seat and backrest and should remain present",
                form_factor_note="the visible furniture form is a folding chair with a rigid slatted seat and backrest",
            ),
        ),
    )


def _localized_quilt() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/quilt.png",
        phrase="blue quilt",
        bbox=BoundingBox(x0=14, y0=18, x1=150, y1=120),
        confidence=0.9,
        crop_path="/tmp/quilt.crop.png",
        identity=ProductIdentitySpec(
            phrase="blue quilt",
            category="bedding",
            canonical_product_type="quilt",
            source_title="Better Homes & Gardens Floral Matelasse Quilt, King, Blue Bedding",
            support_mode="externally_supported_soft",
            default_scene_family="furnished_interior",
            interaction_mode="placed",
            style_persona="cozy_home",
            stable_base=False,
            rigid_vs_soft="soft",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as blue with low-luster tonal variation from textured fabric",
                color_confidence=0.74,
                coverage_class="low_variation_surface",
                coverage_note="most of the visible bedding surface reads as a tonal blue textile field",
                hard_facts=["the product remains a quilt"],
            ),
        ),
    )


def _localized_pet_bed() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/pet_bed.png",
        phrase="flat plush pet bed",
        bbox=BoundingBox(x0=8, y0=12, x1=150, y1=88),
        confidence=0.84,
        crop_path="/tmp/pet_bed.crop.png",
        identity=ProductIdentitySpec(
            phrase="flat plush pet bed",
            category="pet home",
            canonical_product_type="pet bed",
            source_title="Large Plush Pet Bed",
            support_mode="externally_supported_soft",
            default_scene_family="furnished_interior",
            interaction_mode="placed",
            style_persona="cozy_home",
            stable_base=False,
            rigid_vs_soft="soft",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as black with subtle tonal variation from plush fabric",
                color_confidence=0.78,
                coverage_class="low_variation_surface",
                form_factor_note="the visible product form is a low soft pet bed meant to rest on the floor",
                edge_profile_note="the visible pet bed perimeter remains low and softly graded around the resting surface rather than rising into bulky bolsters",
                soft_structure_note="the visible soft product structure reads as a flat plush pad with no bulky bolster, boxed sidewall, or nested inner tray",
                soft_structure_class="flat_surface",
                hard_facts=["the product remains a pet bed"],
            ),
        ),
    )


def _localized_office_chair() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/office_chair.png",
        phrase="black office chair",
        bbox=BoundingBox(x0=10, y0=10, x1=130, y1=180),
        confidence=0.87,
        crop_path="/tmp/office_chair.crop.png",
        identity=ProductIdentitySpec(
            phrase="gtracing office chair",
            category="furniture",
            canonical_product_type="office chair",
            source_title="GTRACING Mesh & Faux Leather Office Gaming Chair with Footrest White Office Chair",
            support_mode="self_supporting_display",
            default_scene_family="editorial_interior",
            interaction_mode="placed",
            style_persona="refined_neutral",
            stable_base=True,
            rigid_vs_soft="rigid",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as black",
                color_confidence=0.8,
                coverage_class="low_variation_surface",
                lower_region_note="the visible lower support frame continues below the seat and backrest and should remain present",
                hard_facts=["the product remains an office chair"],
            ),
        ),
    )


def _localized_toaster() -> LocalizedProduct:
    return LocalizedProduct(
        source_image="/tmp/toaster.png",
        phrase="blue-gray toaster",
        bbox=BoundingBox(x0=16, y0=18, x1=134, y1=120),
        confidence=0.86,
        crop_path="/tmp/toaster.crop.png",
        identity=ProductIdentitySpec(
            phrase="two-slice toaster",
            category="kitchen appliance",
            canonical_product_type="toaster",
            source_title="Mainstays 2-Slice Toaster, Black",
            support_mode="self_supporting_display",
            default_scene_family="tabletop_display",
            interaction_mode="placed",
            style_persona="refined_neutral",
            stable_base=True,
            rigid_vs_soft="rigid",
            requires_human_model=False,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as blue-gray with cool-toned reflective variation",
                color_confidence=0.8,
                palette=["gray", "blue"],
                coverage_class="low_variation_surface",
                hard_facts=["the product remains a toaster"],
            ),
        ),
    )


def test_baseline_prompt_has_flux_order_and_constraints() -> None:
    prompt = PromptComposer().compose_baseline(_localized_product(), seed=3)
    text = prompt.to_prompt_text()
    assert text.startswith("Subject:")
    assert "Action:" in text and "Style:" in text and "Context:" in text
    assert prompt.reference_images
    assert any("preserve" in item.lower() for item in prompt.preservation_constraints)
    assert "featured tote bag" in prompt.subject.lower()
    assert any("must remain a tote bag" in item.lower() for item in prompt.preservation_constraints)
    assert "one coherent support relation" in prompt.to_prompt_text().lower()
    assert "fashion-oriented lifestyle scene" in prompt.context.lower()
    assert "preserving all observed source evidence" in prompt.action.lower()
    assert any("invented details must remain compatible" in item.lower() for item in prompt.preservation_constraints)
    assert any("hard facts" in item.lower() or "retain observed fact" in item.lower() for item in prompt.preservation_constraints)
    assert "cleaner or more flattering angle" in prompt.context.lower()
    assert any("preserve compatible coverage across newly revealed adjacent surfaces" in item.lower() for item in prompt.preservation_constraints)
    assert any("keep them within the observed palette family" in item.lower() for item in prompt.preservation_constraints)
    assert any("upper component" in item.lower() for item in prompt.preservation_constraints)
    assert any("material and texture cues" in item.lower() for item in prompt.preservation_constraints)


def test_business_prior_prompt_uses_restrained_fallback_context_without_duplicate_atoms() -> None:
    localized = _localized_product()
    prior = CampaignPriorSpec(
        style_atoms=["clear hero framing", "human-in-use framing", "keep the featured tote bag unmistakable"],
        scenario_slots=["fashion_lifestyle"],
        scene_family="fashion_lifestyle",
        support_relation="carried_by_hand",
        metadata={"retrieval_mode": "evidence_fallback"},
    )

    prompt = PromptComposer().compose_business_prior(localized, prior, seed=7)
    baseline = PromptComposer().compose_baseline(localized, seed=7)

    assert prompt.style != baseline.style
    assert "fallback prior conservatively" in prompt.context.lower()
    assert "distinct evidence-compatible environment and framing" in prompt.context.lower()
    assert "use the following high-ctr style cues" not in prompt.context.lower()


def test_prompts_block_flat_gray_background_collapse_language() -> None:
    localized = _localized_dress()
    prompt = PromptComposer().compose_business_prior(
        localized,
        CampaignPriorSpec(
            style_atoms=["human-in-use framing"],
            scenario_slots=["fashion_lifestyle"],
            scene_family="fashion_lifestyle",
            support_relation="worn_on_body",
            metadata={"retrieval_mode": "retrieval", "creative_direction": "architectural lifestyle storytelling"},
        ),
        seed=29,
    )
    text = prompt.to_prompt_text().lower()

    assert "flat gray seamless" in text
    assert "partially rendered gray-and-white geometric backdrop" in text
    assert "clean studio cyclorama with lifestyle casting" not in text
    assert "monochrome gallery-like campaign room" not in text


def test_pet_bed_prompt_preserves_flat_soft_structure() -> None:
    prompt = PromptComposer().compose_baseline(_localized_pet_bed(), seed=11)
    text = prompt.to_prompt_text().lower()

    assert "flat plush pad" in text
    assert "low-profile plush pet resting pad" in text


def test_pet_bed_subject_stays_concise_when_presentation_label_differs() -> None:
    prompt = PromptComposer().compose_baseline(_localized_pet_bed(), seed=13)

    assert "large dog bed washable" not in prompt.subject.lower()
    assert "low-profile plush pet resting pad" in prompt.subject.lower()


def test_structured_low_variation_subject_drops_conflicting_title_color() -> None:
    prompt = PromptComposer().compose_baseline(_localized_office_chair(), seed=17)

    assert "white office chair" not in prompt.subject.lower()
    assert "office chair" in prompt.subject.lower()


def test_high_uncertainty_evidence_disables_crop_only_color_lock() -> None:
    localized = _localized_product()
    localized.identity.observed_evidence.uncertainty_level = "high"
    localized.identity.observed_evidence.surface_scope = "partial_or_occluded"
    localized.identity.observed_evidence.reference_crop_path = "/tmp/source.evidence_crop.png"

    refs = PromptComposer()._references(localized)

    assert [ref.role for ref in refs] == ["product crop", "product cutout", "product silhouette"]


def test_high_uncertainty_evidence_becomes_soft_guidance_in_prompt() -> None:
    localized = _localized_product()
    localized.identity.observed_evidence.uncertainty_level = "high"
    localized.identity.observed_evidence.surface_scope = "partial_or_occluded"
    localized.identity.observed_evidence.reference_crop_path = "/tmp/source.evidence_crop.png"

    prompt = PromptComposer().compose_baseline(localized, seed=7)
    constraints = "\n".join(prompt.preservation_constraints)

    assert "Use these source cues from image 1 as compatibility guidance:" in prompt.context
    assert "stay compatible with source cue: the main visible body reads as blue with compatible printed accents in purple, pink" in constraints
    assert "retain observed fact: the main visible body reads as blue with compatible printed accents in purple, pink" not in constraints


def test_pet_bed_prompt_preserves_low_profile_structure() -> None:
    localized = _localized_pet_bed().model_copy(
        update={
            "identity": _localized_pet_bed().identity.model_copy(
                update={
                    "observed_evidence": _localized_pet_bed().identity.observed_evidence.model_copy(
                        update={
                            "soft_structure_note": "the visible soft product structure shows only a modest perimeter rise around the resting surface",
                            "soft_structure_class": "low_perimeter_relief",
                        }
                    )
                }
            )
        }
    )

    prompt = PromptComposer().compose_baseline(localized, seed=13)

    assert any("gentle perimeter rise" in item.lower() for item in prompt.preservation_constraints)
    assert any("boxed pet sofa" in item.lower() for item in prompt.preservation_constraints)


def test_business_prior_prompt_uses_typed_creative_hints_for_retrieval() -> None:
    localized = _localized_product()
    prior = CampaignPriorSpec(
        style_atoms=["human-in-use framing", "editorial apparel styling"],
        scenario_slots=["fashion_lifestyle"],
        scene_family="fashion_lifestyle",
        support_relation="carried_by_hand",
        metadata={
            "retrieval_mode": "retrieval",
            "creative_seed": 101,
            "lighting_hint": "high-end studio key light with grounded shadows",
            "camera_hint": "designer-grade three-quarter or frontal campaign framing with believable perspective",
            "creative_direction": "architectural lifestyle storytelling with polished casual confidence",
            "cast_hint": "a clearly different model with polished casual styling and restrained body language",
        },
    )

    prompt = PromptComposer().compose_business_prior(localized, prior, seed=7)
    text = prompt.to_prompt_text().lower()

    assert "architectural lifestyle storytelling with polished casual confidence" in text
    assert "materially influence environment choice, framing, casting, and pacing" in text
    assert "a clearly different model with polished casual styling and restrained body language" in text
    assert prompt.style.lower().startswith("high-end studio key light with grounded shadows")


def test_business_prior_prompt_uses_retrieved_scene_direction_in_scene_fallback_mode() -> None:
    localized = _localized_product()
    prior = CampaignPriorSpec(
        style_atoms=["clear hero framing"],
        scenario_slots=["fashion_lifestyle"],
        scene_family="fashion_lifestyle",
        support_relation="carried_by_hand",
        metadata={
            "retrieval_mode": "scene_retrieval_fallback",
            "creative_seed": 101,
            "lighting_hint": "warm natural light with controlled contrast",
            "camera_hint": "designer-grade three-quarter or frontal campaign framing with believable perspective",
            "creative_direction": "retrieval-informed casual lifestyle pacing",
            "cast_hint": "a clearly different model with polished casual styling and restrained body language",
        },
    )

    prompt = PromptComposer().compose_business_prior(localized, prior, seed=7)
    text = prompt.to_prompt_text().lower()

    assert "retrieved creative signals" in text
    assert "retrieval-informed scene direction: retrieval-informed casual lifestyle pacing" in prompt.style.lower()


def test_business_prior_cast_hint_keeps_product_specific_casting_constraints() -> None:
    localized = _localized_product()
    localized.identity.casting_note = (
        "If a person appears, use playful, soft, or feminine-coded casting that matches the product."
    )
    prior = CampaignPriorSpec(
        style_atoms=["human-in-use framing"],
        scenario_slots=["fashion_lifestyle"],
        scene_family="fashion_lifestyle",
        support_relation="carried_by_hand",
        metadata={
            "retrieval_mode": "retrieval",
            "cast_hint": "a clearly different model with polished casual styling and restrained body language",
        },
    )

    prompt = PromptComposer().compose_business_prior(localized, prior, seed=7)
    text = prompt.to_prompt_text().lower()

    assert "feminine-coded casting" in text
    assert "a clearly different model with polished casual styling and restrained body language" in text
    assert "no duplicated or impossible limbs" in text


def test_prompt_locks_confident_dominant_body_color() -> None:
    prompt = PromptComposer().compose_baseline(_localized_comforter(), seed=13)
    constraints = " ".join(prompt.preservation_constraints).lower()
    roles = [reference.role for reference in prompt.reference_images]

    assert "dominant visible body color black" in constraints
    assert "different or lighter color family" in constraints
    assert roles[0] == "product crop"
    assert roles == ["product crop"]


def test_prompt_adds_color_fidelity_guardrails_for_chromatic_soft_goods() -> None:
    localized = _localized_comforter().model_copy(
        update={
            "identity": _localized_comforter().identity.model_copy(
                update={
                    "phrase": "dark green comforter",
                    "source_title": "Dark Green Comforter Set",
                    "observed_evidence": _localized_comforter().identity.observed_evidence.model_copy(
                        update={
                            "color_note": "the main visible body reads as dark green with low-luster tonal variation from textured fabric",
                            "palette": ["green", "black", "gray"],
                            "color_confidence": 0.78,
                            "soft_structure_class": "low_perimeter_relief",
                        }
                    ),
                }
            )
        }
    )
    prompt = PromptComposer().compose_baseline(localized, seed=13)
    text = prompt.to_prompt_text().lower()

    assert "neutral or daylight-balanced lighting" in text
    assert "broad surface-led composition" in text


def test_prompt_adds_low_profile_surface_guardrails_for_pet_bed() -> None:
    prompt = PromptComposer().compose_baseline(_localized_pet_bed(), seed=23)
    text = prompt.to_prompt_text().lower()

    assert "broad surface-dominant framing" in text
    assert "exaggerates loft or edge thickness" in text


def test_subject_phrase_strips_conflicting_title_color_tokens() -> None:
    localized = _localized_lamp().model_copy(
        update={
            "identity": _localized_lamp().identity.model_copy(
                update={
                    "canonical_product_type": "office chair",
                    "category": "furniture",
                    "source_title": "GTRACING Office Chair With Footrest, White",
                    "phrase": "office chair",
                    "observed_evidence": _localized_lamp().identity.observed_evidence.model_copy(
                        update={
                            "color_note": "the main visible body reads as black",
                            "palette": ["black", "gray"],
                            "color_confidence": 0.82,
                        }
                    ),
                }
            )
        }
    )

    prompt = PromptComposer().compose_baseline(localized, seed=21)

    assert "white office chair" not in prompt.subject.lower()
    assert "black" not in prompt.subject.lower()


def test_subject_phrase_uses_concise_evidence_label_when_presentation_differs() -> None:
    prompt = PromptComposer().compose_baseline(_localized_pet_bed(), seed=23)

    assert "large dog bed washable" not in prompt.subject.lower()
    assert "low-profile plush pet resting pad" in prompt.subject.lower()


def test_subject_phrase_prefers_evidence_led_identity_when_no_visible_brand_signal() -> None:
    localized = _localized_lamp().model_copy(
        update={
            "identity": _localized_lamp().identity.model_copy(
                update={
                    "canonical_product_type": "coffee maker",
                    "category": "kitchen appliance",
                    "source_title": "Mainstays Black 12 Cup Drip Coffee Maker",
                    "phrase": "coffee maker",
                    "support_mode": "self_supporting_display",
                    "default_scene_family": "tabletop_display",
                    "rigid_vs_soft": "rigid",
                    "observed_evidence": _localized_lamp().identity.observed_evidence.model_copy(
                        update={
                            "color_note": "the main visible body reads as black",
                            "hard_facts": ["the product remains a coffee maker"],
                        }
                    ),
                }
            )
        }
    )

    prompt = PromptComposer().compose_business_prior(
        localized,
        CampaignPriorSpec(
            style_atoms=["clear hero framing"],
            scenario_slots=["tabletop_display"],
            scene_family="tabletop_display",
            support_relation="standing_on_surface",
            metadata={"retrieval_mode": "retrieval", "creative_direction": "clean countertop storytelling"},
        ),
        seed=21,
    )

    assert "mainstays" not in prompt.subject.lower()
    assert "coffee maker" in prompt.subject.lower()


def test_subject_phrase_preserves_functional_backpack_subtype() -> None:
    prompt = PromptComposer().compose_baseline(_localized_backpack_cooler(), seed=31)

    assert "backpack cooler" in prompt.subject.lower()
    assert "generic school or laptop backpack" in " ".join(prompt.preservation_constraints).lower()


def test_references_include_source_for_functional_subtype_context() -> None:
    refs = PromptComposer()._references(_localized_backpack_cooler())

    assert any(ref.role == "base image" and ref.path == "/tmp/backpack_cooler.png" for ref in refs)


def test_uncertain_apparel_prioritizes_cutout_reference() -> None:
    refs = PromptComposer()._references(_localized_uncertain_tshirt())

    assert [ref.role for ref in refs] == ["product cutout", "product crop", "product silhouette"]


def test_uncertain_tshirt_prompt_avoids_alternate_top_styling_language() -> None:
    prompt = PromptComposer().compose_business_prior(
        _localized_uncertain_tshirt(),
        CampaignPriorSpec(
            style_atoms=["editorial apparel styling"],
            scenario_slots=["fashion_lifestyle"],
            scene_family="fashion_lifestyle",
            support_relation="worn_on_body",
            metadata={"retrieval_mode": "retrieval", "creative_direction": "gallery-adjacent editorial rhythm"},
        ),
        seed=41,
    )
    text = prompt.to_prompt_text().lower()

    assert "like knitwear, denim, or relaxed separates" not in text
    assert "do not reinterpret the featured shirt as knitwear" in text


def test_noisy_soft_goods_omit_cutout_reference_when_crop_evidence_is_visually_incompatible() -> None:
    refs = PromptComposer()._references(_localized_noisy_quilt())

    assert [ref.role for ref in refs] == ["product crop", "product silhouette"]


def test_dress_prompt_blocks_invented_contrasting_sleeves() -> None:
    prompt = PromptComposer().compose_business_prior(
        _localized_dress(),
        CampaignPriorSpec(
            style_atoms=["editorial apparel styling"],
            scenario_slots=["fashion_lifestyle"],
            scene_family="fashion_lifestyle",
            support_relation="worn_on_body",
            metadata={"retrieval_mode": "retrieval", "creative_direction": "gallery-adjacent editorial rhythm"},
        ),
        seed=43,
    )
    text = prompt.to_prompt_text().lower()

    assert "do not invent contrasting sleeves" in text
    assert "do not reinterpret the dress as a mixed-material outfit with separate gray knit sleeves" in text
    assert "do not add visible jeans, trousers, leggings" in text
    assert "use a private or tightly controlled corner" in text


def test_business_prior_dress_prompt_does_not_reintroduce_denim_or_relaxed_separates() -> None:
    prompt = PromptComposer().compose_business_prior(
        _localized_dress(),
        CampaignPriorSpec(
            style_atoms=["editorial apparel styling"],
            scenario_slots=["fashion_lifestyle"],
            scene_family="fashion_lifestyle",
            support_relation="worn_on_body",
            metadata={"retrieval_mode": "retrieval", "creative_direction": "architectural lifestyle storytelling"},
        ),
        seed=47,
    )
    text = prompt.to_prompt_text().lower()

    assert "through denim" not in text
    assert "relaxed separates" not in text


def test_low_variation_soft_goods_prompt_blocks_patchwork_insets() -> None:
    prompt = PromptComposer().compose_business_prior(
        _localized_noisy_quilt(),
        CampaignPriorSpec(
            style_atoms=["quiet home storytelling"],
            scenario_slots=["furnished_interior"],
            scene_family="furnished_interior",
            support_relation="resting_on_surface",
            metadata={"retrieval_mode": "fallback", "creative_direction": "broad surface readability"},
        ),
        seed=47,
    )
    text = prompt.to_prompt_text().lower()

    assert "do not invent patchwork blocks" in text


def test_folding_chair_prompt_requires_single_coherent_articulation() -> None:
    prompt = PromptComposer().compose_baseline(_localized_folding_chair(), seed=45)
    text = prompt.to_prompt_text().lower()

    assert "one coherent folding-chair structure" in text
    assert "do not mix folded and unfolded geometry" in text
    assert "fully opaque and structurally continuous" in text
    assert "do not merge nearby plinths, tables, or background furniture planes" in text


def test_subject_phrase_treats_control_window_as_non_brand_signal() -> None:
    localized = _localized_lamp().model_copy(
        update={
            "identity": _localized_lamp().identity.model_copy(
                update={
                    "canonical_product_type": "coffee maker",
                    "category": "kitchen appliance",
                    "source_title": "Mainstays Black 12 Cup Drip Coffee Maker",
                    "phrase": "Mainstays Black 12 Cup Drip Coffee Maker",
                    "support_mode": "self_supporting_display",
                    "default_scene_family": "tabletop_display",
                    "rigid_vs_soft": "rigid",
                    "observed_evidence": _localized_lamp().identity.observed_evidence.model_copy(
                        update={
                            "coverage_note": (
                                "a localized high-contrast front panel or control window interrupts one region of the visible product surface"
                            ),
                            "value_relation_note": (
                                "the localized front panel or control window is visibly lighter than the main body"
                            ),
                            "hard_facts": ["the product remains a coffee maker"],
                        }
                    ),
                }
            )
        }
    )

    prompt = PromptComposer().compose_business_prior(
        localized,
        CampaignPriorSpec(
            style_atoms=["clear hero framing"],
            scenario_slots=["tabletop_display"],
            scene_family="tabletop_display",
            support_relation="standing_on_surface",
            metadata={"retrieval_mode": "retrieval", "creative_direction": "clean countertop storytelling"},
        ),
        seed=33,
    )

    assert "mainstays" not in prompt.subject.lower()
    assert "coffee maker" in prompt.subject.lower()


def test_baseline_prompt_uses_persona_clause_for_playful_products() -> None:
    prompt = PromptComposer().compose_baseline(_localized_product(), seed=11)

    assert "casual, approachable" in prompt.context.lower()
    assert "formal businesswear" in prompt.context.lower()


def test_baseline_prompt_requires_direct_grip_for_handleless_wallets() -> None:
    composer = PromptComposer()
    localized = _localized_wallet().model_copy(
        update={
            "identity": _localized_wallet().identity.model_copy(
                update={
                    "observed_evidence": _localized_wallet().identity.observed_evidence.model_copy(
                        update={
                            "color_note": "the main visible body reads as blue with compatible printed accents in gray, beige",
                            "color_confidence": 0.72,
                        }
                    )
                }
            )
        }
    )
    prompt = composer.compose_baseline(localized, seed=5)
    context = prompt.context.lower()
    constraints = " ".join(prompt.preservation_constraints).lower()

    assert "directly gripped in one hand" in context
    assert "waist-up, half-body, or close hand-focused composition" in context
    assert "do not add handles, straps, wrist loops, shoulder attachments, crossbody carry hardware, or dangling carry components" in context
    assert "crossbody carry hardware" in context
    assert "shoulder bag, crossbody bag, or hanging pouch" in context
    assert "direct hand contact on the product body itself" in constraints
    assert "do not reinterpret a compact hand-held accessory as a shoulder bag" in constraints
    assert "different model in a clean waist-up or hand-focused pose" in prompt.action.lower()
    assert composer._dominant_body_color_hint(localized) == "blue"
    assert "do not spread the product's dominant body color" in constraints
    assert "muted neutral solids" in constraints
    assert "avoid blazers, suit jackets, office shirts, ties, tailored suiting, or corporate styling" in prompt.action.lower()
    assert "do not include background people" in prompt.action.lower()


def test_backpack_prompt_blocks_duplicated_harness_straps_when_back_side_is_observed() -> None:
    localized = _localized_backpack().model_copy(
        update={
            "identity": _localized_backpack().identity.model_copy(
                update={
                    "observed_evidence": _localized_backpack().identity.observed_evidence.model_copy(
                        update={
                            "upper_component_count": 2,
                            "upper_region_note": "the visible upper component splits into multiple narrow segments above the main body",
                            "hard_facts": [
                                "the product remains a backpack",
                                "the visible backpack body includes darker harness, panel, or attachment zones against a gray main body",
                            ],
                        }
                    )
                }
            )
        }
    )
    prompt = PromptComposer().compose_baseline(localized, seed=37)
    text = prompt.to_prompt_text().lower()

    assert "one coherent strap system" in text
    assert "do not add a second duplicated strap set" in text


def test_soft_goods_prompt_keeps_people_and_body_fragments_out_of_frame() -> None:
    prompt = PromptComposer().compose_baseline(_localized_comforter(), seed=13)
    text = prompt.to_prompt_text().lower()

    assert "keep all people, hands, sleeves, and body fragments completely out of frame" in text
    assert "do not show it worn, wrapped around, touched by, or draped across any person" in text
    assert "exactly one human subject" not in text


def test_product_only_clarity_mode_does_not_inject_human_subject_language() -> None:
    prompt = PromptComposer().compose_baseline(_localized_quilt(), seed=13)
    text = prompt.to_prompt_text().lower()

    assert "exactly one human subject" not in text
    assert "keep the frame product-only" in text
    assert "clearly visible bed or daybed surface" in text
    assert "do not stage it as a throw blanket on a bench, chair, sofa, or occupied seat" in text


def test_artifact_flags_add_non_product_noise_guardrails() -> None:
    localized = _localized_product().model_copy(
        update={
            "identity": _localized_product().identity.model_copy(
                update={
                    "observed_evidence": _localized_product().identity.observed_evidence.model_copy(
                        update={"artifact_flags": ["border_text_overlay", "border_human_fragment"]}
                    )
                }
            )
        }
    )

    prompt = PromptComposer().compose_business_prior(
        localized,
        CampaignPriorSpec(
            style_atoms=["clear hero framing"],
            scenario_slots=["fashion_lifestyle"],
            scene_family="fashion_lifestyle",
            support_relation="worn_on_body",
            metadata={"retrieval_mode": "retrieval", "creative_direction": "product-led lifestyle direction"},
        ),
        seed=31,
    )
    text = prompt.to_prompt_text().lower()

    assert "camera watermarks" in text
    assert "interface chrome" in text
    assert "do not reinterpret incidental source hands" in text


def test_wearable_prompt_references_exclude_full_source_image() -> None:
    prompt = PromptComposer().compose_baseline(_localized_product(), seed=17)

    assert all(reference.path != "/tmp/source.png" for reference in prompt.reference_images)


def test_business_prior_locks_view_control_for_artifact_flagged_low_profile_soft_goods() -> None:
    composer = PromptComposer()
    localized = _localized_pet_bed().model_copy(
        update={
            "identity": _localized_pet_bed().identity.model_copy(
                update={
                    "observed_evidence": _localized_pet_bed().identity.observed_evidence.model_copy(
                        update={"artifact_flags": ["border_text_overlay"], "uncertainty_level": "high"}
                    )
                }
            )
        }
    )

    baseline = composer.compose_baseline(localized, seed=19)
    business = composer.compose_business_prior(
        localized,
        CampaignPriorSpec(
            style_atoms=["clear hero framing"],
            scenario_slots=["furnished_interior"],
            scene_family="furnished_interior",
            support_relation="resting_on_surface",
            metadata={
                "retrieval_mode": "retrieval",
                "background_hint": "a bright editorial breakfast nook",
                "lighting_hint": "high-end studio key light with grounded shadows",
                "camera_hint": "commercial hero framing that may reveal new surfaces without contradicting the source evidence",
                "creative_direction": "premium home editorial styling",
            },
        ),
        seed=19,
    )

    baseline_parts = [part.strip() for part in baseline.style.split(";")]
    business_parts = [part.strip() for part in business.style.split(";")]

    assert business_parts[0] == baseline_parts[0]
    assert business_parts[1] == baseline_parts[1]


def test_clarity_mode_adds_single_human_anatomy_guardrails() -> None:
    prompt = PromptComposer().compose_baseline(_localized_product(), seed=19, reinvention_mode="clarity")
    text = prompt.to_prompt_text().lower()

    assert "exactly one different model in a calm stable pose" in text
    assert "do not include background people" in text
    assert "no other people visible anywhere in the background" in text
    assert any("no duplicated or impossible limbs" in item.lower() for item in prompt.preservation_constraints)


def test_refined_neutral_persona_explicitly_avoids_officewear() -> None:
    prompt = PromptComposer().compose_baseline(_localized_wallet(), seed=9)
    context = prompt.context.lower()

    assert "avoid suits, blazers, sharp tailoring, and officewear" in context


def test_business_prior_compact_hand_focus_adds_neutral_wardrobe_clause() -> None:
    localized = _localized_wallet()
    prior = CampaignPriorSpec(
        style_atoms=["clear hero framing"],
        scenario_slots=["fashion_lifestyle"],
        scene_family="fashion_lifestyle",
        support_relation="carried_by_hand",
        metadata={
            "retrieval_mode": "scene_retrieval_fallback",
            "creative_seed": 101,
            "lighting_hint": "warm natural light with controlled contrast",
            "camera_hint": "close hand-detail editorial framing that keeps the product large and dominant with minimal torso overlap",
            "creative_direction": "tight accessory storytelling with calm neutral wardrobe separation and direct hand focus",
            "cast_hint": "a clearly different model with minimal neutral wardrobe, simple hand posing, and restrained body language",
        },
    )

    prompt = PromptComposer().compose_business_prior(localized, prior, seed=7)
    text = prompt.to_prompt_text().lower()

    assert "quiet and neutral around the product" in text
    assert "avoid saturated overshirts or large color-blocked layers" in text


def test_subject_phrase_strips_unsupported_pattern_tokens_from_soft_goods_title() -> None:
    prompt = PromptComposer().compose_baseline(_localized_quilt(), seed=29)

    assert "floral" not in prompt.subject.lower()
    assert "matelasse" in prompt.subject.lower()


def test_business_prior_category_fallback_uses_fallback_lighting_hints() -> None:
    localized = _localized_comforter().model_copy(
        update={
            "identity": _localized_comforter().identity.model_copy(
                update={
                    "category": "drinkware",
                    "canonical_product_type": "water bottle",
                    "support_mode": "self_supporting_display",
                    "default_scene_family": "tabletop_display",
                    "interaction_mode": "handheld_or_display",
                    "rigid_vs_soft": "rigid",
                    "requires_human_model": False,
                    "observed_evidence": _localized_comforter().identity.observed_evidence.model_copy(
                        update={
                            "color_note": "the main visible body reads as black with reflective highlight variation",
                            "color_confidence": 0.82,
                            "coverage_class": "localized_visible_pattern",
                            "upper_region_note": "the visible upper attachment remains visually distinct in gray above a black main body",
                        }
                    ),
                }
            )
        }
    )
    prior = CampaignPriorSpec(
        style_atoms=["clear hero framing"],
        scenario_slots=["tabletop_display"],
        scene_family="tabletop_display",
        support_relation="standing_on_surface",
        metadata={
            "retrieval_mode": "category_fallback",
            "lighting_hint": "high-end studio key light with grounded shadows",
            "camera_hint": "commercial hero framing that may reveal new surfaces without contradicting the source evidence",
            "creative_direction": "merchandising-forward tabletop pacing with bright utility clarity",
        },
    )

    prompt = PromptComposer().compose_business_prior(localized, prior, seed=7)

    assert prompt.style.lower().startswith("high-end studio key light with grounded shadows")


def test_backpack_prompt_uses_worn_or_carried_support_language() -> None:
    prompt = PromptComposer().compose_baseline(_localized_backpack(), seed=13)
    text = prompt.to_prompt_text().lower()

    assert "worn naturally on the back or shoulder" in text or "carried with visible hand or shoulder support" in text
    assert "coherent strap attachment" in text
    assert "rather than reinterpreting it as a tote or handbag" in text


def test_backpack_prompt_adds_back_panel_guardrail_and_strips_conflicting_title_color_alias() -> None:
    localized = _localized_backpack().model_copy(
        update={
            "identity": _localized_backpack().identity.model_copy(
                update={
                    "source_title": "Rawlings Scarlet Baseball Backpack",
                    "observed_evidence": _localized_backpack().identity.observed_evidence.model_copy(
                        update={
                            "palette": ["black", "gray"],
                            "color_note": "the main visible body reads as black with gray trim",
                            "hard_facts": [
                                "the product remains a backpack",
                                "the observed source shows the back of the backpack with harness-facing structure",
                            ],
                        }
                    ),
                }
            )
        }
    )

    prompt = PromptComposer().compose_baseline(localized, seed=19)
    text = prompt.to_prompt_text().lower()

    assert "scarlet" not in prompt.subject.lower()
    assert "contradictory front-face colorway" in text


def test_playful_compact_bag_prompt_blocks_masculine_hand_styling_and_duplicate_bag_instances() -> None:
    localized = _localized_wallet().model_copy(
        update={
            "identity": _localized_wallet().identity.model_copy(
                update={
                    "style_persona": "playful_casual",
                    "casting_note": (
                        "If a person appears, use casting and hand styling that feel playful, soft, or feminine-coded "
                        "in a way that matches the product's visual language. Avoid masculine-coded or business-formal "
                        "presentation when it conflicts with the observed design."
                    ),
                }
            )
        }
    )

    prompt = PromptComposer().compose_baseline(localized, seed=23)
    text = prompt.to_prompt_text().lower()

    assert "avoid masculine-coded hands" in text
    assert "show exactly one coherent bag or accessory instance" in text


def test_toaster_subject_strips_conflicting_title_color_when_cool_rigid_evidence_disagrees() -> None:
    prompt = PromptComposer().compose_baseline(_localized_toaster(), seed=29)

    assert "black toaster" not in prompt.subject.lower()
    assert "toaster" in prompt.subject.lower()


def test_toaster_prompt_forbids_human_fragments_for_display_product() -> None:
    prompt = PromptComposer().compose_business_prior(
        _localized_toaster(),
        CampaignPriorSpec(
            style_atoms=["clear hero framing"],
            scenario_slots=["tabletop_display"],
            scene_family="tabletop_display",
            support_relation="standing_on_surface",
            metadata={"retrieval_mode": "category_fallback"},
        ),
        seed=31,
    )

    text = prompt.to_prompt_text().lower()
    assert "partial body fragments" in text
    assert "do not add people" in text
    assert "completely out of frame" in text


def test_lamp_prompt_avoids_soft_deformation_language() -> None:
    prompt = PromptComposer().compose_baseline(_localized_lamp(), seed=15)
    text = prompt.to_prompt_text().lower()

    assert "coherent relationship between the base, stem, and shade" in text
    assert "do not imply soft deformation" in text
    assert "printed label or surface graphic" in text


@pytest.mark.parametrize(
    ("scene_family", "category", "canonical_product_type", "support_mode", "style_persona"),
    [
        ("tabletop_display", "drinkware", "water bottle", "self_supporting_display", "sport_utility"),
        ("furnished_interior", "home decor", "decorative pillow", "externally_supported_soft", "cozy_home"),
        ("fashion_lifestyle", "bag", "tote bag", "portable_flexible", "playful_casual"),
        ("outdoor_lifestyle", "bag", "backpack", "wearable", "sport_utility"),
        ("retail_display", "bag", "handbag", "portable_flexible", "refined_neutral"),
        ("editorial_interior", "home lighting", "table lamp", "supported_display", "refined_neutral"),
    ],
)
def test_scene_palettes_produce_multiple_context_variants(
    scene_family: str,
    category: str,
    canonical_product_type: str,
    support_mode: str,
    style_persona: str,
) -> None:
    localized = LocalizedProduct(
        source_image="/tmp/source.png",
        phrase=canonical_product_type,
        bbox=BoundingBox(x0=10, y0=20, x1=110, y1=180),
        confidence=0.9,
        crop_path="/tmp/source.crop.png",
        identity=ProductIdentitySpec(
            phrase=canonical_product_type,
            category=category,
            canonical_product_type=canonical_product_type,
            source_title=f"Sample {canonical_product_type.title()}",
            support_mode=support_mode,
            default_scene_family=scene_family,
            style_persona=style_persona,
            stable_base=scene_family in {"tabletop_display", "retail_display", "editorial_interior"},
            rigid_vs_soft="soft" if category in {"apparel", "home decor"} else "rigid",
            requires_human_model=category in {"apparel", "footwear"},
            observed_evidence=ObservedEvidenceSpec(
                hard_facts=[f"the product remains a {canonical_product_type}"],
            ),
        ),
    )

    contexts = {PromptComposer().compose_baseline(localized, seed=seed).context for seed in range(4)}

    assert len(contexts) >= 2
