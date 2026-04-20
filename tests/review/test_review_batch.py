from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from product_campaign_pipeline.review_batch import (
    HUMAN_REINVENTION_CANDIDATE_MODES,
    LocalizationArtifactRecord,
    REINVENTION_CANDIDATE_MODES,
    RetrievalCandidate,
    ReviewSeedRecord,
    _conditioning_reference_images,
    _should_prepare_color_anchor_asset,
    _distinct_person_mask_count,
    _select_generated_focus_mask,
    _primary_generation_input_image,
    build_category_fallback_prior,
    build_business_prior,
    build_business_prior_creative_metadata,
    build_generation_request,
    build_localized_product,
    build_scene_retrieval_fallback_prior,
    build_style_plan,
    build_visual_evidence_profile,
    choose_scene_family,
    choose_support_relation,
    apply_dominant_body_color_correction,
    correct_structured_display_surface_inference,
    candidate_has_hard_evidence_conflict,
    default_support_relation_for_identity,
    detect_any_person_in_scene,
    backpack_harness_face_observed,
    evaluate_category_scores,
    evaluate_prompt_scene_conflicts,
    filter_retrieval_candidates,
    filter_scene_retrieval_candidates,
    infer_canonical_product_type,
    infer_category,
    infer_evidence_caption,
    infer_functional_subtype_hint,
    extract_caption_colors,
    infer_core_body_palette_ranked,
    infer_dark_reflective_body_override,
    infer_affordance_profile,
    infer_localized_contrast_panel,
    infer_low_saturation_cool_body_override,
    _should_prefer_localization_crop_reference,
    infer_material_note,
    infer_named_palette_with_strategy,
    infer_neutral_textile_surface_notes,
    infer_soft_textile_chromatic_override,
    soften_uncertain_neutral_apparel_color_evidence,
    infer_style_persona,
    infer_support_relations,
    identity_has_chromatic_soft_textile_lock,
    identity_has_low_profile_soft_structure,
    is_low_information_retrieval_caption,
    load_retrieval_index,
    load_localization_report,
    load_review_seed_manifest,
    prepare_observed_evidence_assets,
    reconcile_color_note_with_caption,
    refine_canonical_product_type,
    repair_removed_reference_regions,
    repair_rigid_body_notches,
    render_review_board,
    render_upstream_review_board,
    rewrite_evidence_for_canonical_type,
    refine_retrieval_visual_classification,
    sanitize_review_rows_for_bundle,
    sanitize_evidence_caption,
    sanitize_evidence_caption_against_color_evidence,
    sanitize_retrieval_candidates_for_planning,
    select_casting_alignment_eval_prompts,
    select_functional_subtype_eval_prompts,
    select_reinvention_candidate_modes,
    select_reinvention_candidate_modes_for_line,
    score_generation_candidate,
    smooth_reference_export_mask,
    score_retrieval_evidence_compatibility,
    synchronize_coverage_note_with_dominant_color,
    synchronize_pattern_note_with_dominant_color,
    infer_backpack_structure_note,
    compare_dominant_body_color_alignment,
    compare_dominant_body_value_alignment,
    compare_compact_product_prominence,
    compare_coverage_alignment,
    compare_soft_structure_alignment,
    correct_supported_soft_surface_inference,
    detect_compact_accessory_wardrobe_color_spill,
    detect_background_collapse_artifact,
    detect_human_ghost_composite_artifact,
    extract_generated_focus_artifacts,
    extract_dominant_body_color,
    generated_focus_mask_is_safe_for_color_repair,
    harmonize_supported_soft_structure,
    assess_category_consistency,
    assess_semantic_plausibility,
    assess_source_validity,
    identity_requires_functional_context,
    identity_requires_people_out_of_frame,
    semantic_support_margin_threshold,
    soft_surface_color_ok,
    soft_surface_coverage_ok,
    soft_surface_value_ok,
    suppress_border_attached_reference_artifacts,
    should_apply_caption_color_override,
    should_apply_post_generation_color_repair,
    should_prefer_crop_only_color_lock,
    should_strengthen_dominant_body_color_guidance,
    should_use_reference_only_conditioning,
    should_fallback_to_category_prior,
    infer_soft_structure_profile,
)
from product_campaign_pipeline.flux import Flux2KleinClient
from product_campaign_pipeline.localization.models import (
    BoundingBox as LocalizationBoundingBox,
    LocalizationResult,
    MaskCandidate,
    PhraseCandidate,
    ProductPhoto,
)
from product_campaign_pipeline.types import FluxPromptSpec, ObservedEvidenceSpec, ProductIdentitySpec
from product_campaign_pipeline.types import BoundingBox, CampaignPriorSpec, LocalizedProduct


def _seed_record() -> ReviewSeedRecord:
    return ReviewSeedRecord(
        id="disney_tote_03",
        platform="walmart",
        source_page_url="https://example.com/product",
        source_image_url="https://example.com/image.jpg",
        product_title="Disney Mickey and Minnie Mouse Tote Bag",
        hint_phrases=("tote bag", "beach bag", "disney bag"),
        capture_date="2026-04-09",
        local_image_path=Path("/tmp/disney_tote_03.jpg"),
    )


def _localization_record(selected_phrase: str = "disney mickey") -> LocalizationArtifactRecord:
    return LocalizationArtifactRecord(
        id="disney_tote_03",
        product_title="Disney Mickey and Minnie Mouse Tote Bag",
        source_page_url="https://example.com/product",
        source_image_url="https://example.com/image.jpg",
        local_image_path=Path("/tmp/disney_tote_03.jpg"),
        selected_phrase=selected_phrase,
        selected_confidence=0.91,
        selected_box=None,
        overlay_path=None,
        crop_path=Path("/tmp/disney_tote_03.crop.png"),
        mask_path=Path("/tmp/disney_tote_03.mask.png"),
    )


def test_build_localized_product_uses_canonical_type_for_weak_phrase() -> None:
    localized = build_localized_product(_seed_record(), _localization_record())
    assert localized.identity.category == "bag"
    assert localized.identity.canonical_product_type == "tote bag"
    assert localized.identity.source_title == "Disney Mickey and Minnie Mouse Tote Bag"
    assert localized.identity.weak_shape_evidence is True
    assert localized.identity.support_mode == "portable_flexible"
    assert localized.identity.default_scene_family == "fashion_lifestyle"
    assert "tote bag" in localized.identity.phrase


def test_wallet_token_does_not_trigger_wall_mount_affordance() -> None:
    localized = build_localized_product(
        ReviewSeedRecord(
            id="dasein_handbag_01",
            platform="walmart",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            product_title="Dasein Medium Size Tote Handbag Set",
            hint_phrases=("handbag", "tote bag", "wallet"),
            capture_date="2026-04-09",
            local_image_path=Path("/tmp/dasein_handbag_01.jpg"),
        ),
        LocalizationArtifactRecord(
            id="dasein_handbag_01",
            product_title="Dasein Medium Size Tote Handbag Set",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            local_image_path=Path("/tmp/dasein_handbag_01.jpg"),
            selected_phrase="dasein medium size tote handbag set",
            selected_confidence=0.9,
            selected_box=None,
            overlay_path=None,
            crop_path=Path("/tmp/dasein_handbag_01.crop.png"),
            mask_path=Path("/tmp/dasein_handbag_01.mask.png"),
        ),
    )

    assert localized.identity.support_mode == "portable_flexible"
    assert localized.identity.default_scene_family == "fashion_lifestyle"


def test_countertop_token_does_not_trigger_apparel_category() -> None:
    assert infer_category("countertop blender kitchen appliance") == "kitchen appliance"


def test_build_localized_product_keeps_blender_as_kitchen_appliance() -> None:
    localized = build_localized_product(
        ReviewSeedRecord(
            id="kitchen_blender_01",
            platform="walmart",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            product_title="Mainstays 500 W 6-Speed Blender With 48 oz/1.5 L Jar, Black",
            hint_phrases=("blender", "kitchen appliance", "countertop blender"),
            capture_date="2026-04-11",
            local_image_path=Path("/tmp/kitchen_blender_01.jpg"),
        ),
        LocalizationArtifactRecord(
            id="kitchen_blender_01",
            product_title="Mainstays 500 W 6-Speed Blender With 48 oz/1.5 L Jar, Black",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            local_image_path=Path("/tmp/kitchen_blender_01.jpg"),
            selected_phrase="blender countertop blender",
            selected_confidence=0.9,
            selected_box=None,
            overlay_path=None,
            crop_path=Path("/tmp/kitchen_blender_01.crop.png"),
            mask_path=Path("/tmp/kitchen_blender_01.mask.png"),
        ),
    )

    assert localized.identity.category == "kitchen appliance"
    assert localized.identity.canonical_product_type == "blender"
    assert localized.identity.support_mode == "self_supporting_display"
    assert localized.identity.default_scene_family == "tabletop_display"


def test_backpack_harness_face_prefers_carried_support_relation() -> None:
    identity = ProductIdentitySpec(
        category="bag",
        canonical_product_type="backpack",
        support_mode="wearable",
        observed_evidence=ObservedEvidenceSpec(
            upper_component_count=2,
            upper_region_note="the visible upper component splits into multiple narrow segments above the main body",
            hard_facts=[
                "the visible backpack body includes darker harness, panel, or attachment zones against a gray main body",
                "the visible source shows the backpack back-panel and harness side",
            ],
        ),
    )

    assert backpack_harness_face_observed(identity) is True
    assert default_support_relation_for_identity(identity) == "carried_by_hand"


def test_filter_retrieval_candidates_rejects_off_category_drinkware_neighbors() -> None:
    seed = ReviewSeedRecord(
        id="camelbak_bottle_01",
        platform="walmart",
        source_page_url="https://example.com/product",
        source_image_url="https://example.com/image.jpg",
        product_title="CamelBak 25 oz Non-Insulated Peak Fitness Chill Bottle",
        hint_phrases=("water bottle", "bottle", "camelbak"),
        capture_date="2026-04-09",
        local_image_path=Path("/tmp/camelbak.jpg"),
    )
    record = LocalizationArtifactRecord(
        id="camelbak_bottle_01",
        product_title=seed.product_title,
        source_page_url=seed.source_page_url,
        source_image_url=seed.source_image_url,
        local_image_path=seed.local_image_path,
        selected_phrase="camelbak bottle",
        selected_confidence=0.92,
        selected_box=None,
        overlay_path=None,
        crop_path=Path("/tmp/camelbak.crop.png"),
        mask_path=Path("/tmp/camelbak.mask.png"),
    )
    candidates = [
        RetrievalCandidate(
            item_id="a",
            image_name="a.png",
            image_path=Path("/tmp/a.png"),
            score=1.0,
            page_views=100,
            clicks=10,
            caption="a jar of chili sauce next to a bowl of chili sauce",
            embedding=(1.0, 0.0),
            scenario_slots=("tabletop_display",),
            style_atoms=("clear hero framing",),
            scene_families=("tabletop_display",),
            support_relations=("standing_on_surface",),
        ),
        RetrievalCandidate(
            item_id="b",
            image_name="b.png",
            image_path=Path("/tmp/b.png"),
            score=1.0,
            page_views=100,
            clicks=10,
            caption="a reusable water bottle on a countertop",
            embedding=(0.0, 1.0),
            scenario_slots=("tabletop_display",),
            style_atoms=("anchored support-surface composition",),
            scene_families=("tabletop_display",),
            support_relations=("standing_on_surface",),
        ),
    ]

    filtered = filter_retrieval_candidates(
        seed,
        record,
        candidates,
        category="drinkware",
        canonical_product_type="water bottle",
        support_mode="self_supporting_display",
    )

    assert [candidate.item_id for candidate in filtered] == ["b"]


def test_load_retrieval_index_rewrites_migrated_image_paths(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    images_root = data_root / "images"
    images_root.mkdir(parents=True)
    image_path = images_root / "example.png"
    image_path.write_bytes(b"fake")

    index_path = tmp_path / "retrieval_index.json"
    index_path.write_text(
        """
        [
          {
            "item_id": "item-1",
            "image_name": "example.png",
            "image_path": "/home/nyle_j_huang/data/images/example.png",
            "score": 1.0,
            "page_views": 10,
            "clicks": 2,
            "caption": "a reusable water bottle on a countertop",
            "embedding": [0.1, 0.2]
          }
        ]
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("PCP_DATA_ROOT", str(data_root))

    candidates = load_retrieval_index(index_path)

    assert len(candidates) == 1
    assert candidates[0].image_path == image_path


def test_manifest_and_localization_loaders_rewrite_migrated_paths(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "product_campaign_pipeline"
    data_root = workspace_root / "data"
    project_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    local_image = data_root / "source.jpg"
    overlay = project_root / "outputs" / "overlay.png"
    crop = project_root / "outputs" / "crop.png"
    mask = project_root / "outputs" / "mask.png"
    overlay.parent.mkdir(parents=True)
    for path in (local_image, overlay, crop, mask):
        path.write_bytes(b"x")

    legacy_project_root = Path("/home/nyle_j_huang/product_campaign_pipeline")
    legacy_data_root = Path("/home/nyle_j_huang/data")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "id": "seed-1",
                    "platform": "walmart",
                    "source_page_url": "https://example.com/product",
                    "source_image_url": "https://example.com/image.jpg",
                    "product_title": "Test Product",
                    "hint_phrases": ["wallet"],
                    "capture_date": "2026-04-16",
                    "local_image_path": str(legacy_data_root / "source.jpg"),
                }
            ]
        ),
        encoding="utf-8",
    )
    localization_path = tmp_path / "localization.json"
    localization_path.write_text(
        json.dumps(
            [
                {
                    "id": "seed-1",
                    "product_title": "Test Product",
                    "source_page_url": "https://example.com/product",
                    "source_image_url": "https://example.com/image.jpg",
                    "local_image_path": str(legacy_data_root / "source.jpg"),
                    "selected_phrase": "wallet",
                    "selected_confidence": 0.8,
                    "selected_box": None,
                    "overlay_path": str(legacy_project_root / "outputs" / "overlay.png"),
                    "crop_path": str(legacy_project_root / "outputs" / "crop.png"),
                    "mask_path": str(legacy_project_root / "outputs" / "mask.png"),
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("PCP_WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("PCP_PROJECT_ROOT", str(project_root))

    seeds = load_review_seed_manifest(manifest_path)
    records = load_localization_report(localization_path)

    assert seeds[0].local_image_path == local_image
    assert records["seed-1"].local_image_path == local_image
    assert records["seed-1"].overlay_path == overlay
    assert records["seed-1"].crop_path == crop
    assert records["seed-1"].mask_path == mask


def test_extract_generated_focus_artifacts_passes_category_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "generated.png"
    Image.new("RGB", (32, 32), color="white").save(image_path)
    seen: dict[str, object] = {}

    def fake_product_photo_factory(**kwargs):
        seen["photo"] = SimpleNamespace(**kwargs)
        return seen["photo"]

    class FakeLocalizer:
        def localize(self, photo):
            seen["localized_photo"] = photo
            return "result"

    def fake_select_mask(result):
        assert result == "result"
        return "selected"

    def fake_save_artifacts(result, cache_dir, selected_mask):
        assert result == "result"
        assert selected_mask == "selected"
        return SimpleNamespace(
            crop_path=str(tmp_path / "crop.png"),
            mask_path=str(tmp_path / "mask.png"),
            overlay_path=str(tmp_path / "overlay.png"),
        )

    localized = build_localized_product(
        ReviewSeedRecord(
            id="wallet_review",
            platform="walmart",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            product_title="Blue Floral Wallet",
            hint_phrases=("wallet", "bag"),
            capture_date="2026-04-09",
            local_image_path=image_path,
        ),
        LocalizationArtifactRecord(
            id="wallet_review",
            product_title="Blue Floral Wallet",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            local_image_path=image_path,
            selected_phrase="blue floral wallet",
            selected_confidence=0.8,
            selected_box=None,
            overlay_path=None,
            crop_path=None,
            mask_path=None,
        ),
    )
    artifacts = extract_generated_focus_artifacts(
        image_path,
        localized,
        generated_localizer=FakeLocalizer(),
        product_photo_factory=fake_product_photo_factory,
        save_artifacts=fake_save_artifacts,
        select_mask=fake_select_mask,
    )

    photo = seen["localized_photo"]
    assert artifacts is not None
    assert photo.metadata["category"] == "bag"
    assert photo.metadata["canonical_product_type"] == "wallet"
    assert photo.title == "wallet"
    assert photo.hint_phrases == ("wallet", "bag")
    assert "bag" in photo.hint_phrases


def test_candidate_has_hard_evidence_conflict_rejects_structured_candidate_with_conflicting_caption() -> None:
    candidate = RetrievalCandidate(
        item_id="weld",
        image_name="weld.png",
        image_path=Path("/tmp/weld.png"),
        score=0.5,
        page_views=10,
        clicks=4,
        caption="red power welding machine",
        embedding=(0.1, 0.2),
        scenario_slots=("editorial_interior",),
        style_atoms=("clear hero framing",),
        scene_families=("editorial_interior",),
        support_relations=("standing_on_surface",),
        category="kitchen appliance",
        canonical_product_type="toaster",
        observed_evidence=ObservedEvidenceSpec(
            coverage_class="low_variation_surface",
            hard_facts=["the product remains a toaster"],
        ),
    )
    source_evidence = ObservedEvidenceSpec(
        color_note="the main visible body reads as blue with cool-toned reflective variation",
        color_confidence=0.72,
        coverage_class="low_variation_surface",
        hard_facts=["the product remains a toaster"],
    )

    assert candidate_has_hard_evidence_conflict(
        candidate,
        source_evidence=source_evidence,
        category="kitchen appliance",
        source_canonical_product_type="toaster",
    )


def test_is_low_information_retrieval_caption_detects_spam_repetition() -> None:
    assert is_low_information_retrieval_caption(
        "maison maison maison maison maison maison maison maison maison maison maison"
    )
    assert not is_low_information_retrieval_caption(
        "a backpack with a zipper closure and a zipper closure"
    )


def test_sanitize_retrieval_candidates_for_planning_strips_spam_caption_signal() -> None:
    candidate = RetrievalCandidate(
        item_id="spam",
        image_name="spam.png",
        image_path=Path("/tmp/spam.png"),
        score=1.0,
        page_views=100,
        clicks=10,
        caption="maison maison maison maison maison maison maison maison maison",
        embedding=(1.0, 0.0),
        scenario_slots=("fashion_lifestyle",),
        style_atoms=("human-in-use framing",),
        scene_families=("fashion_lifestyle",),
        support_relations=("carried_by_hand",),
    )

    sanitized = sanitize_retrieval_candidates_for_planning([candidate])[0]

    assert sanitized.caption == ""
    assert sanitized.style_atoms == ()
    assert sanitized.scenario_slots == ()


def test_candidate_has_hard_evidence_conflict_rejects_type_less_structured_furniture_neighbor() -> None:
    evidence = ObservedEvidenceSpec(
        color_note="the main visible body reads as black",
        palette=["black", "gray"],
        lower_region_note="the visible lower support frame continues below the seat and backrest and should remain present",
        lower_component_state="present",
    )
    candidate = RetrievalCandidate(
        item_id="glasses",
        image_name="glasses.png",
        image_path=Path("/tmp/glasses.png"),
        score=1.0,
        page_views=100,
        clicks=10,
        caption="a pair of glasses on a wooden table",
        embedding=(1.0, 0.0),
        scenario_slots=("editorial_interior",),
        style_atoms=("clear hero framing",),
        scene_families=("editorial_interior",),
        support_relations=("standing_on_surface",),
    )

    assert candidate_has_hard_evidence_conflict(
        candidate,
        source_evidence=evidence,
        category="furniture",
        source_canonical_product_type="office chair",
    )


def test_candidate_has_hard_evidence_conflict_rejects_conflicting_color_for_compact_direct_grip() -> None:
    evidence = ObservedEvidenceSpec(
        color_note="the main visible body reads as blue with compatible printed accents in gray, beige",
        color_confidence=0.72,
        palette=["blue", "gray", "beige"],
        coverage_class="full_visible_surface_pattern",
        upper_component_state="absent",
        form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
    )
    candidate = RetrievalCandidate(
        item_id="white_wallet",
        image_name="white_wallet.png",
        image_path=Path("/tmp/white_wallet.png"),
        score=1.0,
        page_views=100,
        clicks=10,
        caption="the mini wallet in white",
        embedding=(1.0, 0.0),
        scenario_slots=("fashion_lifestyle",),
        style_atoms=("human-in-use framing",),
        scene_families=("fashion_lifestyle",),
        support_relations=("carried_by_hand",),
    )

    assert candidate_has_hard_evidence_conflict(
        candidate,
        source_evidence=evidence,
        category="bag",
        source_canonical_product_type="wallet",
    )


def test_infer_support_relations_leaves_ambiguous_caption_unassigned() -> None:
    assert infer_support_relations("the mini wallet in white") == ()


def test_extract_caption_colors_preserves_phrase_order() -> None:
    assert extract_caption_colors("dark green comforter with black trim and beige accents")[:3] == [
        "green",
        "black",
        "beige",
    ]


def test_extract_dominant_body_color_prefers_color_note_for_broad_patterned_surface() -> None:
    evidence = ObservedEvidenceSpec(
        body_region_color="gray",
        color_note="the main visible body reads as blue with compatible printed accents in gray, beige",
        color_confidence=0.72,
        coverage_class="full_visible_surface_pattern",
        palette=["blue", "gray", "beige"],
    )

    assert extract_dominant_body_color(evidence) == "blue"


def test_extract_dominant_body_color_prefers_confident_color_note_over_neutral_body_region() -> None:
    evidence = ObservedEvidenceSpec(
        body_region_color="gray",
        color_note="the main visible body reads as blue with cool-toned reflective variation",
        color_confidence=0.72,
        coverage_class="localized_visible_pattern",
        palette=["blue", "gray", "black"],
    )

    assert extract_dominant_body_color(evidence) == "blue"


def test_sanitize_evidence_caption_against_color_evidence_strips_conflicting_colors() -> None:
    assert (
        sanitize_evidence_caption_against_color_evidence(
            "a black comforter",
            canonical_product_type="comforter",
            palette=["green", "black", "gray"],
            color_note="the main visible body reads as dark green with low-luster tonal variation from textured fabric",
            color_confidence=0.78,
        )
        == "a comforter"
    )


def test_sanitize_evidence_caption_against_color_evidence_repairs_sparse_caption_after_color_strip() -> None:
    assert (
        sanitize_evidence_caption_against_color_evidence(
            "a blue and mug",
            canonical_product_type="mug",
            palette=["gray", "blue", "black"],
            color_note="the main visible body reads as gray",
            color_confidence=0.78,
        )
        == "a mug"
    )


def test_evaluate_category_scores_flags_product_type_drift() -> None:
    result = evaluate_category_scores(
        expected_category="bag",
        expected_product_type="tote bag",
        category_scores={
            "bag": 0.12,
            "apparel": 0.27,
            "home decor": 0.09,
            "drinkware": 0.01,
        },
    )

    assert result["is_consistent"] is False
    assert result["predicted_category"] == "apparel"
    assert "category drift flagged" in result["warning"]


def test_crop_only_color_lock_is_disabled_for_soft_structural_products() -> None:
    identity = ProductIdentitySpec(
        category="pet home",
        canonical_product_type="pet bed",
        rigid_vs_soft="soft",
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as blue",
            color_confidence=0.78,
            coverage_class="low_variation_surface",
            reference_crop_path="/tmp/pet_bed.crop.png",
            edge_profile_note="the visible pet bed perimeter remains low and softly graded around the resting surface rather than rising into bulky bolsters",
            soft_structure_note="the visible soft product structure reads as a flat plush pad with no bulky bolster, boxed sidewall, or nested inner tray",
            soft_structure_class="flat_surface",
        ),
    )

    assert should_prefer_crop_only_color_lock(identity) is False


def test_support_planning_avoids_tabletop_for_externally_supported_soft_product() -> None:
    identity = ProductIdentitySpec(
        phrase="rizzy home decorative pillow",
        category="home decor",
        canonical_product_type="decorative pillow",
        support_mode="externally_supported_soft",
        default_scene_family="furnished_interior",
        rigid_vs_soft="soft",
        stable_base=False,
    )
    top_matches = [
        RetrievalCandidate(
            item_id="room-1",
            image_name="room-1.png",
            image_path=Path("/tmp/room-1.png"),
            score=1.0,
            page_views=100,
            clicks=10,
            caption="a room with a bed and a chair",
            embedding=(1.0, 0.0),
            scenario_slots=("furnished_interior",),
            style_atoms=("soft furnished-environment context",),
            scene_families=("furnished_interior",),
            support_relations=("resting_with_back_support",),
        ),
        RetrievalCandidate(
            item_id="table-1",
            image_name="table-1.png",
            image_path=Path("/tmp/table-1.png"),
            score=1.0,
            page_views=100,
            clicks=10,
            caption="a product on a table",
            embedding=(0.0, 1.0),
            scenario_slots=("tabletop_display",),
            style_atoms=("anchored support-surface composition",),
            scene_families=("tabletop_display",),
            support_relations=("standing_on_surface",),
        ),
    ]

    support_relation = choose_support_relation(identity, top_matches)
    scene_family = choose_scene_family(identity, top_matches, support_relation=support_relation)
    style_atoms = build_style_plan(
        identity,
        top_matches,
        scene_family=scene_family,
        support_relation=support_relation,
    )

    assert support_relation == "resting_with_back_support"
    assert scene_family == "furnished_interior"
    assert "anchored support-surface composition" not in style_atoms


def test_choose_support_relation_does_not_make_chair_hand_carried() -> None:
    identity = ProductIdentitySpec(
        phrase="office chair",
        category="furniture",
        canonical_product_type="office chair",
        support_mode="self_supporting_display",
        default_scene_family="editorial_interior",
        rigid_vs_soft="rigid",
        stable_base=True,
    )
    top_matches = [
        RetrievalCandidate(
            item_id="bad-1",
            image_name="bad-1.png",
            image_path=Path("/tmp/bad-1.png"),
            score=1.0,
            page_views=100,
            clicks=10,
            caption="a person carrying a product",
            embedding=(1.0, 0.0),
            scenario_slots=("fashion_lifestyle",),
            style_atoms=("human-in-use framing",),
            scene_families=("fashion_lifestyle",),
            support_relations=("carried_by_hand",),
        ),
        RetrievalCandidate(
            item_id="good-1",
            image_name="good-1.png",
            image_path=Path("/tmp/good-1.png"),
            score=0.8,
            page_views=100,
            clicks=10,
            caption="a chair standing on the floor",
            embedding=(0.0, 1.0),
            scenario_slots=("editorial_interior",),
            style_atoms=("anchored support-surface composition",),
            scene_families=("editorial_interior",),
            support_relations=("standing_on_surface",),
        ),
    ]

    support_relation = choose_support_relation(identity, top_matches)

    assert support_relation == "standing_on_surface"


def test_default_support_relation_uses_surface_support_for_comforter() -> None:
    identity = ProductIdentitySpec(
        phrase="comforter",
        category="bedding",
        canonical_product_type="comforter",
        support_mode="externally_supported_soft",
        default_scene_family="furnished_interior",
        rigid_vs_soft="soft",
        stable_base=False,
    )

    assert default_support_relation_for_identity(identity) == "resting_on_surface"


def test_scene_planning_biases_sport_utility_drinkware_toward_tabletop() -> None:
    identity = ProductIdentitySpec(
        phrase="camelbak bottle",
        category="drinkware",
        canonical_product_type="water bottle",
        support_mode="self_supporting_display",
        default_scene_family="tabletop_display",
        style_persona="sport_utility",
        rigid_vs_soft="rigid",
        stable_base=True,
    )
    top_matches = [
        RetrievalCandidate(
            item_id="editorial-1",
            image_name="editorial-1.png",
            image_path=Path("/tmp/editorial-1.png"),
            score=1.0,
            page_views=100,
            clicks=10,
            caption="a bottle in an interior",
            embedding=(1.0, 0.0),
            scenario_slots=("editorial_interior",),
            style_atoms=("clear hero framing",),
            scene_families=("editorial_interior",),
            support_relations=("standing_on_surface",),
            category="drinkware",
            canonical_product_type="water bottle",
        ),
    ]

    assert choose_scene_family(identity, top_matches, support_relation="standing_on_surface") == "tabletop_display"


def test_build_business_prior_creative_metadata_emits_typed_hints() -> None:
    identity = ProductIdentitySpec(
        phrase="disney tote bag",
        category="bag",
        canonical_product_type="tote bag",
        support_mode="portable_flexible",
        default_scene_family="fashion_lifestyle",
        interaction_mode="carried_or_resting",
        style_persona="playful_casual",
        requires_human_model=True,
    )
    top_matches = [
        RetrievalCandidate(
            item_id="match-1",
            image_name="match-1.png",
            image_path=Path("/tmp/match-1.png"),
            score=1.0,
            page_views=200,
            clicks=20,
            caption="a colorful tote bag carried in a street scene",
            embedding=(1.0, 0.0),
            scenario_slots=("fashion_lifestyle",),
            style_atoms=("human-in-use framing",),
            scene_families=("fashion_lifestyle",),
            support_relations=("carried_by_hand",),
        ),
    ]

    metadata = build_business_prior_creative_metadata(
        identity,
        top_matches,
        scene_family="fashion_lifestyle",
        support_relation="carried_by_hand",
    )

    assert metadata["creative_seed"] > 0
    assert "lighting_hint" in metadata
    assert "camera_hint" in metadata
    assert "creative_direction" in metadata
    assert "cast_hint" in metadata


def test_candidate_hard_evidence_conflict_rejects_structured_backpack_mismatch() -> None:
    candidate = RetrievalCandidate(
        item_id="wallet-1",
        image_name="wallet-1.png",
        image_path=Path("/tmp/wallet-1.png"),
        score=1.0,
        page_views=100,
        clicks=10,
        caption="the mini wallet in white",
        embedding=(1.0, 0.0),
        scenario_slots=("fashion_lifestyle",),
        style_atoms=("human-in-use framing",),
        scene_families=("fashion_lifestyle",),
        support_relations=("carried_by_hand",),
        category="bag",
        canonical_product_type="wallet",
    )

    assert candidate_has_hard_evidence_conflict(
        candidate,
        source_evidence=ObservedEvidenceSpec(),
        category="bag",
        source_canonical_product_type="backpack",
    ) is True


def test_infer_style_persona_uses_playful_text_cues_without_backbone() -> None:
    persona = infer_style_persona(
        category="bag",
        canonical_product_type="tote bag",
        product_title="Disney Mickey and Minnie Mouse Tote Bag",
        hint_phrases=("disney bag", "tote bag"),
        observed_evidence=ObservedEvidenceSpec(
            coverage_class="broad_visible_surface_pattern",
            evidence_caption="a purple and yellow bag with a pattern of small cartoon characters",
        ),
        source_image=Path("/tmp/source.png"),
        crop_path=None,
        backbone=None,
    )

    assert persona == "playful_casual"


def test_infer_category_supports_backpack_footwear_and_lighting() -> None:
    assert infer_category("Rawlings baseball backpack") == "bag"
    assert infer_category("Easy Spirit walking shoe sneaker") == "footwear"
    assert infer_category("Pineapple table lamp") == "home lighting"


def test_infer_category_supports_mug_quilt_and_appliance_subtypes() -> None:
    assert infer_category("The Pioneer Woman stoneware mug") == "drinkware"
    assert infer_category("Mainstays 12 cup drip coffee maker") == "kitchen appliance"
    assert infer_category("The Pioneer Woman slow cooker") == "kitchen appliance"
    assert infer_category("Better Homes and Gardens floral quilt bedding") == "bedding"


def test_infer_canonical_product_type_prefers_dress_over_t_shirt_phrase_overlap() -> None:
    canonical = infer_canonical_product_type(
        "ANYJOIN Women's Summer Casual T Shirt Dresses Short Sleeve Swing Dress Pockets L",
        ("dress", "swing dress", "women's dress"),
        "summer casual t shirt dresses",
    )
    assert canonical == "dress"


def test_infer_canonical_product_type_supports_shoe_backpack_and_lamp() -> None:
    assert infer_canonical_product_type("Easy Spirit Womens Pippa Lace-Up Sneaker", ("shoe",), "easy spirit womens") == "shoe"
    assert infer_canonical_product_type("Rawlings Impulse Player's Baseball Backpack", ("backpack",), "sports backpack baseball backpack") == "backpack"
    assert infer_canonical_product_type("PINEAPPLE TABLE LAMP", ("lamp",), "pineapple table lamp") == "table lamp"


def test_infer_canonical_product_type_supports_mug_quilt_and_appliance_subtypes() -> None:
    assert infer_canonical_product_type("The Pioneer Woman Colette Cream Mug", ("mug", "stoneware mug"), "coffee mug") == "mug"
    assert infer_canonical_product_type("Mainstays Black 12-Cup Drip Coffee Maker", ("coffee maker",), "drip coffee maker") == "coffee maker"
    assert infer_canonical_product_type("The Pioneer Woman 6 Qt Digital Slow Cooker", ("slow cooker",), "slow cooker") == "slow cooker"
    assert infer_canonical_product_type("Better Homes & Gardens Floral Matelasse Quilt", ("quilt", "bedding"), "blue floral quilt") == "quilt"


def test_infer_style_persona_does_not_default_apparel_to_cozy_home_from_texture_only() -> None:
    persona = infer_style_persona(
        category="apparel",
        canonical_product_type="shirt",
        product_title="Women's Long Sleeve Cotton Shirt",
        hint_phrases=("shirt", "long sleeve shirt"),
        observed_evidence=ObservedEvidenceSpec(
            coverage_class="broad_visible_surface_pattern",
            material_note="visible material cues suggest a fabric or textile surface",
            evidence_caption="a gray long sleeve shirt",
        ),
        source_image=Path("/tmp/source.png"),
        crop_path=None,
        backbone=None,
    )

    assert persona != "cozy_home"


def test_infer_style_persona_prefers_sport_utility_for_backpack() -> None:
    persona = infer_style_persona(
        category="bag",
        canonical_product_type="backpack",
        product_title="Rawlings Impulse Player's Baseball Backpack",
        hint_phrases=("backpack", "sports backpack", "baseball backpack"),
        observed_evidence=ObservedEvidenceSpec(
            coverage_class="full_visible_surface_pattern",
            evidence_caption="the back of a black and white backpack with a white logo on it",
        ),
        source_image=Path("/tmp/source.png"),
        crop_path=None,
        backbone=None,
    )

    assert persona == "sport_utility"


def test_infer_style_persona_prefers_refined_or_cozy_for_mug() -> None:
    persona = infer_style_persona(
        category="drinkware",
        canonical_product_type="mug",
        product_title="The Pioneer Woman Stoneware Coffee Mug",
        hint_phrases=("mug", "coffee mug", "stoneware mug"),
        observed_evidence=ObservedEvidenceSpec(
            coverage_class="low_variation_surface",
            material_note="visible material cues suggest a ceramic or stoneware surface",
            evidence_caption="a blue stoneware mug",
        ),
        source_image=Path("/tmp/source.png"),
        crop_path=None,
        backbone=None,
    )

    assert persona in {"refined_neutral", "cozy_home"}


def test_infer_affordance_profile_treats_backpack_as_wearable() -> None:
    affordance = infer_affordance_profile(
        "bag",
        canonical_product_type="backpack",
        product_title="Rawlings Impulse Player's Baseball Backpack",
        hint_phrases=("backpack", "sports backpack", "baseball backpack"),
    )

    assert affordance["support_mode"] == "wearable"
    assert affordance["interaction_mode"] == "worn_or_carried"
    assert affordance["default_scene_family"] == "outdoor_lifestyle"


def test_rewrite_evidence_for_canonical_type_adds_structural_fact() -> None:
    evidence = rewrite_evidence_for_canonical_type(
        ObservedEvidenceSpec(hard_facts=["the product remains a product"]),
        canonical_product_type="table lamp",
    )

    assert any("stable base and upper light shade" in fact for fact in evidence.hard_facts)


def test_prompt_scene_conflict_detects_mixed_tabletop_and_back_support_language() -> None:
    from product_campaign_pipeline.composer import PromptComposer
    from product_campaign_pipeline.types import BoundingBox, LocalizedProduct

    localized = LocalizedProduct(
        source_image="/tmp/source.png",
        phrase="decorative pillow",
        bbox=BoundingBox(x0=0, y0=0, x1=10, y1=10),
        confidence=1.0,
        identity=ProductIdentitySpec(
            phrase="decorative pillow",
            category="home decor",
            canonical_product_type="decorative pillow",
            support_mode="externally_supported_soft",
            default_scene_family="furnished_interior",
            rigid_vs_soft="soft",
            stable_base=False,
        ),
    )
    prompt = PromptComposer().compose_baseline(localized, seed=1)
    conflicts = evaluate_prompt_scene_conflicts(
        prompt,
        scene_family="furnished_interior",
        support_relation="resting_with_back_support",
    )

    assert conflicts == []


def test_render_review_board_stages_external_assets_with_relative_paths(tmp_path: Path) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    source = external_dir / "source.jpg"
    crop = external_dir / "crop.png"
    baseline = review_dir / "images" / "sample.baseline.png"
    business = review_dir / "images" / "sample.business.png"
    baseline.parent.mkdir(parents=True, exist_ok=True)

    for path in (source, crop, baseline, business):
        Image.new("RGB", (8, 8), (120, 90, 60)).save(path)

    report_rows = [
        {
            "id": "sample",
            "line": "baseline",
            "product_title": "Sample Product",
            "expected_category": "bag",
            "canonical_product_type": "tote bag",
            "scene_family": "fashion_lifestyle",
            "support_relation": "carried_by_hand",
            "weak_shape_evidence": False,
            "source_image_path": str(source),
            "crop_path": str(crop),
            "output_path": str(baseline),
            "category_consistency": {"is_consistent": True, "predicted_category": "bag"},
            "semantic_plausibility": {"is_plausible": True, "score": 0.7},
        },
        {
            "id": "sample",
            "line": "business_prior",
            "product_title": "Sample Product",
            "expected_category": "bag",
            "canonical_product_type": "tote bag",
            "scene_family": "fashion_lifestyle",
            "support_relation": "carried_by_hand",
            "weak_shape_evidence": False,
            "source_image_path": str(source),
            "crop_path": str(crop),
            "output_path": str(business),
            "category_consistency": {"is_consistent": True, "predicted_category": "bag"},
            "semantic_plausibility": {"is_plausible": True, "score": 0.7},
        },
    ]

    board_path = render_review_board(report_rows, review_dir / "human_review_board.html")
    html = board_path.read_text(encoding="utf-8")

    assert "board_assets/source/sample.source.jpg" in html
    assert "board_assets/crop/sample.crop.png" in html
    assert "images/sample.baseline.png" in html
    assert "images/sample.business.png" in html
    assert str(external_dir) not in html
    assert (review_dir / "board_assets" / "source" / "sample.source.jpg").exists()
    assert (review_dir / "board_assets" / "crop" / "sample.crop.png").exists()


def test_sanitize_review_rows_for_bundle_rewrites_report_asset_paths(tmp_path: Path) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    source = external_dir / "source.jpg"
    crop = external_dir / "crop.png"
    baseline = external_dir / "sample.baseline.png"
    focus_crop = external_dir / "sample.focus.crop.png"
    focus_mask = external_dir / "sample.focus.mask.png"
    ref_crop = external_dir / "sample.ref.crop.png"
    ref_cutout = external_dir / "sample.ref.cutout.png"
    ref_silhouette = external_dir / "sample.ref.silhouette.png"
    ref_mask = external_dir / "sample.ref.mask.png"
    prompt_ref = external_dir / "sample.prompt.ref.png"
    for path in (source, crop, baseline, focus_crop, focus_mask, ref_crop, ref_cutout, ref_silhouette, ref_mask, prompt_ref):
        Image.new("RGB", (8, 8), (120, 90, 60)).save(path)

    sanitized_rows = sanitize_review_rows_for_bundle(
        [
            {
                "id": "sample",
                "line": "baseline",
                "source_image_path": str(source),
                "crop_path": str(crop),
                "output_path": str(baseline),
                "observed_evidence": {
                    "reference_crop_path": str(ref_crop),
                    "reference_cutout_path": str(ref_cutout),
                    "reference_silhouette_path": str(ref_silhouette),
                    "reference_mask_path": str(ref_mask),
                },
                "prompt": {
                    "reference_images": [
                        {"role": "product crop", "path": str(prompt_ref)},
                    ]
                },
                "evidence_consistency": {
                    "reference_image_path": str(prompt_ref),
                    "focus_crop_path": str(focus_crop),
                    "focus_mask_path": str(focus_mask),
                },
            }
        ],
        bundle_dir,
    )
    row = sanitized_rows[0]

    assert row["source_image_path"] == "board_assets/source/sample.source.jpg"
    assert row["crop_path"] == "board_assets/crop/sample.crop.png"
    assert row["output_path"] == "board_assets/generated/sample.baseline.png"
    assert row["observed_evidence"]["reference_crop_path"] == "board_assets/reference/sample.evidence_crop.png"
    assert row["prompt"]["reference_images"][0]["path"] == "board_assets/reference/sample.baseline.reference_0.png"
    assert row["evidence_consistency"]["focus_crop_path"] == "board_assets/focus/sample.baseline.focus_crop.png"
    assert row["evidence_consistency"]["focus_mask_path"] == "board_assets/focus/sample.baseline.focus_mask.png"


def test_render_review_board_keeps_sanitized_relative_asset_paths(tmp_path: Path) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    source = external_dir / "source.jpg"
    crop = external_dir / "crop.png"
    baseline = external_dir / "sample.baseline.png"
    business = external_dir / "sample.business_prior.png"
    for path in (source, crop, baseline, business):
        Image.new("RGB", (8, 8), (120, 90, 60)).save(path)

    sanitized_rows = sanitize_review_rows_for_bundle(
        [
            {
                "id": "sample",
                "line": "baseline",
                "product_title": "Sample Product",
                "expected_category": "bag",
                "canonical_product_type": "tote bag",
                "scene_family": "fashion_lifestyle",
                "support_relation": "carried_by_hand",
                "weak_shape_evidence": False,
                "source_image_path": str(source),
                "crop_path": str(crop),
                "output_path": str(baseline),
                "category_consistency": {"is_consistent": True, "predicted_category": "bag"},
                "semantic_plausibility": {"is_plausible": True, "score": 0.7},
                "evidence_consistency": {"is_consistent": True, "score": 0.8},
            },
            {
                "id": "sample",
                "line": "business_prior",
                "product_title": "Sample Product",
                "expected_category": "bag",
                "canonical_product_type": "tote bag",
                "scene_family": "fashion_lifestyle",
                "support_relation": "carried_by_hand",
                "weak_shape_evidence": False,
                "source_image_path": str(source),
                "crop_path": str(crop),
                "output_path": str(business),
                "category_consistency": {"is_consistent": True, "predicted_category": "bag"},
                "semantic_plausibility": {"is_plausible": True, "score": 0.75},
                "evidence_consistency": {"is_consistent": True, "score": 0.85},
            },
        ],
        bundle_dir,
    )

    board_path = render_review_board(sanitized_rows, bundle_dir / "human_review_board.html")
    html = board_path.read_text(encoding="utf-8")

    assert 'src=""' not in html
    assert 'src="board_assets/source/sample.source.jpg"' in html
    assert 'src="board_assets/crop/sample.crop.png"' in html
    assert 'src="board_assets/generated/sample.baseline.png"' in html
    assert 'src="board_assets/generated/sample.business_prior.png"' in html


def test_render_upstream_review_board_stages_external_assets_with_relative_paths(tmp_path: Path) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    review_dir = tmp_path / "upstream_review"
    review_dir.mkdir()

    source = external_dir / "source.jpg"
    crop = external_dir / "crop.png"
    overlay = external_dir / "overlay.png"
    for path in (source, crop, overlay):
        Image.new("RGB", (8, 8), (120, 90, 60)).save(path)

    report_rows = [
        {
            "id": "sample",
            "line": "baseline",
            "product_title": "Sample Product",
            "selected_phrase": "sample tote bag",
            "selected_confidence": 0.91,
            "expected_category": "bag",
            "canonical_product_type": "tote bag",
            "support_mode": "portable_flexible",
            "interaction_mode": "carried_or_resting",
            "stable_base": False,
            "rigid_vs_soft": "semi-rigid",
            "weak_shape_evidence": False,
            "source_image_path": str(source),
            "crop_path": str(crop),
            "overlay_path": str(overlay),
            "scene_family": "fashion_lifestyle",
            "support_relation": "carried_by_hand",
            "observed_evidence": {"palette": ["blue", "white"]},
            "candidate_prompts": [{"mode": "balanced", "prompt": {"subject": "x"}}],
            "retrieval_metadata": {},
        },
        {
            "id": "sample",
            "line": "business_prior",
            "product_title": "Sample Product",
            "selected_phrase": "sample tote bag",
            "selected_confidence": 0.91,
            "expected_category": "bag",
            "canonical_product_type": "tote bag",
            "support_mode": "portable_flexible",
            "interaction_mode": "carried_or_resting",
            "stable_base": False,
            "rigid_vs_soft": "semi-rigid",
            "weak_shape_evidence": False,
            "source_image_path": str(source),
            "crop_path": str(crop),
            "overlay_path": str(overlay),
            "scene_family": "fashion_lifestyle",
            "support_relation": "carried_by_hand",
            "observed_evidence": {"palette": ["blue", "white"]},
            "candidate_prompts": [{"mode": "balanced", "prompt": {"subject": "x"}}],
            "retrieval_metadata": {"retrieval_mode": "evidence_fallback"},
            "style_atoms": ["clear hero framing"],
            "semantic_constraints": ["show the product carried by hand"],
        },
    ]

    board_path = render_upstream_review_board(report_rows, review_dir / "human_review_board.html")
    html = board_path.read_text(encoding="utf-8")

    assert "board_assets/source/sample.source.jpg" in html
    assert "board_assets/crop/sample.crop.png" in html
    assert "board_assets/overlay/sample.overlay.png" in html
    assert str(external_dir) not in html
    assert (review_dir / "board_assets" / "source" / "sample.source.jpg").exists()
    assert (review_dir / "board_assets" / "crop" / "sample.crop.png").exists()
    assert (review_dir / "board_assets" / "overlay" / "sample.overlay.png").exists()


def test_build_generation_request_uses_mask_conditioning_references(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    cutout = tmp_path / "source.evidence_cutout.png"
    silhouette = tmp_path / "source.evidence_silhouette.png"
    output = tmp_path / "images" / "sample.baseline.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (24, 24), (180, 140, 90)).save(source)
    Image.new("RGB", (24, 24), (240, 240, 235)).save(cutout)
    Image.new("RGB", (24, 24), (30, 30, 30)).save(silhouette)

    prompt = FluxPromptSpec(
        subject="The featured tote bag from image 1: sample tote bag",
        action="Recompose the product into a campaign image while preserving identity.",
        style="premium commercial photography",
        context="Change the background while preserving the observed silhouette.",
        preservation_constraints=["preserve the dominant observed silhouette"],
        seed=7,
    )

    localized = build_localized_product(_seed_record(), _localization_record())
    localized.identity.observed_evidence.reference_cutout_path = str(cutout)
    localized.identity.observed_evidence.reference_silhouette_path = str(silhouette)
    reference_images = _conditioning_reference_images(localized)

    request = build_generation_request(
        Flux2KleinClient(),
        prompt,
        source_image=source,
        reference_images=reference_images,
        primary_input_image=_primary_generation_input_image(localized),
        output_path=output,
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=1.0,
    )

    assert len(request.input_images) == 2
    assert request.input_images[0] == str(cutout.resolve())
    assert request.input_images[1] == str(silhouette.resolve())
    assert Path(request.input_images[1]).exists()


def test_sanitize_evidence_caption_removes_source_person_for_wearables() -> None:
    sanitized = sanitize_evidence_caption(
        "a woman in a blue shirt and glasses",
        canonical_product_type="shirt",
        category="apparel",
        requires_human_model=True,
    )

    assert sanitized == "a blue shirt"


def test_sanitize_evidence_caption_drops_competing_wearable_type_after_human_strip() -> None:
    sanitized = sanitize_evidence_caption(
        "woman's shirt with a floral pattern",
        canonical_product_type="dress",
        category="apparel",
        requires_human_model=True,
    )

    assert sanitized is None


def test_sanitize_evidence_caption_drops_ambient_scene_description() -> None:
    sanitized = sanitize_evidence_caption(
        "a bed with a black comforter and a white background",
        canonical_product_type="comforter",
        category="bedding",
        requires_human_model=False,
    )

    assert sanitized is None


def test_caption_color_override_is_conservative_for_ambient_soft_goods() -> None:
    assert (
        should_apply_caption_color_override(
            evidence_caption="a bed with a black comforter and a white background",
            canonical_product_type="comforter",
            category="bedding",
            color_confidence=0.22,
            coverage_class="broad_visible_surface_pattern",
        )
        is False
    )


def test_human_supported_identities_use_clarity_candidate_set() -> None:
    identity = ProductIdentitySpec(
        phrase="sample shirt",
        category="apparel",
        canonical_product_type="shirt",
        support_mode="wearable",
        interaction_mode="worn",
        requires_human_model=True,
        observed_evidence=ObservedEvidenceSpec(uncertainty_level="low"),
    )

    assert select_reinvention_candidate_modes(identity) == HUMAN_REINVENTION_CANDIDATE_MODES


def test_structured_display_surface_inference_suppresses_false_pattern_reading() -> None:
    coverage_class, coverage_note, pattern_note = correct_structured_display_surface_inference(
        category="home lighting",
        canonical_product_type="table lamp",
        stable_base=True,
        shape_profile={"aspect_ratio": 2.0, "top_width_ratio": 1.2},
        evidence_caption="a lamp with a brown shade on it",
        coverage_class="broad_visible_surface_pattern",
        coverage_note="the visible print or color treatment spans a broad portion of the observed product surface",
        pattern_note="the visible product body carries a multicolor or printed treatment",
    )

    assert coverage_class == "low_variation_surface"
    assert "structured color zoning" in coverage_note
    assert pattern_note is None


def test_structured_display_identities_use_multi_candidate_modes() -> None:
    identity = ProductIdentitySpec(
        phrase="sample lamp",
        category="home lighting",
        canonical_product_type="table lamp",
        stable_base=True,
        interaction_mode="placed",
        observed_evidence=ObservedEvidenceSpec(
            aspect_ratio=2.1,
            top_width_ratio=1.1,
            uncertainty_level="low",
        ),
    )

    assert select_reinvention_candidate_modes(identity) == REINVENTION_CANDIDATE_MODES


def test_generate_upstream_review_batch_uses_distinct_candidate_seeds_across_lines(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from product_campaign_pipeline import review_batch as rb

    seed = _seed_record()
    localization = _localization_record()
    localized = LocalizedProduct(
        source_image="/tmp/disney_tote_03.jpg",
        phrase="disney tote bag",
        bbox=BoundingBox(x0=10, y0=20, x1=110, y1=140),
        confidence=0.9,
        crop_path="/tmp/disney_tote_03.crop.png",
        mask_path="/tmp/disney_tote_03.mask.png",
        identity=ProductIdentitySpec(
            phrase="disney tote bag",
            category="bag",
            canonical_product_type="tote bag",
            source_title=seed.product_title,
            support_mode="portable_flexible",
            default_scene_family="fashion_lifestyle",
            interaction_mode="carried_or_resting",
            style_persona="playful_casual",
            stable_base=False,
            rigid_vs_soft="soft",
            requires_human_model=True,
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as white",
                hard_facts=["the product remains a tote bag"],
            ),
        ),
    )
    prior = CampaignPriorSpec(
        style_atoms=["clear hero framing"],
        scenario_slots=["fashion_lifestyle"],
        scene_family="fashion_lifestyle",
        support_relation="carried_by_hand",
        metadata={"retrieval_mode": "evidence_fallback"},
    )

    monkeypatch.setattr(rb, "load_review_seed_manifest", lambda path: [seed])
    monkeypatch.setattr(rb, "load_localization_report", lambda path: {seed.id: localization})
    monkeypatch.setattr(rb, "load_retrieval_index", lambda path: {})
    monkeypatch.setattr(rb, "VisionBackbone", lambda device="cpu": object())
    monkeypatch.setattr(rb, "build_localized_product", lambda *args, **kwargs: localized)
    monkeypatch.setattr(rb, "build_business_prior", lambda *args, **kwargs: prior)

    def _render_stub(rows, destination):
        destination = Path(destination)
        destination.write_text("stub", encoding="utf-8")
        return destination

    monkeypatch.setattr(rb, "render_upstream_review_board", _render_stub)

    rb.generate_upstream_review_batch(
        "manifest.json",
        "localization.json",
        "retrieval.json",
        output_dir=tmp_path,
        device="cpu",
    )

    report = (tmp_path / "reports" / "upstream_review_report.json").read_text(encoding="utf-8")
    import json

    rows = json.loads(report)
    seeds_by_line = {
        row["line"]: [candidate["seed"] for candidate in row["candidate_prompts"]]
        for row in rows
    }

    assert seeds_by_line["baseline"] != seeds_by_line["business_prior"]
    assert seeds_by_line["business_prior"][0] - seeds_by_line["baseline"][0] == 10007


def test_dominant_body_color_alignment_flags_gray_drift_from_black() -> None:
    evidence = ObservedEvidenceSpec(
        color_note="the main visible body reads as black with subtle tonal variation from textured fabric",
        color_confidence=0.78,
        coverage_class="full_visible_surface_pattern",
    )

    assert compare_dominant_body_color_alignment(
        evidence,
        {"dominant_body_color": "gray"},
    ) < 0.6


def test_dominant_body_value_alignment_flags_large_luminance_drift() -> None:
    assert compare_dominant_body_value_alignment(
        {"mean_luma": 36.0},
        {"mean_luma": 128.0},
    ) < 0.7


def test_strengthen_guidance_for_confident_soft_surface_body_color() -> None:
    identity = ProductIdentitySpec(
        phrase="black comforter",
        category="bedding",
        canonical_product_type="comforter",
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as black with subtle tonal variation from textured fabric",
            color_confidence=0.78,
            coverage_class="full_visible_surface_pattern",
            reference_crop_path="/tmp/comforter.crop.png",
        ),
    )

    assert should_strengthen_dominant_body_color_guidance(identity) is True
    assert should_prefer_crop_only_color_lock(identity) is False


def test_apply_dominant_body_color_correction_darkens_gray_product_to_black(tmp_path: Path) -> None:
    image_path = tmp_path / "comforter.png"
    mask_path = tmp_path / "comforter.mask.png"

    source = Image.new("RGB", (24, 24), (240, 240, 240))
    draw = ImageDraw.Draw(source)
    draw.rectangle((4, 4, 20, 20), fill=(150, 150, 150))
    source.save(image_path)

    mask = Image.new("L", (24, 24), 0)
    ImageDraw.Draw(mask).rectangle((4, 4, 20, 20), fill=255)
    mask.save(mask_path)

    apply_dominant_body_color_correction(
        image_path,
        mask_path=mask_path,
        dominant_color="black",
        coverage_class="full_visible_surface_pattern",
        category="bedding",
        canonical_product_type="comforter",
        rigid_vs_soft="soft",
    )

    corrected = Image.open(image_path).convert("RGB")
    corrected_pixels = corrected.crop((4, 4, 21, 21))
    mean_value = sum(sum(pixel) for pixel in corrected_pixels.getdata()) / (len(corrected_pixels.getdata()) * 3.0)

    assert mean_value < 110.0
    assert mean_value < 150.0


def test_apply_dominant_body_color_correction_tints_chromatic_rigid_product_without_touching_background(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "toaster.png"
    mask_path = tmp_path / "toaster.mask.png"

    source = Image.new("RGB", (32, 32), (236, 232, 226))
    draw = ImageDraw.Draw(source)
    draw.rounded_rectangle((6, 6, 25, 25), radius=4, fill=(244, 242, 238))
    source.save(image_path)

    mask = Image.new("L", (32, 32), 0)
    ImageDraw.Draw(mask).rounded_rectangle((6, 6, 25, 25), radius=4, fill=255)
    mask.save(mask_path)

    apply_dominant_body_color_correction(
        image_path,
        mask_path=mask_path,
        dominant_color="blue",
        coverage_class="localized_visible_pattern",
        category="kitchen appliance",
        canonical_product_type="toaster",
        rigid_vs_soft="rigid",
    )

    corrected = Image.open(image_path).convert("RGB")
    center = corrected.getpixel((16, 16))
    inner_edge = corrected.getpixel((8, 16))
    background = corrected.getpixel((2, 2))

    assert center[2] > center[0]
    assert inner_edge[2] > inner_edge[0]
    assert background == (236, 232, 226)


def _write_patterned_bag_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "patterned_bag.jpg"
    crop_path = tmp_path / "patterned_bag.crop.png"
    mask_path = tmp_path / "patterned_bag.mask.png"

    image = Image.new("RGB", (120, 120), (245, 242, 236))
    mask = Image.new("L", (120, 120), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    body_box = (18, 36, 102, 108)
    draw.rectangle(body_box, fill=(84, 88, 180))
    mask_draw.rectangle(body_box, fill=255)

    for y in range(40, 104, 8):
        for x in range(22, 98, 10):
            fill = (246, 220, 86) if (x + y) % 20 == 0 else (246, 176, 196)
            draw.ellipse((x, y, x + 6, y + 6), fill=fill)

    draw.rectangle((20, 20, 34, 44), fill=(22, 22, 22))
    draw.rectangle((86, 20, 100, 44), fill=(22, 22, 22))
    mask_draw.rectangle((20, 20, 34, 44), fill=255)
    mask_draw.rectangle((86, 20, 100, 44), fill=255)

    image.save(image_path)
    image.crop((18, 20, 102, 108)).save(crop_path)
    mask.save(mask_path)
    return image_path, crop_path, mask_path


def _write_dark_printed_bottle_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "dark_bottle.jpg"
    mask_path = tmp_path / "dark_bottle.mask.png"

    image = Image.new("RGB", (120, 200), (245, 242, 236))
    mask = Image.new("L", (120, 200), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    draw.rounded_rectangle((38, 24, 82, 184), radius=12, fill=(26, 28, 32))
    draw.rectangle((46, 8, 74, 34), fill=(26, 28, 32))
    mask_draw.rounded_rectangle((38, 24, 82, 184), radius=12, fill=255)
    mask_draw.rectangle((46, 8, 74, 34), fill=255)

    for y in range(42, 170, 18):
        draw.rectangle((44, y, 76, y + 6), fill=(205, 163, 74))
        draw.ellipse((50, y + 8, 70, y + 18), fill=(221, 185, 92))

    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def _write_dark_labeled_bottle_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "dark_labeled_bottle.jpg"
    crop_path = tmp_path / "dark_labeled_bottle.crop.png"
    mask_path = tmp_path / "dark_labeled_bottle.mask.png"

    image = Image.new("RGB", (120, 220), (246, 242, 235))
    mask = Image.new("L", (120, 220), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    draw.rounded_rectangle((34, 26, 86, 202), radius=12, fill=(28, 31, 36))
    draw.rounded_rectangle((40, 6, 80, 40), radius=8, fill=(22, 24, 28))
    draw.rectangle((30, 78, 90, 122), fill=(244, 241, 236))
    draw.text((43, 90), "LOGO", fill=(150, 65, 60))
    mask_draw.rounded_rectangle((34, 26, 86, 202), radius=12, fill=255)
    mask_draw.rounded_rectangle((40, 6, 80, 40), radius=8, fill=255)
    mask_draw.rectangle((30, 78, 90, 122), fill=255)

    image.save(image_path)
    image.crop((28, 6, 92, 206)).save(crop_path)
    mask.save(mask_path)
    return image_path, crop_path, mask_path


def _write_wallet_like_bag_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "wallet_like_bag.jpg"
    crop_path = tmp_path / "wallet_like_bag.crop.png"
    mask_path = tmp_path / "wallet_like_bag.mask.png"

    image = Image.new("RGB", (160, 120), (246, 243, 237))
    mask = Image.new("L", (160, 120), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    body_box = (24, 34, 136, 86)
    draw.rounded_rectangle(body_box, radius=10, fill=(82, 112, 182))
    mask_draw.rounded_rectangle(body_box, radius=10, fill=255)
    for x in range(34, 128, 14):
        draw.ellipse((x, 44, x + 8, 52), fill=(228, 182, 92))
        draw.ellipse((x + 3, 60, x + 11, 68), fill=(243, 218, 118))

    image.save(image_path)
    image.crop((20, 30, 140, 90)).save(crop_path)
    mask.save(mask_path)
    return image_path, crop_path, mask_path


def _write_tall_sideview_bag_without_visible_handles_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "tall_sideview_bag.jpg"
    crop_path = tmp_path / "tall_sideview_bag.crop.png"
    mask_path = tmp_path / "tall_sideview_bag.mask.png"

    image = Image.new("RGB", (140, 180), (246, 243, 237))
    mask = Image.new("L", (140, 180), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    body = [(30, 22), (102, 18), (114, 150), (40, 162)]
    draw.polygon(body, fill=(84, 108, 166))
    mask_draw.polygon(body, fill=255)
    draw.rectangle((62, 44, 88, 132), fill=(214, 196, 176))
    for y in range(38, 146, 20):
        draw.ellipse((42, y, 56, y + 10), fill=(236, 218, 98))
        draw.ellipse((82, y + 8, 96, y + 18), fill=(240, 172, 196))

    image.save(image_path)
    image.crop((28, 18, 116, 164)).save(crop_path)
    mask.save(mask_path)
    return image_path, crop_path, mask_path


def _write_connected_handle_bag_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "connected_handle_bag.jpg"
    mask_path = tmp_path / "connected_handle_bag.mask.png"

    image = Image.new("RGB", (180, 180), (244, 239, 232))
    mask = Image.new("L", (180, 180), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    draw.rounded_rectangle((26, 72, 154, 158), radius=12, fill=(114, 78, 168))
    mask_draw.rounded_rectangle((26, 72, 154, 158), radius=12, fill=255)
    for x in range(36, 146, 18):
        draw.ellipse((x, 84, x + 10, 94), fill=(246, 218, 84))
        draw.ellipse((x + 4, 110, x + 14, 120), fill=(242, 176, 202))
    draw.rounded_rectangle((54, 16, 126, 92), radius=26, outline=(24, 24, 24), width=16)
    mask_draw.rounded_rectangle((54, 16, 126, 92), radius=26, outline=255, width=16)

    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def _write_dark_handle_patterned_shoulder_bag_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "dark_handle_patterned_shoulder_bag.jpg"
    mask_path = tmp_path / "dark_handle_patterned_shoulder_bag.mask.png"

    image = Image.new("RGB", (220, 180), (244, 239, 232))
    mask = Image.new("L", (220, 180), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    draw.rounded_rectangle((28, 68, 192, 156), radius=12, fill=(116, 74, 170))
    mask_draw.rounded_rectangle((28, 68, 192, 156), radius=12, fill=255)
    draw.polygon([(36, 68), (76, 34), (102, 68)], fill=(122, 82, 176))
    draw.polygon([(118, 68), (144, 34), (184, 68)], fill=(124, 84, 178))
    mask_draw.polygon([(36, 68), (76, 34), (102, 68)], fill=255)
    mask_draw.polygon([(118, 68), (144, 34), (184, 68)], fill=255)

    for x in range(40, 186, 18):
        draw.ellipse((x, 84, x + 10, 94), fill=(246, 218, 84))
        draw.ellipse((x + 4, 112, x + 14, 122), fill=(242, 176, 202))

    draw.rounded_rectangle((54, 12, 98, 96), radius=22, outline=(18, 18, 18), width=12)
    draw.rounded_rectangle((122, 12, 166, 96), radius=22, outline=(18, 18, 18), width=12)
    mask_draw.rounded_rectangle((54, 12, 98, 96), radius=22, outline=255, width=12)
    mask_draw.rounded_rectangle((122, 12, 166, 96), radius=22, outline=255, width=12)

    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def _write_omitted_handle_tote_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "omitted_handle_tote.jpg"
    mask_path = tmp_path / "omitted_handle_tote.mask.png"

    image = Image.new("RGB", (180, 160), (38, 34, 40))
    mask = Image.new("L", (180, 160), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    body_points = [(20, 34), (78, 30), (92, 74), (118, 30), (164, 28), (172, 146), (16, 150), (10, 54)]
    draw.polygon(body_points, fill=(110, 78, 170))
    mask_draw.polygon(body_points, fill=255)

    for x in range(24, 156, 18):
        draw.ellipse((x, 64, x + 12, 76), fill=(246, 218, 84))
        draw.ellipse((x + 4, 94, x + 14, 106), fill=(242, 176, 202))

    handle_path = [(62, 18), (72, 6), (108, 6), (124, 18), (118, 30), (100, 18), (82, 18), (70, 30)]
    draw.line(handle_path, fill=(18, 18, 18), width=10, joint="curve")

    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def _write_neutral_pillow_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "neutral_pillow.jpg"
    mask_path = tmp_path / "neutral_pillow.mask.png"

    image = Image.new("RGB", (160, 160), (140, 38, 32))
    mask = Image.new("L", (160, 160), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    draw.rounded_rectangle((20, 22, 140, 142), radius=10, fill=(222, 206, 183))
    mask_draw.rounded_rectangle((20, 22, 140, 142), radius=10, fill=255)
    for y in range(36, 134, 14):
        draw.line((28, y, 132, y), fill=(176, 156, 136), width=4)
        draw.line((28, y + 5, 132, y + 5), fill=(234, 222, 205), width=2)

    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def _write_table_lamp_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_path = tmp_path / "table_lamp.jpg"
    crop_path = tmp_path / "table_lamp.crop.png"
    mask_path = tmp_path / "table_lamp.mask.png"

    image = Image.new("RGB", (160, 220), (244, 240, 232))
    mask = Image.new("L", (160, 220), 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    draw.rectangle((44, 26, 116, 96), fill=(238, 230, 206))
    draw.rectangle((72, 96, 88, 162), fill=(58, 42, 26))
    draw.ellipse((42, 156, 118, 198), fill=(142, 102, 54))
    mask_draw.rectangle((44, 26, 116, 96), fill=255)
    mask_draw.rectangle((72, 96, 88, 162), fill=255)
    mask_draw.ellipse((42, 156, 118, 198), fill=255)

    image.save(image_path)
    image.crop((36, 22, 124, 202)).save(crop_path)
    mask.save(mask_path)
    return image_path, crop_path, mask_path


def test_visual_evidence_profile_detects_broad_pattern_and_upper_component(tmp_path: Path) -> None:
    image_path, _, mask_path = _write_patterned_bag_artifacts(tmp_path)

    profile = build_visual_evidence_profile(image_path, mask_path)

    assert profile["coverage_class"] in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
    assert profile["coverage_ratio"] is not None and profile["coverage_ratio"] >= 0.4
    assert profile["upper_component_count"] is not None and profile["upper_component_count"] >= 2
    assert profile["upper_region_color"] in {"black", "gray"}


def test_build_localized_product_uses_lamp_structure_instead_of_fake_label(tmp_path: Path) -> None:
    image_path, crop_path, mask_path = _write_table_lamp_artifacts(tmp_path)
    localized = build_localized_product(
        ReviewSeedRecord(
            id="table_lamp",
            platform="walmart",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            product_title="PINEAPPLE TABLE LAMP",
            hint_phrases=("lamp", "table lamp", "home lighting"),
            capture_date="2026-04-10",
            local_image_path=image_path,
        ),
        LocalizationArtifactRecord(
            id="table_lamp",
            product_title="PINEAPPLE TABLE LAMP",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            local_image_path=image_path,
            selected_phrase="pineapple table lamp",
            selected_confidence=0.9,
            selected_box=None,
            overlay_path=None,
            crop_path=crop_path,
            mask_path=mask_path,
        ),
    )

    evidence = localized.identity.observed_evidence
    assert evidence.form_factor_note is not None and "lamp form is upright" in evidence.form_factor_note
    assert evidence.coverage_note != "a localized high-contrast front panel or label interrupts one region of the visible product surface"
    assert any("stable base and upper light shade" in fact for fact in evidence.hard_facts)


def test_structural_palette_prefers_body_tone_over_small_bright_accents(tmp_path: Path) -> None:
    image_path, mask_path = _write_dark_printed_bottle_artifacts(tmp_path)

    structural_palette = infer_named_palette_with_strategy(
        image_path,
        mask_path,
        top_k=3,
        use_smoothed=True,
        erode_steps=2,
    )

    assert structural_palette
    assert structural_palette[0] in {"black", "gray", "brown"}


def test_build_localized_product_treats_labeled_bottle_as_dark_body_with_localized_panel(tmp_path: Path) -> None:
    image_path, crop_path, mask_path = _write_dark_labeled_bottle_artifacts(tmp_path)
    localized = build_localized_product(
        ReviewSeedRecord(
            id="dark_labeled_bottle",
            platform="walmart",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            product_title="Reusable sports bottle",
            hint_phrases=("water bottle", "bottle", "drinkware"),
            capture_date="2026-04-10",
            local_image_path=image_path,
        ),
        LocalizationArtifactRecord(
            id="dark_labeled_bottle",
            product_title="Reusable sports bottle",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            local_image_path=image_path,
            selected_phrase="sports bottle",
            selected_confidence=0.95,
            selected_box=None,
            overlay_path=None,
            crop_path=crop_path,
            mask_path=mask_path,
        ),
    )

    evidence = localized.identity.observed_evidence
    assert evidence.coverage_class == "localized_visible_pattern"
    assert evidence.color_note is not None and ("black" in evidence.color_note or "gray" in evidence.color_note)
    assert evidence.coverage_note is not None and "localized" in evidence.coverage_note
    assert evidence.pattern_note is None
    assert evidence.upper_component_state == "present"


def test_visual_evidence_profile_detects_connected_upper_attachment(tmp_path: Path) -> None:
    image_path, mask_path = _write_connected_handle_bag_artifacts(tmp_path)

    profile = build_visual_evidence_profile(image_path, mask_path)

    assert profile["upper_component_state"] == "present"
    assert profile["upper_component_count"] is not None and profile["upper_component_count"] >= 1
    assert profile["upper_region_color"] in {"black", "gray"}


def test_visual_evidence_profile_prefers_dark_handles_over_patterned_upper_shoulders(tmp_path: Path) -> None:
    image_path, mask_path = _write_dark_handle_patterned_shoulder_bag_artifacts(tmp_path)

    profile = build_visual_evidence_profile(image_path, mask_path)

    assert profile["upper_component_state"] == "present"
    assert profile["upper_region_color"] in {"black", "gray"}
    assert profile["body_region_color"] in {"purple", "blue", "gray"}
    assert profile["upper_component_count"] is not None and profile["upper_component_count"] >= 1


def test_visual_evidence_profile_marks_omitted_handle_structure_without_fake_handle_color(tmp_path: Path) -> None:
    image_path, mask_path = _write_omitted_handle_tote_artifacts(tmp_path)

    profile = build_visual_evidence_profile(image_path, mask_path)

    assert profile["upper_component_state"] == "present"
    assert profile["upper_region_note"] is not None and "partially shown" in profile["upper_region_note"]
    assert profile["upper_region_color"] is None
    assert profile["upper_component_count"] is None


def test_core_body_palette_keeps_neutral_textile_in_neutral_family(tmp_path: Path) -> None:
    image_path, mask_path = _write_neutral_pillow_artifacts(tmp_path)

    ranked = infer_core_body_palette_ranked(image_path, mask_path, top_k=3)

    assert ranked
    assert ranked[0][0] in {"beige", "white", "gray"}


def test_build_localized_product_refines_wallet_like_object_away_from_tote(tmp_path: Path) -> None:
    image_path, crop_path, mask_path = _write_wallet_like_bag_artifacts(tmp_path)
    localized = build_localized_product(
        ReviewSeedRecord(
            id="wallet_like_bag",
            platform="walmart",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            product_title="Large Tote Set With Matching Wallet",
            hint_phrases=("handbag", "tote bag", "wallet"),
            capture_date="2026-04-10",
            local_image_path=image_path,
        ),
        LocalizationArtifactRecord(
            id="wallet_like_bag",
            product_title="Large Tote Set With Matching Wallet",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            local_image_path=image_path,
            selected_phrase="matching wallet from set",
            selected_confidence=0.93,
            selected_box=None,
            overlay_path=None,
            crop_path=crop_path,
            mask_path=mask_path,
        ),
    )

    assert localized.identity.canonical_product_type == "wallet"
    assert localized.identity.observed_evidence.upper_component_state != "present"


def test_build_localized_product_keeps_tall_sideview_bag_as_bag_not_wallet(tmp_path: Path) -> None:
    image_path, crop_path, mask_path = _write_tall_sideview_bag_without_visible_handles_artifacts(tmp_path)
    localized = build_localized_product(
        ReviewSeedRecord(
            id="tall_sideview_bag",
            platform="walmart",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            product_title="Structured Tote Handbag Set",
            hint_phrases=("handbag", "tote bag", "wallet"),
            capture_date="2026-04-10",
            local_image_path=image_path,
        ),
        LocalizationArtifactRecord(
            id="tall_sideview_bag",
            product_title="Structured Tote Handbag Set",
            source_page_url="https://example.com/product",
            source_image_url="https://example.com/image.jpg",
            local_image_path=image_path,
            selected_phrase="structured tote handbag set",
            selected_confidence=0.91,
            selected_box=None,
            overlay_path=None,
            crop_path=crop_path,
            mask_path=mask_path,
        ),
    )

    assert localized.identity.canonical_product_type in {"tote bag", "handbag"}
    assert localized.identity.canonical_product_type != "wallet"


def test_refine_canonical_product_type_keeps_partial_backpack_body_as_backpack() -> None:
    canonical = refine_canonical_product_type(
        category="bag",
        initial_canonical_product_type="backpack",
        product_title="RTIC Backpack Cooler",
        hint_phrases=("backpack cooler", "cooler backpack"),
        selected_phrase="backpack cooler backpack backpack",
        observed_evidence=ObservedEvidenceSpec(
            surface_scope="partial_or_ambiguous",
            aspect_ratio=0.79,
            top_width_ratio=0.64,
            upper_component_state="absent",
            form_factor_note="the product reads as a backpack body rather than a handbag or tote",
            hard_facts=[
                "the visible backpack body includes darker harness, panel, or attachment zones against a blue main body"
            ],
        ),
    )

    assert canonical == "backpack"


def test_infer_functional_subtype_hint_preserves_backpack_cooler_modifier() -> None:
    subtype = infer_functional_subtype_hint(
        category="bag",
        canonical_product_type="backpack",
        product_title="Titan 24 Can Backpack Cooler",
        hint_phrases=("backpack cooler", "cooler backpack", "insulated backpack"),
        selected_phrase="backpack cooler backpack backpack",
    )

    assert subtype == "backpack cooler"


def test_identity_requires_functional_context_for_backpack_cooler() -> None:
    identity = ProductIdentitySpec(
        category="bag",
        canonical_product_type="backpack",
        subtype_hint="backpack cooler",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            source_validity_issues=[],
            artifact_flags=[],
        ),
    )

    assert identity_requires_functional_context(identity) is True
    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("clarity", "hero")


def test_select_functional_subtype_eval_prompts_for_backpack_cooler() -> None:
    prompts = select_functional_subtype_eval_prompts(
        ProductIdentitySpec(
            category="bag",
            canonical_product_type="backpack",
            subtype_hint="backpack cooler",
        ),
        "backpack",
    )

    assert prompts is not None
    assert any("insulated opening" in prompt for prompt in prompts["positive"])


def test_infer_evidence_caption_prefers_source_context_for_backpack_cooler(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    cutout_path = tmp_path / "cutout.png"
    crop_path = tmp_path / "crop.png"
    for path in (source_path, cutout_path, crop_path):
        path.write_bytes(b"x")

    class _Backbone:
        def caption_image(self, path: Path) -> str:
            if path == source_path:
                return "a bag full of cans of sodas"
            return "a blue backpack"

    caption = infer_evidence_caption(
        _Backbone(),
        source_image=source_path,
        cutout_path=cutout_path,
        crop_path=crop_path,
        canonical_product_type="backpack",
        category="bag",
        prefer_source_context=True,
    )

    assert caption == "a bag full of cans of sodas"


def test_sanitize_evidence_caption_drops_conflicting_drinkware_subtype() -> None:
    assert (
        sanitize_evidence_caption(
            "a black and white coffee cup with a lid",
            canonical_product_type="water bottle",
            category="drinkware",
        )
        is None
    )


def test_sanitize_evidence_caption_drops_conflicting_structured_appliance_type() -> None:
    assert (
        sanitize_evidence_caption(
            "a blender with a blender in it",
            canonical_product_type="coffee maker",
            category="kitchen appliance",
        )
        is None
    )


def test_sanitize_evidence_caption_repairs_sparse_caption_phrase() -> None:
    assert (
        sanitize_evidence_caption(
            "a and mug",
            canonical_product_type="mug",
            category="drinkware",
        )
        == "a mug"
    )


def test_infer_material_note_ignores_title_only_fabric_colorway_for_rigid_appliance() -> None:
    assert (
        infer_material_note(
            category="kitchen appliance",
            canonical_product_type="slow cooker",
            product_title="The Pioneer Woman 6 Qt Slow Cooker Linen Speckle",
            hint_phrases=("slow cooker", "kitchen appliance"),
            evidence_caption=None,
        )
        is None
    )


def test_refine_retrieval_visual_classification_promotes_generic_product_to_kitchen_appliance() -> None:
    class _Backbone:
        def encode_texts(self, texts: tuple[str, ...] | list[str]) -> list[tuple[float, ...]]:
            vectors: list[tuple[float, ...]] = []
            for text in texts:
                lowered = str(text).lower()
                if "countertop kitchen appliance" in lowered:
                    vectors.append((1.0, 0.0, 0.0))
                elif "drip coffee maker" in lowered:
                    vectors.append((1.0, 0.0, 0.0))
                else:
                    vectors.append((0.0, 1.0, 0.0))
            return vectors

    category, canonical_type = refine_retrieval_visual_classification(
        backbone=_Backbone(),
        image_embedding=(1.0, 0.0, 0.0),
        category="product",
        canonical_product_type="product",
    )

    assert category == "kitchen appliance"
    assert canonical_type == "coffee maker"


def test_reconcile_color_note_with_caption_prefers_caption_colors_for_broad_prints() -> None:
    note = reconcile_color_note_with_caption(
        "the main visible body reads as beige with compatible printed accents in orange, brown",
        caption_colors=["beige", "white", "brown"],
        coverage_class="full_visible_surface_pattern",
    )

    assert note == "the main visible body reads as beige with compatible printed accents in white, brown"


def test_infer_neutral_textile_surface_notes_prefers_woven_tonal_language() -> None:
    notes = infer_neutral_textile_surface_notes(
        material_note="visible material cues suggest a woven or interlaced texture; visible material cues suggest a fabric or textile surface",
        palette=["beige", "brown", "white"],
        coverage_class="full_visible_surface_pattern",
    )

    assert notes[0] == "the main visible body reads as beige with subtle tonal variation from woven texture"
    assert notes[1] == "the visible woven or ribbed texture spans most of the observed product surface"
    assert notes[2] == "the visible surface is defined by woven or ribbed texture rather than a bold printed graphic"


def test_infer_neutral_textile_surface_notes_uses_visual_soft_good_cues_without_material_text(tmp_path: Path) -> None:
    image_path, mask_path = _write_neutral_pillow_artifacts(tmp_path)

    notes = infer_neutral_textile_surface_notes(
        material_note=None,
        palette=["beige", "orange", "brown"],
        coverage_class="full_visible_surface_pattern",
        category="home decor",
        canonical_product_type="decorative pillow",
        source_image=image_path,
        mask_path=mask_path,
    )

    assert notes[0] == "the main visible body reads as beige with subtle tonal variation from textured fabric"
    assert notes[1] == "the visible tonal texture spans most of the observed product surface"
    assert notes[2] == "the visible surface is defined by tonal textile texture rather than a bold printed graphic"


def test_infer_dark_reflective_body_override_prefers_dark_base_for_localized_label(tmp_path: Path) -> None:
    image_path, _, mask_path = _write_dark_labeled_bottle_artifacts(tmp_path)

    note, palette, confidence = infer_dark_reflective_body_override(
        source_image=image_path,
        mask_path=mask_path,
        category="drinkware",
        canonical_product_type="water bottle",
        coverage_class="localized_visible_pattern",
        palette=["brown", "gold", "white"],
    )

    assert note == "the main visible body reads as black with reflective highlight variation"
    assert palette is not None and palette[0] == "black"
    assert confidence == 0.82


def test_infer_dark_reflective_body_override_handles_glossy_black_appliance(tmp_path: Path) -> None:
    image_path = tmp_path / "coffee_maker.jpg"
    mask_path = tmp_path / "coffee_maker.mask.png"
    image = Image.new("RGB", (160, 180), (232, 232, 232))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 20, 120, 160), fill=(36, 42, 54))
    draw.rectangle((52, 34, 110, 78), fill=(78, 92, 112))
    draw.rectangle((48, 86, 112, 154), fill=(28, 34, 42))
    image.save(image_path)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((40, 20, 120, 160), fill=255)
    mask.save(mask_path)

    note, palette, confidence = infer_dark_reflective_body_override(
        source_image=image_path,
        mask_path=mask_path,
        category="kitchen appliance",
        canonical_product_type="coffee maker",
        coverage_class="low_variation_surface",
        palette=["gray", "blue", "beige"],
    )

    assert note == "the main visible body reads as black with reflective highlight variation"
    assert palette is not None and palette[0] == "black"
    assert confidence == 0.82


def test_infer_dark_reflective_body_override_skips_warm_wood_furniture(tmp_path: Path) -> None:
    image_path = tmp_path / "folding_chair.jpg"
    mask_path = tmp_path / "folding_chair.mask.png"
    image = Image.new("RGB", (180, 220), (226, 216, 206))
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 24, 132, 196), fill=(148, 104, 58))
    draw.rectangle((56, 38, 124, 74), fill=(164, 118, 68))
    draw.rectangle((60, 92, 120, 126), fill=(154, 110, 62))
    image.save(image_path)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((48, 24, 132, 196), fill=255)
    mask.save(mask_path)

    note, palette, confidence = infer_dark_reflective_body_override(
        source_image=image_path,
        mask_path=mask_path,
        category="furniture",
        canonical_product_type="folding chair",
        coverage_class="low_variation_surface",
        palette=["brown", "orange", "beige"],
    )

    assert note is None
    assert palette is None
    assert confidence is None


def test_infer_low_saturation_cool_body_override_detects_cool_rigid_mug(tmp_path: Path) -> None:
    image_path = tmp_path / "mug.jpg"
    mask_path = tmp_path / "mug.mask.png"
    image = Image.new("RGB", (160, 160), (240, 236, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle((34, 28, 114, 132), fill=(146, 171, 182))
    draw.ellipse((102, 52, 140, 108), outline=(132, 160, 173), width=8)
    image.save(image_path)
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle((34, 28, 114, 132), fill=255)
    mask_draw.ellipse((102, 52, 140, 108), outline=255, width=8)
    mask.save(mask_path)

    note, palette, confidence = infer_low_saturation_cool_body_override(
        source_image=image_path,
        mask_path=mask_path,
        category="drinkware",
        canonical_product_type="mug",
        coverage_class="full_visible_surface_pattern",
        palette=["gray", "blue", "brown"],
    )

    assert note is not None and ("teal" in note or "blue" in note)
    assert palette is not None and palette[0] in {"teal", "blue"}
    assert confidence == 0.72


def test_infer_low_saturation_cool_body_override_detects_cool_rigid_toaster(tmp_path: Path) -> None:
    image_path = tmp_path / "toaster.jpg"
    mask_path = tmp_path / "toaster.mask.png"
    image = Image.new("RGB", (160, 180), (242, 236, 226))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((38, 28, 124, 152), radius=16, fill=(147, 176, 186))
    draw.rectangle((68, 34, 94, 78), fill=(82, 97, 106))
    image.save(image_path)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((38, 28, 124, 152), radius=16, fill=255)
    mask.save(mask_path)

    note, palette, confidence = infer_low_saturation_cool_body_override(
        source_image=image_path,
        mask_path=mask_path,
        category="kitchen appliance",
        canonical_product_type="toaster",
        coverage_class="localized_visible_pattern",
        palette=["black", "gray", "white"],
    )

    assert note is not None and ("teal" in note or "blue" in note)
    assert palette is not None and palette[0] in {"teal", "blue"}
    assert confidence == 0.72


def test_infer_low_saturation_cool_body_override_accepts_moderately_dark_cool_rigid_body(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "cool_rigid_dark.jpg"
    mask_path = tmp_path / "cool_rigid_dark.mask.png"
    image = Image.new("RGB", (160, 180), (238, 233, 225))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((34, 28, 126, 152), radius=14, fill=(83, 104, 118))
    draw.rectangle((68, 38, 96, 78), fill=(58, 70, 76))
    image.save(image_path)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((34, 28, 126, 152), radius=14, fill=255)
    mask.save(mask_path)

    note, palette, confidence = infer_low_saturation_cool_body_override(
        source_image=image_path,
        mask_path=mask_path,
        category="kitchen appliance",
        canonical_product_type="toaster",
        coverage_class="low_variation_surface",
        palette=["black", "gray"],
    )

    assert note is not None and ("teal" in note or "blue" in note)
    assert palette is not None and palette[0] in {"teal", "blue"}
    assert confidence == 0.72


def test_generated_focus_mask_is_safe_for_color_repair_rejects_sparse_open_structure(tmp_path: Path) -> None:
    sparse_mask = tmp_path / "sparse.mask.png"
    compact_mask = tmp_path / "compact.mask.png"

    sparse = Image.new("L", (200, 200), 0)
    draw = ImageDraw.Draw(sparse)
    draw.rectangle((85, 20, 115, 180), fill=255)
    draw.rectangle((35, 150, 165, 180), fill=255)
    draw.rectangle((45, 40, 75, 70), fill=255)
    draw.rectangle((125, 40, 155, 70), fill=255)
    sparse.save(sparse_mask)

    compact = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(compact).rounded_rectangle((40, 30, 160, 170), radius=18, fill=255)
    compact.save(compact_mask)

    assert not generated_focus_mask_is_safe_for_color_repair(
        sparse_mask,
        category="furniture",
        canonical_product_type="office chair",
    )
    assert generated_focus_mask_is_safe_for_color_repair(
        compact_mask,
        category="kitchen appliance",
        canonical_product_type="toaster",
    )


def test_synchronize_coverage_note_with_dominant_color_updates_soft_goods_field() -> None:
    note = synchronize_coverage_note_with_dominant_color(
        coverage_note="most of the visible bedding surface reads as a relatively uniform gray treatment",
        coverage_class="low_variation_surface",
        dominant_color="blue",
        category="bedding",
        canonical_product_type="quilt",
    )

    assert note == "most of the visible bedding surface reads as a tonal blue textile field"


def test_synchronize_pattern_note_with_dominant_color_updates_base_phrase() -> None:
    note = synchronize_pattern_note_with_dominant_color(
        pattern_note="the visible product body carries a multicolor or printed treatment on a gray base",
        dominant_color="teal",
    )

    assert note == "the visible product body carries a multicolor or printed treatment on a teal base"


def test_infer_backpack_structure_note_preserves_dark_panel_relationship() -> None:
    note = infer_backpack_structure_note(
        category="bag",
        canonical_product_type="backpack",
        palette=["blue", "black"],
        accent_palette=["blue", "black"],
        evidence_caption="a black backpack with a blue handle",
    )

    assert note is not None
    assert "darker" in note and "attachment" in note


def test_should_apply_post_generation_color_repair_skips_human_handled_items() -> None:
    identity = ProductIdentitySpec(
        category="bag",
        canonical_product_type="wallet",
        interaction_mode="held_in_hand",
        requires_human_model=True,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as blue with compatible printed accents in gray",
            color_confidence=0.74,
            coverage_class="full_visible_surface_pattern",
            upper_component_state="absent",
        ),
    )

    assert should_apply_post_generation_color_repair(identity) is False


def test_should_apply_post_generation_color_repair_skips_low_saturation_glazed_drinkware() -> None:
    identity = ProductIdentitySpec(
        category="drinkware",
        canonical_product_type="mug",
        interaction_mode="handheld_or_display",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as teal with low-saturation glazed variation",
            color_confidence=0.72,
            coverage_class="full_visible_surface_pattern",
            upper_component_state="present",
        ),
    )

    assert should_apply_post_generation_color_repair(identity) is False


def test_infer_localized_contrast_panel_prefers_large_bright_control_region(tmp_path: Path) -> None:
    image_path, _, mask_path = _write_dark_labeled_bottle_artifacts(tmp_path)

    panel_note, value_relation_note = infer_localized_contrast_panel(
        image_path,
        mask_path,
        category="drinkware",
        canonical_product_type="water bottle",
        body_color="black",
    )

    assert panel_note is not None and ("label band" in panel_note or "printed wrap" in panel_note)
    assert value_relation_note == "the localized label band or printed wrap is visibly lighter than the main body"


def test_score_retrieval_evidence_compatibility_penalizes_solid_generic_neighbor() -> None:
    evidence = ObservedEvidenceSpec(
        palette=["blue", "purple", "pink"],
        coverage_class="broad_visible_surface_pattern",
        coverage_note="the visible print or color treatment spans a broad portion of the observed product surface on a blue base",
        upper_region_note="the visible upper component splits into multiple narrow segments in black above a blue main body",
        evidence_tags=["broad_surface_treatment", "distinct_upper_component", "distinct_boundary_trim"],
    )

    incompatible = score_retrieval_evidence_compatibility(
        "a white bag with pearls on the handle",
        evidence=evidence,
        canonical_type_tokens={"tote", "bag"},
        category="bag",
    )
    compatible = score_retrieval_evidence_compatibility(
        "a patterned tote bag with black handles",
        evidence=evidence,
        canonical_type_tokens={"tote", "bag"},
        category="bag",
    )

    assert incompatible < 0.0
    assert compatible > incompatible
    assert should_fallback_to_category_prior(evidence, [incompatible, incompatible - 0.1]) is True


def test_should_fallback_to_category_prior_keeps_visually_strong_candidates() -> None:
    evidence = ObservedEvidenceSpec(
        palette=["blue", "purple", "pink"],
        coverage_class="broad_visible_surface_pattern",
        upper_component_state="absent",
    )

    assert (
        should_fallback_to_category_prior(
            evidence,
            [0.03, -0.04],
            retrieval_scores=[0.41, 0.24],
            image_similarities=[0.78, 0.52],
        )
        is False
    )


def test_should_fallback_to_category_prior_for_compact_direct_grip_with_weak_compatibility() -> None:
    evidence = ObservedEvidenceSpec(
        palette=["blue", "gray", "beige"],
        color_note="the main visible body reads as blue with compatible printed accents in gray, beige",
        color_confidence=0.72,
        coverage_class="full_visible_surface_pattern",
        upper_component_state="absent",
        form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
    )

    assert (
        should_fallback_to_category_prior(
            evidence,
            [0.232],
            retrieval_scores=[0.9905],
            image_similarities=[0.6029],
        )
        is True
    )


def test_build_business_prior_falls_back_when_neighbors_conflict_with_evidence(tmp_path: Path) -> None:
    image_path, crop_path, mask_path = _write_patterned_bag_artifacts(tmp_path)
    seed = ReviewSeedRecord(
        id="synthetic_tote",
        platform="walmart",
        source_page_url="https://example.com/product",
        source_image_url="https://example.com/image.jpg",
        product_title="Synthetic Patterned Tote Bag",
        hint_phrases=("tote bag", "patterned tote", "bag"),
        capture_date="2026-04-10",
        local_image_path=image_path,
    )
    record = LocalizationArtifactRecord(
        id="synthetic_tote",
        product_title=seed.product_title,
        source_page_url=seed.source_page_url,
        source_image_url=seed.source_image_url,
        local_image_path=image_path,
        selected_phrase="patterned tote bag",
        selected_confidence=0.95,
        selected_box=None,
        overlay_path=None,
        crop_path=crop_path,
        mask_path=mask_path,
    )
    localized = build_localized_product(seed, record)

    class _Backbone:
        def encode_image(self, image_path: str | Path) -> tuple[float, ...]:
            return (1.0, 0.0)

    prior = build_business_prior(
        seed,
        localized,
        record,
        [
            RetrievalCandidate(
                item_id="bad-1",
                image_name="bad-1.png",
                image_path=tmp_path / "bad-1.png",
                score=1.0,
                page_views=100,
                clicks=10,
                caption="a white bag with pearls on the handle",
                embedding=(1.0, 0.0),
                scenario_slots=("fashion_lifestyle",),
                style_atoms=("clear hero framing", "human-in-use framing"),
                scene_families=("fashion_lifestyle",),
                support_relations=("carried_by_hand",),
            ),
            RetrievalCandidate(
                item_id="bad-2",
                image_name="bad-2.png",
                image_path=tmp_path / "bad-2.png",
                score=1.0,
                page_views=90,
                clicks=9,
                caption="a woman holding a brown and white bag",
                embedding=(0.9, 0.1),
                scenario_slots=("fashion_lifestyle",),
                style_atoms=("clear hero framing", "human-in-use framing"),
                scene_families=("fashion_lifestyle",),
                support_relations=("carried_by_hand",),
            ),
        ],
        _Backbone(),
        top_k=2,
    )

    assert prior.metadata["retrieval_mode"] == "category_fallback"


def test_build_business_prior_uses_visual_match_when_caption_is_generic(tmp_path: Path) -> None:
    image_path, crop_path, mask_path = _write_patterned_bag_artifacts(tmp_path)
    seed = ReviewSeedRecord(
        id="synthetic_tote",
        platform="walmart",
        source_page_url="https://example.com/product",
        source_image_url="https://example.com/image.jpg",
        product_title="Synthetic Patterned Tote Bag",
        hint_phrases=("tote bag", "patterned tote", "bag"),
        capture_date="2026-04-10",
        local_image_path=image_path,
    )
    record = LocalizationArtifactRecord(
        id="synthetic_tote",
        product_title=seed.product_title,
        source_page_url=seed.source_page_url,
        source_image_url=seed.source_image_url,
        local_image_path=image_path,
        selected_phrase="patterned tote bag",
        selected_confidence=0.95,
        selected_box=None,
        overlay_path=None,
        crop_path=crop_path,
        mask_path=mask_path,
    )
    localized = build_localized_product(seed, record)

    class _Backbone:
        def encode_image(self, image_path: str | Path) -> tuple[float, ...]:
            if str(image_path).endswith(("good-1.png", ".crop.png")):
                return (1.0, 0.0)
            return (0.1, 0.9)

    compatible_evidence = ObservedEvidenceSpec(
        palette=["blue", "purple", "pink"],
        coverage_class="broad_visible_surface_pattern",
        coverage_note="the visible print or color treatment spans a broad portion of the observed product surface on a blue base",
        upper_component_state="absent",
        evidence_tags=["broad_surface_treatment", "no_distinct_upper_component"],
        aspect_ratio=0.72,
        top_width_ratio=0.84,
    )

    prior = build_business_prior(
        seed,
        localized,
        record,
        [
            RetrievalCandidate(
                item_id="good-1",
                image_name="good-1.png",
                image_path=tmp_path / "good-1.png",
                score=1.0,
                page_views=100,
                clicks=10,
                caption="a product on display",
                embedding=(1.0, 0.0),
                scenario_slots=("fashion_lifestyle",),
                style_atoms=("clear hero framing",),
                scene_families=("fashion_lifestyle",),
                support_relations=("carried_by_hand",),
                category="bag",
                canonical_product_type="tote bag",
                support_mode="portable_flexible",
                default_scene_family="fashion_lifestyle",
                interaction_mode="carried_or_resting",
                observed_evidence=compatible_evidence,
            ),
            RetrievalCandidate(
                item_id="bad-1",
                image_name="bad-1.png",
                image_path=tmp_path / "bad-1.png",
                score=1.0,
                page_views=90,
                clicks=9,
                caption="a white bag with pearls on the handle",
                embedding=(0.1, 0.9),
                scenario_slots=("fashion_lifestyle",),
                style_atoms=("human-in-use framing",),
                scene_families=("fashion_lifestyle",),
                support_relations=("carried_by_hand",),
                category="bag",
                canonical_product_type="handbag",
                support_mode="portable_flexible",
                default_scene_family="fashion_lifestyle",
                interaction_mode="carried_or_resting",
                observed_evidence=ObservedEvidenceSpec(
                    palette=["white", "beige"],
                    coverage_class="localized_visible_pattern",
                    upper_component_state="present",
                    upper_region_note="a visible handle appears above the main body",
                    aspect_ratio=1.3,
                    top_width_ratio=0.55,
                ),
            ),
        ],
        _Backbone(),
        top_k=2,
    )

    assert prior.metadata["retrieval_mode"] == "retrieval"
    assert prior.neighbor_item_ids[0] == "good-1"


def test_build_category_fallback_prior_avoids_generic_category_storytelling() -> None:
    prior = build_category_fallback_prior(
        "bag",
        "tote bag",
        support_mode="portable_flexible",
        default_scene_family="fashion_lifestyle",
    )

    assert "commercial fashion-accessory storytelling" not in prior.style_atoms


def test_build_scene_retrieval_fallback_prior_preserves_retrieved_creative_metadata() -> None:
    identity = ProductIdentitySpec(
        phrase="black office chair",
        category="furniture",
        canonical_product_type="office chair",
        support_mode="self_supporting_display",
        default_scene_family="editorial_interior",
        interaction_mode="placed",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as black",
            color_confidence=0.82,
        ),
    )
    candidate = RetrievalCandidate(
        item_id="chair-1",
        image_name="chair-1.png",
        image_path=Path("/tmp/chair-1.png"),
        score=1.0,
        page_views=100,
        clicks=10,
        caption="a product on display",
        embedding=(1.0, 0.0),
        scenario_slots=("editorial_interior",),
        style_atoms=("clear hero framing",),
        scene_families=("editorial_interior",),
        support_relations=("standing_on_surface",),
        category="furniture",
        canonical_product_type="office chair",
        support_mode="self_supporting_display",
        default_scene_family="editorial_interior",
        interaction_mode="placed",
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as black",
            color_confidence=0.82,
        ),
    )

    prior = build_scene_retrieval_fallback_prior(
        identity,
        [candidate],
        category="furniture",
        canonical_product_type="office chair",
        support_mode="self_supporting_display",
    )

    assert prior.metadata["retrieval_mode"] == "scene_retrieval_fallback"


def test_filter_scene_retrieval_candidates_keeps_evidence_safe_same_category_matches() -> None:
    seed = ReviewSeedRecord(
        id="control_dasein_handbag_03",
        platform="walmart",
        source_page_url="https://example.com/product",
        source_image_url="https://example.com/image.jpg",
        product_title="Dasein Medium Size Tote Handbag Set",
        hint_phrases=("wallet", "handbag"),
        capture_date="2026-04-12",
        local_image_path=Path("/tmp/dasein_wallet.jpg"),
    )
    record = LocalizationArtifactRecord(
        id=seed.id,
        product_title=seed.product_title,
        source_page_url=seed.source_page_url,
        source_image_url=seed.source_image_url,
        local_image_path=seed.local_image_path,
        selected_phrase="wallet",
        selected_confidence=0.9,
        selected_box=None,
        overlay_path=None,
        crop_path=Path("/tmp/dasein_wallet.crop.png"),
        mask_path=Path("/tmp/dasein_wallet.mask.png"),
    )
    evidence = ObservedEvidenceSpec(
        color_note="the main visible body reads as blue with compatible printed accents in gray, beige",
        color_confidence=0.72,
        coverage_class="full_visible_surface_pattern",
        upper_component_state="absent",
        form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
        hard_facts=["the product remains a wallet"],
    )
    candidate = RetrievalCandidate(
        item_id="scene-wallet-1",
        image_name="scene_wallet.png",
        image_path=Path("/tmp/scene_wallet.png"),
        score=1.0,
        page_views=120,
        clicks=8,
        caption="a woman holding a patterned wallet in a hotel hallway",
        embedding=(0.95, 0.05),
        scenario_slots=("fashion_lifestyle",),
        style_atoms=("clear hero framing",),
        scene_families=("fashion_lifestyle",),
        support_relations=("carried_by_hand",),
        category="bag",
        canonical_product_type="wallet",
        observed_evidence=ObservedEvidenceSpec(
            coverage_class="full_visible_surface_pattern",
            palette=["blue", "gray"],
            upper_component_state="absent",
            form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
        ),
    )

    relaxed_filtered = filter_scene_retrieval_candidates(
        seed,
        record,
        [candidate],
        category="bag",
        canonical_product_type="wallet",
        support_mode="portable_flexible",
        query_embedding=np.asarray((1.0, 0.0), dtype=np.float32),
        evidence=evidence,
    )

    assert [row.item_id for row in relaxed_filtered] == ["scene-wallet-1"]


def test_correct_supported_soft_surface_inference_rewrites_false_localized_pattern() -> None:
    coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence = (
        correct_supported_soft_surface_inference(
            category="bedding",
            canonical_product_type="comforter",
            product_title="Dark Green Comforter Set",
            hint_phrases=("comforter", "bedding"),
            evidence_caption=None,
            coverage_class="localized_visible_pattern",
            coverage_ratio=0.34,
            coverage_note="a localized multicolor, printed, or contrast zone appears on one region of the visible product surface",
            pattern_note="the visible product surface includes multiple compatible colors on a black base",
            color_note="the main visible body reads as dark green with low-luster tonal variation from textured fabric",
            color_confidence=0.72,
            palette=["green", "black", "beige"],
        )
    )

    assert coverage_class == "low_variation_surface"
    assert coverage_ratio == 0.0
    assert pattern_note is None
    assert "tonal textile field" in str(coverage_note)
    assert "dark green" in str(color_note)
    assert color_confidence >= 0.72


def test_harmonize_supported_soft_structure_caps_contradictory_raised_relief() -> None:
    structure_class, note, confidence = harmonize_supported_soft_structure(
        category="bedding",
        canonical_product_type="comforter",
        edge_thickness_class="low_profile_edge",
        structure_class="raised_perimeter_relief",
        note="the visible soft product structure shows pronounced loft and raised edges rather than a tray-like boxed form",
        confidence=0.72,
    )

    assert structure_class == "low_perimeter_relief"
    assert "moderate loft" in str(note)
    assert confidence == 0.62


def test_assess_category_consistency_prefers_focus_crop(tmp_path: Path, monkeypatch) -> None:
    import product_campaign_pipeline.review_batch as rb

    full_path = tmp_path / "full.png"
    focus_path = tmp_path / "focus.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(full_path)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(focus_path)

    monkeypatch.setattr(
        rb,
        "CATEGORY_CLASSIFICATION_TEXTS",
        {
            "bag": ("bag",),
            "footwear": ("footwear",),
        },
    )

    class _Backbone:
        def encode_image(self, image_path: str | Path) -> np.ndarray:
            if Path(image_path).name == "focus.png":
                return np.asarray((1.0, 0.0), dtype=np.float32)
            return np.asarray((0.0, 1.0), dtype=np.float32)

        def encode_texts(self, prompts: list[str] | tuple[str, ...]) -> list[np.ndarray]:
            outputs: list[np.ndarray] = []
            for prompt in prompts:
                outputs.append(
                    np.asarray((1.0, 0.0), dtype=np.float32)
                    if prompt == "bag"
                    else np.asarray((0.0, 1.0), dtype=np.float32)
                )
            return outputs

    result = assess_category_consistency(
        full_path,
        expected_category="bag",
        expected_product_type="wallet",
        backbone=_Backbone(),
        focus_image_path=focus_path,
    )

    assert result["predicted_category"] == "bag"
    assert result["is_consistent"] is True


def test_semantic_support_margin_threshold_relaxes_for_compact_hand_focus() -> None:
    identity = ProductIdentitySpec(
        category="bag",
        canonical_product_type="wallet",
        interaction_mode="held_in_hand",
        observed_evidence=ObservedEvidenceSpec(
            upper_component_state="absent",
            form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
            aspect_ratio=1.2,
            color_note="the main visible body reads as blue",
            color_confidence=0.8,
        ),
    )

    assert semantic_support_margin_threshold(identity, support_relation="carried_by_hand") == -0.03
    assert semantic_support_margin_threshold(identity, support_relation="resting_on_surface") == -0.01


def test_soft_surface_gates_relax_false_negative_metric_failures() -> None:
    assert soft_surface_coverage_ok(
        base_ok=False,
        soft_textile_color_lock=True,
        low_profile_soft_lock=False,
        image_similarity=0.85,
        dominant_body_color_alignment=1.0,
        edge_profile_alignment=0.99,
        soft_structure_alignment=1.0,
    )
    assert soft_surface_value_ok(
        base_ok=False,
        soft_textile_color_lock=False,
        low_profile_soft_lock=True,
        image_similarity=0.74,
        dominant_body_color_alignment=0.58,
        dominant_body_value_alignment=0.0,
        edge_profile_alignment=0.95,
    )
    assert soft_surface_color_ok(
        base_ok=False,
        low_profile_soft_lock=True,
        image_similarity=0.74,
        dominant_body_color_alignment=0.58,
        edge_profile_alignment=0.95,
    )


def test_low_profile_soft_baseline_prefers_clarity_only() -> None:
    identity = ProductIdentitySpec(
        category="pet home",
        canonical_product_type="pet bed",
        support_mode="externally_supported_soft",
        rigid_vs_soft="soft",
        observed_evidence=ObservedEvidenceSpec(
            soft_structure_class="flat_surface",
            edge_thickness_class="thin",
        ),
    )

    assert select_reinvention_candidate_modes_for_line(identity, line_name="baseline") == ("clarity",)
    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("clarity", "balanced")


def test_low_profile_soft_business_prior_uses_clarity_only_when_artifact_flagged() -> None:
    identity = ProductIdentitySpec(
        category="pet home",
        canonical_product_type="pet bed",
        support_mode="externally_supported_soft",
        rigid_vs_soft="soft",
        observed_evidence=ObservedEvidenceSpec(
            soft_structure_class="flat_surface",
            edge_thickness_class="thin",
            uncertainty_level="high",
            artifact_flags=["border_text_overlay"],
        ),
    )

    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("clarity",)


def test_low_profile_soft_business_prior_keeps_balanced_candidate_for_border_foreground_intrusion() -> None:
    identity = ProductIdentitySpec(
        category="bedding",
        canonical_product_type="quilt",
        support_mode="externally_supported_soft",
        rigid_vs_soft="soft",
        observed_evidence=ObservedEvidenceSpec(
            soft_structure_class="flat_surface",
            edge_thickness_class="thin",
            uncertainty_level="medium",
            artifact_flags=["border_foreground_intrusion"],
        ),
    )

    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("clarity", "balanced")


def test_score_generation_candidate_strongly_penalizes_anatomy_and_evidence_failures() -> None:
    anatomy_weak = score_generation_candidate(
        category_consistency={"is_consistent": True},
        semantic_plausibility={
            "score": 0.64,
            "is_plausible": False,
            "human_supported": True,
            "anatomy_margin": -0.02,
        },
        evidence_consistency={"score": 0.53, "is_consistent": True},
    )
    clean = score_generation_candidate(
        category_consistency={"is_consistent": True},
        semantic_plausibility={
            "score": 0.64,
            "is_plausible": True,
            "human_supported": True,
            "anatomy_margin": 0.01,
        },
        evidence_consistency={"score": 0.53, "is_consistent": True},
    )
    evidence_bad = score_generation_candidate(
        category_consistency={"is_consistent": True},
        semantic_plausibility={
            "score": 0.64,
            "is_plausible": True,
            "human_supported": False,
            "anatomy_margin": 0.0,
        },
        evidence_consistency={"score": 0.41, "is_consistent": False},
    )

    assert anatomy_weak < clean
    assert evidence_bad < clean


def test_score_generation_candidate_penalizes_negative_casting_margin() -> None:
    casting_mismatch = score_generation_candidate(
        category_consistency={"is_consistent": True},
        semantic_plausibility={
            "score": 0.66,
            "is_plausible": False,
            "human_supported": True,
            "anatomy_margin": 0.02,
            "casting_margin": -0.03,
        },
        evidence_consistency={"score": 0.84, "is_consistent": True},
    )
    casting_aligned = score_generation_candidate(
        category_consistency={"is_consistent": True},
        semantic_plausibility={
            "score": 0.66,
            "is_plausible": True,
            "human_supported": True,
            "anatomy_margin": 0.02,
            "casting_margin": 0.04,
        },
        evidence_consistency={"score": 0.84, "is_consistent": True},
    )

    assert casting_aligned > casting_mismatch


def test_score_generation_candidate_penalizes_person_in_product_only_frame() -> None:
    contaminated = score_generation_candidate(
        category_consistency={"is_consistent": True},
        semantic_plausibility={
            "score": 0.74,
            "is_plausible": False,
            "human_supported": False,
            "people_out_of_frame_required": True,
            "person_presence_flag": True,
        },
        evidence_consistency={"score": 0.88, "is_consistent": True},
    )
    clean = score_generation_candidate(
        category_consistency={"is_consistent": True},
        semantic_plausibility={
            "score": 0.74,
            "is_plausible": True,
            "human_supported": False,
            "people_out_of_frame_required": True,
            "person_presence_flag": False,
        },
        evidence_consistency={"score": 0.88, "is_consistent": True},
    )

    assert clean > contaminated


def test_assess_semantic_plausibility_flags_person_when_product_only_is_required(tmp_path: Path) -> None:
    image_path = tmp_path / "quilt.png"
    Image.new("RGB", (256, 256), color="white").save(image_path)

    class _Backbone:
        def encode_image(self, image_path: str | Path) -> np.ndarray:
            return np.asarray((1.0, 0.0), dtype=np.float32)

        def encode_texts(self, prompts: list[str] | tuple[str, ...]) -> list[np.ndarray]:
            return [np.asarray((1.0, 0.0), dtype=np.float32) for _ in prompts]

    def fake_product_photo_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    class FakeLocalizer:
        def localize(self, photo):
            return SimpleNamespace(
                masks=[
                    MaskCandidate(
                        phrase=PhraseCandidate(text="person woman model", confidence=0.86, source="test"),
                        box=BoundingBox(x0=52, y0=22, x1=172, y1=242),
                        polygon=((52, 22), (172, 22), (172, 242), (52, 242)),
                        area_pixels=(172 - 52) * (242 - 22),
                        confidence=0.86,
                        source="test",
                    )
                ]
            )

    identity = ProductIdentitySpec(
        category="bedding",
        canonical_product_type="quilt",
        interaction_mode="placed",
        requires_human_model=False,
    )
    prompt_spec = FluxPromptSpec(
        subject="The featured quilt from image 1",
        action="Reinvent the product into a product-only campaign image with no people in frame.",
        style="premium commercial photography",
        context="Show the quilt resting naturally on furniture with no people, hands, sleeves, or body fragments visible.",
    )

    result = assess_semantic_plausibility(
        image_path,
        identity,
        prompt_spec=prompt_spec,
        scene_family="furnished_interior",
        support_relation="resting_on_surface",
        backbone=_Backbone(),
        generated_localizer=FakeLocalizer(),
        product_photo_factory=fake_product_photo_factory,
    )

    assert result["people_out_of_frame_required"] is True
    assert result["person_presence_flag"] is True
    assert result["is_plausible"] is False
    assert "product-only frame" in (result["warning"] or "")


def test_select_casting_alignment_eval_prompts_uses_feminine_wallet_prompt_family() -> None:
    identity = ProductIdentitySpec(
        phrase="blue floral wallet",
        category="bag",
        canonical_product_type="wallet",
        casting_note=(
            "If a person appears, use casting and hand styling that feel playful, soft, or feminine-coded in a way "
            "that matches the product's visual language."
        ),
    )

    prompts = select_casting_alignment_eval_prompts(identity, "wallet")

    assert prompts is not None
    assert any("feminine-coded" in prompt for prompt in prompts["positive"])
    assert any("masculine-coded" in prompt for prompt in prompts["negative"])


def test_compare_compact_product_prominence_prefers_larger_product_presence() -> None:
    assert compare_compact_product_prominence({"mask_area_ratio": 0.058, "bbox_area_ratio": 0.12}) == 1.0
    assert compare_compact_product_prominence({"mask_area_ratio": 0.012, "bbox_area_ratio": 0.03}) <= 0.4


def test_infer_soft_textile_chromatic_override_recovers_dark_green_bedding(tmp_path: Path) -> None:
    image_path = tmp_path / "comforter.png"
    mask_path = tmp_path / "comforter.mask.png"
    Image.new("RGB", (64, 64), (36, 48, 34)).save(image_path)
    Image.new("L", (64, 64), 255).save(mask_path)

    note, palette, confidence = infer_soft_textile_chromatic_override(
        source_image=image_path,
        mask_path=mask_path,
        category="bedding",
        canonical_product_type="comforter",
        coverage_class="low_variation_surface",
        palette=["black", "gray"],
    )

    assert note is not None
    assert "green" in note
    assert palette is not None and palette[0] == "green"
    assert confidence is not None and confidence >= 0.72


def test_infer_soft_textile_chromatic_override_does_not_reinterpret_near_black_apparel_as_blue(tmp_path: Path) -> None:
    image_path = tmp_path / "shirt.png"
    mask_path = tmp_path / "shirt.mask.png"
    image = Image.new("RGB", (72, 72), (24, 24, 28))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 71, 71), fill=(24, 24, 28))
    draw.rectangle((8, 8, 64, 64), fill=(30, 31, 35))
    image.save(image_path)
    Image.new("L", (72, 72), 255).save(mask_path)

    note, palette, confidence = infer_soft_textile_chromatic_override(
        source_image=image_path,
        mask_path=mask_path,
        category="apparel",
        canonical_product_type="shirt",
        coverage_class="low_variation_surface",
        palette=["black", "gray"],
    )

    assert note is None
    assert palette is None
    assert confidence is None


def test_soften_uncertain_neutral_apparel_color_evidence_removes_false_beige_lock(tmp_path: Path) -> None:
    image_path = tmp_path / "shirt.png"
    mask_path = tmp_path / "shirt.mask.png"
    image = Image.new("RGB", (72, 72), (230, 206, 184))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 66, 66), fill=(226, 205, 185))
    image.save(image_path)
    Image.new("L", (72, 72), 255).save(mask_path)

    note, confidence, palette = soften_uncertain_neutral_apparel_color_evidence(
        source_image=image_path,
        mask_path=mask_path,
        category="apparel",
        canonical_product_type="shirt",
        product_title="Perfect-T Women's T-Shirt",
        hint_phrases=("shirt", "t-shirt"),
        evidence_caption="the back of a shirt with a logo on it",
        coverage_class="low_variation_surface",
        color_note="the main visible body reads as beige with tonal variation from folds, pile, or soft-surface texture",
        color_confidence=0.76,
        palette=["beige", "gold", "orange"],
    )

    assert note is not None
    assert "light neutral fabric" in note
    assert confidence is not None and confidence <= 0.54
    assert palette == ["white", "beige", "gray"]


def test_suppress_border_attached_reference_artifacts_removes_text_overlay() -> None:
    source = np.full((64, 64, 3), (70, 140, 180), dtype=np.float32)
    source[48:62, 56:63] = (245, 245, 245)
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:60, 6:60] = True
    cleaned, flags = suppress_border_attached_reference_artifacts(
        source,
        mask,
        category="apparel",
        canonical_product_type="dress",
        requires_human_model=True,
    )

    assert "border_text_overlay" in flags
    assert cleaned.sum() < mask.sum()
    assert not cleaned[52:60, 57:63].any()


def test_suppress_border_attached_reference_artifacts_removes_skin_fragment() -> None:
    source = np.full((80, 96, 3), (28, 32, 34), dtype=np.float32)
    source[30:58, 0:22] = (214, 168, 144)
    mask = np.zeros((80, 96), dtype=bool)
    mask[10:72, 0:92] = True
    cleaned, flags = suppress_border_attached_reference_artifacts(
        source,
        mask,
        category="pet home",
        canonical_product_type="pet bed",
        requires_human_model=False,
    )

    assert "border_human_fragment" in flags
    assert cleaned.sum() < mask.sum()
    assert not cleaned[34:54, 0:18].any()


def test_suppress_border_attached_reference_artifacts_removes_soft_surface_foreground_intrusion() -> None:
    source = np.full((96, 128, 3), (214, 210, 198), dtype=np.float32)
    source[10:34, 12:76] = (198, 120, 54)
    source[66:94, 84:126] = (248, 248, 248)
    mask = np.zeros((96, 128), dtype=bool)
    mask[8:94, 6:126] = True

    cleaned, flags = suppress_border_attached_reference_artifacts(
        source,
        mask,
        category="bedding",
        canonical_product_type="quilt",
        requires_human_model=False,
    )

    assert "border_foreground_intrusion" in flags
    assert cleaned.sum() < mask.sum()
    assert not cleaned[14:30, 18:70].any()
    assert not cleaned[72:92, 92:122].any()


def test_repair_removed_reference_regions_inpaints_suppressed_artifact_pixels() -> None:
    source = np.full((48, 48, 3), (24, 24, 24), dtype=np.uint8)
    source[16:30, 0:12] = (220, 220, 220)
    keep_mask = np.ones((48, 48), dtype=bool)
    removed = np.zeros((48, 48), dtype=bool)
    removed[16:30, 0:12] = True
    keep_mask &= ~removed

    repaired = repair_removed_reference_regions(
        Image.fromarray(source),
        removed_region_mask=removed,
        keep_mask=keep_mask,
    )
    repaired_arr = np.asarray(repaired.convert("RGB"))

    assert repaired_arr[20, 4].mean() < 120
    assert not np.array_equal(repaired_arr[20, 4], source[20, 4])


def test_smooth_reference_export_mask_closes_border_notch_for_soft_goods() -> None:
    mask = np.zeros((64, 96), dtype=bool)
    mask[14:54, 8:88] = True
    notch = np.array([[24, 8], [18, 18], [22, 30], [30, 36], [34, 22]], dtype=np.int32)
    notch_mask = Image.new("L", (96, 64), 0)
    ImageDraw.Draw(notch_mask).polygon([tuple(point) for point in notch], fill=255)
    mask &= ~(np.asarray(notch_mask, dtype=np.uint8) > 0)

    smoothed = smooth_reference_export_mask(
        mask,
        category="pet home",
        canonical_product_type="pet bed",
        artifact_flags=["border_text_overlay"],
    )

    assert smoothed.sum() > mask.sum()
    assert smoothed[24:30, 12:20].any()


def test_repair_rigid_body_notches_fills_label_band_notch_for_drinkware() -> None:
    mask = np.zeros((128, 96), dtype=bool)
    mask[10:122, 26:74] = True
    notch = Image.new("L", (96, 128), 0)
    ImageDraw.Draw(notch).polygon([(26, 44), (36, 34), (42, 48), (38, 62), (26, 62)], fill=255)
    mask &= ~(np.asarray(notch, dtype=np.uint8) > 0)

    repaired = repair_rigid_body_notches(
        mask,
        category="drinkware",
        canonical_product_type="water bottle",
    )

    assert repaired.sum() > mask.sum()
    assert repaired[46:58, 28:36].any()


def test_compare_coverage_alignment_penalizes_invented_pattern_on_low_variation_soft_goods() -> None:
    evidence = ObservedEvidenceSpec(
        coverage_class="low_variation_surface",
        coverage_ratio=0.12,
    )
    generated_profile = {
        "coverage_class": "full_visible_surface_pattern",
        "coverage_ratio": 0.62,
    }

    score = compare_coverage_alignment(evidence, generated_profile)

    assert score < 0.2


def test_select_generated_focus_mask_reranks_soft_goods_away_from_small_bright_artifact(tmp_path: Path) -> None:
    image_path = tmp_path / "pet_bed_generated.png"
    image = Image.new("RGB", (160, 120), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 42, 142, 96), radius=18, fill=(26, 26, 28))
    draw.polygon([(62, 58), (84, 52), (104, 62), (98, 82), (70, 86)], fill=(232, 232, 232))
    image.save(image_path)

    phrase = PhraseCandidate(text="pet bed", confidence=0.9, source="test")
    shard = MaskCandidate(
        phrase=phrase,
        box=LocalizationBoundingBox(58, 52, 106, 88),
        polygon=((62, 58), (84, 52), (104, 62), (98, 82), (70, 86)),
        area_pixels=950,
        confidence=0.95,
        source="test",
    )
    bed = MaskCandidate(
        phrase=phrase,
        box=LocalizationBoundingBox(18, 40, 144, 98),
        polygon=((24, 44), (138, 44), (146, 66), (140, 96), (26, 98), (16, 72)),
        area_pixels=5600,
        confidence=0.8,
        source="test",
    )
    result = LocalizationResult(
        photo=ProductPhoto(
            image_path=image_path,
            product_id="pet_bed_generated",
            title="pet bed",
            hint_phrases=("pet bed",),
            metadata={"category": "pet home", "canonical_product_type": "pet bed"},
        ),
        phrases=(phrase,),
        proposals=(),
        masks=(shard, bed),
    )
    localized = LocalizedProduct(
        source_image=str(image_path),
        phrase="pet bed",
        bbox=BoundingBox(x0=18, y0=40, x1=144, y1=98),
        confidence=0.9,
        identity=ProductIdentitySpec(
            phrase="pet bed",
            category="pet home",
            canonical_product_type="pet bed",
            rigid_vs_soft="soft",
            observed_evidence=ObservedEvidenceSpec(
                palette=["black", "gray"],
                soft_structure_class="flat_surface",
                color_note="the main visible body reads as black",
            ),
        ),
    )

    selected = _select_generated_focus_mask(result, localized, default_selector=lambda _: shard)

    assert selected == bed


def test_infer_localized_contrast_panel_avoids_label_language(tmp_path: Path) -> None:
    image_path = tmp_path / "coffee_maker.png"
    mask_path = tmp_path / "coffee_maker.mask.png"
    image = Image.new("RGB", (96, 132), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 12, 78, 120), radius=10, fill=(26, 26, 28))
    draw.rounded_rectangle((26, 22, 70, 56), radius=6, fill=(210, 212, 214))
    image.save(image_path)
    mask = Image.new("L", (96, 132), 0)
    ImageDraw.Draw(mask).rounded_rectangle((18, 12, 78, 120), radius=10, fill=255)
    mask.save(mask_path)

    coverage_note, value_note = infer_localized_contrast_panel(
        image_path,
        mask_path,
        category="kitchen appliance",
        canonical_product_type="coffee maker",
        body_color="black",
    )

    assert coverage_note is not None and "label" not in coverage_note
    assert value_note is not None and "label" not in value_note


def test_infer_localized_contrast_panel_is_suppressed_for_apparel(tmp_path: Path) -> None:
    image_path, _, mask_path = _write_dark_labeled_bottle_artifacts(tmp_path)

    panel_note, value_note = infer_localized_contrast_panel(
        image_path,
        mask_path,
        category="apparel",
        canonical_product_type="dress",
        body_color="beige",
    )

    assert panel_note is None
    assert value_note is None


def test_assess_source_validity_marks_incomplete_blender_invalid(tmp_path: Path) -> None:
    image_path = tmp_path / "blender.png"
    mask_path = tmp_path / "blender.mask.png"
    Image.new("RGB", (96, 96), (40, 40, 40)).save(image_path)
    mask = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(mask).rectangle((10, 18, 88, 54), fill=255)
    mask.save(mask_path)

    validity, score, issues = assess_source_validity(
        source_image=image_path,
        mask_path=mask_path,
        category="kitchen appliance",
        canonical_product_type="blender",
        observed_evidence=ObservedEvidenceSpec(
            surface_scope="partial_or_occluded",
            uncertainty_level="high",
            aspect_ratio=0.82,
            artifact_flags=[],
        ),
        weak_shape_evidence=False,
    )

    assert validity == "invalid"
    assert score < 0.5
    assert "multipart_appliance_structure_incomplete" in issues


def test_assess_source_validity_marks_low_confidence_visual_type_conflict_invalid(tmp_path: Path) -> None:
    image_path = tmp_path / "mug.png"
    mask_path = tmp_path / "mug.mask.png"
    Image.new("RGB", (96, 96), (40, 40, 40)).save(image_path)
    mask = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(mask).rectangle((18, 18, 74, 82), fill=255)
    mask.save(mask_path)

    validity, score, issues = assess_source_validity(
        source_image=image_path,
        mask_path=mask_path,
        category="drinkware",
        canonical_product_type="mug",
        observed_evidence=ObservedEvidenceSpec(
            surface_scope="partial_or_occluded",
            uncertainty_level="high",
            raw_evidence_caption="the camera is shown in the image",
            evidence_caption=None,
            artifact_flags=[],
        ),
        weak_shape_evidence=True,
        localization_confidence=0.28,
    )

    assert validity == "invalid"
    assert score < 0.5
    assert "localized_crop_visual_type_conflict" in issues


def test_assess_source_validity_marks_competing_subobject_caption_invalid_for_structured_appliance(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "coffee_maker.png"
    mask_path = tmp_path / "coffee_maker.mask.png"
    Image.new("RGB", (96, 128), (40, 40, 40)).save(image_path)
    mask = Image.new("L", (96, 128), 0)
    ImageDraw.Draw(mask).rectangle((18, 12, 78, 120), fill=255)
    mask.save(mask_path)

    validity, score, issues = assess_source_validity(
        source_image=image_path,
        mask_path=mask_path,
        category="kitchen appliance",
        canonical_product_type="coffee maker",
        observed_evidence=ObservedEvidenceSpec(
            surface_scope="single_photo_limited",
            uncertainty_level="low",
            raw_evidence_caption="a cup of coffee",
            evidence_caption=None,
            artifact_flags=[],
            aspect_ratio=1.35,
            top_width_ratio=0.82,
        ),
        weak_shape_evidence=False,
        localization_confidence=0.82,
    )

    assert validity == "invalid"
    assert score < 0.5
    assert "localized_crop_visual_type_conflict" in issues


def test_assess_source_validity_marks_soft_surface_foreground_intrusion_conflict_invalid(tmp_path: Path) -> None:
    image_path = tmp_path / "quilt.png"
    mask_path = tmp_path / "quilt.mask.png"
    Image.new("RGB", (128, 96), (220, 214, 202)).save(image_path)
    mask = Image.new("L", (128, 96), 0)
    ImageDraw.Draw(mask).polygon([(0, 28), (90, 12), (127, 34), (127, 95), (0, 95)], fill=255)
    mask.save(mask_path)

    validity, score, issues = assess_source_validity(
        source_image=image_path,
        mask_path=mask_path,
        category="bedding",
        canonical_product_type="quilt",
        observed_evidence=ObservedEvidenceSpec(
            surface_scope="partial_or_occluded",
            uncertainty_level="medium",
            coverage_class="low_variation_surface",
            raw_evidence_caption="a map of the state of california",
            evidence_caption=None,
            artifact_flags=["border_foreground_intrusion"],
            aspect_ratio=0.7,
        ),
        weak_shape_evidence=False,
    )

    assert validity == "invalid"
    assert score < 0.5
    assert "source_contains_border_foreground_intrusion" in issues
    assert "localized_crop_visual_type_conflict" in issues


def test_assess_source_validity_marks_pet_home_animal_fragment_intrusion_invalid(tmp_path: Path) -> None:
    image_path = tmp_path / "pet_bed.png"
    mask_path = tmp_path / "pet_bed.mask.png"
    Image.new("RGB", (128, 96), (30, 30, 30)).save(image_path)
    mask = Image.new("L", (128, 96), 0)
    ImageDraw.Draw(mask).polygon([(0, 24), (86, 12), (127, 28), (127, 95), (0, 95)], fill=255)
    mask.save(mask_path)

    validity, score, issues = assess_source_validity(
        source_image=image_path,
        mask_path=mask_path,
        category="pet home",
        canonical_product_type="pet bed",
        observed_evidence=ObservedEvidenceSpec(
            surface_scope="partial_or_occluded",
            uncertainty_level="medium",
            coverage_class="low_variation_surface",
            raw_evidence_caption="a black cat with a white face and a black coat",
            evidence_caption=None,
            artifact_flags=["border_foreground_intrusion"],
            aspect_ratio=0.7,
        ),
        weak_shape_evidence=False,
    )

    assert validity == "invalid"
    assert score < 0.5
    assert "source_contains_border_foreground_intrusion" in issues
    assert "localized_crop_visual_type_conflict" in issues


def test_detect_compact_accessory_wardrobe_color_spill_flags_large_non_product_color_panel(tmp_path: Path) -> None:
    image_path = tmp_path / "wallet_scene.png"
    mask_path = tmp_path / "wallet.mask.png"

    image = Image.new("RGB", (128, 128), (232, 228, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((72, 10, 122, 118), fill=(47, 105, 176))
    draw.rectangle((38, 58, 74, 88), fill=(47, 105, 176))
    image.save(image_path)

    mask = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(mask).rectangle((38, 58, 74, 88), fill=255)
    mask.save(mask_path)

    localized = LocalizedProduct(
        source_image=str(image_path),
        phrase="blue floral wallet",
        bbox=BoundingBox(x0=38, y0=58, x1=74, y1=88),
        confidence=0.9,
        crop_path=str(image_path),
        identity=ProductIdentitySpec(
            phrase="blue floral wallet",
            category="bag",
            canonical_product_type="wallet",
            support_mode="portable_flexible",
            interaction_mode="held_in_hand",
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as blue with compatible printed accents in gray, beige",
                color_confidence=0.72,
                coverage_class="full_visible_surface_pattern",
                upper_component_state="absent",
                form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
                hard_facts=["the product remains a wallet"],
            ),
        ),
    )

    flagged, metrics = detect_compact_accessory_wardrobe_color_spill(
        image_path,
        focus_mask_path=mask_path,
        localized=localized,
    )

    assert flagged is True
    assert metrics["component_area_ratio"] > 1.4


def test_distinct_person_mask_count_flags_multiple_people_only_when_distinct() -> None:
    masks = [
        MaskCandidate(
            phrase=PhraseCandidate(text="person woman model", confidence=0.8, source="test"),
            box=BoundingBox(x0=10, y0=10, x1=80, y1=180),
            polygon=((10, 10), (80, 10), (80, 180), (10, 180)),
            area_pixels=(80 - 10) * (180 - 10),
            confidence=0.8,
            source="test",
        ),
        MaskCandidate(
            phrase=PhraseCandidate(text="woman model", confidence=0.72, source="test"),
            box=BoundingBox(x0=12, y0=12, x1=82, y1=182),
            polygon=((12, 12), (82, 12), (82, 182), (12, 182)),
            area_pixels=(82 - 12) * (182 - 12),
            confidence=0.72,
            source="test",
        ),
        MaskCandidate(
            phrase=PhraseCandidate(text="person man", confidence=0.78, source="test"),
            box=BoundingBox(x0=150, y0=24, x1=220, y1=176),
            polygon=((150, 24), (220, 24), (220, 176), (150, 176)),
            area_pixels=(220 - 150) * (176 - 24),
            confidence=0.78,
            source="test",
        ),
        MaskCandidate(
            phrase=PhraseCandidate(text="man", confidence=0.7, source="test"),
            box=BoundingBox(x0=240, y0=240, x1=248, y1=248),
            polygon=((240, 240), (248, 240), (248, 248), (240, 248)),
            area_pixels=(248 - 240) * (248 - 240),
            confidence=0.7,
            source="test",
        ),
    ]

    count, metrics = _distinct_person_mask_count(masks, image_area=256 * 256)

    assert count == 2
    assert metrics["person_candidate_count"] == 3.0
    assert metrics["distinct_person_count"] == 2.0


def test_identity_requires_people_out_of_frame_for_bedding() -> None:
    identity = ProductIdentitySpec(
        category="bedding",
        canonical_product_type="quilt",
        interaction_mode="placed",
        requires_human_model=False,
    )

    assert identity_requires_people_out_of_frame(identity) is True


def test_detect_any_person_in_scene_flags_single_person(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (256, 256), color="white").save(image_path)

    def fake_product_photo_factory(**kwargs):
        return SimpleNamespace(**kwargs)

    class FakeLocalizer:
        def localize(self, photo):
            return SimpleNamespace(
                masks=[
                    MaskCandidate(
                        phrase=PhraseCandidate(text="person woman model", confidence=0.82, source="test"),
                        box=BoundingBox(x0=32, y0=20, x1=160, y1=240),
                        polygon=((32, 20), (160, 20), (160, 240), (32, 240)),
                        area_pixels=(160 - 32) * (240 - 20),
                        confidence=0.82,
                        source="test",
                    )
                ]
            )

    flagged, metrics = detect_any_person_in_scene(
        image_path,
        generated_localizer=FakeLocalizer(),
        product_photo_factory=fake_product_photo_factory,
    )

    assert flagged is True
    assert metrics["distinct_person_count"] == 1.0


def test_infer_soft_structure_profile_detects_flat_pet_pad(tmp_path: Path) -> None:
    image_path = tmp_path / "flat_pad.png"
    mask_path = tmp_path / "flat_pad.mask.png"
    image = Image.new("RGB", (120, 80), (28, 28, 28))
    mask = Image.new("L", (120, 80), 0)
    ImageDraw.Draw(mask).rectangle((10, 12, 110, 68), fill=255)
    image.save(image_path)
    mask.save(mask_path)

    profile = infer_soft_structure_profile(
        source_image=image_path,
        mask_path=mask_path,
        category="pet home",
        canonical_product_type="pet bed",
        edge_thickness_class="low_profile_edge",
    )

    assert profile["structure_class"] == "flat_surface"
    assert "flat plush pad" in str(profile["note"])


def test_infer_soft_structure_profile_penalizes_raised_perimeter_against_flat_expectation(tmp_path: Path) -> None:
    image_path = tmp_path / "raised_bed.png"
    mask_path = tmp_path / "raised_bed.mask.png"
    image = Image.new("RGB", (120, 80), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 12, 110, 68), fill=(72, 94, 152))
    draw.rectangle((24, 24, 96, 56), fill=(24, 28, 42))
    mask = Image.new("L", (120, 80), 0)
    ImageDraw.Draw(mask).rectangle((10, 12, 110, 68), fill=255)
    image.save(image_path)
    mask.save(mask_path)

    profile = infer_soft_structure_profile(
        source_image=image_path,
        mask_path=mask_path,
        category="pet home",
        canonical_product_type="pet bed",
        edge_thickness_class="low_profile_edge",
    )

    assert profile["structure_class"] in {"low_perimeter_relief", "raised_perimeter_relief"}
    assert compare_soft_structure_alignment(
        ObservedEvidenceSpec(
            soft_structure_note="the visible soft product structure reads as a flat plush pad with no bulky bolster, boxed sidewall, or nested inner tray",
            soft_structure_class="flat_surface",
        ),
        {"soft_structure_class": profile["structure_class"]},
    ) <= 0.1


def test_select_reinvention_candidate_modes_prefers_non_balanced_for_low_profile_pet_bed() -> None:
    identity = ProductIdentitySpec(
        phrase="plush pet bed",
        category="pet home",
        canonical_product_type="pet bed",
        support_mode="externally_supported_soft",
        default_scene_family="furnished_interior",
        stable_base=False,
        rigid_vs_soft="soft",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            soft_structure_note="the visible soft product structure shows only a modest perimeter rise around the resting surface",
            soft_structure_class="low_perimeter_relief",
        ),
    )

    assert select_reinvention_candidate_modes(identity) == ("balanced", "clarity")


def test_select_reinvention_candidate_modes_prefers_clean_modes_for_compact_direct_grip() -> None:
    identity = ProductIdentitySpec(
        phrase="floral wallet",
        category="bag",
        canonical_product_type="wallet",
        interaction_mode="held_in_hand",
        support_mode="portable_flexible",
        default_scene_family="fashion_lifestyle",
        stable_base=False,
        rigid_vs_soft="semi-rigid",
        requires_human_model=True,
        observed_evidence=ObservedEvidenceSpec(
            upper_component_state="absent",
            form_factor_note="the visible bag form is compact and hand-held with no visible handles or shoulder straps",
        ),
    )

    assert select_reinvention_candidate_modes(identity) == ("clarity", "hero")
    assert select_reinvention_candidate_modes_for_line(identity, line_name="baseline") == ("clarity", "hero")
    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("clarity", "hero")
    assert should_use_reference_only_conditioning(identity) is True


def test_select_reinvention_candidate_modes_prefers_balanced_clarity_for_chromatic_soft_goods() -> None:
    identity = ProductIdentitySpec(
        phrase="dark green comforter",
        category="bedding",
        canonical_product_type="comforter",
        interaction_mode="placed",
        support_mode="externally_supported_soft",
        default_scene_family="furnished_interior",
        stable_base=False,
        rigid_vs_soft="soft",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as dark green with low-luster tonal variation from textured fabric",
            color_confidence=0.78,
            coverage_class="low_variation_surface",
            soft_structure_class="low_perimeter_relief",
        ),
    )

    assert identity_has_chromatic_soft_textile_lock(identity) is True
    assert identity_has_low_profile_soft_structure(identity) is True
    assert select_reinvention_candidate_modes(identity) == ("balanced", "clarity")
    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("clarity", "balanced")


def test_select_reinvention_candidate_modes_avoids_hero_for_rigid_placed_display_products() -> None:
    identity = ProductIdentitySpec(
        phrase="office chair",
        category="furniture",
        canonical_product_type="office chair",
        interaction_mode="placed",
        support_mode="self_supporting_display",
        default_scene_family="editorial_interior",
        stable_base=True,
        rigid_vs_soft="rigid",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as black with reflective highlight variation",
            color_confidence=0.82,
            coverage_class="low_variation_surface",
        ),
    )

    assert select_reinvention_candidate_modes(identity) == ("balanced", "reveal")
    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("balanced", "reveal")


def test_select_reinvention_candidate_modes_for_open_frame_folding_chair_prefers_clarity() -> None:
    identity = ProductIdentitySpec(
        phrase="folding chair",
        category="furniture",
        canonical_product_type="folding chair",
        interaction_mode="placed",
        support_mode="self_supporting_display",
        default_scene_family="editorial_interior",
        stable_base=True,
        rigid_vs_soft="rigid",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as black with reflective highlight variation",
            color_confidence=0.82,
            coverage_class="low_variation_surface",
        ),
    )

    assert select_reinvention_candidate_modes(identity) == ("balanced", "reveal")
    assert select_reinvention_candidate_modes_for_line(identity, line_name="baseline") == ("clarity", "reveal")
    assert select_reinvention_candidate_modes_for_line(identity, line_name="business_prior") == ("clarity", "reveal")


def test_should_prefer_crop_only_color_lock_for_chromatic_rigid_display_product() -> None:
    identity = ProductIdentitySpec(
        phrase="blue toaster",
        category="kitchen appliance",
        canonical_product_type="toaster",
        interaction_mode="placed",
        support_mode="self_supporting_display",
        default_scene_family="tabletop_display",
        stable_base=True,
        rigid_vs_soft="rigid",
        requires_human_model=False,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as blue with cool-toned reflective variation",
            color_confidence=0.72,
            coverage_class="localized_visible_pattern",
            reference_crop_path="/tmp/toaster.crop.png",
            palette=["blue", "gray", "black"],
        ),
    )

    assert should_strengthen_dominant_body_color_guidance(identity) is True
    assert should_prefer_crop_only_color_lock(identity) is True


def test_primary_generation_input_prefers_source_frame_for_chromatic_soft_goods(tmp_path: Path) -> None:
    source = tmp_path / "comforter.png"
    crop = tmp_path / "comforter.crop.png"
    cutout = tmp_path / "comforter.cutout.png"
    for path in (source, crop, cutout):
        Image.new("RGB", (64, 64), (10, 80, 30)).save(path)

    localized = LocalizedProduct(
        source_image=str(source),
        phrase="dark green comforter",
        bbox=BoundingBox(x0=0, y0=0, x1=64, y1=64),
        confidence=0.9,
        crop_path=str(crop),
        identity=ProductIdentitySpec(
            phrase="dark green comforter",
            category="bedding",
            canonical_product_type="comforter",
            support_mode="externally_supported_soft",
            default_scene_family="furnished_interior",
            interaction_mode="placed",
            rigid_vs_soft="soft",
            observed_evidence=ObservedEvidenceSpec(
                color_note="the main visible body reads as dark green with low-luster tonal variation from textured fabric",
                color_confidence=0.8,
                coverage_class="low_variation_surface",
                soft_structure_class="low_perimeter_relief",
                reference_crop_path=str(crop),
                reference_cutout_path=str(cutout),
            ),
        ),
    )

    assert _primary_generation_input_image(localized) == source


def test_build_generation_request_skips_primary_input_for_reference_only_conditioning(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    crop = tmp_path / "crop.png"
    silhouette = tmp_path / "silhouette.png"
    for path, color in ((source, (240, 240, 240)), (crop, (12, 100, 180)), (silhouette, (0, 0, 0))):
        Image.new("RGB", (64, 64), color).save(path)
    client = Flux2KleinClient(pipeline_factory=object())
    prompt = FluxPromptSpec(
        subject="wallet",
        action="hand-focused campaign image",
        style="studio",
        context="close hand-focused composition",
        preservation_constraints=["preserve the product"],
        seed=5,
    )

    request = build_generation_request(
        client,
        prompt,
        source_image=source,
        reference_images=[crop, silhouette],
        primary_input_image=crop,
        allow_reference_only=True,
        output_path=tmp_path / "out.png",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=1.0,
    )

    assert request.input_images == (str(crop), str(silhouette))


def test_detect_human_ghost_composite_artifact_flags_washed_out_subject(tmp_path: Path) -> None:
    image_path = tmp_path / "ghosted.png"
    image = Image.new("RGB", (200, 260), (230, 220, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((62, 32, 138, 238), fill=(165, 165, 165))
    draw.ellipse((72, 18, 128, 72), fill=(175, 175, 175))
    image.save(image_path)

    flagged, metrics = detect_human_ghost_composite_artifact(image_path)

    assert flagged is True
    assert metrics["central_saturation_mean"] < 0.12


def test_detect_human_ghost_composite_artifact_ignores_regular_portrait(tmp_path: Path) -> None:
    image_path = tmp_path / "portrait.png"
    image = Image.new("RGB", (200, 260), (240, 235, 228))
    draw = ImageDraw.Draw(image)
    draw.rectangle((62, 32, 138, 238), fill=(110, 145, 210))
    draw.ellipse((72, 18, 128, 72), fill=(208, 166, 132))
    image.save(image_path)

    flagged, _ = detect_human_ghost_composite_artifact(image_path)

    assert flagged is False


def test_detect_human_ghost_composite_artifact_flags_low_saturation_midtone_subject(tmp_path: Path) -> None:
    image_path = tmp_path / "ghosted_midtone.png"
    image = Image.new("RGB", (256, 256), (244, 238, 232))
    draw = ImageDraw.Draw(image)
    draw.rectangle((82, 44, 174, 232), fill=(142, 148, 152))
    draw.ellipse((96, 24, 160, 82), fill=(154, 156, 160))
    draw.rectangle((116, 122, 140, 150), fill=(58, 118, 176))
    image.save(image_path)

    flagged, metrics = detect_human_ghost_composite_artifact(image_path)

    assert flagged is True
    assert metrics["central_saturation_mean"] < 0.18


def test_detect_background_collapse_artifact_flags_flat_gray_scene(tmp_path: Path) -> None:
    image_path = tmp_path / "gray_scene.png"
    image = Image.new("RGB", (256, 256), (214, 214, 214))
    draw = ImageDraw.Draw(image)
    draw.rectangle((84, 46, 172, 236), fill=(150, 92, 82))
    draw.ellipse((104, 24, 152, 72), fill=(222, 186, 160))
    image.save(image_path)

    flagged, metrics = detect_background_collapse_artifact(image_path)

    assert flagged is True
    assert metrics["border_saturation_mean"] < 0.09


def test_detect_background_collapse_artifact_ignores_warm_resolved_interior(tmp_path: Path) -> None:
    image_path = tmp_path / "resolved_interior.png"
    image = Image.new("RGB", (256, 256), (232, 220, 202))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 176, 256, 256), fill=(118, 83, 53))
    draw.rectangle((0, 0, 54, 160), fill=(196, 180, 156))
    draw.rectangle((84, 46, 172, 236), fill=(30, 34, 42))
    draw.ellipse((104, 24, 152, 72), fill=(222, 186, 160))
    draw.rectangle((188, 58, 236, 156), fill=(142, 168, 92))
    image.save(image_path)

    flagged, metrics = detect_background_collapse_artifact(image_path)

    assert flagged is False
    assert metrics["border_neutral_fraction"] < 0.84


def test_color_anchor_asset_not_used_for_footwear() -> None:
    assert _should_prepare_color_anchor_asset(
        category="footwear",
        canonical_product_type="shoe",
        soft_structure_class=None,
    ) is False


def test_post_generation_color_repair_allows_carried_bags_and_worn_footwear() -> None:
    bag_identity = ProductIdentitySpec(
        category="bag",
        canonical_product_type="tote bag",
        interaction_mode="carried_or_resting",
        rigid_vs_soft="semi-rigid",
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as blue",
            color_confidence=0.82,
            coverage_class="low_variation_surface",
            reference_crop_path="/tmp/bag.crop.png",
        ),
    )
    footwear_identity = ProductIdentitySpec(
        category="footwear",
        canonical_product_type="shoe",
        interaction_mode="worn",
        rigid_vs_soft="semi-rigid",
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as white",
            color_confidence=0.82,
            coverage_class="low_variation_surface",
            reference_crop_path="/tmp/shoe.crop.png",
        ),
    )
    apparel_identity = ProductIdentitySpec(
        category="apparel",
        canonical_product_type="dress",
        interaction_mode="worn",
        rigid_vs_soft="soft",
        requires_human_model=True,
        observed_evidence=ObservedEvidenceSpec(
            color_note="the main visible body reads as red",
            color_confidence=0.82,
            coverage_class="low_variation_surface",
            reference_crop_path="/tmp/dress.crop.png",
        ),
    )

    assert should_apply_post_generation_color_repair(bag_identity) is True
    assert should_apply_post_generation_color_repair(footwear_identity) is True
    assert should_apply_post_generation_color_repair(apparel_identity) is False


def test_localization_crop_reference_preferred_when_apparel_evidence_crop_is_truncated(tmp_path: Path) -> None:
    localization_crop = tmp_path / "dress.localization.png"
    evidence_crop = tmp_path / "dress.evidence.png"
    Image.new("RGB", (160, 320), (240, 220, 210)).save(localization_crop)
    Image.new("RGB", (124, 146), (236, 214, 204)).save(evidence_crop)

    assert (
        _should_prefer_localization_crop_reference(
            localization_crop_path=localization_crop,
            evidence_crop_path=evidence_crop,
            category="apparel",
            canonical_product_type="dress",
            artifact_flags=(),
            structure_completeness=1.0,
        )
        is True
    )


def test_localization_crop_reference_preferred_for_artifact_prone_bag_crop(tmp_path: Path) -> None:
    localization_crop = tmp_path / "bag.localization.png"
    evidence_crop = tmp_path / "bag.evidence.png"
    Image.new("RGB", (400, 300), (200, 160, 220)).save(localization_crop)
    Image.new("RGB", (400, 300), (188, 150, 212)).save(evidence_crop)

    assert (
        _should_prefer_localization_crop_reference(
            localization_crop_path=localization_crop,
            evidence_crop_path=evidence_crop,
            category="bag",
            canonical_product_type="tote bag",
            artifact_flags=("border_text_overlay",),
            structure_completeness=0.88,
        )
        is True
    )


def test_prepare_observed_evidence_assets_drops_mask_conditioning_when_localization_crop_is_preferred(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "dress.source.jpg"
    mask_path = tmp_path / "dress.source.mask.png"
    localization_crop = tmp_path / "dress.source.crop.png"
    Image.new("RGB", (120, 120), (240, 220, 210)).save(source_path)
    Image.new("RGB", (160, 320), (240, 220, 210)).save(localization_crop)
    mask = Image.new("L", (120, 120), 0)
    ImageDraw.Draw(mask).rectangle((40, 48, 82, 110), fill=255)
    mask.save(mask_path)

    prepared = prepare_observed_evidence_assets(
        source_image=source_path,
        mask_path=mask_path,
        localization_crop_path=localization_crop,
        category="apparel",
        canonical_product_type="dress",
        requires_human_model=True,
    )

    assert prepared.crop_path == localization_crop
    assert prepared.cutout_path is None
    assert prepared.silhouette_path is None


def test_soft_textile_chromatic_override_recovers_dark_olive_bedding(tmp_path: Path) -> None:
    source_path = tmp_path / "comforter.png"
    mask_path = tmp_path / "comforter.mask.png"
    image = Image.new("RGB", (96, 96), (52, 50, 38))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 88, 88), fill=(78, 76, 52))
    image.save(source_path)
    Image.new("L", (96, 96), 255).save(mask_path)

    note, palette, confidence = infer_soft_textile_chromatic_override(
        source_image=source_path,
        mask_path=mask_path,
        category="bedding",
        canonical_product_type="comforter",
        coverage_class="low_variation_surface",
        palette=["black", "gray", "brown"],
    )

    assert note is not None
    assert "green" in note
    assert palette is not None and palette[0] == "green"
    assert confidence is not None and confidence >= 0.7
