from __future__ import annotations

from pathlib import Path

from PIL import Image

from product_campaign_pipeline.localization import (
    ProductLocalizationPipeline,
    save_localization_artifacts,
    select_primary_mask,
)
from product_campaign_pipeline.localization.models import (
    BoundingBox,
    LocalizationResult,
    MaskCandidate,
    PhraseCandidate,
    ProductPhoto,
)


def test_placeholder_localization_pipeline_runs_without_model_weights(tmp_path: Path) -> None:
    image_path = tmp_path / "sparkling_water_bottle.png"
    Image.new("RGB", (320, 240), (245, 245, 245)).save(image_path)

    pipeline = ProductLocalizationPipeline()
    result = pipeline.localize(
        ProductPhoto(
            image_path=image_path,
            product_id="sku-55",
            title="Sparkling Water Bottle",
            hint_phrases=("label",),
        )
    )

    assert result.photo.product_id == "sku-55"
    assert [phrase.text for phrase in result.phrases] == [
        "sparkling water bottle",
        "label",
        "sparkling water bottle",
        "product",
    ][: len(result.phrases)]
    assert len(result.proposals) == len(result.phrases)
    assert len(result.masks) == len(result.proposals)
    assert all(mask.area_pixels > 0 for mask in result.masks)
    assert result.proposals[0].box.width > 0


def test_localization_artifacts_are_written_for_selected_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "sparkling_water_bottle.png"
    Image.new("RGB", (320, 240), (245, 245, 245)).save(image_path)

    result = ProductLocalizationPipeline().localize(
        ProductPhoto(
            image_path=image_path,
            product_id="sku-55",
            title="Sparkling Water Bottle",
            hint_phrases=("label",),
        )
    )

    selected = select_primary_mask(result)
    artifacts = save_localization_artifacts(result, tmp_path / "artifacts", selected_mask=selected)

    assert artifacts is not None
    assert Path(artifacts.crop_path).exists()
    assert Path(artifacts.mask_path).exists()
    assert Path(artifacts.overlay_path).exists()


def test_select_primary_mask_uses_type_aware_reranking_for_footwear(tmp_path: Path) -> None:
    image_path = tmp_path / "easy_spirit.png"
    Image.new("RGB", (320, 240), (245, 245, 245)).save(image_path)

    photo = ProductPhoto(
        image_path=image_path,
        product_id="easy-spirit-01",
        title="Easy Spirit Womens Pippa Lace-Up Sneaker",
        hint_phrases=("shoe", "walking shoe"),
    )
    wire_phrase = PhraseCandidate(text="green wire", confidence=0.88, source="test")
    shoe_phrase = PhraseCandidate(text="walking shoe", confidence=0.74, source="test")
    result = LocalizationResult(
        photo=photo,
        phrases=(wire_phrase, shoe_phrase),
        proposals=(),
        masks=(
            MaskCandidate(
                phrase=wire_phrase,
                box=BoundingBox(x0=140, y0=18, x1=152, y1=224),
                polygon=((140, 18), (152, 18), (152, 224), (140, 224)),
                area_pixels=2472,
                confidence=0.88,
                source="test",
            ),
            MaskCandidate(
                phrase=shoe_phrase,
                box=BoundingBox(x0=48, y0=108, x1=260, y1=214),
                polygon=((48, 108), (260, 108), (260, 214), (48, 214)),
                area_pixels=22472,
                confidence=0.74,
                source="test",
            ),
        ),
    )

    selected = select_primary_mask(result)

    assert selected is not None
    assert selected.phrase.text == "walking shoe"


def test_select_primary_mask_prefers_large_bedding_surface_for_comforter(tmp_path: Path) -> None:
    image_path = tmp_path / "comforter.png"
    Image.new("RGB", (320, 240), (245, 245, 245)).save(image_path)

    photo = ProductPhoto(
        image_path=image_path,
        product_id="comforter-01",
        title="Bare Home Down Alternative Comforter",
        hint_phrases=("comforter", "bedding"),
    )
    narrow_phrase = PhraseCandidate(text="bare home ultra soft 1800 collection", confidence=0.82, source="test")
    bedding_phrase = PhraseCandidate(text="comforter bedding", confidence=0.71, source="test")
    result = LocalizationResult(
        photo=photo,
        phrases=(narrow_phrase, bedding_phrase),
        proposals=(),
        masks=(
            MaskCandidate(
                phrase=narrow_phrase,
                box=BoundingBox(x0=0, y0=0, x1=320, y1=42),
                polygon=((0, 0), (320, 0), (320, 42), (0, 42)),
                area_pixels=13440,
                confidence=0.82,
                source="test",
            ),
            MaskCandidate(
                phrase=bedding_phrase,
                box=BoundingBox(x0=8, y0=52, x1=312, y1=228),
                polygon=((8, 52), (312, 52), (312, 228), (8, 228)),
                area_pixels=53504,
                confidence=0.71,
                source="test",
            ),
        ),
    )

    selected = select_primary_mask(result)

    assert selected is not None
    assert selected.phrase.text == "comforter bedding"


def test_select_primary_mask_prefers_complete_blender_over_jar_only_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "blender.png"
    Image.new("RGB", (320, 320), (245, 245, 245)).save(image_path)

    photo = ProductPhoto(
        image_path=image_path,
        product_id="blender-02",
        title="Mainstays 6-Speed Blender With Jar",
        hint_phrases=("blender", "kitchen appliance"),
        metadata={"canonical_product_type": "blender", "category": "kitchen appliance"},
    )
    full_phrase = PhraseCandidate(text="blender", confidence=0.72, source="test")
    jar_phrase = PhraseCandidate(text="blender jar", confidence=0.9, source="test")
    result = LocalizationResult(
        photo=photo,
        phrases=(full_phrase, jar_phrase),
        proposals=(),
        masks=(
            MaskCandidate(
                phrase=full_phrase,
                box=BoundingBox(x0=92, y0=38, x1=226, y1=286),
                polygon=((110, 38), (208, 38), (226, 188), (196, 286), (122, 286), (92, 188)),
                area_pixels=21400,
                confidence=0.72,
                source="test",
            ),
            MaskCandidate(
                phrase=jar_phrase,
                box=BoundingBox(x0=102, y0=34, x1=214, y1=222),
                polygon=((112, 34), (204, 34), (214, 170), (176, 222), (140, 222), (102, 170)),
                area_pixels=15200,
                confidence=0.9,
                source="test",
            ),
        ),
    )

    selected = select_primary_mask(result)

    assert selected is not None
    assert selected.phrase.text == "blender"


def test_save_localization_artifacts_expands_incomplete_blender_crop_for_missing_base(tmp_path: Path) -> None:
    image_path = tmp_path / "blender_source.png"
    Image.new("RGB", (400, 300), (245, 245, 245)).save(image_path)

    photo = ProductPhoto(
        image_path=image_path,
        product_id="blender-03",
        title="Mainstays 6-Speed Blender With Jar",
        hint_phrases=("blender", "kitchen appliance"),
        metadata={"canonical_product_type": "blender", "category": "kitchen appliance"},
    )
    jar_phrase = PhraseCandidate(text="blender jar", confidence=0.92, source="test")
    result = LocalizationResult(
        photo=photo,
        phrases=(jar_phrase,),
        proposals=(),
        masks=(
            MaskCandidate(
                phrase=jar_phrase,
                box=BoundingBox(x0=92, y0=24, x1=214, y1=214),
                polygon=((112, 24), (204, 24), (214, 168), (176, 214), (140, 214), (92, 168)),
                area_pixels=14600,
                confidence=0.92,
                source="test",
            ),
        ),
    )

    artifacts = save_localization_artifacts(result, tmp_path / "artifacts")

    assert artifacts is not None
    with Image.open(artifacts.crop_path) as crop:
        assert crop.size[0] > 220
        assert crop.size[1] > 240


def test_select_primary_mask_prefers_complete_coffee_maker_over_carafe_only_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "coffee_maker.png"
    Image.new("RGB", (320, 320), (245, 245, 245)).save(image_path)

    photo = ProductPhoto(
        image_path=image_path,
        product_id="coffee-maker-01",
        title="Mainstays Black 12-Cup Drip Coffee Maker",
        hint_phrases=("coffee maker", "kitchen appliance"),
        metadata={"canonical_product_type": "coffee maker", "category": "kitchen appliance"},
    )
    full_phrase = PhraseCandidate(text="coffee maker", confidence=0.72, source="test")
    carafe_phrase = PhraseCandidate(text="coffee maker carafe", confidence=0.9, source="test")
    result = LocalizationResult(
        photo=photo,
        phrases=(full_phrase, carafe_phrase),
        proposals=(),
        masks=(
            MaskCandidate(
                phrase=full_phrase,
                box=BoundingBox(x0=88, y0=44, x1=232, y1=294),
                polygon=((96, 44), (224, 44), (232, 248), (204, 294), (116, 294), (88, 248)),
                area_pixels=23800,
                confidence=0.72,
                source="test",
            ),
            MaskCandidate(
                phrase=carafe_phrase,
                box=BoundingBox(x0=118, y0=132, x1=222, y1=292),
                polygon=((124, 132), (214, 132), (222, 244), (198, 292), (138, 292), (118, 244)),
                area_pixels=13600,
                confidence=0.9,
                source="test",
            ),
        ),
    )

    selected = select_primary_mask(result)

    assert selected is not None
    assert selected.phrase.text == "coffee maker"


def test_select_primary_mask_penalizes_generic_brand_phrase_for_multipart_appliance(tmp_path: Path) -> None:
    image_path = tmp_path / "slow_cooker.png"
    Image.new("RGB", (360, 360), (245, 245, 245)).save(image_path)

    photo = ProductPhoto(
        image_path=image_path,
        product_id="slow-cooker-01",
        title="The Pioneer Woman 6 Qt Slow Cooker",
        hint_phrases=("slow cooker", "kitchen appliance", "crock pot"),
        metadata={"canonical_product_type": "slow cooker", "category": "kitchen appliance"},
    )
    cooker_phrase = PhraseCandidate(text="slow cooker", confidence=0.62, source="test")
    brand_phrase = PhraseCandidate(text="the pioneer woman", confidence=0.92, source="test")
    result = LocalizationResult(
        photo=photo,
        phrases=(cooker_phrase, brand_phrase),
        proposals=(),
        masks=(
            MaskCandidate(
                phrase=cooker_phrase,
                box=BoundingBox(x0=82, y0=74, x1=278, y1=318),
                polygon=((96, 74), (264, 74), (278, 132), (270, 318), (90, 318), (82, 132)),
                area_pixels=37200,
                confidence=0.62,
                source="test",
            ),
            MaskCandidate(
                phrase=brand_phrase,
                box=BoundingBox(x0=0, y0=0, x1=356, y1=352),
                polygon=((0, 0), (356, 0), (356, 352), (0, 352)),
                area_pixels=125312,
                confidence=0.92,
                source="test",
            ),
        ),
    )

    selected = select_primary_mask(result)

    assert selected is not None
    assert selected.phrase.text == "slow cooker"
