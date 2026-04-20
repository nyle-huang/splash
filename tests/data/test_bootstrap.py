from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from product_campaign_pipeline.data import (
    ManifestRow,
    PseudoHandheldBootstrapper,
    SyntheticBootstrapConfig,
    generate_bootstrap_examples,
)


def _make_product_cutout(path: Path) -> None:
    image = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 16, 112, 124), radius=20, fill=(235, 88, 52, 255))
    draw.rounded_rectangle((46, 34, 94, 58), radius=8, fill=(252, 252, 252, 235))
    draw.rectangle((52, 64, 88, 102), fill=(255, 218, 48, 255))
    image.save(path)


def test_pseudo_handheld_bootstrapper_generates_images_and_records(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    source_path = image_root / "product.png"
    _make_product_cutout(source_path)

    bootstrapper = PseudoHandheldBootstrapper(canvas_size=(256, 256), seed=11)
    rows = [ManifestRow(item_id="sku-1", image_name="product.png", ds=0, pv=9, clk=2)]

    records = bootstrapper.generate_from_manifest_rows(
        rows,
        image_root=image_root,
        output_dir=tmp_path / "synthetic",
        split_name="train",
        variants_per_image=2,
    )

    assert len(records) == 2
    assert records[0].source_item_id == "sku-1"
    assert records[0].synthetic_image_name != records[1].synthetic_image_name

    for record in records:
        assert record.synthetic_image_path.exists()
        image = Image.open(record.synthetic_image_path)
        assert image.size == (256, 256)
        assert record.output_width == 256
        assert record.output_height == 256
        assert 0.0 <= record.blur_radius <= 1.0


def test_generate_bootstrap_examples_uses_config_defaults(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    source_path = image_root / "detail.png"
    _make_product_cutout(source_path)

    rows = [ManifestRow(item_id="sku-2", image_name="detail.png", ds=0, pv=4, clk=1)]
    config = SyntheticBootstrapConfig(canvas_size=(192, 192), seed=7, split_name="val", variants_per_image=1)

    records = generate_bootstrap_examples(
        rows,
        image_root=image_root,
        output_dir=tmp_path / "synthetic",
        config=config,
    )

    assert len(records) == 1
    assert records[0].split_name == "val"
    assert records[0].output_width == 192
    assert records[0].synthetic_image_path.exists()
