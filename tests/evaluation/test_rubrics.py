from pathlib import Path

from PIL import Image

from product_campaign_pipeline.composer import PromptComposer
from product_campaign_pipeline.evaluation import evaluate_output, evaluate_prompt
from product_campaign_pipeline.types import BoundingBox, LocalizedProduct, ProductIdentitySpec


def _localized_product(path: str) -> LocalizedProduct:
    return LocalizedProduct(
        source_image=path,
        phrase="featured bottle",
        bbox=BoundingBox(x0=10, y0=10, x1=90, y1=90),
        confidence=0.7,
        identity=ProductIdentitySpec(phrase="featured bottle", category="bottle"),
    )


def test_evaluate_prompt_scores_reasonably(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 100), (180, 90, 60)).save(source)
    prompt = PromptComposer().compose_baseline(_localized_product(str(source)), seed=1)
    evaluation = evaluate_prompt(prompt)
    assert evaluation.guideline_alignment >= 4.0
    assert evaluation.preservation_conformity >= 4.0


def test_evaluate_output_returns_scores(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    generated = tmp_path / "generated.png"
    Image.new("RGB", (64, 64), (120, 90, 80)).save(source)
    Image.new("RGB", (64, 64), (124, 88, 84)).save(generated)

    evaluation = evaluate_output(source, generated)
    assert 1.0 <= evaluation.product_preservation <= 5.0
    assert 1.0 <= evaluation.aesthetic_performance <= 5.0
