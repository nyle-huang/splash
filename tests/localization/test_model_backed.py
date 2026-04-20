from __future__ import annotations

from product_campaign_pipeline.localization.model_backed import _build_grounding_prompt, _normalize_grounding_phrase
from product_campaign_pipeline.localization.models import PhraseCandidate


def test_build_grounding_prompt_flattens_and_dedupes_phrase_text() -> None:
    phrases = (
        PhraseCandidate(text="Walking Shoe", confidence=0.9, source="hint"),
        PhraseCandidate(text=" walking shoe. ", confidence=0.8, source="title"),
        PhraseCandidate(text="Easy Spirit sneaker", confidence=0.7, source="caption"),
    )

    prompt = _build_grounding_prompt(phrases)

    assert prompt == "walking shoe . easy spirit sneaker ."


def test_normalize_grounding_phrase_removes_trailing_periods_and_extra_spaces() -> None:
    assert _normalize_grounding_phrase("  Office   Chair. ") == "office chair"
