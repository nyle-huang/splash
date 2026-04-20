from __future__ import annotations

import re
from typing import Protocol, Sequence

from PIL import Image

from .models import BoundingBox, LocalizationResult, MaskCandidate, PhraseCandidate, ProductPhoto, RegionProposal


class PhraseExtractionAdapter(Protocol):
    def extract_phrases(self, photo: ProductPhoto) -> Sequence[PhraseCandidate]:
        ...


class GroundingAdapter(Protocol):
    def propose_regions(
        self, photo: ProductPhoto, phrases: Sequence[PhraseCandidate]
    ) -> Sequence[RegionProposal]:
        ...


class MaskRefinementAdapter(Protocol):
    def refine_masks(
        self, photo: ProductPhoto, proposals: Sequence[RegionProposal]
    ) -> Sequence[MaskCandidate]:
        ...


class PlaceholderVlmPhraseExtractor:
    """Deterministic placeholder for a future VLM phrase extractor."""

    def __init__(self, *, max_phrases: int = 4) -> None:
        self.max_phrases = max_phrases

    def extract_phrases(self, photo: ProductPhoto) -> Sequence[PhraseCandidate]:
        phrases: list[str] = []
        if photo.title:
            phrases.append(photo.title.strip().lower())
        phrases.extend(phrase.strip().lower() for phrase in photo.hint_phrases if phrase.strip())

        stem_tokens = [
            token
            for token in re.split(r"[^a-zA-Z0-9]+", photo.image_path.stem.lower())
            if len(token) > 2 and not token.isdigit()
        ]
        if stem_tokens:
            phrases.append(" ".join(stem_tokens[:3]))

        phrases.extend(["product", "package", "brand mark"])

        bounded_phrases = phrases[: self.max_phrases]
        return tuple(
            PhraseCandidate(
                text=text, confidence=max(0.35, 0.92 - index * 0.12), source="placeholder_vlm"
            )
            for index, text in enumerate(bounded_phrases)
        )


class PlaceholderGroundingDinoProposer:
    """Geometry-only stand-in for Grounding DINO proposals."""

    def propose_regions(
        self, photo: ProductPhoto, phrases: Sequence[PhraseCandidate]
    ) -> Sequence[RegionProposal]:
        with Image.open(photo.image_path) as handle:
            width, height = handle.size

        if not phrases:
            return ()

        base_width = max(24, int(width * 0.58))
        base_height = max(24, int(height * 0.72))
        center_x = width / 2.0
        center_y = height / 2.0

        proposals: list[RegionProposal] = []
        for index, phrase in enumerate(phrases):
            horizontal_shift = int((index - (len(phrases) - 1) / 2.0) * width * 0.05)
            vertical_shift = int(((-1) ** index) * height * 0.03)
            x0 = int(center_x - base_width / 2 + horizontal_shift)
            y0 = int(center_y - base_height / 2 + vertical_shift)
            box = BoundingBox(
                x0=max(0, x0),
                y0=max(0, y0),
                x1=min(width, x0 + base_width),
                y1=min(height, y0 + base_height),
            ).clamp(width, height)
            proposals.append(
                RegionProposal(
                    phrase=phrase,
                    box=box,
                    confidence=max(0.3, phrase.confidence - 0.08),
                    source="placeholder_grounding_dino",
                )
            )

        return tuple(proposals)


class PlaceholderSam2MaskRefiner:
    """Box-to-polygon placeholder for a future SAM2 mask refiner."""

    def __init__(self, *, expand_pixels: int = 8) -> None:
        self.expand_pixels = expand_pixels

    def refine_masks(
        self, photo: ProductPhoto, proposals: Sequence[RegionProposal]
    ) -> Sequence[MaskCandidate]:
        with Image.open(photo.image_path) as handle:
            width, height = handle.size

        masks: list[MaskCandidate] = []
        for proposal in proposals:
            expanded = BoundingBox(
                x0=max(0, proposal.box.x0 - self.expand_pixels),
                y0=max(0, proposal.box.y0 - self.expand_pixels),
                x1=min(width, proposal.box.x1 + self.expand_pixels),
                y1=min(height, proposal.box.y1 + self.expand_pixels),
            ).clamp(width, height)
            polygon = (
                (expanded.x0, expanded.y0),
                (expanded.x1, expanded.y0),
                (expanded.x1, expanded.y1),
                (expanded.x0, expanded.y1),
            )
            masks.append(
                MaskCandidate(
                    phrase=proposal.phrase,
                    box=expanded,
                    polygon=polygon,
                    area_pixels=expanded.area,
                    confidence=max(0.25, proposal.confidence - 0.04),
                    source="placeholder_sam2",
                )
            )
        return tuple(masks)


class ProductLocalizationPipeline:
    """Adapter-based localization pipeline with lightweight defaults."""

    def __init__(
        self,
        *,
        phrase_extractor: PhraseExtractionAdapter | None = None,
        proposer: GroundingAdapter | None = None,
        mask_refiner: MaskRefinementAdapter | None = None,
    ) -> None:
        self.phrase_extractor = phrase_extractor or PlaceholderVlmPhraseExtractor()
        self.proposer = proposer or PlaceholderGroundingDinoProposer()
        self.mask_refiner = mask_refiner or PlaceholderSam2MaskRefiner()

    def localize(self, photo: ProductPhoto) -> LocalizationResult:
        phrases = tuple(self.phrase_extractor.extract_phrases(photo))
        proposals = tuple(self.proposer.propose_regions(photo, phrases))
        masks = tuple(self.mask_refiner.refine_masks(photo, proposals))
        return LocalizationResult(photo=photo, phrases=phrases, proposals=proposals, masks=masks)
