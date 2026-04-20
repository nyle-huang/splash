from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ProductPhoto:
    """Input product photo to localize."""

    image_path: Path
    product_id: str | None = None
    title: str | None = None
    hint_phrases: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_path.name:
            raise ValueError("image_path must point to a file")


@dataclass(frozen=True, slots=True)
class PhraseCandidate:
    text: str
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must be non-empty")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding boxes must have positive area")
        if min(self.x0, self.y0, self.x1, self.y1) < 0:
            raise ValueError("bounding boxes cannot have negative coordinates")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def clamp(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x0=max(0, min(self.x0, width - 1)),
            y0=max(0, min(self.y0, height - 1)),
            x1=max(1, min(self.x1, width)),
            y1=max(1, min(self.y1, height)),
        )


@dataclass(frozen=True, slots=True)
class RegionProposal:
    phrase: PhraseCandidate
    box: BoundingBox
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    phrase: PhraseCandidate
    box: BoundingBox
    polygon: tuple[tuple[int, int], ...]
    area_pixels: int
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    photo: ProductPhoto
    phrases: tuple[PhraseCandidate, ...]
    proposals: tuple[RegionProposal, ...]
    masks: tuple[MaskCandidate, ...]

