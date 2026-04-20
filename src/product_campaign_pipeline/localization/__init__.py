"""Product localization abstractions with placeholder and model-backed adapters."""

from .artifacts import SavedLocalizationArtifacts, save_localization_artifacts, select_primary_mask
from .models import BoundingBox, LocalizationResult, MaskCandidate, PhraseCandidate, ProductPhoto, RegionProposal
from .model_backed import (
    DEFAULT_CAPTION_MODEL_ID,
    DEFAULT_GROUNDING_MODEL_ID,
    DEFAULT_SAM2_MODEL_ID,
    BlipPhraseExtractor,
    GroundingDinoProposer,
    Sam2MaskRefiner,
    build_model_backed_localization_pipeline,
)
from .pipeline import (
    GroundingAdapter,
    MaskRefinementAdapter,
    PhraseExtractionAdapter,
    PlaceholderGroundingDinoProposer,
    PlaceholderSam2MaskRefiner,
    PlaceholderVlmPhraseExtractor,
    ProductLocalizationPipeline,
)

__all__ = [
    "BoundingBox",
    "BlipPhraseExtractor",
    "DEFAULT_CAPTION_MODEL_ID",
    "DEFAULT_GROUNDING_MODEL_ID",
    "DEFAULT_SAM2_MODEL_ID",
    "GroundingAdapter",
    "GroundingDinoProposer",
    "LocalizationResult",
    "MaskCandidate",
    "MaskRefinementAdapter",
    "PhraseCandidate",
    "PhraseExtractionAdapter",
    "PlaceholderGroundingDinoProposer",
    "PlaceholderSam2MaskRefiner",
    "PlaceholderVlmPhraseExtractor",
    "ProductLocalizationPipeline",
    "ProductPhoto",
    "RegionProposal",
    "Sam2MaskRefiner",
    "SavedLocalizationArtifacts",
    "build_model_backed_localization_pipeline",
    "save_localization_artifacts",
    "select_primary_mask",
]
