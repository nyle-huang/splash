from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .models import BoundingBox, MaskCandidate, PhraseCandidate, ProductPhoto, RegionProposal
from .pipeline import GroundingAdapter, MaskRefinementAdapter, PhraseExtractionAdapter, ProductLocalizationPipeline


DEFAULT_CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"
DEFAULT_GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "in",
    "of",
    "on",
    "the",
    "with",
}


def _normalize_grounding_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip(".")).strip()


def _build_grounding_prompt(phrases: Sequence[PhraseCandidate]) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = _normalize_grounding_phrase(phrase.text)
        if not normalized or normalized in seen:
            continue
        tokens.append(normalized)
        seen.add(normalized)
    return " . ".join(tokens) + (" ." if tokens else "")


def build_model_backed_localization_pipeline(
    *,
    device: str = "cuda",
    caption_model_id: str = DEFAULT_CAPTION_MODEL_ID,
    grounding_model_id: str = DEFAULT_GROUNDING_MODEL_ID,
    sam2_model_id: str = DEFAULT_SAM2_MODEL_ID,
    max_phrases: int = 5,
    box_threshold: float = 0.25,
    text_threshold: float = 0.2,
) -> ProductLocalizationPipeline:
    return ProductLocalizationPipeline(
        phrase_extractor=BlipPhraseExtractor(
            model_id=caption_model_id,
            device=device,
            max_phrases=max_phrases,
        ),
        proposer=GroundingDinoProposer(
            model_id=grounding_model_id,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        ),
        mask_refiner=Sam2MaskRefiner(
            model_id=sam2_model_id,
            device=device,
        ),
    )


class BlipPhraseExtractor(PhraseExtractionAdapter):
    """Image-caption-backed phrase extractor for messy product photos."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_CAPTION_MODEL_ID,
        device: str = "cuda",
        max_phrases: int = 5,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_phrases = max_phrases
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def extract_phrases(self, photo: ProductPhoto) -> Sequence[PhraseCandidate]:
        processor, model, torch, device = self._ensure_runtime()
        with Image.open(photo.image_path) as handle:
            image = handle.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = inputs.to(device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=24)
        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip().lower()

        phrases: list[tuple[str, float, str]] = []
        if photo.title:
            phrases.append((photo.title.strip().lower(), 0.96, "photo_title"))
        for hint in photo.hint_phrases:
            if hint.strip():
                phrases.append((hint.strip().lower(), 0.88, "hint_phrase"))
        if caption:
            phrases.append((caption, 0.84, "blip_caption"))
            simplified = _simplify_caption_phrase(caption)
            if simplified and simplified != caption:
                phrases.append((simplified, 0.78, "caption_chunk"))
        phrases.append(("product", 0.52, "generic_fallback"))

        bounded_phrases = _dedupe_phrase_specs(phrases)[: self.max_phrases]
        return tuple(
            PhraseCandidate(text=text, confidence=confidence, source=source)
            for text, confidence, source in bounded_phrases
        )

    def _ensure_runtime(self) -> tuple[Any, Any, Any, str]:
        if self._processor is not None and self._model is not None and self._torch is not None:
            return self._processor, self._model, self._torch, _resolve_device(self._torch, self.device)

        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("Transformers with BLIP support is required for model-backed localization.") from exc

        device = _resolve_device(torch, self.device)
        processor = BlipProcessor.from_pretrained(self.model_id)
        model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(device)
        model.eval()

        self._processor = processor
        self._model = model
        self._torch = torch
        return processor, model, torch, device


class GroundingDinoProposer(GroundingAdapter):
    """Grounding DINO region proposer for phrase-conditioned localization."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_GROUNDING_MODEL_ID,
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.2,
        max_proposals: int = 8,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_proposals = max_proposals
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def propose_regions(
        self,
        photo: ProductPhoto,
        phrases: Sequence[PhraseCandidate],
    ) -> Sequence[RegionProposal]:
        if not phrases:
            return ()

        processor, model, torch, device = self._ensure_runtime()
        with Image.open(photo.image_path) as handle:
            image = handle.convert("RGB")

        priority_phrases = [
            phrase for phrase in phrases if phrase.source in {"photo_title", "hint_phrase"}
        ]
        search_batches = [priority_phrases] if priority_phrases else []
        if len(priority_phrases) != len(phrases):
            search_batches.append(list(phrases))

        for phrase_batch in search_batches:
            proposals = self._run_grounding_pass(
                image=image,
                phrases=phrase_batch,
                processor=processor,
                model=model,
                torch=torch,
                device=device,
            )
            if proposals:
                return proposals
        return ()

    def _run_grounding_pass(
        self,
        *,
        image: Image.Image,
        phrases: Sequence[PhraseCandidate],
        processor: Any,
        model: Any,
        torch: Any,
        device: str,
    ) -> tuple[RegionProposal, ...]:
        grounding_prompt = _build_grounding_prompt(phrases)
        if not grounding_prompt:
            return ()
        inputs = processor(images=image, text=[grounding_prompt], return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )
        if not results:
            return ()

        result = results[0]
        raw_labels = result.get("text_labels") or result.get("labels") or ()
        phrase_map = {_normalize_grounding_phrase(phrase.text): phrase for phrase in phrases}
        proposals: list[RegionProposal] = []

        for box, score, label in zip(result.get("boxes", ()), result.get("scores", ()), raw_labels):
            label_text = _normalize_grounding_phrase(str(label))
            phrase = phrase_map.get(label_text) or PhraseCandidate(
                text=label_text,
                confidence=float(score),
                source="grounding_dino_label",
            )
            box_values = [int(round(value)) for value in box.tolist()]
            proposal_box = BoundingBox(
                x0=max(0, box_values[0]),
                y0=max(0, box_values[1]),
                x1=max(box_values[0] + 1, box_values[2]),
                y1=max(box_values[1] + 1, box_values[3]),
            ).clamp(image.width, image.height)
            proposals.append(
                RegionProposal(
                    phrase=phrase,
                    box=proposal_box,
                    confidence=float(score),
                    source="grounding_dino",
                )
            )

        proposals.sort(key=lambda proposal: proposal.confidence, reverse=True)
        return tuple(proposals[: self.max_proposals])

    def _ensure_runtime(self) -> tuple[Any, Any, Any, str]:
        if self._processor is not None and self._model is not None and self._torch is not None:
            return self._processor, self._model, self._torch, _resolve_device(self._torch, self.device)

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "Transformers with Grounding DINO support is required for model-backed localization."
            ) from exc

        device = _resolve_device(torch, self.device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(device)
        model.eval()

        self._processor = processor
        self._model = model
        self._torch = torch
        return processor, model, torch, device


class Sam2MaskRefiner(MaskRefinementAdapter):
    """SAM2 box-prompted mask refinement for candidate regions."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SAM2_MODEL_ID,
        device: str = "cuda",
        contour_epsilon_ratio: float = 0.01,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.contour_epsilon_ratio = contour_epsilon_ratio
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def refine_masks(
        self,
        photo: ProductPhoto,
        proposals: Sequence[RegionProposal],
    ) -> Sequence[MaskCandidate]:
        if not proposals:
            return ()

        processor, model, torch, device = self._ensure_runtime()
        with Image.open(photo.image_path) as handle:
            image = handle.convert("RGB")

        input_boxes = [[[proposal.box.x0, proposal.box.y0, proposal.box.x1, proposal.box.y1] for proposal in proposals]]
        inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=False)

        processed_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks: list[MaskCandidate] = []
        for index, proposal in enumerate(proposals):
            candidate_mask = processed_masks[index]
            while getattr(candidate_mask, "ndim", 0) > 2:
                candidate_mask = candidate_mask[0]
            mask_array = candidate_mask.numpy() > 0

            if not mask_array.any():
                masks.append(
                    MaskCandidate(
                        phrase=proposal.phrase,
                        box=proposal.box,
                        polygon=_rectangle_polygon(proposal.box),
                        area_pixels=proposal.box.area,
                        confidence=proposal.confidence,
                        source="sam2_box_fallback",
                    )
                )
                continue

            mask_box = _mask_to_box(mask_array, proposal.box)
            polygon = _mask_to_polygon(
                mask_array,
                fallback_box=mask_box,
                contour_epsilon_ratio=self.contour_epsilon_ratio,
            )
            masks.append(
                MaskCandidate(
                    phrase=proposal.phrase,
                    box=mask_box,
                    polygon=polygon,
                    area_pixels=int(mask_array.sum()),
                    confidence=proposal.confidence,
                    source="sam2",
                )
            )
        return tuple(masks)

    def _ensure_runtime(self) -> tuple[Any, Any, Any, str]:
        if self._processor is not None and self._model is not None and self._torch is not None:
            return self._processor, self._model, self._torch, _resolve_device(self._torch, self.device)

        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError("Transformers with SAM2 support is required for model-backed localization.") from exc

        device = _resolve_device(torch, self.device)
        processor = Sam2Processor.from_pretrained(self.model_id)
        model = Sam2Model.from_pretrained(self.model_id).to(device)
        model.eval()

        self._processor = processor
        self._model = model
        self._torch = torch
        return processor, model, torch, device


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for localization, but torch.cuda.is_available() is false.")
    return device


def _simplify_caption_phrase(caption: str) -> str:
    normalized = caption.strip().lower()
    if normalized.startswith(("a ", "an ", "the ")):
        normalized = normalized.split(" ", 1)[1]

    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", normalized)
        if token and token not in _STOPWORDS and not token.isdigit()
    ]
    return " ".join(tokens[:4]).strip()


def _dedupe_phrase_specs(
    phrases: Sequence[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    deduped: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    for text, confidence, source in phrases:
        normalized = " ".join(text.split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        deduped.append((normalized, float(confidence), source))
        seen.add(normalized)
    return deduped


def _mask_to_box(mask_array: np.ndarray, fallback_box: BoundingBox) -> BoundingBox:
    ys, xs = np.nonzero(mask_array)
    if len(xs) == 0 or len(ys) == 0:
        return fallback_box
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _mask_to_polygon(
    mask_array: np.ndarray,
    *,
    fallback_box: BoundingBox,
    contour_epsilon_ratio: float,
) -> tuple[tuple[int, int], ...]:
    binary_mask = mask_array.astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _rectangle_polygon(fallback_box)

    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, contour_epsilon_ratio * cv2.arcLength(contour, True))
    approximated = cv2.approxPolyDP(contour, epsilon, True)
    polygon = tuple((int(point[0][0]), int(point[0][1])) for point in approximated)
    if len(polygon) < 3:
        return _rectangle_polygon(fallback_box)
    return polygon


def _rectangle_polygon(box: BoundingBox) -> tuple[tuple[int, int], ...]:
    return (
        (box.x0, box.y0),
        (box.x1, box.y0),
        (box.x1, box.y1),
        (box.x0, box.y1),
    )
