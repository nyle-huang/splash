from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from product_campaign_pipeline.taxonomy import (
    BEDDING_CANONICAL_TYPES,
    DRINKWARE_CANONICAL_TYPES,
    KITCHEN_APPLIANCE_CANONICAL_TYPES,
    MULTIPART_LOCALIZATION_CANONICAL_TYPES,
    STRUCTURED_DISPLAY_CANONICAL_TYPES,
)
from .models import BoundingBox, LocalizationResult, MaskCandidate


@dataclass(frozen=True, slots=True)
class SavedLocalizationArtifacts:
    crop_path: str
    mask_path: str
    overlay_path: str
    phrase: str
    confidence: float
    box: BoundingBox


def select_primary_mask(result: LocalizationResult) -> MaskCandidate | None:
    if not result.masks:
        return None

    with Image.open(result.photo.image_path) as handle:
        image_width, image_height = handle.size
        image_area = max(1, image_width * image_height)

    target_text = " ".join(
        part
        for part in (
            result.photo.metadata.get("canonical_product_type", ""),
            result.photo.metadata.get("category", ""),
            result.photo.title or "",
            *result.photo.hint_phrases,
        )
        if part
    )
    target_tokens = _text_tokens(target_text)
    target_category = result.photo.metadata.get("category", "") or _infer_target_category(target_text)
    target_type = result.photo.metadata.get("canonical_product_type", "") or _infer_target_type(target_text)

    scored_masks: list[tuple[float, float, MaskCandidate]] = []
    for mask in result.masks:
        structural_score = _structural_completeness_score(
            mask,
            image_width=image_width,
            image_height=image_height,
            target_category=target_category,
            target_type=target_type,
        )
        total_score = (
            1.2 * _token_overlap_score(mask.phrase.text, target_tokens)
            + 0.8 * _type_phrase_score(mask.phrase.text, target_type, target_category)
            - 0.7 * _component_phrase_penalty(mask.phrase.text, target_type)
            - 0.6 * _generic_non_specific_phrase_penalty(mask.phrase.text, target_tokens)
            + 0.95
            * _geometry_compatibility_score(
                mask,
                image_width=image_width,
                image_height=image_height,
                image_area=image_area,
                target_category=target_category,
                target_type=target_type,
            )
            + 1.1 * structural_score
            + 0.8 * mask.confidence
            + min(mask.area_pixels / image_area, 0.22)
        )
        scored_masks.append((total_score, structural_score, mask))

    multipart_types = MULTIPART_LOCALIZATION_CANONICAL_TYPES
    if target_type in multipart_types or target_category in {"kitchen appliance", "home lighting", "furniture"}:
        best_structural = max(score[1] for score in scored_masks)
        if best_structural >= 0.52:
            structural_pool = [
                score for score in scored_masks if score[1] >= max(0.48, best_structural - 0.12)
            ]
            if structural_pool:
                scored_masks = structural_pool

    return max(scored_masks, key=lambda item: (item[0], item[1], item[2].confidence))[2]


def save_localization_artifacts(
    result: LocalizationResult,
    output_dir: str | Path,
    *,
    selected_mask: MaskCandidate | None = None,
) -> SavedLocalizationArtifacts | None:
    selected = selected_mask or select_primary_mask(result)
    if selected is None:
        return None

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = result.photo.image_path.stem
    crop_path = target_dir / f"{stem}.crop.png"
    mask_path = target_dir / f"{stem}.mask.png"
    overlay_path = target_dir / f"{stem}.overlay.png"
    target_text = " ".join(
        part
        for part in (
            result.photo.metadata.get("canonical_product_type", ""),
            result.photo.metadata.get("category", ""),
            result.photo.title or "",
            *result.photo.hint_phrases,
        )
        if part
    )
    target_category = result.photo.metadata.get("category", "") or _infer_target_category(target_text)
    target_type = result.photo.metadata.get("canonical_product_type", "") or _infer_target_type(target_text)

    with Image.open(result.photo.image_path) as handle:
        image = handle.convert("RGB")
        image_area = max(1, image.size[0] * image.size[1])
        structural_completeness = _structural_completeness_score(
            selected,
            image_width=image.size[0],
            image_height=image.size[1],
            target_category=target_category,
            target_type=target_type,
        )
        crop_box = _expand_crop_box_for_target(
            selected.box,
            image_size=image.size,
            target_category=target_category,
            target_type=target_type,
            structural_completeness=structural_completeness,
        )
        crop = image.crop(crop_box)
        crop.save(crop_path)

        mask_image = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask_image).polygon(selected.polygon, fill=255)
        mask_image.save(mask_path)

        overlay = image.copy()
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.polygon(selected.polygon, outline=(255, 64, 64), width=3)
        overlay_draw.rectangle(
            (selected.box.x0, selected.box.y0, selected.box.x1, selected.box.y1),
            outline=(64, 196, 255),
            width=2,
        )
        overlay.save(overlay_path)

    return SavedLocalizationArtifacts(
        crop_path=str(crop_path),
        mask_path=str(mask_path),
        overlay_path=str(overlay_path),
        phrase=selected.phrase.text,
        confidence=selected.confidence,
        box=selected.box,
    )


def _expand_crop_box_for_target(
    box: BoundingBox,
    *,
    image_size: tuple[int, int],
    target_category: str,
    target_type: str,
    structural_completeness: float | None = None,
) -> tuple[int, int, int, int]:
    width = max(1, box.width)
    height = max(1, box.height)
    margin_ratio = 0.08
    top_boost = 1.0
    left_boost = 1.0
    bottom_boost = 1.0
    right_boost = 1.0
    if target_type == "blender":
        margin_ratio = 0.18
        left_boost = 1.8
        bottom_boost = 3.0
        right_boost = 2.8
    elif target_type in {"office chair", "folding chair"} or target_category == "furniture":
        margin_ratio = 0.16
        left_boost = 1.2
        bottom_boost = 2.15
    elif target_type in {"toaster", "table lamp"} or target_category in {
        "kitchen appliance",
        "home lighting",
    }:
        margin_ratio = 0.14
        left_boost = 1.1
        bottom_boost = 1.85
    elif target_type in {"comforter", "pet bed"} or target_category in {"bedding", "pet home"}:
        margin_ratio = 0.1
        bottom_boost = 1.15
    elif target_category == "apparel":
        margin_ratio = 0.1
        top_boost = 1.45
        bottom_boost = 1.1
    if structural_completeness is not None and structural_completeness < 0.62:
        margin_ratio = max(margin_ratio, 0.22 if target_type == "blender" else 0.16)
        left_boost = max(left_boost, 1.7 if target_type == "blender" else 1.25)
        right_boost = max(right_boost, 3.5 if target_type == "blender" else 1.7)
        bottom_boost = max(bottom_boost, 3.6 if target_type == "blender" else 2.35)
    margin_x = max(10, int(round(width * margin_ratio)))
    margin_y = max(10, int(round(height * margin_ratio)))
    image_width, image_height = image_size
    return (
        max(0, box.x0 - int(round(margin_x * left_boost))),
        max(0, box.y0 - int(round(margin_y * top_boost))),
        min(image_width, box.x1 + int(round(margin_x * right_boost))),
        min(image_height, box.y1 + int(round(margin_y * bottom_boost))),
    )


def _text_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if token and len(token) > 2
    }


def _token_overlap_score(candidate: str, target_tokens: set[str]) -> float:
    if not target_tokens:
        return 0.0
    candidate_tokens = _text_tokens(candidate)
    if not candidate_tokens:
        return 0.0
    return len(candidate_tokens & target_tokens) / len(target_tokens)


def _component_phrase_penalty(candidate: str, target_type: str) -> float:
    lowered = candidate.lower()
    component_tokens: tuple[str, ...] = ()
    if target_type == "blender":
        component_tokens = ("jar", "pitcher", "lid")
    elif target_type == "coffee maker":
        component_tokens = ("carafe", "pot", "filter", "basket", "reservoir")
    elif target_type == "slow cooker":
        component_tokens = ("lid", "insert", "crock", "pot")
    elif target_type == "food chopper":
        component_tokens = ("bowl", "cup", "lid", "container")
    elif target_type == "table lamp":
        component_tokens = ("shade", "base")
    if not component_tokens:
        return 0.0
    return 1.0 if any(token in lowered for token in component_tokens) else 0.0


def _generic_non_specific_phrase_penalty(candidate: str, target_tokens: set[str]) -> float:
    candidate_tokens = _text_tokens(candidate)
    if not candidate_tokens or candidate_tokens.intersection(target_tokens):
        return 0.0
    if len(candidate_tokens) <= 2:
        return 1.0
    if len(candidate_tokens) <= 3:
        return 0.6
    return 0.0


def _infer_target_category(text: str) -> str:
    lowered = text.lower()
    if any(
        token in lowered
        for token in ("blender", "toaster", "coffee maker", "slow cooker", "food chopper", "food processor", "air fryer", "appliance", "kettle", "microwave")
    ):
        return "kitchen appliance"
    if any(token in lowered for token in ("bottle", "mug", "cup", "tumbler", "drinkware")):
        return "drinkware"
    if any(
        token in lowered for token in ("comforter", "duvet", "blanket", "quilt", "bedding", "coverlet", "sheet set")
    ):
        return "bedding"
    if any(token in lowered for token in ("lamp", "lighting", "shade", "light fixture")):
        return "home lighting"
    if any(token in lowered for token in ("pillow", "cushion")):
        return "home decor"
    if any(token in lowered for token in ("office chair", "folding chair", "desk chair", "chair", "stool", "bench")):
        return "furniture"
    if any(token in lowered for token in ("backpack", "bookbag", "knapsack", "bag", "handbag", "tote", "wallet", "satchel", "purse", "clutch")):
        return "bag"
    if any(token in lowered for token in ("dog bed", "pet bed", "cat bed")):
        return "pet home"
    if any(token in lowered for token in ("shoe", "sneaker", "sandal", "boot", "loafer", "trainer")):
        return "footwear"
    if any(token in lowered for token in ("dress", "shirt", "jacket", "pants", "skirt", "blouse", "hoodie", "tunic", "top")):
        return "apparel"
    return "product"


def _infer_target_type(text: str) -> str:
    lowered = text.lower()
    if "table lamp" in lowered or "desk lamp" in lowered or "lamp" in lowered:
        return "table lamp"
    if "dog bed" in lowered or "pet bed" in lowered or "cat bed" in lowered:
        return "pet bed"
    if "office chair" in lowered or "desk chair" in lowered:
        return "office chair"
    if "folding chair" in lowered or "event chair" in lowered:
        return "folding chair"
    if "comforter" in lowered or "duvet" in lowered:
        return "comforter"
    if "blanket" in lowered or "quilt" in lowered or "coverlet" in lowered or "bedspread" in lowered:
        return "quilt"
    if "blender" in lowered:
        return "blender"
    if "toaster" in lowered:
        return "toaster"
    if "coffee maker" in lowered or "coffeemaker" in lowered:
        return "coffee maker"
    if "slow cooker" in lowered or "crock pot" in lowered or "crock-pot" in lowered or "crockpot" in lowered:
        return "slow cooker"
    if "food chopper" in lowered or "electric chopper" in lowered or "food processor" in lowered or "garlic chopper" in lowered:
        return "food chopper"
    if "water bottle" in lowered or "fitness bottle" in lowered or "bottle" in lowered:
        return "water bottle"
    if "mug" in lowered or "tumbler" in lowered or "cup" in lowered:
        return "mug"
    if "backpack" in lowered or "bookbag" in lowered or "knapsack" in lowered:
        return "backpack"
    if "wallet" in lowered or "clutch" in lowered or "wristlet" in lowered or "pouch" in lowered:
        return "wallet"
    if "tote bag" in lowered or "tote" in lowered or "shopping bag" in lowered or "beach bag" in lowered:
        return "tote bag"
    if "handbag" in lowered or "purse" in lowered:
        return "handbag"
    if "decorative pillow" in lowered or "throw pillow" in lowered or "pillow" in lowered or "cushion" in lowered:
        return "decorative pillow"
    if "dress" in lowered:
        return "dress"
    if any(token in lowered for token in ("shoe", "sneaker", "sandal", "boot", "loafer", "trainer")):
        return "shoe"
    if any(token in lowered for token in ("shirt", "tee", "blouse", "tunic", "top", "hoodie", "sweatshirt")):
        return "shirt"
    return _infer_target_category(lowered)


def _type_phrase_score(candidate: str, target_type: str, target_category: str) -> float:
    candidate_tokens = _text_tokens(candidate)
    if not candidate_tokens:
        return 0.0
    score = 0.0
    target_type_tokens = _text_tokens(target_type)
    if target_type_tokens:
        overlap = len(candidate_tokens & target_type_tokens)
        if overlap:
            score += 0.55 * overlap / max(len(target_type_tokens), 1)
    category_tokens = _text_tokens(target_category)
    if category_tokens:
        overlap = len(candidate_tokens & category_tokens)
        if overlap:
            score += 0.25 * overlap / max(len(category_tokens), 1)
    return score


def _touches_border(box: BoundingBox, *, image_width: int, image_height: int) -> bool:
    return box.x0 <= 0 or box.y0 <= 0 or box.x1 >= image_width or box.y1 >= image_height


def _geometry_compatibility_score(
    mask: MaskCandidate,
    *,
    image_width: int,
    image_height: int,
    image_area: int,
    target_category: str,
    target_type: str,
) -> float:
    box = mask.box
    area_ratio = mask.area_pixels / float(max(1, image_area))
    fill_ratio = mask.area_pixels / float(max(1, box.area))
    width = max(1, box.width)
    height = max(1, box.height)
    aspect_ratio = width / float(height)
    inverse_aspect_ratio = height / float(width)
    thinness = min(width, height) / float(max(width, height))

    score = 0.0
    if area_ratio < 0.01:
        score -= 0.55
    elif area_ratio >= 0.04:
        score += 0.14
    if fill_ratio < 0.12:
        score -= 0.45
    elif fill_ratio >= 0.42:
        score += 0.08
    if thinness < 0.08:
        score -= 0.35
    if _touches_border(box, image_width=image_width, image_height=image_height) and target_category not in {
        "bedding",
        "pet home",
    }:
        score -= 0.08

    if target_category == "drinkware" or target_type in DRINKWARE_CANONICAL_TYPES:
        if inverse_aspect_ratio >= 1.4:
            score += 0.36
        elif inverse_aspect_ratio < 0.95:
            score -= 0.24
    elif target_category == "bedding" or target_type in BEDDING_CANONICAL_TYPES | {"blanket"}:
        if area_ratio >= 0.12:
            score += 0.42
        elif area_ratio < 0.05:
            score -= 0.3
        if 1.0 <= aspect_ratio <= 3.8:
            score += 0.18
        if fill_ratio < 0.08:
            score -= 0.2
    elif target_category == "home lighting" or target_type == "table lamp":
        if inverse_aspect_ratio >= 1.35:
            score += 0.34
        elif inverse_aspect_ratio < 1.0:
            score -= 0.22
    elif target_category == "furniture" or target_type in {"office chair", "folding chair"}:
        if area_ratio >= 0.05:
            score += 0.2
        if 0.35 <= aspect_ratio <= 1.6 or 0.6 <= inverse_aspect_ratio <= 2.8:
            score += 0.2
        if thinness < 0.12:
            score -= 0.14
    elif target_type == "backpack":
        if inverse_aspect_ratio >= 1.0:
            score += 0.28
        elif aspect_ratio > 1.55:
            score -= 0.16
    elif target_category == "bag":
        if 0.45 <= aspect_ratio <= 1.8 or 0.55 <= inverse_aspect_ratio <= 1.9:
            score += 0.22
    elif target_category == "footwear" or target_type == "shoe":
        if 0.35 <= aspect_ratio <= 4.2:
            score += 0.24
        if inverse_aspect_ratio > 2.8:
            score -= 0.32
    elif target_category == "pet home" or target_type == "pet bed":
        if area_ratio >= 0.06:
            score += 0.24
        if 1.0 <= aspect_ratio <= 4.0:
            score += 0.18
        if inverse_aspect_ratio > 1.6:
            score -= 0.16
    elif target_category == "apparel":
        if area_ratio >= 0.05:
            score += 0.12
        if thinness < 0.05:
            score -= 0.22
    elif target_category == "kitchen appliance" or target_type in KITCHEN_APPLIANCE_CANONICAL_TYPES:
        if area_ratio >= 0.03:
            score += 0.18
        if 0.35 <= aspect_ratio <= 1.8 or 0.55 <= inverse_aspect_ratio <= 2.6:
            score += 0.18
        if thinness < 0.12:
            score -= 0.16

    return score


def _structural_completeness_score(
    mask: MaskCandidate,
    *,
    image_width: int,
    image_height: int,
    target_category: str,
    target_type: str,
) -> float:
    tracked_types = MULTIPART_LOCALIZATION_CANONICAL_TYPES
    if target_type not in tracked_types and target_category not in {"kitchen appliance", "home lighting", "furniture"}:
        return 0.0
    raster = _rasterize_candidate_mask(mask, image_width=image_width, image_height=image_height)
    if raster is None or not raster.any():
        return -0.2
    lower_extent = mask.box.y1 / float(max(image_height, 1))
    box_height_ratio = mask.box.height / float(max(image_height, 1))
    ys, xs = np.nonzero(raster)
    crop = raster[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    total = float(crop.sum())
    if total <= 0:
        return -0.2
    height, width = crop.shape
    if height < 6 or width < 6:
        return -0.2
    top = float(crop[: max(1, int(round(height * 0.35)))].sum()) / total
    middle = float(crop[int(round(height * 0.35)) : max(1, int(round(height * 0.7)))].sum()) / total
    bottom = float(crop[max(0, int(round(height * 0.7))) :].sum()) / total
    bottom_rows = crop[max(0, int(round(height * 0.78))) :, :]
    row_widths = bottom_rows.sum(axis=1) if bottom_rows.size else np.array([], dtype=np.float32)
    bottom_width_ratio = 0.0 if row_widths.size == 0 else float(np.max(row_widths) / max(width, 1))

    if target_type == "blender":
        score = 0.0
        score += 0.22 if top >= 0.22 else -0.12
        score += 0.18 if middle >= 0.18 else -0.08
        score += 0.28 if bottom >= 0.1 else -0.28
        score += 0.16 if bottom_width_ratio >= 0.16 else -0.12
        score += 0.2 if lower_extent >= 0.78 else -0.2
        score += 0.1 if box_height_ratio >= 0.55 else -0.08
        return score

    if target_type in {"office chair", "folding chair", "table lamp"}:
        score = 0.0
        score += 0.18 if top >= 0.2 else -0.08
        score += 0.24 if bottom >= 0.08 else -0.24
        score += 0.12 if bottom_width_ratio >= 0.08 else -0.1
        score += 0.14 if lower_extent >= 0.72 else -0.14
        return score

    if target_type == "toaster":
        return 0.24 if middle >= 0.35 and bottom >= 0.12 else -0.18

    if target_type in {"coffee maker", "slow cooker", "food chopper"} or target_category == "kitchen appliance":
        score = 0.0
        score += 0.14 if top >= 0.14 else -0.05
        score += 0.18 if middle >= 0.18 else -0.08
        score += 0.22 if bottom >= 0.12 else -0.22
        score += 0.14 if lower_extent >= 0.72 else -0.12
        score += 0.08 if bottom_width_ratio >= 0.12 else -0.06
        return score

    return 0.0


def _rasterize_candidate_mask(
    mask: MaskCandidate,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    if not mask.polygon:
        return None
    canvas = Image.new("L", (image_width, image_height), 0)
    ImageDraw.Draw(canvas).polygon(mask.polygon, fill=255)
    return np.asarray(canvas, dtype=np.uint8) > 0
