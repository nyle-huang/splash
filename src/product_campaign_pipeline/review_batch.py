"""Human-review batch utilities for retrieval prep and local generation."""

from __future__ import annotations

import csv
import colorsys
import copy
import hashlib
import heapq
import json
import math
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from product_campaign_pipeline.composer.prompts import PromptComposer
from product_campaign_pipeline.flux import Flux2KleinClient
from product_campaign_pipeline.taxonomy import (
    BEDDING_CANONICAL_TYPES,
    DRINKWARE_CANONICAL_TYPES,
    FURNITURE_CANONICAL_TYPES,
    KITCHEN_APPLIANCE_CANONICAL_TYPES,
    MULTIPART_LOCALIZATION_CANONICAL_TYPES,
    SOFT_HOME_CANONICAL_TYPES,
    STRUCTURED_DISPLAY_CANONICAL_TYPES,
)
from product_campaign_pipeline.types import (
    BoundingBox,
    CampaignPriorSpec,
    FluxPromptSpec,
    LocalizedProduct,
    ObservedEvidenceSpec,
    ProductIdentitySpec,
)


@dataclass(frozen=True, slots=True)
class PreparedEvidenceAssets:
    crop_path: Path | None
    cutout_path: Path | None
    silhouette_path: Path | None
    mask_path: Path | None
    artifact_flags: tuple[str, ...] = ()


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
SCENE_PATTERNS: dict[str, tuple[str, ...]] = {
    "tabletop kitchen setting": ("table", "counter", "kitchen", "cup", "mug", "bottle"),
    "living room sofa setting": ("couch", "sofa", "pillow", "living", "chair"),
    "fashion lifestyle setting": ("handbag", "bag", "shirt", "dress", "woman", "person"),
    "outdoor lifestyle setting": ("grass", "beach", "outside", "street", "road", "park"),
    "home interior setting": ("home", "room", "bed", "floor", "indoor", "interior"),
}
CATEGORY_CAPTION_HINTS: dict[str, tuple[str, ...]] = {
    "drinkware": ("bottle", "water", "drink", "mug", "cup", "tumbler", "flask"),
    "home decor": ("pillow", "cushion", "sofa", "couch", "bed", "chair", "rug"),
    "home lighting": ("lamp", "lighting", "shade", "light"),
    "bedding": ("comforter", "quilt", "blanket", "duvet", "bedding", "bedspread", "coverlet"),
    "furniture": ("chair", "stool", "bench", "seat", "office chair", "folding chair"),
    "kitchen appliance": (
        "appliance",
        "blender",
        "toaster",
        "coffee",
        "coffee maker",
        "slow cooker",
        "cooker",
        "chopper",
        "processor",
        "kettle",
    ),
    "pet home": ("pet bed", "dog bed", "cat bed", "pet", "dog", "cat"),
    "bag": ("bag", "handbag", "tote", "purse", "wallet", "backpack"),
    "apparel": ("shirt", "dress", "jacket", "shoe", "pants", "skirt", "coat"),
}
CATEGORY_NEGATIVE_HINTS: dict[str, tuple[str, ...]] = {
    "drinkware": ("sauce", "scissors", "brush", "comb", "shirt", "dress", "pillow", "wallet", "bag"),
    "home decor": ("shirt", "dress", "bottle", "wallet", "purse"),
    "home lighting": ("shirt", "dress", "wallet", "purse", "bottle", "mug"),
    "bedding": ("shirt", "dress", "wallet", "purse", "bottle", "mug", "lamp"),
    "furniture": ("shirt", "dress", "wallet", "purse", "bottle", "mug", "lamp"),
    "kitchen appliance": ("shirt", "dress", "wallet", "purse", "pillow", "shoe"),
    "pet home": ("shirt", "dress", "wallet", "purse", "bottle", "mug"),
    "bag": ("shirt", "dress", "pillow", "cushion", "bottle", "mug", "cup"),
    "apparel": ("pillow", "cushion", "bottle", "mug", "wallet"),
}
STRUCTURED_SUBOBJECT_COMPETING_TOKENS: dict[str, frozenset[str]] = {
    "kitchen appliance": frozenset({"cup", "mug", "glass", "tumbler", "bottle", "pitcher", "carafe", "bowl"}),
}
MULTI_OBJECT_NOISE_TOKENS = ("including", "variety", "items", "assortment", "collection")
CATEGORY_FALLBACK_STYLE_ATOMS: dict[str, tuple[str, ...]] = {
    "drinkware": ("clear hero framing", "clean hydration-product storytelling", "anchored support-surface composition"),
    "home decor": ("clear hero framing", "soft furnished-environment context", "commercial product storytelling"),
    "bag": ("clear hero framing", "commercial fashion-accessory storytelling", "human-in-use framing"),
    "apparel": ("clear hero framing", "editorial apparel styling", "human-in-use framing"),
    "product": ("clear hero framing", "commercial product storytelling"),
}
CATEGORY_FALLBACK_SCENARIOS: dict[str, tuple[str, ...]] = {
    "drinkware": ("tabletop_display",),
    "home decor": ("furnished_interior",),
    "home lighting": ("furnished_interior",),
    "bedding": ("furnished_interior",),
    "furniture": ("editorial_interior", "furnished_interior"),
    "kitchen appliance": ("tabletop_display", "editorial_interior"),
    "pet home": ("furnished_interior",),
    "bag": ("fashion_lifestyle",),
    "apparel": ("fashion_lifestyle",),
    "footwear": ("fashion_lifestyle",),
    "product": ("editorial_interior",),
}
CATEGORY_SUPPORT_DEFAULTS: dict[str, tuple[str, str, str, bool]] = {
    "drinkware": ("self_supporting_display", "tabletop_display", "handheld_or_display", True),
    "home decor": ("externally_supported_soft", "furnished_interior", "placed", False),
    "home lighting": ("supported_display", "furnished_interior", "placed", True),
    "bedding": ("externally_supported_soft", "furnished_interior", "placed", False),
    "furniture": ("self_supporting_display", "editorial_interior", "placed", True),
    "kitchen appliance": ("self_supporting_display", "tabletop_display", "placed", True),
    "pet home": ("externally_supported_soft", "furnished_interior", "placed", False),
    "bag": ("portable_flexible", "fashion_lifestyle", "carried_or_resting", False),
    "apparel": ("wearable", "fashion_lifestyle", "worn", False),
    "footwear": ("wearable", "fashion_lifestyle", "worn", False),
    "product": ("supported_display", "editorial_interior", "placed", False),
}
SCENE_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tabletop_display": ("table", "counter", "desk", "vanity", "shelf", "box"),
    "furnished_interior": ("couch", "sofa", "bed", "chair", "living", "bedroom", "room", "interior"),
    "fashion_lifestyle": ("handbag", "bag", "shirt", "dress", "woman", "person", "holding", "wearing"),
    "outdoor_lifestyle": ("grass", "beach", "outside", "street", "road", "park"),
    "retail_display": ("retail", "store", "boutique", "rack", "display"),
    "editorial_interior": ("home", "indoor", "studio", "interior"),
}
SUPPORT_RELATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "standing_on_surface": ("table", "counter", "desk", "shelf", "box", "vanity"),
    "resting_with_back_support": ("couch", "sofa", "bed", "chair", "armchair", "seat"),
    "resting_on_surface": ("floor", "rug", "surface", "table", "desk", "shelf"),
    "carried_by_hand": ("holding", "hand", "woman", "man", "person"),
    "worn_on_body": ("wearing", "model", "shirt", "dress", "jacket", "pants", "shoe"),
    "mounted_or_hanging": ("hanging", "hook", "wall", "mounted"),
}
SUPPORT_MODE_RELATION_COMPATIBILITY: dict[str, tuple[str, ...]] = {
    "self_supporting_display": ("standing_on_surface", "carried_by_hand"),
    "externally_supported_soft": ("resting_with_back_support", "resting_on_surface", "carried_by_hand"),
    "portable_flexible": ("carried_by_hand", "resting_on_surface", "mounted_or_hanging"),
    "wearable": ("worn_on_body",),
    "supported_display": ("standing_on_surface", "resting_on_surface", "mounted_or_hanging"),
    "mounted": ("mounted_or_hanging",),
}
SUPPORT_RELATION_DEFAULTS: dict[str, str] = {
    "self_supporting_display": "standing_on_surface",
    "externally_supported_soft": "resting_with_back_support",
    "portable_flexible": "carried_by_hand",
    "wearable": "worn_on_body",
    "supported_display": "resting_on_surface",
    "mounted": "mounted_or_hanging",
}
SCENE_SUPPORT_COMPATIBILITY: dict[str, tuple[str, ...]] = {
    "standing_on_surface": ("tabletop_display", "retail_display", "editorial_interior", "furnished_interior"),
    "resting_with_back_support": ("furnished_interior", "editorial_interior"),
    "resting_on_surface": ("editorial_interior", "furnished_interior", "tabletop_display"),
    "carried_by_hand": ("fashion_lifestyle", "outdoor_lifestyle", "editorial_interior"),
    "worn_on_body": ("fashion_lifestyle", "outdoor_lifestyle"),
    "mounted_or_hanging": ("retail_display", "editorial_interior"),
}
SCENE_FAMILY_DEFAULTS_BY_SUPPORT: dict[str, str] = {
    "standing_on_surface": "tabletop_display",
    "resting_with_back_support": "furnished_interior",
    "resting_on_surface": "editorial_interior",
    "carried_by_hand": "fashion_lifestyle",
    "worn_on_body": "fashion_lifestyle",
    "mounted_or_hanging": "retail_display",
}
SUPPORT_RELATION_STYLE_ATOMS: dict[str, tuple[str, ...]] = {
    "standing_on_surface": ("anchored support-surface composition",),
    "resting_with_back_support": ("soft furnished-environment context",),
    "resting_on_surface": ("anchored support-surface composition",),
    "carried_by_hand": ("human-in-use framing",),
    "worn_on_body": ("human-in-use framing",),
    "mounted_or_hanging": ("structured display presentation",),
}
SCENE_FAMILY_STYLE_ATOMS: dict[str, tuple[str, ...]] = {
    "tabletop_display": ("anchored support-surface composition",),
    "furnished_interior": ("soft furnished-environment context",),
    "fashion_lifestyle": ("human-in-use framing",),
    "outdoor_lifestyle": ("outdoor lifestyle context",),
    "retail_display": ("retail display context",),
    "editorial_interior": ("commercial product storytelling",),
}
SUPPORT_RELATION_CONSTRAINTS: dict[str, str] = {
    "standing_on_surface": "show the product standing securely on a stable surface with visible contact and grounded shadow",
    "resting_with_back_support": "show the product resting naturally with visible support from furniture or another object, with believable contact and compression",
    "resting_on_surface": "show the product resting naturally on a surface with visible support, contact, and plausible deformation",
    "carried_by_hand": "show the product carried or held with visible support from a person, or hanging naturally under gravity",
    "worn_on_body": "show the product worn naturally on a person so its shape follows the body and gravity",
    "mounted_or_hanging": "show the product attached to a visible support point with believable tension, contact, and gravity",
}
SUPPORT_RELATION_EVAL_PROMPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "standing_on_surface": {
        "positive": (
            "a {product} standing securely on a flat surface with a visible base and contact shadow",
            "a {product} displayed upright on a stable countertop or shelf",
        ),
        "negative": (
            "a {product} floating without visible support",
            "a {product} leaning impossibly without stable contact",
        ),
    },
    "resting_with_back_support": {
        "positive": (
            "a {product} resting naturally with visible support from furniture or another object",
            "a soft {product} leaning or nestled into a supported interior setting",
        ),
        "negative": (
            "a soft {product} standing upright unsupported on a hard table",
            "a {product} floating without visible support",
        ),
    },
    "resting_on_surface": {
        "positive": (
            "a {product} resting naturally on a surface with visible contact and support",
            "a {product} lying or resting with believable contact and material behavior",
        ),
        "negative": (
            "a {product} floating without support",
            "a {product} standing rigidly without believable contact",
        ),
    },
    "carried_by_hand": {
        "positive": (
            "a {product} carried by hand with visible support and natural gravity",
            "a {product} hanging naturally from handles or straps while being carried",
        ),
        "negative": (
            "a flexible {product} standing rigidly upright without support",
            "a {product} floating without visible support",
        ),
    },
    "worn_on_body": {
        "positive": (
            "a {product} worn naturally on a person",
            "a wearable {product} shaped by the body and gravity",
        ),
        "negative": (
            "a wearable {product} standing upright by itself without a body",
            "a {product} floating without visible support",
        ),
    },
    "mounted_or_hanging": {
        "positive": (
            "a {product} hanging from a visible support point with believable tension",
            "a {product} mounted securely to a structure with clear contact",
        ),
        "negative": (
            "a {product} floating without visible support",
            "a {product} standing in space without contact",
        ),
    },
}
HUMAN_ANATOMY_EVAL_PROMPTS: dict[str, tuple[str, ...]] = {
    "positive": (
        "a campaign photo of one person with natural arm, hand, and shoulder anatomy interacting with a {product}",
        "a fashion or lifestyle campaign image with a single human subject and believable limbs while holding or wearing a {product}",
    ),
    "negative": (
        "a campaign photo of a person with extra arms, extra hands, duplicated limbs, or merged anatomy while holding or wearing a {product}",
        "a fashion or lifestyle image with broken or impossible human anatomy around a {product}",
    ),
}
CASTING_ALIGNMENT_EVAL_PROMPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "soft_feminine": {
        "positive": (
            "a {product} shown with playful, soft, feminine-coded casual casting and accessory-compatible styling",
            "a {product} held or worn by a casually styled feminine or soft-coded model with approachable everyday presentation",
        ),
        "negative": (
            "a {product} shown with masculine-coded or business-formal casting that conflicts with a playful or delicate design language",
            "a {product} styled with a formal masculine presentation that feels incompatible with a floral or playful accessory",
        ),
    },
    "active_neutral": {
        "positive": (
            "a {product} shown with active neutral casting and grounded everyday styling",
            "a {product} presented with practical casual casting rather than luxury-editorial formality",
        ),
        "negative": (
            "a {product} shown with luxury-editorial or business-formal casting that conflicts with active everyday utility",
            "a {product} styled with overly formal or impractical casting for a sporty everyday product",
        ),
    },
    "accessory_compatible": {
        "positive": (
            "a {product} shown with understated accessory-compatible casting and relaxed everyday styling",
            "a {product} held by a casual model with quiet, product-compatible presentation instead of formal business cues",
        ),
        "negative": (
            "a {product} shown with masculine-formal or business-coded casting that overpowers a small accessory",
            "a {product} held in a business-formal presentation that conflicts with an understated accessory role",
        ),
    },
}
FUNCTIONAL_SUBTYPE_EVAL_PROMPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "backpack cooler": {
        "positive": (
            "a {product} shown as a backpack cooler with a visible insulated opening or cooler-style top compartment",
            "a {product} presented as a utility cooler backpack designed to carry drinks or food rather than a plain school bag",
        ),
        "negative": (
            "a generic school backpack with no cooler or insulated compartment function",
            "a plain laptop backpack or everyday bookbag without cooler features",
        ),
    },
}
SCENE_FAMILY_EVAL_PROMPTS: dict[str, tuple[str, ...]] = {
    "tabletop_display": (
        "a {product} shown in a coherent tabletop or counter display scene",
        "a {product} presented on a support surface in an editorial display",
    ),
    "furnished_interior": (
        "a {product} shown in a furnished interior with believable support context",
        "a {product} placed naturally within a home interior scene",
    ),
    "fashion_lifestyle": (
        "a {product} shown in a fashion-oriented lifestyle scene with natural human interaction",
        "a {product} integrated into an editorial lifestyle scene",
    ),
    "outdoor_lifestyle": (
        "a {product} shown in an outdoor lifestyle scene with believable contact and support",
        "a {product} integrated into a grounded outdoor campaign scene",
    ),
    "retail_display": (
        "a {product} shown in a structured retail display scene",
        "a {product} presented in a merchandising or boutique display",
    ),
    "editorial_interior": (
        "a {product} shown in an editorial interior scene with believable placement",
        "a {product} integrated into a coherent interior campaign set",
    ),
}
CATEGORY_CLASSIFICATION_TEXTS: dict[str, tuple[str, ...]] = {
    "drinkware": (
        "a commercial product photo of a reusable water bottle or mug",
        "a commercial product photo of a tumbler, mug, or bottle",
    ),
    "home decor": (
        "a commercial product photo of a decorative pillow",
        "a commercial product photo of a throw cushion",
    ),
    "home lighting": (
        "a commercial product photo of a table lamp",
        "a commercial product photo of a lamp in a home interior",
    ),
    "bedding": (
        "a commercial product photo of a comforter, quilt, duvet, or bedding set",
        "a commercial product photo of bedding spread across a bed",
    ),
    "furniture": (
        "a commercial product photo of a chair or seating furniture item",
        "a commercial product photo of a standalone chair in an interior or display scene",
    ),
    "kitchen appliance": (
        "a commercial product photo of a countertop kitchen appliance",
        "a commercial product photo of a blender, toaster, coffee maker, slow cooker, or small kitchen machine",
    ),
    "pet home": (
        "a commercial product photo of a dog bed or pet bed",
        "a commercial product photo of a soft pet resting bed",
    ),
    "bag": (
        "a commercial product photo of a tote bag",
        "a commercial product photo of a handbag, purse, or backpack",
    ),
    "apparel": (
        "a commercial product photo of clothing or a garment",
        "a commercial product photo of a shirt or dress",
    ),
    "footwear": (
        "a commercial product photo of a shoe or sneaker",
        "a commercial product photo of footwear",
    ),
    "product": ("a commercial product photo",),
}
PRODUCT_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("water bottle", ("water bottle", "fitness bottle", "chill bottle", "bottle")),
    ("mug", ("travel mug", "coffee mug", "stoneware mug", "ceramic mug", "mug", "tumbler")),
    ("table lamp", ("table lamp", "desk lamp", "nightstand lamp", "lamp")),
    ("comforter", ("comforter", "duvet", "duvet insert")),
    ("quilt", ("quilt", "matelasse quilt", "bedspread", "coverlet")),
    ("pet bed", ("dog bed", "pet bed", "cat bed")),
    ("office chair", ("office chair", "desk chair", "gaming chair", "task chair")),
    ("folding chair", ("folding chair", "event chair", "resin chair")),
    ("blender", ("countertop blender", "kitchen blender", "blender")),
    ("toaster", ("toaster", "2 slice toaster", "two slice toaster")),
    ("coffee maker", ("coffee maker", "coffeemaker", "drip coffee maker", "single serve coffee maker")),
    ("slow cooker", ("slow cooker", "crock pot", "crock-pot", "crockpot")),
    ("food chopper", ("food chopper", "electric chopper", "mini food processor", "food processor", "garlic chopper")),
    ("backpack", ("backpack", "bookbag", "knapsack")),
    ("tote bag", ("tote bag", "beach bag", "shopping bag", "tote")),
    ("handbag", ("handbag", "purse")),
    ("wallet", ("wallet",)),
    ("decorative pillow", ("decorative pillow", "throw pillow", "pillow", "cushion")),
    ("dress", ("maxi dress", "swing dress", "sundress", "dress")),
    ("shoe", ("walking shoe", "sneaker", "shoe", "sandal", "loafer", "boot", "trainer")),
    ("shirt", ("t shirt", "shirt", "tee", "blouse", "tunic", "top", "hoodie", "sweatshirt")),
)
TYPE_STRUCTURAL_FACTS: dict[str, tuple[str, ...]] = {
    "shirt": ("the product remains a shirt worn on the upper body",),
    "dress": ("the product remains a dress worn on the body",),
    "shoe": ("the product remains a shoe with a structured upper and a visible sole",),
    "backpack": ("the product remains a backpack with a main body and visible carry straps",),
    "mug": ("the product remains a mug with an open drinking vessel body and a visible handle or graspable side profile",),
    "table lamp": ("the product remains an upright table lamp with a stable base and upper light shade",),
    "comforter": ("the product remains a broad soft bedding surface meant to rest across a bed",),
    "quilt": ("the product remains a broad soft quilt meant to rest across a bed surface with readable stitched or layered bedding structure",),
    "pet bed": ("the product remains a low soft pet bed with a visible resting surface",),
    "office chair": ("the product remains a chair with a seat, backrest, and supporting frame",),
    "folding chair": ("the product remains a rigid folding chair with a visible seat and backrest",),
    "blender": ("the product remains a countertop blender with a base and upper blending vessel",),
    "toaster": ("the product remains a compact countertop toaster with a rigid appliance body",),
    "coffee maker": ("the product remains a countertop coffee maker with a stable base and a visible brewing body or carafe assembly",),
    "slow cooker": ("the product remains a countertop slow cooker with a stable base and a visible cooking vessel with lid",),
    "food chopper": ("the product remains a compact countertop food chopper with a base below a small processing bowl",),
}
EVIDENCE_COLOR_SWATCHES: dict[str, tuple[int, int, int]] = {
    "white": (238, 236, 232),
    "black": (34, 34, 34),
    "gray": (148, 148, 148),
    "blue": (47, 105, 176),
    "teal": (41, 128, 134),
    "green": (66, 133, 74),
    "red": (188, 73, 70),
    "pink": (212, 134, 164),
    "purple": (123, 91, 159),
    "yellow": (219, 181, 82),
    "gold": (184, 149, 79),
    "orange": (207, 126, 59),
    "brown": (123, 86, 62),
    "beige": (212, 197, 173),
}
PATTERN_TEXT_TOKENS: tuple[str, ...] = (
    "pattern",
    "patterned",
    "printed",
    "print",
    "graphic",
    "graphics",
    "floral",
    "stripe",
    "striped",
    "woven",
    "braided",
    "quilted",
    "cartoon",
    "motif",
    "plaid",
    "textured",
)
EXPLICIT_PRINT_TOKENS: tuple[str, ...] = (
    "pattern",
    "patterned",
    "printed",
    "print",
    "graphic",
    "graphics",
    "floral",
    "stripe",
    "striped",
    "cartoon",
    "motif",
    "plaid",
)
STYLE_PERSONA_TEXT_HINTS: dict[str, tuple[str, ...]] = {
    "playful_casual": ("disney", "cartoon", "character", "cute", "playful", "floral", "flower", "bow", "kids"),
    "refined_neutral": ("minimal", "refined", "classic", "tailored", "structured", "clean", "lamp", "lighting"),
    "sport_utility": ("fitness", "sport", "sports", "hydration", "outdoor", "running", "gym", "utility", "backpack"),
    "cozy_home": ("pillow", "cushion", "cozy", "home", "woven", "knit", "soft", "textile", "lamp"),
}
STYLE_PERSONA_IMAGE_PROMPTS: dict[str, str] = {
    "playful_casual": "a playful colorful cartoon or floral product styled in a casual approachable way",
    "refined_neutral": "a refined neutral product styled in a polished but non-business way",
    "sport_utility": "a sporty utility product styled for active everyday use",
    "cozy_home": "a cozy home decor textile styled in a relaxed warm setting",
}
RETRIEVAL_CATEGORY_IMAGE_PROMPTS: dict[str, str] = {
    "drinkware": "a commercial product photo centered on drinkware such as a reusable bottle or ceramic mug",
    "home decor": "a commercial product photo centered on a decorative pillow or soft home decor accent",
    "home lighting": "a commercial product photo centered on a table lamp or home lighting object",
    "bedding": "a commercial product photo centered on a comforter, quilt, or broad bedding surface",
    "furniture": "a commercial product photo centered on a chair, stool, or bench",
    "kitchen appliance": "a commercial product photo centered on a countertop kitchen appliance such as a toaster, blender, coffee maker, slow cooker, or food chopper",
    "pet home": "a commercial product photo centered on a dog bed or pet bed",
    "bag": "a commercial product photo centered on a handbag, tote bag, wallet, or backpack",
    "apparel": "a commercial product photo centered on clothing such as a shirt or dress",
    "footwear": "a commercial product photo centered on shoes or sneakers",
}
RETRIEVAL_TYPE_IMAGE_PROMPTS: dict[str, dict[str, str]] = {
    "drinkware": {
        "water bottle": "a commercial product photo of a reusable water bottle",
        "mug": "a commercial product photo of a ceramic mug with a handle",
    },
    "home decor": {
        "decorative pillow": "a commercial product photo of a decorative pillow or throw pillow",
    },
    "home lighting": {
        "table lamp": "a commercial product photo of a table lamp with a base and shade",
    },
    "bedding": {
        "comforter": "a commercial product photo of a comforter spread across a bed",
        "quilt": "a commercial product photo of a quilt or coverlet spread across a bed",
    },
    "furniture": {
        "office chair": "a commercial product photo of an office chair or desk chair",
        "folding chair": "a commercial product photo of a folding chair",
    },
    "kitchen appliance": {
        "blender": "a commercial product photo of a countertop blender with a base and jar",
        "toaster": "a commercial product photo of a toaster on a countertop",
        "coffee maker": "a commercial product photo of a drip coffee maker with a carafe",
        "slow cooker": "a commercial product photo of a slow cooker or crock pot with a lid",
        "food chopper": "a commercial product photo of a food chopper or food processor",
    },
    "pet home": {
        "pet bed": "a commercial product photo of a pet bed or dog bed",
    },
    "bag": {
        "tote bag": "a commercial product photo of a tote bag",
        "handbag": "a commercial product photo of a handbag or purse",
        "backpack": "a commercial product photo of a backpack",
        "wallet": "a commercial product photo of a wallet or clutch",
    },
    "apparel": {
        "shirt": "a commercial product photo of a shirt or top",
        "dress": "a commercial product photo of a dress",
    },
    "footwear": {
        "shoe": "a commercial product photo of shoes or sneakers",
    },
}
BUSINESS_PRIOR_DIRECTION_PALETTES: dict[tuple[str, str], tuple[str, ...]] = {
    ("fashion_lifestyle", "playful_casual"): (
        "approachable street-style storytelling with bright social energy",
        "casual city-lifestyle pacing with playful movement and easy confidence",
        "travel-day fashion storytelling with relaxed premium ease",
    ),
    ("fashion_lifestyle", "refined_neutral"): (
        "architectural lifestyle storytelling with polished casual confidence",
        "gallery-adjacent editorial rhythm with calm refined restraint",
        "quiet travel-lifestyle pacing with disciplined negative space",
    ),
    ("fashion_lifestyle", "sport_utility"): (
        "active everyday lifestyle storytelling with clean utility emphasis",
        "movement-led campaign pacing with bright functional clarity",
        "sport-adjacent city storytelling with purposeful body mechanics",
    ),
    ("outdoor_lifestyle", "sport_utility"): (
        "performance-adjacent outdoor storytelling with clean active energy",
        "travel-ready lifestyle pacing with grounded movement and open depth",
        "bright utility-first outdoor rhythm with disciplined motion",
    ),
    ("furnished_interior", "cozy_home"): (
        "domestic comfort storytelling with tactile warmth and intentional styling",
        "quiet residential campaign pacing with layered but controlled comfort cues",
        "hospitality-inspired home storytelling with calm material richness",
    ),
    ("furnished_interior", "refined_neutral"): (
        "editorial interior storytelling with sculpted depth and calm restraint",
        "architectural home styling with measured luxury and clean hierarchy",
        "residential campaign pacing with polished restraint and clear product focus",
    ),
    ("tabletop_display", "sport_utility"): (
        "functional hydration storytelling with clear label-first readability",
        "merchandising-forward tabletop pacing with bright utility clarity",
        "clean everyday-use storytelling with crisp support-surface discipline",
    ),
    ("retail_display", "playful_casual"): (
        "approachable merchandising storytelling with cheerful retail energy",
        "boutique display pacing with bright commercial polish",
        "storefront campaign styling with playful but disciplined presentation",
    ),
    ("editorial_interior", "refined_neutral"): (
        "sculptural editorial storytelling with precise spatial rhythm",
        "gallery-like campaign art direction with clean set control",
        "studio interior storytelling with polished material contrast",
    ),
    ("default", "default"): (
        "high-performing campaign direction with stronger environmental contrast and differentiated framing",
        "creative campaign pacing with deliberate scene identity and controlled hero focus",
        "evidence-compatible commercial direction with clearer scene personality",
    ),
}
BUSINESS_PRIOR_LIGHTING_PALETTES: dict[tuple[str, str], tuple[str, ...]] = {
    ("fashion_lifestyle", "playful_casual"): (
        "window-lit editorial mood with clean highlight rolloff",
        "warm natural light with controlled contrast",
    ),
    ("fashion_lifestyle", "refined_neutral"): (
        "softbox hero lighting with subtle rim light",
        "high-end studio key light with grounded shadows",
    ),
    ("fashion_lifestyle", "sport_utility"): (
        "window-lit editorial mood with clean highlight rolloff",
        "high-end studio key light with grounded shadows",
    ),
    ("outdoor_lifestyle", "sport_utility"): (
        "warm natural light with controlled contrast",
        "window-lit editorial mood with clean highlight rolloff",
    ),
    ("furnished_interior", "cozy_home"): (
        "warm natural light with controlled contrast",
        "window-lit editorial mood with clean highlight rolloff",
    ),
    ("furnished_interior", "refined_neutral"): (
        "softbox hero lighting with subtle rim light",
        "window-lit editorial mood with clean highlight rolloff",
    ),
    ("tabletop_display", "sport_utility"): (
        "high-end studio key light with grounded shadows",
        "softbox hero lighting with subtle rim light",
    ),
    ("default", "default"): (
        "softbox hero lighting with subtle rim light",
        "window-lit editorial mood with clean highlight rolloff",
        "high-end studio key light with grounded shadows",
        "warm natural light with controlled contrast",
    ),
}
BUSINESS_PRIOR_CAMERA_PALETTES: dict[tuple[str, str], tuple[str, ...]] = {
    ("carried_by_hand", "fashion_lifestyle"): (
        "designer-grade three-quarter or frontal campaign framing with believable perspective",
        "commercial hero framing that may reveal new surfaces without contradicting the source evidence",
    ),
    ("worn_on_body", "fashion_lifestyle"): (
        "high-end editorial product angle with natural depth and evidence-consistent proportions",
        "designer-grade three-quarter or frontal campaign framing with believable perspective",
    ),
    ("standing_on_surface", "tabletop_display"): (
        "premium hero angle chosen to flatter the product while staying compatible with the source evidence",
        "commercial hero framing that may reveal new surfaces without contradicting the source evidence",
    ),
    ("resting_on_surface", "furnished_interior"): (
        "high-end editorial product angle with natural depth and evidence-consistent proportions",
        "premium hero angle chosen to flatter the product while staying compatible with the source evidence",
    ),
    ("default", "default"): (
        "premium hero angle chosen to flatter the product while staying compatible with the source evidence",
        "designer-grade three-quarter or frontal campaign framing with believable perspective",
        "high-end editorial product angle with natural depth and evidence-consistent proportions",
        "commercial hero framing that may reveal new surfaces without contradicting the source evidence",
    ),
}
CHROMATIC_SOFT_TEXTILE_LIGHTING_PALETTE: tuple[str, ...] = (
    "neutral daylight-balanced interior light with preserved textile color separation",
    "soft window daylight with accurate white balance and readable chromatic fabric detail",
)
CHROMATIC_SOFT_TEXTILE_DIRECTION_PALETTE: tuple[str, ...] = (
    "color-faithful residential storytelling with restrained contrast and honest textile tonality",
    "evidence-led home styling that preserves chromatic textile color instead of warming it toward black or brown",
)
CHROMATIC_SOFT_TEXTILE_CAMERA_PALETTE: tuple[str, ...] = (
    "balanced three-quarter bedding framing with readable surface color and full textile field continuity",
    "surface-led residential framing that keeps the broad textile plane visible without low-key darkening",
)
LOW_PROFILE_SOFT_SURFACE_CAMERA_PALETTE: tuple[str, ...] = (
    "broad surface-dominant framing that keeps the resting plane and perimeter height clearly readable",
    "clean three-quarter top view that preserves low perimeter relief without exaggerating edge thickness",
)
LOW_PROFILE_SOFT_SURFACE_DIRECTION_PALETTE: tuple[str, ...] = (
    "quiet home storytelling with broad surface readability and controlled perimeter relief",
    "evidence-led domestic styling that preserves a low-profile resting surface instead of plush bulk",
)
BUSINESS_PRIOR_CAST_PALETTES: dict[str, tuple[str, ...]] = {
    "playful_casual": (
        "a clearly different model with easy, casual body language and product-compatible everyday styling",
        "a new model with approachable relaxed energy and informal wardrobe",
    ),
    "refined_neutral": (
        "a clearly different model with polished casual styling, restrained body language, and no office-formal cues",
        "a new model with calm refined posture, modern everyday wardrobe, and no businesswear",
    ),
    "sport_utility": (
        "a clearly different model with active posture, grounded movement, and functional everyday styling",
        "a new model with purposeful athletic-adjacent body language and bright practical wardrobe",
    ),
    "cozy_home": (
        "a clearly different model with relaxed domestic posture and soft natural movement",
        "a new model with calm home-oriented styling and gentle believable body mechanics",
    ),
}
COMPACT_HAND_FOCUS_DIRECTION_PALETTES: dict[str, tuple[str, ...]] = {
    "playful_casual": (
        "close accessory storytelling with bright approachable energy and quiet wardrobe separation",
        "product-led hand-detail styling with cheerful movement and restrained neutral layering",
    ),
    "refined_neutral": (
        "tight accessory storytelling with calm neutral wardrobe separation and direct hand focus",
        "close hand-detail campaign pacing with restrained styling and strong product dominance",
    ),
    "sport_utility": (
        "compact accessory storytelling with clean utility clarity and quiet wardrobe separation",
        "direct-grip campaign pacing with functional hand detail and restrained body styling",
    ),
    "default": (
        "product-led close accessory storytelling with strong hand focus and neutral wardrobe separation",
        "tight hand-detail campaign pacing that keeps styling secondary to the product",
    ),
}
COMPACT_HAND_FOCUS_CAMERA_PALETTE: tuple[str, ...] = (
    "close hand-detail editorial framing that keeps the product large and dominant with minimal torso overlap",
    "waist-up accessory campaign framing with the product held near the camera and limited wardrobe area",
    "tight half-body accessory framing with direct grip, readable hands, and quiet body-adjacent styling",
)
COMPACT_HAND_FOCUS_CAST_PALETTES: dict[str, tuple[str, ...]] = {
    "playful_casual": (
        "a clearly different model with simple readable hand posing, muted casual wardrobe, and no large saturated color blocks",
        "a new model with approachable relaxed energy, neutral everyday layers, and product-first hand styling",
    ),
    "refined_neutral": (
        "a clearly different model with minimal neutral wardrobe, simple hand posing, and restrained body language",
        "a new model with calm refined posture, quiet tonal styling, and direct product-first handling",
    ),
    "sport_utility": (
        "a clearly different model with clean practical wardrobe, restrained color, and readable product-first hand posing",
        "a new model with functional casual styling, simple hand mechanics, and no saturated torso panels near the product",
    ),
    "default": (
        "a clearly different model with neutral wardrobe, simple readable hand posing, and product-first composition",
        "a new model with restrained everyday styling and direct hand focus instead of torso-led fashion posing",
    ),
}
EXPLICIT_HOME_PERSONA_TOKENS: frozenset[str] = frozenset(
    {
        "home",
        "cozy",
        "lounge",
        "loungewear",
        "sleep",
        "sleepwear",
        "pajama",
        "robe",
        "bed",
        "bedroom",
        "indoor",
    }
)
PERSON_CAPTION_TOKENS: frozenset[str] = frozenset(
    {
        "woman",
        "women",
        "man",
        "men",
        "person",
        "people",
        "model",
        "girl",
        "boy",
        "lady",
        "gentleman",
        "female",
        "male",
        "wearing",
        "holding",
        "carrying",
        "posing",
        "standing",
        "sitting",
    }
)
PERSON_ACCESSORY_TOKENS: frozenset[str] = frozenset(
    {
        "glasses",
        "sunglasses",
        "hat",
        "cap",
        "hair",
        "hairstyle",
        "face",
        "smile",
        "earrings",
        "necklace",
        "bracelet",
        "watch",
        "makeup",
    }
)
ANIMAL_FRAGMENT_TOKENS: frozenset[str] = frozenset(
    {
        "dog",
        "cat",
        "puppy",
        "kitten",
        "paw",
        "fur",
        "coat",
    }
)
COMPETING_CATEGORY_TYPE_TOKENS: dict[str, frozenset[str]] = {
    "apparel": frozenset({"dress", "shirt", "tee", "top", "blouse", "tunic", "hoodie", "skirt", "pants"}),
    "footwear": frozenset({"shoe", "sneaker", "sandal", "boot", "loafer", "trainer", "sock"}),
    "bag": frozenset({"bag", "handbag", "tote", "wallet", "backpack", "purse", "clutch"}),
    "drinkware": frozenset({"bottle", "cup", "mug", "glass", "tumbler", "flask"}),
    "home lighting": frozenset({"lamp", "shade", "lantern"}),
}
MATERIAL_TEXT_TOKENS: dict[str, tuple[str, ...]] = {
    "woven": ("woven", "weave", "woven texture"),
    "braided": ("braided", "braid", "corded"),
    "textured": ("textured", "embossed", "tufted", "ribbed"),
    "transparent": ("transparent", "clear", "translucent", "plastic window"),
    "metallic": ("metal", "metallic", "gold", "silver", "chrome"),
    "fabric": ("fabric", "cotton", "canvas", "linen", "textile"),
    "glossy": ("glossy", "shiny", "polished"),
    "matte": ("matte", "soft-touch"),
}
UPPER_COMPONENT_TEXT_TOKENS: tuple[str, ...] = (
    "handle",
    "handles",
    "strap",
    "straps",
    "loop",
    "loops",
    "lid",
    "cap",
    "zipper",
    "chain",
    "closure",
)
REINVENTION_CANDIDATE_MODES: tuple[str, ...] = ("balanced", "reveal", "hero")
HUMAN_REINVENTION_CANDIDATE_MODES: tuple[str, ...] = ("balanced", "clarity", "reveal", "hero")


@dataclass(frozen=True, slots=True)
class ReviewSeedRecord:
    id: str
    platform: str
    source_page_url: str
    source_image_url: str
    product_title: str
    hint_phrases: tuple[str, ...]
    capture_date: str
    local_image_path: Path


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    item_id: str
    image_name: str
    image_path: Path
    score: float
    page_views: int
    clicks: int
    caption: str
    embedding: tuple[float, ...]
    scenario_slots: tuple[str, ...]
    style_atoms: tuple[str, ...]
    scene_families: tuple[str, ...]
    support_relations: tuple[str, ...]
    category: str = "product"
    canonical_product_type: str = "product"
    support_mode: str | None = None
    default_scene_family: str | None = None
    interaction_mode: str | None = None
    observed_evidence: ObservedEvidenceSpec = field(default_factory=ObservedEvidenceSpec)


@dataclass(frozen=True, slots=True)
class LocalizationArtifactRecord:
    id: str
    product_title: str
    source_page_url: str
    source_image_url: str
    local_image_path: Path
    selected_phrase: str | None
    selected_confidence: float | None
    selected_box: BoundingBox | None
    overlay_path: Path | None
    crop_path: Path | None
    mask_path: Path | None


def load_review_seed_manifest(path: str | Path) -> list[ReviewSeedRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ReviewSeedRecord(
            id=str(record["id"]),
            platform=str(record["platform"]),
            source_page_url=str(record["source_page_url"]),
            source_image_url=str(record["source_image_url"]),
            product_title=str(record["product_title"]),
            hint_phrases=tuple(str(item) for item in record.get("hint_phrases", ())),
            capture_date=str(record["capture_date"]),
            local_image_path=_resolve_migrated_workspace_path(Path(str(record["local_image_path"]))),
        )
        for record in raw
    ]


def load_localization_report(path: str | Path) -> dict[str, LocalizationArtifactRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records: dict[str, LocalizationArtifactRecord] = {}
    for row in raw:
        box_data = row.get("selected_box")
        box = None
        if isinstance(box_data, dict):
            box = BoundingBox(
                x0=int(box_data["x0"]),
                y0=int(box_data["y0"]),
                x1=int(box_data["x1"]),
                y1=int(box_data["y1"]),
            )
        records[str(row["id"])] = LocalizationArtifactRecord(
            id=str(row["id"]),
            product_title=str(row["product_title"]),
            source_page_url=str(row["source_page_url"]),
            source_image_url=str(row["source_image_url"]),
            local_image_path=_resolve_migrated_workspace_path(Path(str(row["local_image_path"]))),
            selected_phrase=None if row.get("selected_phrase") is None else str(row["selected_phrase"]),
            selected_confidence=None
            if row.get("selected_confidence") is None
            else float(row["selected_confidence"]),
            selected_box=box,
            overlay_path=None
            if row.get("overlay_path") is None
            else _resolve_migrated_workspace_path(Path(str(row["overlay_path"]))),
            crop_path=None
            if row.get("crop_path") is None
            else _resolve_migrated_workspace_path(Path(str(row["crop_path"]))),
            mask_path=None
            if row.get("mask_path") is None
            else _resolve_migrated_workspace_path(Path(str(row["mask_path"]))),
        )
    return records


def load_retrieval_index(path: str | Path) -> list[RetrievalCandidate]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        RetrievalCandidate(
            item_id=str(record["item_id"]),
            image_name=str(record["image_name"]),
            image_path=_resolve_retrieval_candidate_image_path(Path(str(record["image_path"]))),
            score=float(record["score"]),
            page_views=int(record["page_views"]),
            clicks=int(record["clicks"]),
            caption=str(record["caption"]),
            embedding=tuple(float(value) for value in record["embedding"]),
            scenario_slots=tuple(str(value) for value in record.get("scenario_slots", ())),
            style_atoms=tuple(str(value) for value in record.get("style_atoms", ())),
            scene_families=tuple(
                str(value) for value in record.get("scene_families", infer_scene_families(str(record["caption"])))
            ),
            support_relations=tuple(
                str(value)
                for value in record.get("support_relations", infer_support_relations(str(record["caption"])))
            ),
            category=str(record.get("category", infer_category(str(record.get("caption", "")), str(record.get("image_name", ""))))),
            canonical_product_type=str(
                record.get(
                    "canonical_product_type",
                    infer_canonical_product_type(
                        str(record.get("caption", "")),
                        (),
                        str(record.get("caption", "")),
                    ),
                )
            ),
            support_mode=None if record.get("support_mode") is None else str(record.get("support_mode")),
            default_scene_family=(
                None if record.get("default_scene_family") is None else str(record.get("default_scene_family"))
            ),
            interaction_mode=None if record.get("interaction_mode") is None else str(record.get("interaction_mode")),
            observed_evidence=ObservedEvidenceSpec.model_validate(record.get("observed_evidence", {})),
        )
        for record in raw
    ]


def _resolve_retrieval_candidate_image_path(path: Path) -> Path:
    path = _resolve_migrated_workspace_path(path)
    if path.exists():
        return path

    relative_from_images = _relative_suffix_from_anchor(path.parts, "images")
    if relative_from_images is None:
        return path

    for root in _retrieval_image_roots():
        candidate = root / relative_from_images
        if candidate.exists():
            return candidate
    return path


def _retrieval_image_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[Path] = set()

    env_roots = (
        os.environ.get("PCP_CREATIVE_RANKING_IMAGE_ROOT"),
        os.environ.get("PCP_DATA_ROOT"),
    )
    for raw_root in env_roots:
        if not raw_root:
            continue
        root = Path(raw_root).expanduser()
        if root.name != "images":
            root = root / "images"
        if root not in seen:
            roots.append(root)
            seen.add(root)

    repo_workspace_root = Path(__file__).resolve().parents[3]
    default_roots = (
        repo_workspace_root / "data" / "images",
        Path("/workspace/data/images"),
        Path("/home/nyle_j_huang/data/images"),
    )
    for root in default_roots:
        if root not in seen:
            roots.append(root)
            seen.add(root)

    return tuple(roots)


def _resolve_migrated_workspace_path(path: Path) -> Path:
    if path.exists():
        return path

    workspace_root = Path(os.environ.get("PCP_WORKSPACE_ROOT", str(Path(__file__).resolve().parents[3])))
    project_root = Path(os.environ.get("PCP_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
    if not path.is_absolute():
        for base in (project_root, workspace_root):
            candidate = base / path
            if candidate.exists():
                return candidate
    legacy_mappings = (
        (Path("/home/nyle_j_huang/product_campaign_pipeline"), project_root),
        (Path("/home/nyle_j_huang/data"), workspace_root / "data"),
        (Path("/root/.codex"), Path("/root/.codex")),
    )
    for legacy_root, current_root in legacy_mappings:
        try:
            if path.is_absolute() and path.is_relative_to(legacy_root):
                candidate = current_root / path.relative_to(legacy_root)
                if candidate.exists():
                    return candidate
        except ValueError:
            continue
    return path


def _relative_suffix_from_anchor(parts: tuple[str, ...], anchor: str) -> Path | None:
    if anchor not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index(anchor)
    suffix = parts[index + 1 :]
    if not suffix:
        return None
    return Path(*suffix)


def export_retrieval_index(candidates: Sequence[RetrievalCandidate], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "item_id": candidate.item_id,
            "image_name": candidate.image_name,
            "image_path": str(candidate.image_path),
            "score": candidate.score,
            "page_views": candidate.page_views,
            "clicks": candidate.clicks,
            "caption": candidate.caption,
            "embedding": list(candidate.embedding),
            "scenario_slots": list(candidate.scenario_slots),
            "style_atoms": list(candidate.style_atoms),
            "scene_families": list(candidate.scene_families),
            "support_relations": list(candidate.support_relations),
            "category": candidate.category,
            "canonical_product_type": candidate.canonical_product_type,
            "support_mode": candidate.support_mode,
            "default_scene_family": candidate.default_scene_family,
            "interaction_mode": candidate.interaction_mode,
            "observed_evidence": candidate.observed_evidence.model_dump(mode="json"),
        }
        for candidate in candidates
    ]
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return destination


def select_top_creatives(
    manifest_path: str | Path,
    image_root: str | Path,
    *,
    top_k: int = 256,
    pool_size: int = 4096,
    min_page_views: int = 5,
) -> list[tuple[str, str, int, int, float, Path]]:
    image_root_path = Path(image_root)
    heap: list[tuple[float, str, str, int, int]] = []

    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 5:
                continue
            item_id, image_name, _, page_views_text, clicks_text = row
            page_views = int(page_views_text)
            clicks = int(clicks_text)
            if page_views < min_page_views:
                continue
            ctr = 0.0 if page_views <= 0 else clicks / page_views
            score = ctr + 0.03 * math.log1p(page_views) + 0.02 * math.log1p(clicks)
            entry = (score, item_id, image_name, page_views, clicks)
            if len(heap) < pool_size:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    ranked = sorted(heap, reverse=True)
    deduped: list[tuple[str, str, int, int, float, Path]] = []
    seen_items: set[str] = set()
    for score, item_id, image_name, page_views, clicks in ranked:
        if item_id in seen_items:
            continue
        image_path = image_root_path / image_name
        if not image_path.exists():
            continue
        seen_items.add(item_id)
        deduped.append((item_id, image_name, page_views, clicks, score, image_path))
        if len(deduped) >= top_k:
            break
    return deduped


def _canonical_type_matches_category(canonical_product_type: str, category: str) -> bool:
    if not canonical_product_type or canonical_product_type == "product":
        return False
    if category == "drinkware":
        return canonical_product_type in DRINKWARE_CANONICAL_TYPES
    if category == "home decor":
        return canonical_product_type == "decorative pillow"
    if category == "home lighting":
        return canonical_product_type == "table lamp"
    if category == "bedding":
        return canonical_product_type in BEDDING_CANONICAL_TYPES
    if category == "furniture":
        return canonical_product_type in FURNITURE_CANONICAL_TYPES
    if category == "kitchen appliance":
        return canonical_product_type in KITCHEN_APPLIANCE_CANONICAL_TYPES
    if category == "pet home":
        return canonical_product_type == "pet bed"
    if category == "bag":
        return canonical_product_type in {"tote bag", "handbag", "backpack", "wallet"}
    if category == "apparel":
        return canonical_product_type in {"shirt", "dress"}
    if category == "footwear":
        return canonical_product_type == "shoe"
    return False


def refine_retrieval_visual_classification(
    *,
    backbone: VisionBackbone,
    image_embedding: Sequence[float],
    category: str,
    canonical_product_type: str,
) -> tuple[str, str]:
    embedding = np.asarray(image_embedding, dtype=np.float32)
    resolved_category = category
    resolved_type = canonical_product_type

    should_refine_category = resolved_category == "product"
    if should_refine_category:
        category_names = tuple(RETRIEVAL_CATEGORY_IMAGE_PROMPTS)
        category_embeddings = backbone.encode_texts(
            [RETRIEVAL_CATEGORY_IMAGE_PROMPTS[name] for name in category_names]
        )
        category_scores = [
            float(np.dot(embedding, np.asarray(vector, dtype=np.float32)))
            for vector in category_embeddings
        ]
        best_index = int(np.argmax(category_scores))
        best_score = category_scores[best_index]
        second_score = max((score for i, score in enumerate(category_scores) if i != best_index), default=-1.0)
        if best_score >= 0.19 and (best_score - second_score) >= 0.015:
            resolved_category = category_names[best_index]

    should_refine_type = (
        resolved_type == "product"
        or not _canonical_type_matches_category(resolved_type, resolved_category)
    )
    type_prompts = RETRIEVAL_TYPE_IMAGE_PROMPTS.get(resolved_category)
    if should_refine_type and type_prompts:
        type_names = tuple(type_prompts)
        type_embeddings = backbone.encode_texts([type_prompts[name] for name in type_names])
        type_scores = [
            float(np.dot(embedding, np.asarray(vector, dtype=np.float32)))
            for vector in type_embeddings
        ]
        best_index = int(np.argmax(type_scores))
        best_score = type_scores[best_index]
        second_score = max((score for i, score in enumerate(type_scores) if i != best_index), default=-1.0)
        if best_score >= 0.18 and (best_score - second_score) >= 0.012:
            resolved_type = type_names[best_index]

    return resolved_category, resolved_type


class VisionBackbone:
    """Shared CLIP and BLIP runtime for retrieval prep and query encoding."""

    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._torch: Any | None = None
        self._clip_model: Any | None = None
        self._clip_processor: Any | None = None
        self._caption_model: Any | None = None
        self._caption_processor: Any | None = None
        self._text_embedding_cache: dict[str, tuple[float, ...]] = {}

    def encode_image(self, image_path: str | Path) -> tuple[float, ...]:
        clip_model, clip_processor, torch, device = self._ensure_clip()
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.inference_mode():
            features = clip_model.get_image_features(**inputs)
        normalized = features / features.norm(dim=-1, keepdim=True)
        return tuple(float(value) for value in normalized[0].detach().cpu().tolist())

    def caption_image(self, image_path: str | Path) -> str:
        caption_model, caption_processor, torch, device = self._ensure_caption()
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        inputs = caption_processor(images=image, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.inference_mode():
            output_ids = caption_model.generate(**inputs, max_new_tokens=24)
        return caption_processor.decode(output_ids[0], skip_special_tokens=True).strip().lower()

    def encode_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        embeddings: list[tuple[float, ...] | None] = [None] * len(texts)
        pending_indices: list[int] = []
        pending_texts: list[str] = []
        for index, text in enumerate(texts):
            key = " ".join(str(text).split()).strip().lower()
            cached = self._text_embedding_cache.get(key)
            if cached is not None:
                embeddings[index] = cached
                continue
            pending_indices.append(index)
            pending_texts.append(str(text))

        if pending_texts:
            clip_model, clip_processor, torch, device = self._ensure_clip()
            inputs = clip_processor(text=pending_texts, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            with torch.inference_mode():
                features = clip_model.get_text_features(**inputs)
            normalized = features / features.norm(dim=-1, keepdim=True)
            for index, vector in zip(pending_indices, normalized.detach().cpu().tolist(), strict=False):
                embedding = tuple(float(value) for value in vector)
                key = " ".join(str(texts[index]).split()).strip().lower()
                self._text_embedding_cache[key] = embedding
                embeddings[index] = embedding

        return [embedding for embedding in embeddings if embedding is not None]

    def _ensure_clip(self) -> tuple[Any, Any, Any, str]:
        if self._clip_model is not None and self._clip_processor is not None and self._torch is not None:
            return self._clip_model, self._clip_processor, self._torch, _resolve_device(self._torch, self.device)

        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = _resolve_device(torch, self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
        self._clip_model.eval()
        self._torch = torch
        return self._clip_model, self._clip_processor, torch, device

    def _ensure_caption(self) -> tuple[Any, Any, Any, str]:
        if self._caption_model is not None and self._caption_processor is not None and self._torch is not None:
            return self._caption_model, self._caption_processor, self._torch, _resolve_device(self._torch, self.device)

        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        device = _resolve_device(torch, self.device)
        self._caption_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        self._caption_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to(device)
        self._caption_model.eval()
        self._torch = torch
        return self._caption_model, self._caption_processor, torch, device


def build_retrieval_index(
    manifest_path: str | Path,
    image_root: str | Path,
    *,
    output_path: str | Path,
    top_k: int = 256,
    pool_size: int = 4096,
    min_page_views: int = 5,
    device: str = "cuda",
) -> Path:
    selected = select_top_creatives(
        manifest_path,
        image_root,
        top_k=top_k,
        pool_size=pool_size,
        min_page_views=min_page_views,
    )
    backbone = VisionBackbone(device=device)
    candidates: list[RetrievalCandidate] = []
    for item_id, image_name, page_views, clicks, score, image_path in selected:
        caption = backbone.caption_image(image_path)
        embedding = backbone.encode_image(image_path)
        category = infer_category(caption, image_name)
        initial_canonical_product_type = infer_canonical_product_type(caption, (), caption)
        category, initial_canonical_product_type = refine_retrieval_visual_classification(
            backbone=backbone,
            image_embedding=embedding,
            category=category,
            canonical_product_type=initial_canonical_product_type,
        )
        weak_shape_evidence = has_weak_shape_evidence(
            caption,
            (),
            canonical_product_type=initial_canonical_product_type,
        )
        affordance = infer_affordance_profile(
            category,
            canonical_product_type=initial_canonical_product_type,
            product_title=caption,
            hint_phrases=(),
        )
        observed_evidence = infer_observed_evidence(
            category=category,
            canonical_product_type=initial_canonical_product_type,
            product_title=caption,
            hint_phrases=(),
            source_image=image_path,
            crop_path=None,
            mask_path=None,
            localization_crop_path=None,
            stable_base=affordance["stable_base"],
            weak_shape_evidence=weak_shape_evidence,
            requires_human_model=False,
            backbone=backbone,
        )
        canonical_product_type = refine_canonical_product_type(
            category=category,
            initial_canonical_product_type=initial_canonical_product_type,
            product_title=caption,
            hint_phrases=(),
            selected_phrase=caption,
            observed_evidence=observed_evidence,
        )
        observed_evidence = rewrite_evidence_for_canonical_type(
            observed_evidence,
            canonical_product_type=canonical_product_type,
        )
        affordance = infer_affordance_profile(
            category,
            canonical_product_type=canonical_product_type,
            product_title=caption,
            hint_phrases=(),
        )
        scene_families = infer_scene_families(caption)
        support_relations = infer_support_relations(caption)
        scenario_slots = infer_scenario_slots(caption)
        style_atoms = infer_style_atoms(caption)
        candidates.append(
            RetrievalCandidate(
                item_id=item_id,
                image_name=image_name,
                image_path=image_path,
                score=score,
                page_views=page_views,
                clicks=clicks,
                caption=caption,
                embedding=embedding,
                scenario_slots=scenario_slots,
                style_atoms=style_atoms,
                scene_families=scene_families,
                support_relations=support_relations,
                category=category,
                canonical_product_type=canonical_product_type,
                support_mode=affordance["support_mode"],
                default_scene_family=affordance["default_scene_family"],
                interaction_mode=affordance["interaction_mode"],
                observed_evidence=observed_evidence,
            )
        )
    return export_retrieval_index(candidates, output_path)


def infer_style_persona(
    *,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    observed_evidence: ObservedEvidenceSpec,
    source_image: Path,
    crop_path: Path | None,
    backbone: VisionBackbone | None,
) -> str:
    scores = {
        "playful_casual": 0.0,
        "refined_neutral": 0.0,
        "sport_utility": 0.0,
        "cozy_home": 0.0,
    }
    text = " ".join(
        part
        for part in (
            category,
            product_title,
            *hint_phrases,
            observed_evidence.evidence_caption or "",
            observed_evidence.pattern_note or "",
            observed_evidence.material_note or "",
            observed_evidence.color_note or "",
            observed_evidence.coverage_note or "",
        )
        if part
    ).lower()
    tokens = set(_tokens(text))
    for persona, hints in STYLE_PERSONA_TEXT_HINTS.items():
        scores[persona] += 0.55 * len(tokens.intersection(hints))

    if category == "home decor":
        scores["cozy_home"] += 0.8
    elif category == "home lighting":
        scores["cozy_home"] += 0.55
        scores["refined_neutral"] += 0.35
    elif category == "bedding":
        scores["cozy_home"] += 0.75
    elif category == "furniture":
        scores["refined_neutral"] += 0.45
    elif category == "kitchen appliance":
        scores["refined_neutral"] += 0.55
        scores["sport_utility"] += 0.1
    elif category == "pet home":
        scores["cozy_home"] += 0.6
    elif category == "drinkware":
        if canonical_product_type == "mug":
            scores["refined_neutral"] += 0.45
            scores["cozy_home"] += 0.3
            scores["sport_utility"] -= 0.05
        else:
            scores["sport_utility"] += 0.35
    elif category == "bag":
        scores["refined_neutral"] += 0.2
        if {"sport", "sports", "baseball", "backpack", "utility", "cooler", "insulated", "lunch", "travel"} & tokens:
            scores["sport_utility"] += 0.35
    elif category == "footwear":
        scores["refined_neutral"] += 0.2
        if {"running", "walking", "sport", "athletic"} & tokens:
            scores["sport_utility"] += 0.35
    elif category == "apparel":
        scores["refined_neutral"] += 0.2

    if observed_evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        scores["playful_casual"] += 0.2
    if category in {"apparel", "footwear", "bag", "kitchen appliance", "furniture"}:
        scores["cozy_home"] *= 0.35
    explicit_home_persona = bool(tokens & EXPLICIT_HOME_PERSONA_TOKENS)
    if category == "apparel" and not explicit_home_persona:
        scores["cozy_home"] *= 0.12
        if canonical_product_type in {"shirt", "dress"}:
            scores["refined_neutral"] += 0.1
    if category == "footwear":
        scores["cozy_home"] *= 0.15
    if category == "bag" and canonical_product_type == "backpack":
        scores["cozy_home"] *= 0.08
        scores["sport_utility"] += 0.25

    candidate_path = crop_path if crop_path is not None and crop_path.exists() else source_image
    if backbone is not None and candidate_path.exists():
        if category == "drinkware":
            if canonical_product_type == "mug":
                personas = ("refined_neutral", "cozy_home", "sport_utility")
            else:
                personas = ("sport_utility", "refined_neutral")
        elif category == "home decor":
            personas = ("cozy_home", "refined_neutral")
        elif category == "home lighting":
            personas = ("cozy_home", "refined_neutral")
        elif category == "bedding":
            personas = ("cozy_home", "refined_neutral")
        elif category == "furniture":
            personas = ("refined_neutral", "cozy_home")
        elif category == "kitchen appliance":
            personas = ("refined_neutral", "sport_utility")
        elif category == "pet home":
            personas = ("cozy_home", "refined_neutral")
        elif category == "bag":
            personas = ("playful_casual", "refined_neutral", "sport_utility")
        elif category == "footwear":
            personas = ("playful_casual", "refined_neutral", "sport_utility")
        elif category == "apparel":
            personas = ("playful_casual", "refined_neutral", "sport_utility")
        else:
            personas = tuple(STYLE_PERSONA_IMAGE_PROMPTS)
        try:
            image_embedding = np.asarray(backbone.encode_image(candidate_path), dtype=np.float32)
            text_embeddings = backbone.encode_texts([STYLE_PERSONA_IMAGE_PROMPTS[persona] for persona in personas])
            for persona, vector in zip(personas, text_embeddings, strict=False):
                scores[persona] += 0.45 * float(np.dot(image_embedding, np.asarray(vector, dtype=np.float32)))
        except Exception:
            pass

    default_persona = "refined_neutral"
    if category == "drinkware":
        default_persona = "refined_neutral" if canonical_product_type == "mug" else "sport_utility"
    elif category in {"home decor", "home lighting", "bedding", "pet home"}:
        default_persona = "cozy_home"
    elif category in {"furniture", "kitchen appliance"}:
        default_persona = "refined_neutral"
    elif category == "bag" and canonical_product_type == "backpack":
        default_persona = "sport_utility"
    best_persona, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < 0.25:
        return default_persona
    return best_persona


def infer_casting_note(
    *,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    observed_evidence: ObservedEvidenceSpec,
) -> str | None:
    if category not in {"bag", "apparel", "footwear"}:
        return None
    text = " ".join(
        part
        for part in (
            category,
            canonical_product_type,
            product_title,
            *hint_phrases,
            observed_evidence.evidence_caption or "",
            observed_evidence.pattern_note or "",
            observed_evidence.coverage_note or "",
            observed_evidence.material_note or "",
        )
        if part
    ).lower()
    tokens = set(_tokens(text))
    playful_tokens = {
        "cute",
        "cartoon",
        "character",
        "disney",
        "floral",
        "flower",
        "bow",
        "princess",
        "playful",
        "romantic",
        "delicate",
    }
    sport_tokens = {"sport", "sports", "athletic", "running", "utility", "outdoor", "hydration", "baseball"}
    formal_tokens = {"tailored", "minimal", "structured", "classic", "refined", "office"}
    if tokens.intersection(playful_tokens):
        return (
            "If a person appears, use casting and hand styling that feel playful, soft, or feminine-coded in a way "
            "that matches the product's visual language. Avoid masculine-coded or business-formal presentation when it "
            "conflicts with the observed design."
        )
    if tokens.intersection(sport_tokens):
        return (
            "If a person appears, use active, neutral casting with grounded everyday styling rather than luxury-editorial "
            "or fashion-formal presentation."
        )
    if category == "bag" and canonical_product_type == "wallet":
        return (
            "If a person appears, keep the hand styling and casting understated and accessory-compatible instead of "
            "masculine formal or business-coded."
        )
    if tokens.intersection(formal_tokens):
        return (
            "If a person appears, keep the casting refined and neutral without drifting into office-formal or corporate styling."
        )
    return None


def select_casting_alignment_eval_prompts(
    identity: ProductIdentitySpec,
    canonical_product_type: str,
) -> dict[str, tuple[str, ...]] | None:
    note = str(identity.casting_note or "").strip().lower()
    if not note:
        return None
    if "feminine-coded" in note or ("playful" in note and "soft" in note):
        key = "soft_feminine"
    elif "active, neutral" in note or "active neutral" in note:
        key = "active_neutral"
    elif "accessory-compatible" in note or "accessory compatible" in note:
        key = "accessory_compatible"
    else:
        return None
    prompts = CASTING_ALIGNMENT_EVAL_PROMPTS[key]
    return {
        "positive": tuple(template.format(product=canonical_product_type) for template in prompts["positive"]),
        "negative": tuple(template.format(product=canonical_product_type) for template in prompts["negative"]),
    }


def select_functional_subtype_eval_prompts(
    identity: ProductIdentitySpec,
    canonical_product_type: str,
) -> dict[str, tuple[str, ...]] | None:
    subtype = str(identity.subtype_hint or "").strip().lower()
    if not subtype:
        return None
    prompts = FUNCTIONAL_SUBTYPE_EVAL_PROMPTS.get(subtype)
    if prompts is None:
        return None
    label = subtype if canonical_product_type.lower() not in subtype else canonical_product_type
    return {
        "positive": tuple(template.format(product=label) for template in prompts["positive"]),
        "negative": tuple(template.format(product=label) for template in prompts["negative"]),
    }


def build_localized_product(
    seed: ReviewSeedRecord,
    record: LocalizationArtifactRecord,
    backbone: VisionBackbone | None = None,
) -> LocalizedProduct:
    initial_canonical_product_type = infer_canonical_product_type(
        seed.product_title,
        seed.hint_phrases,
        record.selected_phrase or "",
    )
    phrase = build_identity_phrase(
        seed.product_title,
        seed.hint_phrases,
        record.selected_phrase or "",
        canonical_product_type=initial_canonical_product_type,
    )
    category = infer_category(seed.product_title, " ".join(seed.hint_phrases), phrase)
    requires_human_model = category in {"apparel", "footwear"}
    weak_shape_evidence = has_weak_shape_evidence(
        record.selected_phrase or "",
        seed.hint_phrases,
        canonical_product_type=initial_canonical_product_type,
    )
    initial_affordance = infer_affordance_profile(
        category,
        canonical_product_type=initial_canonical_product_type,
        product_title=seed.product_title,
        hint_phrases=seed.hint_phrases,
    )
    observed_evidence = infer_observed_evidence(
        category=category,
        canonical_product_type=initial_canonical_product_type,
        product_title=seed.product_title,
        hint_phrases=seed.hint_phrases,
        source_image=record.local_image_path,
        crop_path=record.crop_path,
        mask_path=record.mask_path,
        localization_crop_path=record.crop_path,
        stable_base=initial_affordance["stable_base"],
        weak_shape_evidence=weak_shape_evidence,
        requires_human_model=requires_human_model,
        backbone=backbone,
        localization_confidence=record.selected_confidence,
    )
    canonical_product_type = refine_canonical_product_type(
        category=category,
        initial_canonical_product_type=initial_canonical_product_type,
        product_title=seed.product_title,
        hint_phrases=seed.hint_phrases,
        selected_phrase=record.selected_phrase or "",
        observed_evidence=observed_evidence,
    )
    if canonical_product_type != initial_canonical_product_type:
        phrase = build_refined_identity_phrase(
            selected_phrase=record.selected_phrase or "",
            hint_phrases=seed.hint_phrases,
            canonical_product_type=canonical_product_type,
        )
    observed_evidence = rewrite_evidence_for_canonical_type(
        observed_evidence,
        canonical_product_type=canonical_product_type,
    )
    affordance = infer_affordance_profile(
        category,
        canonical_product_type=canonical_product_type,
        product_title=seed.product_title,
        hint_phrases=seed.hint_phrases,
    )
    subtype_hint = infer_functional_subtype_hint(
        category=category,
        canonical_product_type=canonical_product_type,
        product_title=seed.product_title,
        hint_phrases=seed.hint_phrases,
        selected_phrase=record.selected_phrase or "",
    )
    style_persona = infer_style_persona(
        category=category,
        canonical_product_type=canonical_product_type,
        product_title=seed.product_title,
        hint_phrases=seed.hint_phrases,
        observed_evidence=observed_evidence,
        source_image=record.local_image_path,
        crop_path=record.crop_path,
        backbone=backbone,
    )
    casting_note = infer_casting_note(
        category=category,
        canonical_product_type=canonical_product_type,
        product_title=seed.product_title,
        hint_phrases=seed.hint_phrases,
        observed_evidence=observed_evidence,
    )
    box = _coerce_localized_product_bbox(record.selected_box)
    return LocalizedProduct(
        source_image=str(record.local_image_path),
        phrase=phrase,
        bbox=box,
        confidence=0.0 if record.selected_confidence is None else record.selected_confidence,
        crop_path=(
            observed_evidence.reference_crop_path
            if observed_evidence.reference_crop_path
            else (None if record.crop_path is None else str(record.crop_path))
        ),
        mask_path=(
            observed_evidence.reference_mask_path
            if observed_evidence.reference_mask_path
            else (None if record.mask_path is None else str(record.mask_path))
        ),
        identity=ProductIdentitySpec(
            phrase=phrase,
            category=category,
            canonical_product_type=canonical_product_type,
            subtype_hint=subtype_hint or (None if canonical_product_type == initial_canonical_product_type else canonical_product_type),
            source_title=seed.product_title,
            support_mode=affordance["support_mode"],
            default_scene_family=affordance["default_scene_family"],
            interaction_mode=affordance["interaction_mode"],
            style_persona=style_persona,
            casting_note=casting_note,
            stable_base=affordance["stable_base"],
            rigid_vs_soft=affordance["rigid_vs_soft"],
            requires_human_model=requires_human_model,
            weak_shape_evidence=weak_shape_evidence,
            observed_evidence=observed_evidence,
        ),
    )


def _coerce_localized_product_bbox(box: Any | None) -> BoundingBox:
    if box is None:
        return BoundingBox(x0=0, y0=0, x1=512, y1=512)
    if isinstance(box, BoundingBox):
        return box
    if isinstance(box, dict):
        return BoundingBox(
            x0=int(box["x0"]),
            y0=int(box["y0"]),
            x1=int(box["x1"]),
            y1=int(box["y1"]),
        )
    if all(hasattr(box, attr) for attr in ("x0", "y0", "x1", "y1")):
        return BoundingBox(
            x0=int(box.x0),
            y0=int(box.y0),
            x1=int(box.x1),
            y1=int(box.y1),
        )
    raise TypeError(f"unsupported localization bounding box type: {type(box).__name__}")


def infer_observed_evidence(
    *,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    source_image: Path,
    crop_path: Path | None,
    mask_path: Path | None,
    localization_crop_path: Path | None = None,
    stable_base: bool | None,
    weak_shape_evidence: bool,
    requires_human_model: bool,
    backbone: VisionBackbone | None,
    localization_confidence: float | None = None,
) -> ObservedEvidenceSpec:
    prepared_assets = prepare_observed_evidence_assets(
        source_image=source_image,
        mask_path=mask_path,
        localization_crop_path=localization_crop_path,
        category=category,
        canonical_product_type=canonical_product_type,
        requires_human_model=requires_human_model,
    )
    cutout_path = prepared_assets.cutout_path
    silhouette_path = prepared_assets.silhouette_path
    evidence_crop_path = prepared_assets.crop_path or crop_path
    effective_mask_path = prepared_assets.mask_path or mask_path
    surface_mask_path = prepare_surface_evidence_mask(
        source_image=source_image,
        mask_path=effective_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
    )

    structural_ranked = infer_named_palette_ranked(
        source_image,
        surface_mask_path,
        top_k=3,
        use_smoothed=True,
        erode_steps=2,
    )
    accent_ranked = infer_named_palette_ranked(
        source_image,
        surface_mask_path,
        top_k=4,
        use_smoothed=False,
        erode_steps=0,
    )
    core_body_ranked = infer_core_body_palette_ranked(
        source_image,
        surface_mask_path,
        top_k=3,
    )
    structural_palette = [name for name, _ in structural_ranked]
    accent_palette = [name for name, _ in accent_ranked]
    palette = structural_palette or accent_palette
    coverage_class, coverage_ratio, coverage_note = infer_surface_coverage_profile(
        source_image,
        surface_mask_path,
        palette=accent_palette or palette,
        base_palette=structural_palette or palette,
    )
    shape_profile = infer_shape_profile(effective_mask_path)
    contrast_panel_note, value_relation_note = infer_localized_contrast_panel(
        source_image,
        surface_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        body_color=None if not core_body_ranked else core_body_ranked[0][0],
    )
    if _should_suppress_contrast_panel_inference(
        canonical_product_type=canonical_product_type,
        shape_profile=shape_profile,
        contrast_panel_note=contrast_panel_note,
    ):
        contrast_panel_note = None
        value_relation_note = None
    if contrast_panel_note and coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        coverage_class = "localized_visible_pattern"
        coverage_note = contrast_panel_note
    pattern_note = infer_pattern_note(
        source_image,
        surface_mask_path,
        palette=accent_palette or palette,
        base_palette=structural_palette or palette,
        coverage_class=coverage_class,
    )
    color_note, color_confidence = infer_color_note(
        core_body_ranked=core_body_ranked or structural_ranked or accent_ranked,
        structural_ranked=structural_ranked or core_body_ranked or accent_ranked,
        accent_ranked=accent_ranked or structural_ranked or core_body_ranked,
        coverage_class=coverage_class,
    )
    if contrast_panel_note and coverage_class == "localized_visible_pattern":
        pattern_note = None
    if color_confidence is not None and color_confidence < 0.55:
        coverage_note, pattern_note = generalize_low_confidence_surface_notes(
            coverage_class=coverage_class,
            coverage_note=coverage_note,
            pattern_note=pattern_note,
        )
    boundary_color, interior_color, trim_note, trim_confidence = infer_trim_profile(source_image, surface_mask_path)
    silhouette_note = shape_profile["note"]
    upper_region_profile = infer_upper_region_profile(source_image, effective_mask_path)
    mask_has_border_spill = False
    if effective_mask_path is not None and effective_mask_path.exists():
        with Image.open(source_image) as source_handle:
            source_shape = np.asarray(source_handle.convert("RGB")).shape[:2]
        effective_mask = _load_mask_array(effective_mask_path, source_shape=source_shape)
        mask_has_border_spill = bool(effective_mask is not None and _mask_has_heavy_border_spill(effective_mask))
    if mask_has_border_spill and category in {"furniture", "kitchen appliance", "home lighting"}:
        upper_region_profile = {
            "note": None,
            "upper_region_color": None,
            "body_region_color": None,
            "upper_component_count": None,
            "confidence": None,
            "component_state": "uncertain",
        }
    form_factor_note = infer_form_factor_note(
        category=category,
        canonical_product_type=canonical_product_type,
        shape_profile=shape_profile,
        upper_region_profile=upper_region_profile,
    )
    prefer_source_context = bool(
        infer_functional_subtype_hint(
            category=category,
            canonical_product_type=canonical_product_type,
            product_title=product_title,
            hint_phrases=hint_phrases,
            selected_phrase="",
        )
    )
    raw_evidence_caption = infer_raw_evidence_caption(
        backbone,
        source_image=source_image,
        cutout_path=cutout_path,
        crop_path=evidence_crop_path,
        prefer_source_context=prefer_source_context,
    )
    evidence_caption = infer_evidence_caption(
        backbone,
        source_image=source_image,
        cutout_path=cutout_path,
        crop_path=evidence_crop_path,
        canonical_product_type=canonical_product_type,
        category=category,
        prefer_source_context=prefer_source_context,
    )
    evidence_caption = sanitize_evidence_caption(
        evidence_caption,
        canonical_product_type=canonical_product_type,
        category=category,
        requires_human_model=requires_human_model,
    )
    caption_colors = extract_caption_colors(evidence_caption)
    if should_apply_caption_color_override(
        evidence_caption=evidence_caption,
        canonical_product_type=canonical_product_type,
        category=category,
        color_confidence=color_confidence,
        coverage_class=coverage_class,
    ):
        color_note = reconcile_color_note_with_caption(
            color_note,
            caption_colors=caption_colors,
            coverage_class=coverage_class,
        )
        if len(caption_colors) >= 2:
            palette = caption_colors
    if evidence_caption and color_confidence is not None and color_confidence < 0.55:
        color_note = None
    coverage_class, coverage_note, pattern_note, color_note, color_confidence, upper_region_profile = (
        correct_footwear_surface_inference(
            source_image=source_image,
            mask_path=surface_mask_path,
            category=category,
            canonical_product_type=canonical_product_type,
            coverage_class=coverage_class,
            coverage_note=coverage_note,
            pattern_note=pattern_note,
            color_note=color_note,
            color_confidence=color_confidence,
            upper_region_profile=upper_region_profile,
        )
    )
    coverage_class, coverage_note, pattern_note, color_note, color_confidence = (
        correct_soft_textile_surface_inference(
            source_image=source_image,
            mask_path=surface_mask_path,
            category=category,
            canonical_product_type=canonical_product_type,
            coverage_class=coverage_class,
            coverage_note=coverage_note,
            pattern_note=pattern_note,
            color_note=color_note,
            color_confidence=color_confidence,
        )
    )
    material_note = infer_material_note(
        category=category,
        canonical_product_type=canonical_product_type,
        product_title=product_title,
        hint_phrases=hint_phrases,
        evidence_caption=evidence_caption,
    )
    coverage_class, coverage_note, pattern_note = correct_structured_display_surface_inference(
        category=category,
        canonical_product_type=canonical_product_type,
        stable_base=stable_base,
        shape_profile=shape_profile,
        evidence_caption=evidence_caption,
        coverage_class=coverage_class,
        coverage_note=coverage_note,
        pattern_note=pattern_note,
    )
    if mask_has_border_spill and category in {"furniture", "kitchen appliance", "home lighting"} and core_body_ranked:
        base_color = core_body_ranked[0][0]
        structural_palette = [base_color]
        palette = [base_color]
        accent_palette = [color for color in accent_palette if color in {base_color, "black", "gray", "white"}]
    if (
        coverage_class == "low_variation_surface"
        and color_note
        and "compatible printed accents" in color_note
        and core_body_ranked
    ):
        color_note = f"the main visible body reads as {core_body_ranked[0][0]}"
    reflective_color_note, reflective_palette, reflective_confidence = infer_dark_reflective_body_override(
        source_image=source_image,
        mask_path=surface_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        coverage_class=coverage_class,
        palette=palette,
    )
    if reflective_color_note:
        color_note = reflective_color_note
        color_confidence = reflective_confidence
        if reflective_palette:
            palette = reflective_palette
            structural_palette = reflective_palette
            pattern_note = synchronize_pattern_note_with_dominant_color(
                pattern_note=pattern_note,
                dominant_color=reflective_palette[0],
            )
        coverage_note = synchronize_coverage_note_with_dominant_color(
            coverage_note=coverage_note,
            coverage_class=coverage_class,
            dominant_color=reflective_palette[0] if reflective_palette else extract_dominant_body_color(
                ObservedEvidenceSpec(color_note=reflective_color_note, palette=list(palette))
            ),
            category=category,
            canonical_product_type=canonical_product_type,
        )
    textile_color_note, textile_coverage_note, textile_pattern_note = infer_neutral_textile_surface_notes(
        material_note=material_note,
        palette=palette,
        coverage_class=coverage_class,
        category=category,
        canonical_product_type=canonical_product_type,
        source_image=source_image,
        mask_path=surface_mask_path,
    )
    translucent_color_note = infer_translucent_surface_note(
        source_image,
        surface_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        aspect_ratio=shape_profile["aspect_ratio"],
        coverage_class=coverage_class,
    )
    surface_relief_note = infer_surface_relief_note(
        source_image=source_image,
        mask_path=effective_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    lower_region_profile = infer_lower_region_profile(
        source_image=source_image,
        mask_path=effective_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    edge_profile = infer_edge_profile(
        source_image=source_image,
        mask_path=effective_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    soft_structure = infer_soft_structure_profile(
        source_image=source_image,
        mask_path=effective_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        edge_thickness_class=edge_profile["thickness_class"],
    )
    if translucent_color_note:
        color_note = translucent_color_note
    if textile_color_note:
        color_note = textile_color_note
        color_confidence = 0.68
    if textile_coverage_note:
        coverage_note = textile_coverage_note
    if textile_pattern_note:
        pattern_note = textile_pattern_note
    soft_textile_color_note, soft_textile_palette, soft_textile_confidence = infer_soft_textile_chromatic_override(
        source_image=source_image,
        mask_path=surface_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        coverage_class=coverage_class,
        palette=palette,
    )
    if soft_textile_color_note:
        color_note = soft_textile_color_note
        color_confidence = soft_textile_confidence
        if soft_textile_palette:
            palette = soft_textile_palette
            structural_palette = soft_textile_palette
            coverage_note = synchronize_coverage_note_with_dominant_color(
                coverage_note=coverage_note,
                coverage_class=coverage_class,
                dominant_color=soft_textile_palette[0],
                category=category,
                canonical_product_type=canonical_product_type,
            )
            pattern_note = synchronize_pattern_note_with_dominant_color(
                pattern_note=pattern_note,
                dominant_color=soft_textile_palette[0],
            )
    low_saturation_cool_note, low_saturation_cool_palette, low_saturation_cool_confidence = (
        infer_low_saturation_cool_body_override(
            source_image=source_image,
            mask_path=surface_mask_path,
            category=category,
            canonical_product_type=canonical_product_type,
            coverage_class=coverage_class,
            palette=palette,
        )
    )
    if low_saturation_cool_note:
        color_note = low_saturation_cool_note
        color_confidence = low_saturation_cool_confidence
        if low_saturation_cool_palette:
            palette = low_saturation_cool_palette
            structural_palette = low_saturation_cool_palette
            coverage_note = synchronize_coverage_note_with_dominant_color(
                coverage_note=coverage_note,
                coverage_class=coverage_class,
                dominant_color=low_saturation_cool_palette[0],
                category=category,
                canonical_product_type=canonical_product_type,
            )
            pattern_note = synchronize_pattern_note_with_dominant_color(
                pattern_note=pattern_note,
                dominant_color=low_saturation_cool_palette[0],
            )
    color_note, color_confidence, softened_palette = soften_uncertain_neutral_apparel_color_evidence(
        source_image=source_image,
        mask_path=surface_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        product_title=product_title,
        hint_phrases=hint_phrases,
        evidence_caption=evidence_caption,
        coverage_class=coverage_class,
        color_note=color_note,
        color_confidence=color_confidence,
        palette=palette,
    )
    if softened_palette:
        palette = softened_palette
        structural_palette = softened_palette
    coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence = (
        correct_supported_soft_surface_inference(
            category=category,
            canonical_product_type=canonical_product_type,
            product_title=product_title,
            hint_phrases=hint_phrases,
            evidence_caption=evidence_caption,
            coverage_class=coverage_class,
            coverage_ratio=coverage_ratio,
            coverage_note=coverage_note,
            pattern_note=pattern_note,
            color_note=color_note,
            color_confidence=color_confidence,
            palette=palette,
        )
    )
    evidence_caption = sanitize_evidence_caption_against_color_evidence(
        evidence_caption,
        canonical_product_type=canonical_product_type,
        palette=palette,
        color_note=color_note,
        color_confidence=color_confidence,
    )
    backpack_structure_note = infer_backpack_structure_note(
        category=category,
        canonical_product_type=canonical_product_type,
        palette=palette,
        accent_palette=accent_palette,
        evidence_caption=evidence_caption,
    )
    if backpack_structure_note and coverage_class == "low_variation_surface":
        coverage_note = (
            "the visible backpack body includes structured darker panel, harness, or attachment zoning rather than a single uniform color block"
        )
    functional_subtype_facts = infer_functional_subtype_hard_facts(
        category=category,
        canonical_product_type=canonical_product_type,
        product_title=product_title,
        hint_phrases=hint_phrases,
        selected_phrase=evidence_caption or "",
    )

    hard_facts = [f"the product remains a {canonical_product_type}"]
    hard_facts.extend(functional_subtype_facts)
    if evidence_caption:
        hard_facts.append(f"source evidence reads as {evidence_caption}")
    if color_note:
        hard_facts.append(color_note)
    if coverage_note:
        hard_facts.append(coverage_note)
    if backpack_structure_note:
        hard_facts.append(backpack_structure_note)
    if value_relation_note:
        hard_facts.append(value_relation_note)
    if material_note:
        hard_facts.append(material_note)
    if surface_relief_note:
        hard_facts.append(surface_relief_note)
    if lower_region_profile["note"]:
        hard_facts.append(str(lower_region_profile["note"]))
    if edge_profile["note"]:
        hard_facts.append(str(edge_profile["note"]))
    if soft_structure["note"]:
        hard_facts.append(str(soft_structure["note"]))
    if palette and color_confidence is not None and color_confidence >= 0.72 and coverage_class != "localized_visible_pattern":
        hard_facts.append(f"observed palette includes {', '.join(palette[:3])}")
    if pattern_note:
        hard_facts.append(pattern_note)
    if trim_note:
        hard_facts.append(trim_note)
    if upper_region_profile["note"]:
        hard_facts.append(str(upper_region_profile["note"]))
    if form_factor_note:
        hard_facts.append(form_factor_note)
    if silhouette_note:
        hard_facts.append(silhouette_note)

    surface_scope = infer_surface_scope(
        effective_mask_path,
        weak_shape_evidence=weak_shape_evidence,
        stable_base=stable_base,
    )
    uncertainty_level = infer_evidence_uncertainty(
        surface_scope=surface_scope,
        weak_shape_evidence=weak_shape_evidence,
        stable_base=stable_base,
    )
    soft_hypotheses = [
        "unseen surfaces may be reinvented for a cleaner campaign presentation if they stay compatible with the observed evidence",
        "viewpoint, pose, support, and drape may change as long as the observed details remain believable",
    ]
    unknowns = [
        "full front presentation and other unseen faces",
        "rear view, interior details, and hidden surfaces",
        "the exact continuation of localized patterns, wrinkles, or trims across unobserved areas",
    ]
    evidence_tags = build_evidence_tags(
        coverage_class=coverage_class,
        trim_note=trim_note,
        upper_region_note=upper_region_profile["note"],
        upper_component_state=upper_region_profile["component_state"],
        lower_region_note=lower_region_profile["note"],
        lower_component_state=lower_region_profile["component_state"],
        form_factor_note=form_factor_note,
        material_note=material_note,
        surface_relief_note=surface_relief_note,
        edge_profile_note=edge_profile["note"],
        soft_structure_note=soft_structure["note"],
    )
    evidence_tags = _dedupe_strings([*evidence_tags, *prepared_assets.artifact_flags])
    source_validity, source_validity_score, source_validity_issues = assess_source_validity(
        source_image=source_image,
        mask_path=effective_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        observed_evidence=ObservedEvidenceSpec(
            surface_scope=surface_scope,
            uncertainty_level=uncertainty_level,
            palette=palette,
            structural_palette=structural_palette,
            accent_palette=accent_palette,
            color_note=color_note,
            color_confidence=None if color_confidence is None else round(float(color_confidence), 4),
            coverage_class=coverage_class,
            coverage_note=coverage_note,
            trim_note=trim_note,
            silhouette_note=silhouette_note,
            aspect_ratio=shape_profile["aspect_ratio"],
            top_width_ratio=shape_profile["top_width_ratio"],
            form_factor_note=form_factor_note,
            upper_region_note=upper_region_profile["note"],
            lower_region_note=lower_region_profile["note"],
            edge_profile_note=edge_profile["note"],
            soft_structure_note=soft_structure["note"],
            artifact_flags=list(prepared_assets.artifact_flags),
            raw_evidence_caption=raw_evidence_caption,
            evidence_caption=evidence_caption,
        ),
        weak_shape_evidence=weak_shape_evidence,
        localization_confidence=localization_confidence,
    )

    return ObservedEvidenceSpec(
        surface_scope=surface_scope,
        uncertainty_level=uncertainty_level,
        palette=palette,
        structural_palette=structural_palette,
        accent_palette=accent_palette,
        hard_facts=_dedupe_strings(hard_facts),
        soft_hypotheses=_dedupe_strings(soft_hypotheses),
        unknowns=_dedupe_strings(unknowns),
        color_note=color_note,
        color_confidence=None if color_confidence is None else round(float(color_confidence), 4),
        pattern_note=pattern_note,
        coverage_class=coverage_class,
        coverage_note=coverage_note,
        coverage_ratio=None if coverage_ratio is None else round(float(coverage_ratio), 4),
        value_relation_note=value_relation_note,
        trim_note=trim_note,
        trim_confidence=trim_confidence,
        boundary_color=boundary_color,
        interior_color=interior_color,
        silhouette_note=silhouette_note,
        aspect_ratio=shape_profile["aspect_ratio"],
        top_width_ratio=shape_profile["top_width_ratio"],
        form_factor_note=form_factor_note,
        upper_region_note=upper_region_profile["note"],
        upper_region_confidence=upper_region_profile["confidence"],
        upper_component_state=upper_region_profile["component_state"],
        upper_region_color=upper_region_profile["upper_region_color"],
        body_region_color=upper_region_profile["body_region_color"],
        upper_component_count=upper_region_profile["upper_component_count"],
        lower_region_note=lower_region_profile["note"],
        lower_region_confidence=lower_region_profile["confidence"],
        lower_component_state=lower_region_profile["component_state"],
        lower_region_color=lower_region_profile["lower_region_color"],
        edge_profile_note=edge_profile["note"],
        edge_profile_confidence=edge_profile["confidence"],
        edge_thickness_class=edge_profile["thickness_class"],
        edge_inner_ratio=edge_profile["inner_ratio"],
        soft_structure_note=soft_structure["note"],
        soft_structure_confidence=soft_structure["confidence"],
        soft_structure_class=soft_structure["structure_class"],
        material_note=material_note,
        surface_relief_note=surface_relief_note,
        evidence_tags=evidence_tags,
        artifact_flags=list(prepared_assets.artifact_flags),
        source_validity=source_validity,
        source_validity_score=round(float(source_validity_score), 4),
        source_validity_issues=source_validity_issues,
        raw_evidence_caption=raw_evidence_caption,
        evidence_caption=evidence_caption,
        reference_crop_path=None if evidence_crop_path is None else str(evidence_crop_path),
        reference_cutout_path=None if cutout_path is None else str(cutout_path),
        reference_silhouette_path=None if silhouette_path is None else str(silhouette_path),
        reference_mask_path=None if effective_mask_path is None else str(effective_mask_path),
    )


def prepare_observed_evidence_assets(
    *,
    source_image: Path,
    mask_path: Path | None,
    localization_crop_path: Path | None,
    category: str,
    canonical_product_type: str,
    requires_human_model: bool,
) -> PreparedEvidenceAssets:
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return PreparedEvidenceAssets(
            crop_path=None,
            cutout_path=None,
            silhouette_path=None,
            mask_path=mask_path,
            artifact_flags=(),
        )
    with Image.open(source_image) as source_handle, Image.open(mask_path) as mask_handle:
        source = source_handle.convert("RGB")
        mask = mask_handle.convert("L")
        if mask.size != source.size:
            mask = mask.resize(source.size, Image.Resampling.NEAREST)
        mask_array = np.asarray(mask) > 0
    refined_mask = refine_observed_evidence_mask(
        mask_array,
        category=category,
        canonical_product_type=canonical_product_type,
        requires_human_model=requires_human_model,
    )
    suppressed_mask, artifact_flags = suppress_border_attached_reference_artifacts(
        np.asarray(source, dtype=np.float32),
        refined_mask,
        category=category,
        canonical_product_type=canonical_product_type,
        requires_human_model=requires_human_model,
    )
    artifact_removed_regions = refined_mask & ~suppressed_mask
    refined_mask = suppressed_mask
    refined_mask = suppress_interior_cavity_contamination(
        np.asarray(source, dtype=np.float32),
        refined_mask,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    if not refined_mask.any():
        refined_mask = mask_array
    if "border_foreground_intrusion" in artifact_flags:
        reference_export_mask = refined_mask.copy()
    else:
        reference_export_mask = refined_mask | artifact_removed_regions
    reference_export_mask = smooth_reference_export_mask(
        reference_export_mask,
        category=category,
        canonical_product_type=canonical_product_type,
        artifact_flags=artifact_flags,
    )
    reference_source = repair_removed_reference_regions(
        source,
        removed_region_mask=artifact_removed_regions,
        keep_mask=reference_export_mask,
    )
    use_mask_conditioning = (
        not _mask_has_heavy_border_spill(refined_mask)
        and _structure_completeness_score(
            refined_mask,
            category=category,
            canonical_product_type=canonical_product_type,
        )
        >= 0.5
    )
    if canonical_product_type == "blender":
        use_mask_conditioning = False
    structure_completeness = _structure_completeness_score(
        refined_mask,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    prefer_localization_crop_reference = False

    base_dir = mask_path.parent
    stem = mask_path.stem.removesuffix(".mask")
    evidence_mask_path = base_dir / f"{stem}.evidence_mask.png"
    evidence_crop_path = base_dir / f"{stem}.evidence_crop.png"
    color_anchor_path = base_dir / f"{stem}.evidence_color_anchor.png"
    cutout_path = base_dir / f"{stem}.evidence_cutout.png"
    silhouette_path = base_dir / f"{stem}.evidence_silhouette.png"
    _export_evidence_assets(
        source_image=reference_source,
        mask_array=refined_mask,
        display_mask_array=reference_export_mask,
        crop_path=evidence_crop_path,
        cutout_path=cutout_path,
        silhouette_path=silhouette_path,
        mask_output_path=evidence_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        force_masked_crop=bool(
            artifact_flags
            and not _should_prefer_localization_crop_reference(
                localization_crop_path=localization_crop_path,
                evidence_crop_path=evidence_crop_path,
                category=category,
                canonical_product_type=canonical_product_type,
                artifact_flags=artifact_flags,
                structure_completeness=structure_completeness,
            )
        ),
    )
    crop_reference_path = evidence_crop_path
    edge_profile = infer_edge_profile(
        source_image=source_image,
        mask_path=evidence_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    soft_structure = infer_soft_structure_profile(
        source_image=source_image,
        mask_path=evidence_mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        edge_thickness_class=edge_profile["thickness_class"],
    )
    if _should_prepare_color_anchor_asset(
        category=category,
        canonical_product_type=canonical_product_type,
        soft_structure_class=soft_structure["structure_class"],
    ):
        core_body_ranked = infer_core_body_palette_ranked(
            source_image,
            evidence_mask_path,
            top_k=1,
        )
        dominant_color = None if not core_body_ranked else core_body_ranked[0][0]
        if dominant_color in EVIDENCE_COLOR_SWATCHES:
            _export_color_anchor_asset(
                source_image=source,
                mask_array=refined_mask,
                dominant_color=dominant_color,
                output_path=color_anchor_path,
            )
            crop_reference_path = color_anchor_path
    prefer_localization_crop_reference = _should_prefer_localization_crop_reference(
        localization_crop_path=localization_crop_path,
        evidence_crop_path=crop_reference_path,
        category=category,
        canonical_product_type=canonical_product_type,
        artifact_flags=artifact_flags,
        structure_completeness=structure_completeness,
    )
    if prefer_localization_crop_reference:
        crop_reference_path = localization_crop_path
    return PreparedEvidenceAssets(
        crop_path=crop_reference_path,
        cutout_path=cutout_path if use_mask_conditioning and not prefer_localization_crop_reference else None,
        silhouette_path=silhouette_path if use_mask_conditioning and not prefer_localization_crop_reference else None,
        mask_path=evidence_mask_path,
        artifact_flags=tuple(artifact_flags),
    )


def _image_size(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    with Image.open(path) as handle:
        return handle.size


def _should_prefer_localization_crop_reference(
    *,
    localization_crop_path: Path | None,
    evidence_crop_path: Path | None,
    category: str,
    canonical_product_type: str,
    artifact_flags: Sequence[str],
    structure_completeness: float | None,
) -> bool:
    if localization_crop_path is None or not localization_crop_path.exists():
        return False
    if evidence_crop_path is None or not evidence_crop_path.exists():
        return True
    localization_size = _image_size(localization_crop_path)
    evidence_size = _image_size(evidence_crop_path)
    if localization_size is None or evidence_size is None:
        return False
    loc_w, loc_h = localization_size
    ev_w, ev_h = evidence_size
    loc_area = max(1, loc_w * loc_h)
    ev_area = max(1, ev_w * ev_h)
    area_ratio = ev_area / float(loc_area)
    width_ratio = ev_w / float(max(loc_w, 1))
    height_ratio = ev_h / float(max(loc_h, 1))
    artifact_sensitive = bool(artifact_flags) and category in {"apparel", "footwear", "bag"}
    if artifact_sensitive:
        return True
    if category == "apparel":
        return area_ratio < 0.72 or height_ratio < 0.72
    if category in {"bag", "footwear"}:
        return area_ratio < 0.86 or width_ratio < 0.92 or height_ratio < 0.86
    multipart_sensitive = bool(
        canonical_product_type in (MULTIPART_LOCALIZATION_CANONICAL_TYPES | {"table lamp"})
        or category in {"kitchen appliance", "furniture", "home lighting"}
    )
    if multipart_sensitive:
        if structure_completeness is not None and structure_completeness < 0.74:
            return True
        return area_ratio < 0.9 or height_ratio < 0.84
    return False


def prepare_surface_evidence_mask(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
) -> Path | None:
    if mask_path is None or not mask_path.exists():
        return mask_path
    with Image.open(mask_path) as mask_handle:
        mask = np.asarray(mask_handle.convert("L")) > 0
    if not mask.any():
        return mask_path
    if canonical_product_type == "shoe" or category == "footwear":
        surface_mask = _build_footwear_surface_mask(mask)
    elif canonical_product_type in (STRUCTURED_DISPLAY_CANONICAL_TYPES - DRINKWARE_CANONICAL_TYPES) or category in {
        "kitchen appliance",
        "furniture",
        "home lighting",
    }:
        surface_mask = _build_rigid_surface_mask(mask)
    elif canonical_product_type in SOFT_HOME_CANONICAL_TYPES or category in {"bedding", "pet home", "home decor"}:
        surface_mask = _build_soft_surface_mask(
            mask,
            category=category,
            canonical_product_type=canonical_product_type,
        )
    else:
        surface_mask = mask
    if not surface_mask.any():
        return mask_path
    surface_mask_path = mask_path.parent / f"{mask_path.stem.removesuffix('.mask')}.surface_mask.png"
    Image.fromarray((surface_mask.astype(np.uint8) * 255)).save(surface_mask_path)
    return surface_mask_path


def _mask_has_heavy_border_spill(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    ys, xs = np.nonzero(mask)
    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    fill_ratio = float(mask.sum()) / max(bbox_area, 1.0)
    border_band = np.zeros_like(mask, dtype=bool)
    border_band[:3, :] = True
    border_band[-3:, :] = True
    border_band[:, :3] = True
    border_band[:, -3:] = True
    border_ratio = float(np.mean(mask[border_band])) if border_band.any() else 0.0
    return border_ratio >= 0.18 and fill_ratio <= 0.68


def _structure_completeness_score(
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
) -> float:
    tracked_types = MULTIPART_LOCALIZATION_CANONICAL_TYPES
    if canonical_product_type not in tracked_types and category not in {"kitchen appliance", "home lighting", "furniture"}:
        return 1.0
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    lower_extent = y1 / float(max(mask.shape[0], 1))
    box_height_ratio = (y1 - y0) / float(max(mask.shape[0], 1))
    crop = mask[y0:y1, x0:x1]
    total = float(crop.sum())
    if total <= 0:
        return 0.0
    height, width = crop.shape
    if height < 6 or width < 6:
        return 0.0
    top = float(crop[: max(1, int(round(height * 0.35)))].sum()) / total
    middle = float(crop[int(round(height * 0.35)) : max(1, int(round(height * 0.7)))].sum()) / total
    bottom = float(crop[max(0, int(round(height * 0.7))) :].sum()) / total
    bottom_rows = crop[max(0, int(round(height * 0.78))) :, :]
    row_widths = bottom_rows.sum(axis=1) if bottom_rows.size else np.array([], dtype=np.float32)
    bottom_width_ratio = 0.0 if row_widths.size == 0 else float(np.max(row_widths) / max(width, 1))

    if canonical_product_type == "blender":
        score = 0.35
        if top >= 0.22:
            score += 0.2
        if middle >= 0.18:
            score += 0.15
        if bottom >= 0.1:
            score += 0.2
        else:
            score -= 0.2
        if bottom_width_ratio >= 0.16:
            score += 0.1
        if lower_extent >= 0.78:
            score += 0.2
        else:
            score -= 0.2
        if box_height_ratio >= 0.55:
            score += 0.1
        return max(0.0, min(1.0, score))

    if canonical_product_type in (FURNITURE_CANONICAL_TYPES | {"table lamp"}):
        score = 0.4
        if top >= 0.2:
            score += 0.15
        if bottom >= 0.08:
            score += 0.2
        else:
            score -= 0.2
        if bottom_width_ratio >= 0.08:
            score += 0.1
        if lower_extent >= 0.72:
            score += 0.14
        return max(0.0, min(1.0, score))

    if canonical_product_type == "toaster":
        return 1.0 if middle >= 0.35 and bottom >= 0.12 else 0.45

    if canonical_product_type in {"coffee maker", "slow cooker", "food chopper"} or category == "kitchen appliance":
        score = 0.38
        if top >= 0.14:
            score += 0.1
        if middle >= 0.18:
            score += 0.16
        if bottom >= 0.12:
            score += 0.2
        else:
            score -= 0.18
        if bottom_width_ratio >= 0.12:
            score += 0.08
        if lower_extent >= 0.72:
            score += 0.14
        return max(0.0, min(1.0, score))

    return 1.0


def infer_named_palette(source_image: Path, mask_path: Path | None, *, top_k: int = 4) -> list[str]:
    return infer_named_palette_with_strategy(
        source_image,
        mask_path,
        top_k=top_k,
        use_smoothed=False,
        erode_steps=0,
    )


def infer_named_palette_ranked(
    source_image: Path,
    mask_path: Path | None,
    *,
    top_k: int = 4,
    use_smoothed: bool,
    erode_steps: int,
) -> list[tuple[str, float]]:
    if not source_image.exists():
        return []
    with Image.open(source_image) as source_handle:
        source_image_rgb = source_handle.convert("RGB")
        if use_smoothed:
            source_image_rgb = source_image_rgb.filter(ImageFilter.GaussianBlur(radius=1.5))
        source = np.asarray(source_image_rgb, dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is not None and mask.any() and erode_steps > 0:
        eroded_mask = _erode_mask(mask, steps=erode_steps)
        if eroded_mask.any() and eroded_mask.sum() >= max(64, int(mask.sum() * 0.2)):
            mask = eroded_mask
    pixels = source[mask] if mask is not None and mask.any() else source.reshape(-1, 3)
    if pixels.size == 0:
        return []
    stride = max(1, int(math.ceil(len(pixels) / 4000.0)))
    sampled = pixels[::stride]
    if use_smoothed:
        counts = _weighted_structural_color_distribution(sampled)
    else:
        counts = _named_color_distribution(sampled)
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if use_smoothed:
        ranked = _reorder_structural_palette(ranked)
    return [(name, round(float(score), 4)) for name, score in ranked[:top_k]]


def infer_named_palette_with_strategy(
    source_image: Path,
    mask_path: Path | None,
    *,
    top_k: int = 4,
    use_smoothed: bool,
    erode_steps: int,
) -> list[str]:
    ranked = infer_named_palette_ranked(
        source_image,
        mask_path,
        top_k=top_k,
        use_smoothed=use_smoothed,
        erode_steps=erode_steps,
    )
    return [name for name, _ in ranked]


def infer_core_body_palette_ranked(
    source_image: Path,
    mask_path: Path | None,
    *,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    if not source_image.exists():
        return []
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return infer_named_palette_ranked(
            source_image,
            mask_path,
            top_k=top_k,
            use_smoothed=True,
            erode_steps=2,
        )
    core_mask = _extract_core_body_mask(mask)
    if core_mask.sum() < max(64, int(mask.sum() * 0.14)):
        core_mask = _erode_mask(mask, steps=2)
    pixels = source[core_mask] if core_mask.any() else source[mask]
    if pixels.size == 0:
        return []
    stride = max(1, int(math.ceil(len(pixels) / 4000.0)))
    sampled = pixels[::stride]
    counts = _weighted_structural_color_distribution(sampled)
    ranked = _reorder_structural_palette(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    return [(name, round(float(score), 4)) for name, score in ranked[:top_k]]


def infer_surface_coverage_profile(
    source_image: Path,
    mask_path: Path | None,
    *,
    palette: Sequence[str],
    base_palette: Sequence[str] | None = None,
) -> tuple[str | None, float | None, str | None]:
    if not source_image.exists():
        return None, None, None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        pixels = source.reshape(-1, 3)
        bbox_mask = np.ones(source.shape[:2], dtype=bool)
    else:
        pixels = source[mask]
        bbox_mask = mask
    if len(pixels) < 64:
        return None, None, None

    ys, xs = np.nonzero(bbox_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    region = source[y0:y1, x0:x1]
    region_mask = bbox_mask[y0:y1, x0:x1]
    rows = max(2, min(4, region.shape[0] // 24 or 2))
    cols = max(2, min(4, region.shape[1] // 24 or 2))

    qualified_tiles = 0
    patterned_tiles = 0
    patterned_positions: list[tuple[int, int]] = []
    for row_index in range(rows):
        for col_index in range(cols):
            tile_y0 = int(round(row_index * region.shape[0] / rows))
            tile_y1 = int(round((row_index + 1) * region.shape[0] / rows))
            tile_x0 = int(round(col_index * region.shape[1] / cols))
            tile_x1 = int(round((col_index + 1) * region.shape[1] / cols))
            tile_mask = region_mask[tile_y0:tile_y1, tile_x0:tile_x1]
            if not tile_mask.any() or tile_mask.mean() < 0.18:
                continue
            tile_pixels = region[tile_y0:tile_y1, tile_x0:tile_x1][tile_mask]
            if len(tile_pixels) < 32:
                continue
            qualified_tiles += 1
            local_std = float(np.mean(np.std(tile_pixels, axis=0)))
            sampled = tile_pixels[:: max(1, int(math.ceil(len(tile_pixels) / 256.0)))]
            local_palette = {_nearest_color_name(pixel) for pixel in sampled}
            if local_std >= 30.0 or len(local_palette) >= 3:
                patterned_tiles += 1
                patterned_positions.append((row_index, col_index))

    if qualified_tiles == 0:
        return None, None, None

    coverage_ratio = patterned_tiles / float(qualified_tiles)
    base_color = summarize_base_palette_label(base_palette or palette)
    unique_palette = len(set(palette))
    patterned_rows = {row_index for row_index, _ in patterned_positions}
    patterned_cols = {col_index for _, col_index in patterned_positions}
    row_spread = len(patterned_rows) / float(max(rows, 1))
    col_spread = len(patterned_cols) / float(max(cols, 1))
    localized_band = bool(
        patterned_tiles > 0
        and (
            row_spread <= 0.5
            or col_spread <= 0.45
            or (row_spread <= 0.65 and col_spread <= 0.35)
        )
    )
    if coverage_ratio >= 0.72 and unique_palette >= 3:
        if localized_band:
            return (
                "localized_visible_pattern",
                coverage_ratio,
                "a localized high-contrast printed or label-like zone interrupts one region of the visible product surface",
            )
        return (
            "full_visible_surface_pattern",
            coverage_ratio,
            f"the visible print or color treatment covers most of the observed product surface on a {base_color} base",
        )
    if coverage_ratio >= 0.4 and unique_palette >= 3:
        if localized_band:
            return (
                "localized_visible_pattern",
                coverage_ratio,
                "a localized high-contrast printed or label-like zone interrupts one region of the visible product surface",
            )
        return (
            "broad_visible_surface_pattern",
            coverage_ratio,
            f"the visible print or color treatment spans a broad portion of the observed product surface on a {base_color} base",
        )
    if coverage_ratio >= 0.18 and unique_palette >= 2:
        return (
            "localized_visible_pattern",
            coverage_ratio,
            "a localized multicolor, printed, or contrast zone appears on one region of the visible product surface",
        )
    return (
        "low_variation_surface",
        coverage_ratio,
        f"most of the visible product surface reads as a relatively uniform {base_color} treatment",
    )


def infer_pattern_note(
    source_image: Path,
    mask_path: Path | None,
    *,
    palette: Sequence[str],
    base_palette: Sequence[str] | None = None,
    coverage_class: str | None = None,
) -> str | None:
    if not source_image.exists():
        return None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    pixels = source[mask] if mask is not None and mask.any() else source.reshape(-1, 3)
    if len(pixels) < 64:
        return None
    color_std = float(np.mean(np.std(pixels, axis=0)))
    unique_palette = len(set(palette))
    base_color = summarize_base_palette_label(base_palette or palette)
    if coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        return f"the visible product body carries a multicolor or printed treatment on a {base_color} base"
    if unique_palette >= 3 and color_std >= 38.0:
        return f"one observed surface shows a multicolor printed or patterned treatment on a {base_color} base"
    if unique_palette >= 2 and color_std >= 28.0:
        return f"the visible product surface includes multiple compatible colors on a {base_color} base"
    return None


def infer_localized_contrast_panel(
    source_image: Path,
    mask_path: Path | None,
    *,
    category: str,
    canonical_product_type: str,
    body_color: str | None,
) -> tuple[str | None, str | None]:
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return None, None
    if category in {"apparel", "bedding", "pet home", "furniture"} or canonical_product_type in (
        BEDDING_CANONICAL_TYPES | {"shirt", "dress", "pet bed", "decorative pillow", "office chair", "folding chair", "chair"}
    ):
        return None, None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None, None
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = max(1, y1 - y0)
    shoulder_row = _estimate_body_shoulder_row(mask)
    visible_body_end = y0 + int(round(height * 0.9))
    visible_body_mask = np.zeros_like(mask, dtype=bool)
    visible_body_mask[shoulder_row:visible_body_end, :] = mask[
        shoulder_row:visible_body_end,
        :,
    ]
    if not visible_body_mask.any():
        return None, None
    core_body_mask = _extract_core_body_mask(mask)
    reference_mask = core_body_mask if core_body_mask.any() else visible_body_mask
    reference_pixels = source[reference_mask]
    if len(reference_pixels) < 32:
        if body_color is None or body_color not in EVIDENCE_COLOR_SWATCHES:
            return None, None
        body_rgb = np.asarray(EVIDENCE_COLOR_SWATCHES[body_color], dtype=np.float32)
    else:
        body_rgb = np.mean(reference_pixels, axis=0)
    reference_hsv = np.asarray(
        [colorsys.rgb_to_hsv(*(pixel / 255.0)) for pixel in reference_pixels[:: max(1, len(reference_pixels) // 512)]],
        dtype=np.float32,
    )
    reference_value = float(reference_hsv[:, 2].mean()) if reference_hsv.size else 0.0
    reference_saturation = float(reference_hsv[:, 1].mean()) if reference_hsv.size else 0.0
    visible_width = _mask_band_width(visible_body_mask, 0.0, 1.0)
    visible_height = float(np.nonzero(visible_body_mask)[0].max() - np.nonzero(visible_body_mask)[0].min() + 1)
    if visible_height / max(visible_width, 1.0) < 1.15:
        return None, None
    color_distance = np.sqrt(np.sum((source - body_rgb) ** 2, axis=2))
    candidate_mask = visible_body_mask & (color_distance >= 62.0)
    components = _mask_components(candidate_mask, min_pixels=max(24, int(visible_body_mask.sum() * 0.014)))
    if not components:
        return None, None
    best_match: tuple[float, str, str | None] | None = None
    for component in sorted(components, key=lambda item: int(item.sum()), reverse=True):
        comp_ys, comp_xs = np.nonzero(component)
        comp_width = float(comp_xs.max() - comp_xs.min() + 1)
        comp_height = float(comp_ys.max() - comp_ys.min() + 1)
        area_ratio = float(component.sum()) / max(float(visible_body_mask.sum()), 1.0)
        if area_ratio < 0.04 or area_ratio > 0.42:
            continue
        component_pixels = source[component]
        component_hsv = np.asarray(
            [colorsys.rgb_to_hsv(*(pixel / 255.0)) for pixel in component_pixels[:: max(1, len(component_pixels) // 256)]],
            dtype=np.float32,
        )
        component_value = float(component_hsv[:, 2].mean()) if component_hsv.size else 0.0
        component_saturation = float(component_hsv[:, 1].mean()) if component_hsv.size else 0.0
        if component_value < reference_value + 0.08 and component_saturation > reference_saturation + 0.1:
            continue
        value_relation_note = None
        if category == "drinkware":
            if component_value >= reference_value + 0.14:
                value_relation_note = "the localized label band or printed wrap is visibly lighter than the main body"
            elif component_value <= reference_value - 0.14:
                value_relation_note = "the localized label band or printed wrap is visibly darker than the main body"
        else:
            if component_value >= reference_value + 0.14:
                value_relation_note = "the localized front panel or control window is visibly lighter than the main body"
            elif component_value <= reference_value - 0.14:
                value_relation_note = "the localized front panel or control window is visibly darker than the main body"
        if comp_width >= max(visible_width * 0.48, 1.0) and 0.08 <= comp_height / max(visible_height, 1.0) <= 0.42:
            candidate_note = (
                "a localized high-contrast label band or printed wrap interrupts one region of the visible bottle or mug surface"
                if category == "drinkware"
                else "a localized high-contrast front panel or control window interrupts one region of the visible product surface"
            )
            brightness_bonus = max(component_value - reference_value, 0.0)
            neutrality_bonus = max(0.0, 0.35 - component_saturation)
            candidate_score = area_ratio + 0.45 * brightness_bonus + 0.18 * neutrality_bonus
            if best_match is None or candidate_score > best_match[0]:
                best_match = (candidate_score, candidate_note, value_relation_note)
            continue
        if comp_width >= max(visible_width * 0.28, 1.0):
            candidate_note = (
                "a localized label band or printed wrap zone appears on one region of the visible bottle or mug surface"
                if category == "drinkware"
                else "a localized contrast panel or control-window zone appears on one region of the visible product surface"
            )
            brightness_bonus = max(component_value - reference_value, 0.0)
            neutrality_bonus = max(0.0, 0.3 - component_saturation)
            candidate_score = area_ratio + 0.3 * brightness_bonus + 0.12 * neutrality_bonus
            if best_match is None or candidate_score > best_match[0]:
                best_match = (candidate_score, candidate_note, value_relation_note)
    if best_match is None:
        return None, None
    return best_match[1], best_match[2]


def generalize_low_confidence_surface_notes(
    *,
    coverage_class: str | None,
    coverage_note: str | None,
    pattern_note: str | None,
) -> tuple[str | None, str | None]:
    if coverage_class == "full_visible_surface_pattern":
        return (
            "the visible print or color treatment covers most of the observed product surface",
            "the visible product body carries a multicolor or printed treatment",
        )
    if coverage_class == "broad_visible_surface_pattern":
        return (
            "the visible print or color treatment spans a broad portion of the observed product surface",
            "the visible product body carries a multicolor or printed treatment across much of the observed view",
        )
    if coverage_class == "localized_visible_pattern":
        return (
            coverage_note or "a localized multicolor, printed, or contrast zone appears on one region of the visible product surface",
            pattern_note or "a localized printed or contrast treatment appears on one region of the visible product surface",
        )
    return coverage_note, pattern_note


def correct_structured_display_surface_inference(
    *,
    category: str,
    canonical_product_type: str,
    stable_base: bool | None,
    shape_profile: dict[str, Any],
    evidence_caption: str | None,
    coverage_class: str | None,
    coverage_note: str | None,
    pattern_note: str | None,
) -> tuple[str | None, str | None, str | None]:
    if coverage_class not in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        return coverage_class, coverage_note, pattern_note
    if not stable_base:
        return coverage_class, coverage_note, pattern_note

    aspect_ratio = shape_profile.get("aspect_ratio")
    top_width_ratio = shape_profile.get("top_width_ratio")
    lowered_type = canonical_product_type.lower()
    structured_type = canonical_product_type in (STRUCTURED_DISPLAY_CANONICAL_TYPES - DRINKWARE_CANONICAL_TYPES) or (
        category in {"home lighting", "furniture", "kitchen appliance", "product"}
        and any(
            token in lowered_type
            for token in ("lamp", "lantern", "shade", "chair", "stool", "bench", "blender", "toaster", "coffee", "slow cooker", "chopper")
        )
    )
    if not structured_type:
        return coverage_class, coverage_note, pattern_note
    if canonical_product_type not in (STRUCTURED_DISPLAY_CANONICAL_TYPES - DRINKWARE_CANONICAL_TYPES):
        if aspect_ratio is None or aspect_ratio < 1.5:
            return coverage_class, coverage_note, pattern_note
        if top_width_ratio is not None and top_width_ratio < 0.95:
            return coverage_class, coverage_note, pattern_note

    caption_tokens = set(_tokens(evidence_caption or ""))
    if caption_tokens.intersection(PATTERN_TEXT_TOKENS):
        return coverage_class, coverage_note, pattern_note

    return (
        "low_variation_surface",
        "most of the visible product surface reads as structured color zoning, paneling, or tonal treatment rather than an all-over print",
        None,
    )


def infer_trim_profile(source_image: Path, mask_path: Path | None) -> tuple[str | None, str | None, str | None, float | None]:
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return None, None, None, None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB").filter(ImageFilter.GaussianBlur(radius=3.0)), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None, None, None, None
    boundary_mask = _mask_boundary(mask)
    interior_mask = _erode_mask(mask, steps=2)
    if not interior_mask.any():
        interior_mask = mask & ~boundary_mask
    if not boundary_mask.any() or not interior_mask.any():
        return None, None, None, None
    boundary_color, boundary_confidence = _dominant_named_color_with_ratio(source[boundary_mask])
    interior_color, interior_confidence = _dominant_named_color_with_ratio(source[interior_mask])
    confidence = round(min(boundary_confidence, interior_confidence), 4)
    if (
        boundary_color == interior_color
        or confidence < 0.52
        or _named_color_distance(boundary_color, interior_color) < 42.0
    ):
        return boundary_color, interior_color, None, confidence
    return (
        boundary_color,
        interior_color,
        f"visible outer trim or edging reads as {boundary_color} against a {interior_color} interior",
        confidence,
    )


def infer_shape_profile(mask_path: Path | None) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "aspect_ratio": None,
        "top_width_ratio": None,
        "note": None,
    }
    if mask_path is None or not mask_path.exists():
        return profile
    with Image.open(mask_path) as mask_handle:
        mask_array = np.asarray(mask_handle.convert("L")) > 0
    if not mask_array.any():
        return profile
    ys, xs = np.nonzero(mask_array)
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    aspect = height / max(width, 1.0)
    top_width = _mask_band_width(mask_array, 0.05, 0.2)
    mid_width = _mask_band_width(mask_array, 0.4, 0.6)
    top_width_ratio = None
    notes: list[str] = []
    if aspect >= 1.2:
        notes.append("observed silhouette is vertically elongated")
    elif aspect <= 0.8:
        notes.append("observed silhouette is horizontally spread")
    if top_width > 0 and mid_width > 0:
        top_width_ratio = top_width / max(mid_width, 1.0)
        if top_width_ratio <= 0.72:
            notes.append("the observed visible surface narrows or tapers toward the top")
    profile["aspect_ratio"] = round(aspect, 4)
    profile["top_width_ratio"] = None if top_width_ratio is None else round(float(top_width_ratio), 4)
    profile["note"] = "; ".join(notes[:2]) or None
    return profile


def infer_silhouette_note(mask_path: Path | None) -> str | None:
    return infer_shape_profile(mask_path)["note"]


def infer_upper_region_profile(source_image: Path, mask_path: Path | None) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "note": None,
        "upper_region_color": None,
        "body_region_color": None,
        "upper_component_count": None,
        "confidence": None,
        "component_state": "uncertain",
    }
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return profile
    with Image.open(source_image) as source_handle:
        source_rgb = source_handle.convert("RGB")
        raw_source = np.asarray(source_rgb, dtype=np.float32)
        source = np.asarray(source_rgb.filter(ImageFilter.GaussianBlur(radius=3.0)), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return profile
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = max(1, y1 - y0)
    bbox_width = max(1.0, float(xs.max() - xs.min() + 1))
    shape_aspect_ratio = float(height) / bbox_width
    upper_end = _estimate_body_shoulder_row(mask)
    upper_end = max(y0 + 2, min(upper_end, y1 - 1))
    mid_width = _mask_band_width(mask, 0.4, 0.6)
    upper_mask = np.zeros_like(mask)
    upper_mask[y0:upper_end, :] = mask[y0:upper_end, :]
    body_mask = _extract_core_body_mask(mask)
    if not upper_mask.any() or not body_mask.any():
        return profile
    upper_region_components = _mask_components(upper_mask, min_pixels=max(8, int(mask.sum() * 0.008)))
    omitted_attachment_split = _looks_like_omitted_upper_attachment(
        upper_region_components,
        main_body_width=mid_width,
        total_product_pixels=float(mask.sum()),
    )
    dark_structure_mask = _extract_dark_structure_mask(raw_source, upper_mask)
    use_dark_structure = dark_structure_mask.sum() >= max(16, int(mask.sum() * 0.006))
    component_source_mask = dark_structure_mask if use_dark_structure else upper_mask
    component_min_pixels = max(8, int(mask.sum() * (0.0035 if use_dark_structure else 0.008)))
    component_masks = [
        component
        for component in _mask_components(component_source_mask, min_pixels=component_min_pixels)
        if _is_attached_component_candidate(
            component,
            main_body_width=mid_width,
            total_product_pixels=float(mask.sum()),
            shape_aspect_ratio=shape_aspect_ratio,
        )
    ]
    prioritized_components = _prioritize_upper_components(
        component_masks,
        raw_source,
        upper_start=y0,
        upper_end=upper_end,
    )
    selected_components = prioritized_components or sorted(
        component_masks,
        key=lambda item: int(np.nonzero(item)[0].min()) if np.nonzero(item)[0].size else 10**9,
    )[: max(1, min(2, len(component_masks)))]
    component_count = len(selected_components) if selected_components else len(component_masks)
    upper_area_ratio = float(upper_mask.sum()) / max(float(mask.sum()), 1.0)
    upper_bbox_width = _mask_band_width(upper_mask, 0.0, 1.0)
    upper_fill_ratio = 0.0
    if upper_end > y0 and upper_bbox_width > 0:
        upper_fill_ratio = float(upper_mask.sum()) / max((upper_end - y0) * upper_bbox_width, 1.0)
    color_mask = upper_mask
    if component_masks:
        color_mask = np.zeros_like(mask)
        for component in selected_components:
            color_mask |= component
    emphasis_mask = color_mask
    if selected_components:
        candidate_mask = np.zeros_like(color_mask)
        for component in selected_components:
            component_ys, _ = np.nonzero(component)
            if len(component_ys) == 0:
                continue
            emphasis_end = int(component_ys.min() + max(1, round((component_ys.max() - component_ys.min() + 1) * 0.3)))
            component_focus = np.zeros_like(component)
            component_focus[: emphasis_end + 1, :] = component[: emphasis_end + 1, :]
            if component_focus.any():
                candidate_mask |= component_focus
        if candidate_mask.any():
            emphasis_mask = candidate_mask
    else:
        component_ys, _ = np.nonzero(color_mask)
        if len(component_ys) > 0:
            emphasis_end = int(component_ys.min() + max(1, round((component_ys.max() - component_ys.min() + 1) * 0.45)))
            candidate_mask = np.zeros_like(color_mask)
            candidate_mask[: emphasis_end + 1, :] = color_mask[: emphasis_end + 1, :]
            if candidate_mask.any():
                emphasis_mask = candidate_mask
    use_structural_color = use_dark_structure and bool(selected_components)
    upper_source = raw_source if component_count >= 2 or use_structural_color else source
    if component_count >= 2 or use_structural_color:
        upper_color, upper_confidence = _select_attached_component_color(upper_source[emphasis_mask])
    else:
        upper_color, upper_confidence = _dominant_named_color_with_ratio(upper_source[emphasis_mask])
    body_color, body_confidence = _dominant_named_color_with_ratio(source[body_mask])
    shape_profile = infer_shape_profile(mask_path)
    top_width_ratio = shape_profile["top_width_ratio"]
    detached_segments = component_count >= 2
    color_separation = (
        upper_color != body_color
        and min(upper_confidence, body_confidence) >= 0.4
        and _named_color_distance(upper_color, body_color) >= 52.0
    )
    upper_color_confident = upper_color not in {None, "mixed"} and upper_confidence >= 0.38
    narrow_structure = bool(
        component_count >= 1 and top_width_ratio is not None and top_width_ratio <= 0.55 and upper_area_ratio <= 0.23
    )
    attachment_like = bool(
        component_count >= 1
        and upper_fill_ratio <= 0.72
        and upper_end <= y0 + int(round(height * 0.56))
        and upper_bbox_width <= max(mid_width * 0.95, 1.0)
    )
    compact_top_component = bool(
        component_count >= 1
        and upper_area_ratio <= 0.16
        and upper_fill_ratio >= 0.72
        and upper_bbox_width <= max(mid_width * 0.82, 1.0)
    )
    confidence = 0.0
    if detached_segments:
        confidence = 0.72 + 0.06 * min(component_count - 2, 2)
    elif (attachment_like or compact_top_component) and (color_separation or component_count >= 1):
        confidence = 0.62
    elif use_structural_color and component_count >= 1 and (color_separation or upper_color_confident):
        confidence = 0.58
    elif narrow_structure and color_separation:
        confidence = 0.58
    if omitted_attachment_split and confidence < 0.45:
        confidence = 0.54
    component_state = "present" if confidence >= 0.45 else ("absent" if component_count == 0 else "uncertain")
    note = None
    if omitted_attachment_split and not (use_structural_color and (color_separation or upper_color_confident)):
        note = (
            "the visible upper carrying structure continues above the main body, "
            "even though the full attachment is only partially shown"
        )
        upper_color = None
    elif omitted_attachment_split and use_structural_color and (color_separation or upper_color_confident):
        if upper_color and body_color and upper_color != body_color and body_confidence >= 0.32:
            note = (
                f"the visible upper carrying structure is only partially shown but remains visually distinct in "
                f"{upper_color} above a {body_color} main body"
            )
        elif upper_color_confident and upper_color:
            note = (
                f"the visible upper carrying structure is only partially shown but remains visually distinct in "
                f"{upper_color} above the main body"
            )
        else:
            note = (
                "the visible upper carrying structure is only partially shown but remains structurally distinct "
                "above the main body"
            )
    elif detached_segments:
        if upper_color and body_color and upper_color != body_color:
            note = (
                f"the visible upper component splits into multiple narrow segments in {upper_color} "
                f"above a {body_color} main body"
            )
        else:
            note = "the visible upper component splits into multiple narrow segments above the main body"
    elif attachment_like or compact_top_component:
        if upper_color and body_color and upper_color != body_color:
            note = f"the visible upper attachment remains visually distinct in {upper_color} above a {body_color} main body"
        else:
            note = "the visible upper attachment remains structurally distinct above the main body"
    elif use_structural_color and (color_separation or upper_color_confident):
        if upper_color and body_color and upper_color != body_color and body_confidence >= 0.32:
            note = f"the visible upper attachment remains visually distinct in {upper_color} above a {body_color} main body"
        elif upper_color_confident and upper_color:
            note = f"the visible upper attachment remains visually distinct in {upper_color} above the main body"
        else:
            note = "the visible upper attachment remains structurally distinct above the main body"
    elif narrow_structure and color_separation:
        if upper_color and body_color and upper_color != body_color:
            note = f"the visible upper component is narrower and reads as {upper_color} above a {body_color} main body"
        else:
            note = "the visible upper component is narrower than the main body and should remain structurally distinct"
    profile["note"] = note if confidence >= 0.45 else None
    if omitted_attachment_split and not (use_structural_color and (color_separation or upper_color_confident)):
        profile["upper_region_color"] = None
    else:
        profile["upper_region_color"] = None if confidence < 0.45 or upper_color == "mixed" else upper_color
    profile["body_region_color"] = None if confidence < 0.45 or body_color == "mixed" else body_color
    profile["upper_component_count"] = None if omitted_attachment_split else int(component_count)
    profile["confidence"] = round(confidence, 4)
    profile["component_state"] = component_state
    return profile


def infer_lower_region_profile(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "note": None,
        "lower_region_color": None,
        "confidence": None,
        "component_state": "uncertain",
    }
    tracked_types = MULTIPART_LOCALIZATION_CANONICAL_TYPES
    if canonical_product_type not in tracked_types and category not in {"kitchen appliance", "furniture", "home lighting"}:
        return profile
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return profile
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB").filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return profile
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = max(1, y1 - y0)
    lower_start = y0 + int(round(height * 0.68))
    lower_mask = np.zeros_like(mask, dtype=bool)
    lower_mask[lower_start:y1, :] = mask[lower_start:y1, :]
    if not lower_mask.any():
        return profile
    mid_width = _mask_band_width(mask, 0.38, 0.6)
    lower_width = _mask_band_width(lower_mask, 0.0, 1.0)
    lower_area_ratio = float(lower_mask.sum()) / max(float(mask.sum()), 1.0)
    lower_color, lower_confidence = _dominant_named_color_with_ratio(source[lower_mask])
    note = None
    confidence = 0.0
    if canonical_product_type in (KITCHEN_APPLIANCE_CANONICAL_TYPES - {"coffee maker", "slow cooker", "food chopper"}):
        if lower_area_ratio >= 0.18 and lower_width >= max(mid_width * 0.58, 1.0):
            note = "the visible lower base remains attached beneath the main vessel or appliance body and should stay present in a compatible proportion"
            confidence = 0.72
    elif canonical_product_type in {"coffee maker", "slow cooker", "food chopper"} or category == "kitchen appliance":
        if lower_area_ratio >= 0.14 and lower_width >= max(mid_width * 0.46, 1.0):
            note = "the visible lower appliance base remains attached beneath the main body and should stay present in a compatible proportion"
            confidence = 0.7
    elif canonical_product_type in {"office chair", "folding chair"}:
        rows = _mask_row_widths(mask)
        if rows:
            bottom_band = rows[max(0, len(rows) - max(3, int(round(len(rows) * 0.14)))) :]
            stem_band = rows[
                max(0, len(rows) - max(7, int(round(len(rows) * 0.28)))) : max(1, len(rows) - max(3, int(round(len(rows) * 0.14))))
            ]
            bottom_width = float(sum(bottom_band) / max(len(bottom_band), 1))
            stem_width = float(sum(stem_band) / max(len(stem_band), 1)) if stem_band else 0.0
            if bottom_width >= max(stem_width * 1.45, mid_width * 0.42) and lower_area_ratio >= 0.14:
                note = "the visible lower support includes an extended forward support element below the seat and should remain attached"
                confidence = 0.78
            elif lower_area_ratio >= 0.16:
                note = "the visible lower support frame continues below the seat and backrest and should remain present"
                confidence = 0.66
    elif canonical_product_type == "table lamp":
        if lower_area_ratio >= 0.12 and lower_width <= max(mid_width * 0.92, 1.0):
            note = "the visible lower support stays narrower than the upper body and remains attached as a stable base or stem"
            confidence = 0.62
    profile["note"] = note
    profile["lower_region_color"] = None if lower_color == "mixed" or lower_confidence < 0.34 else lower_color
    profile["confidence"] = round(confidence, 4)
    profile["component_state"] = "present" if confidence >= 0.45 else "uncertain"
    return profile


def infer_edge_profile(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "note": None,
        "confidence": None,
        "thickness_class": None,
        "inner_ratio": None,
    }
    tracked_types = {"pet bed", "decorative pillow", *BEDDING_CANONICAL_TYPES}
    if canonical_product_type not in tracked_types and category not in {"pet home", "bedding", "home decor"}:
        return profile
    if mask_path is None or not mask_path.exists():
        return profile
    if source_image.exists():
        with Image.open(source_image) as source_handle:
            source_shape = np.asarray(source_handle.convert("RGB")).shape[:2]
        mask = _load_mask_array(mask_path, source_shape=source_shape)
    else:
        with Image.open(mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L")) > 0
    if mask is None or not mask.any():
        return profile
    total = float(mask.sum())
    if total <= 0:
        return profile
    erode_steps = 3 if total < 80_000 else 5
    inner = _erode_mask(mask, steps=erode_steps)
    if not inner.any():
        inner = _erode_mask(mask, steps=2)
    inner_ratio = float(inner.sum()) / total if inner.any() else 0.0
    if inner_ratio <= 0.48:
        thickness_class = "thick_raised_edge"
        confidence = 0.76
    elif inner_ratio <= 0.66:
        thickness_class = "moderate_edge"
        confidence = 0.64
    else:
        thickness_class = "low_profile_edge"
        confidence = 0.68
    note = None
    if canonical_product_type == "pet bed":
        if thickness_class == "low_profile_edge":
            note = "the visible pet bed perimeter remains low and softly graded around the resting surface rather than rising into bulky bolsters"
        elif thickness_class == "moderate_edge":
            note = "the visible pet bed perimeter forms a modest raised edge around the resting surface"
        else:
            note = "the visible pet bed perimeter forms a thick raised bolster around the resting surface"
    elif canonical_product_type in BEDDING_CANONICAL_TYPES:
        if thickness_class == "low_profile_edge":
            note = "the visible bedding surface reads as relatively low-loft and broadly spread rather than heavily puffed at the edges"
        elif thickness_class == "moderate_edge":
            note = "the visible bedding surface shows moderate loft with softly raised edges"
        else:
            note = "the visible bedding surface shows heavy loft with strongly raised edges"
    elif canonical_product_type == "decorative pillow":
        if thickness_class == "low_profile_edge":
            note = "the visible cushion edge stays broad and soft without exaggerated piping or bolstering"
        elif thickness_class == "thick_raised_edge":
            note = "the visible cushion edge reads as thick and padded around the face of the pillow"
    profile["note"] = note
    profile["confidence"] = round(confidence, 4)
    profile["thickness_class"] = thickness_class
    profile["inner_ratio"] = round(inner_ratio, 4)
    return profile


def infer_soft_structure_profile(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
    edge_thickness_class: str | None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "note": None,
        "confidence": None,
        "structure_class": None,
    }
    tracked_types = {"pet bed", "decorative pillow", *BEDDING_CANONICAL_TYPES}
    if canonical_product_type not in tracked_types and category not in {"pet home", "bedding", "home decor"}:
        return profile
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return profile
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return profile
    mask = _keep_primary_center_component(mask)
    if not mask.any():
        return profile
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return profile
    bbox_height = int(ys.max()) - int(ys.min()) + 1
    bbox_width = int(xs.max()) - int(xs.min()) + 1
    min_dim = max(1, min(bbox_height, bbox_width))
    erode_steps = max(2, int(round(min_dim * 0.04)))
    inner = _erode_mask(mask, steps=erode_steps)
    if not inner.any() or inner.sum() < max(64, int(mask.sum() * 0.18)):
        inner = _erode_mask(mask, steps=max(1, erode_steps // 2))
    if not inner.any():
        return profile
    boundary = np.logical_and(mask, np.logical_not(inner))
    if not boundary.any():
        return profile

    gray = np.mean(source, axis=2)
    boundary_values = gray[boundary]
    inner_values = gray[inner]
    if boundary_values.size == 0 or inner_values.size == 0:
        return profile

    boundary_mean = float(np.mean(boundary_values))
    inner_mean = float(np.mean(inner_values))
    boundary_std = float(np.std(boundary_values))
    inner_std = float(np.std(inner_values))
    luminance_delta = abs(boundary_mean - inner_mean) / 255.0
    texture_delta = abs(boundary_std - inner_std) / float(max(boundary_std, inner_std, 1.0))
    boundary_ratio = float(boundary.sum()) / float(max(1, mask.sum()))
    prominence = 0.45 * luminance_delta + 0.35 * texture_delta + 0.2 * max(0.0, boundary_ratio - 0.28)

    structure_class = None
    confidence = 0.58
    note = None

    if canonical_product_type == "pet bed" or category == "pet home":
        if edge_thickness_class == "thick_raised_edge" or prominence >= 0.2:
            structure_class = "raised_perimeter_relief"
            confidence = 0.76
            note = "the visible soft product structure forms a raised perimeter around the resting surface rather than a flat mat-like body"
        elif edge_thickness_class == "moderate_edge" or prominence >= 0.1:
            structure_class = "low_perimeter_relief"
            confidence = 0.68
            note = "the visible soft product structure shows only a modest perimeter rise around the resting surface"
        else:
            structure_class = "flat_surface"
            confidence = 0.78
            note = "the visible soft product structure reads as a flat plush pad with no bulky bolster, boxed sidewall, or nested inner tray"
    elif canonical_product_type in BEDDING_CANONICAL_TYPES or category == "bedding":
        if edge_thickness_class == "thick_raised_edge" or prominence >= 0.2:
            structure_class = "raised_perimeter_relief"
            confidence = 0.72
            note = "the visible soft product structure shows pronounced loft and raised edges rather than a tray-like boxed form"
        elif edge_thickness_class == "moderate_edge" or prominence >= 0.1:
            structure_class = "low_perimeter_relief"
            confidence = 0.64
            note = "the visible soft product structure shows moderate loft with softly lifted edges"
        else:
            structure_class = "flat_surface"
            confidence = 0.74
            note = "the visible soft product structure reads as broadly spread and low-profile rather than boxed or heavily bolstered"
    elif canonical_product_type == "decorative pillow" or category == "home decor":
        if edge_thickness_class == "thick_raised_edge" or prominence >= 0.22:
            structure_class = "raised_perimeter_relief"
            confidence = 0.7
            note = "the visible soft product structure reads as thickly padded with a pronounced outer edge"
        elif edge_thickness_class == "moderate_edge" or prominence >= 0.12:
            structure_class = "low_perimeter_relief"
            confidence = 0.62
            note = "the visible soft product structure shows a modest padded edge around the face"
        else:
            structure_class = "flat_surface"
            confidence = 0.68
            note = "the visible soft product structure stays broad and softly filled without exaggerated piping or boxed edges"

    structure_class, note, confidence = harmonize_supported_soft_structure(
        category=category,
        canonical_product_type=canonical_product_type,
        edge_thickness_class=edge_thickness_class,
        structure_class=structure_class,
        note=note,
        confidence=confidence,
    )
    profile["note"] = note
    profile["confidence"] = round(confidence, 4) if note else None
    profile["structure_class"] = structure_class
    return profile


def infer_color_note(
    *,
    core_body_ranked: Sequence[tuple[str, float]],
    structural_ranked: Sequence[tuple[str, float]],
    accent_ranked: Sequence[tuple[str, float]],
    coverage_class: str | None,
) -> tuple[str | None, float | None]:
    body_ranked = list(core_body_ranked or structural_ranked or accent_ranked)
    structural = _dedupe_strings(name for name, _ in body_ranked)
    accents = _dedupe_strings(name for name, _ in accent_ranked)
    if not structural and not accents:
        return None, None
    overlap = len(set(structural[:2]).intersection(accents[:3]))
    dominant = structural[0] if structural else accents[0]
    dominant_ratio = float(body_ranked[0][1]) if body_ranked else 0.0
    second_ratio = float(body_ranked[1][1]) if len(body_ranked) > 1 else 0.0
    accent_only = [color for color in accents if color != dominant][:2]
    family_note = describe_palette_family(structural or accents)
    dominant_family = _named_color_family(dominant)
    if coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        if dominant and dominant_ratio >= 0.5 and second_ratio <= 0.3 and dominant in {"black", "gray", "white", "beige", "brown"}:
            if accent_only:
                return (
                    f"the main visible body reads as {dominant} with compatible printed accents in {', '.join(accent_only)}",
                    0.76,
                )
            return (f"the main visible body reads as {dominant} beneath a printed or multicolor treatment", 0.72)
        if dominant and dominant_ratio >= 0.42 and second_ratio <= 0.26 and dominant_family in {"dark-neutral", "cool-toned"} and accent_only:
            return (
                f"the main visible body reads as {dominant} with compatible printed accents in {', '.join(accent_only)}",
                0.72,
            )
        if dominant_ratio < 0.4 or second_ratio >= dominant_ratio * 0.72:
            return (
                "the visible body carries a dense multicolor or printed treatment with no single solid color dominating the observed view",
                0.44,
            )
        if accent_only:
            return (
                f"the visible body carries a {family_note} multicolor treatment with lighter or contrasting accents",
                0.46,
            )
        return (f"the visible body carries a {family_note} multicolor treatment", 0.46)
    if coverage_class == "localized_visible_pattern":
        if dominant and dominant_ratio >= 0.42:
            return (f"the main visible body reads as {dominant}", 0.78)
        return (f"the main visible body reads as a {family_note} treatment", 0.58)
    if dominant and (overlap >= 1 or len(structural) <= 1):
        return (f"the main visible body reads as {dominant}", 0.8)
    return (f"the main visible body reads as a {family_note} treatment", 0.5)


def synchronize_coverage_note_with_dominant_color(
    *,
    coverage_note: str | None,
    coverage_class: str | None,
    dominant_color: str | None,
    category: str,
    canonical_product_type: str,
) -> str | None:
    if not coverage_note or not dominant_color:
        return coverage_note
    normalized = str(dominant_color).lower()
    if coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        updated = re.sub(
            r"on a (?:black|white|gray|grey|blue|teal|green|red|pink|purple|yellow|gold|orange|brown|beige) base",
            f"on a {normalized} base",
            coverage_note,
            flags=re.IGNORECASE,
        )
        return updated
    if coverage_class != "low_variation_surface":
        return coverage_note
    if category == "bedding" or canonical_product_type in BEDDING_CANONICAL_TYPES:
        return f"most of the visible bedding surface reads as a tonal {normalized} textile field"
    if category == "pet home" or canonical_product_type == "pet bed":
        return f"most of the visible pet resting surface reads as a tonal {normalized} plush field"
    if canonical_product_type == "mug":
        return f"most of the visible mug surface reads as a relatively uniform {normalized} glaze treatment"
    return f"most of the visible product surface reads as a relatively uniform {normalized} treatment"


def synchronize_pattern_note_with_dominant_color(
    *,
    pattern_note: str | None,
    dominant_color: str | None,
) -> str | None:
    if not pattern_note or not dominant_color:
        return pattern_note
    return re.sub(
        r"on a (?:black|white|gray|grey|blue|teal|green|red|pink|purple|yellow|gold|orange|brown|beige) base",
        f"on a {str(dominant_color).lower()} base",
        pattern_note,
        flags=re.IGNORECASE,
    )


def describe_palette_family(palette: Sequence[str]) -> str:
    families = [_named_color_family(color) for color in palette if color]
    if not families:
        return "mixed-color"
    ordered_families = _dedupe_strings(families)
    primary = ordered_families[0]
    if primary == "dark-neutral":
        if "cool-toned" in ordered_families[1:]:
            return "dark cool-toned"
        if "warm-toned" in ordered_families[1:]:
            return "dark warm-toned"
        return "dark neutral"
    if primary == "light-neutral":
        if "warm-toned" in ordered_families[1:]:
            return "light warm-toned"
        if "cool-toned" in ordered_families[1:]:
            return "light cool-toned"
        return "light neutral"
    if primary == "cool-toned" and "warm-toned" in ordered_families[1:]:
        return "cool multicolor"
    if primary == "warm-toned" and "cool-toned" in ordered_families[1:]:
        return "warm multicolor"
    return primary.replace("-", " ")


def summarize_base_palette_label(palette: Sequence[str]) -> str:
    if not palette:
        return "mixed"
    dominant = palette[0]
    if dominant in {"brown", "beige", "gold", "yellow", "orange", "pink"}:
        return describe_palette_family(palette)
    return dominant


def infer_form_factor_note(
    *,
    category: str,
    canonical_product_type: str,
    shape_profile: dict[str, Any],
    upper_region_profile: dict[str, Any],
) -> str | None:
    aspect_ratio = shape_profile.get("aspect_ratio")
    if aspect_ratio is None:
        return None
    upper_component_state = upper_region_profile.get("component_state")
    top_width_ratio = shape_profile.get("top_width_ratio")
    if canonical_product_type == "backpack":
        if aspect_ratio >= 1.0:
            return "the visible bag form is a vertically oriented backpack body designed for shoulder or back carry"
        return "the product reads as a backpack body rather than a handbag or tote"
    if canonical_product_type == "table lamp":
        if top_width_ratio is not None and top_width_ratio >= 0.95 and aspect_ratio >= 1.6:
            return "the visible lamp form is upright with a broad upper shade over a narrower lower support"
        if aspect_ratio >= 1.6:
            return "the visible lamp form is upright with a stable lower support and upper light element"
    if canonical_product_type == "blender":
        return "the visible appliance form is an upright blender with a base below a blending vessel"
    if canonical_product_type == "toaster":
        return "the visible appliance form is a compact rigid toaster body meant to rest on a countertop"
    if canonical_product_type == "coffee maker":
        return "the visible appliance form is a countertop coffee maker with a stable brewing body and lower carafe or dispensing base"
    if canonical_product_type == "slow cooker":
        return "the visible appliance form is a countertop slow cooker with a broad cooking vessel and supporting base"
    if canonical_product_type == "food chopper":
        return "the visible appliance form is a compact food chopper with a base below a processing bowl"
    if canonical_product_type == "office chair":
        return "the visible furniture form is an upright chair with a seat and backrest"
    if canonical_product_type == "folding chair":
        return "the visible furniture form is a folding chair with a rigid slatted seat and backrest"
    if canonical_product_type in BEDDING_CANONICAL_TYPES:
        return "the visible bedding form is a broad soft comforter spread across a bed surface"
    if canonical_product_type == "mug":
        return "the visible drinkware form is an open mug vessel with a handle or graspable side profile"
    if canonical_product_type == "pet bed":
        return "the visible product form is a low soft pet bed meant to rest on the floor"
    if canonical_product_type == "shoe":
        return "the visible footwear form combines a structured upper with a grounded sole"
    if canonical_product_type == "shirt":
        return "the visible garment reads as an upper-body top rather than a home textile"
    if canonical_product_type == "dress" and aspect_ratio >= 1.25:
        return "the visible garment extends as a dress rather than a shorter upper-body top"
    if category == "bag":
        if upper_component_state != "present" and aspect_ratio <= 0.82 and top_width_ratio is not None and top_width_ratio <= 0.82:
            return "the visible bag form is compact and hand-held with no visible handles or shoulder straps"
        if upper_component_state == "present" and aspect_ratio >= 1.15:
            return "the visible bag form is a larger carryall with attached carrying components above the main body"
        if upper_component_state == "present":
            return "the visible bag form includes attached carrying components above the main body"
    if category == "drinkware" and aspect_ratio >= 2.2:
        if shape_profile.get("top_width_ratio") is not None and shape_profile["top_width_ratio"] <= 0.72:
            return "the visible vessel form is tall and narrow with a distinctly narrowed top section"
        return "the visible vessel form is tall and narrow"
    if category == "home decor" and canonical_product_type == "decorative pillow" and 0.8 <= aspect_ratio <= 1.2:
        return "the visible soft-good form is broad and cushion-like rather than a structured accessory"
    return None


def correct_footwear_surface_inference(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
    coverage_class: str | None,
    coverage_note: str | None,
    pattern_note: str | None,
    color_note: str | None,
    color_confidence: float | None,
    upper_region_profile: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None, float | None, dict[str, Any]]:
    if category != "footwear" and canonical_product_type != "shoe":
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence, upper_region_profile
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence, upper_region_profile
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence, upper_region_profile
    ys, xs = np.nonzero(mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    width = float(x1 - x0)
    height = float(y1 - y0)
    if height / max(width, 1.0) > 1.0:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence, upper_region_profile
    lower_start = y0 + int(round(height * 0.18))
    lower_mask = np.zeros_like(mask, dtype=bool)
    lower_mask[lower_start:y1, :] = mask[lower_start:y1, :]
    pixels = source[lower_mask]
    if len(pixels) < 64:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence, upper_region_profile
    stride = max(1, int(math.ceil(len(pixels) / 3000.0)))
    sampled = pixels[::stride]
    distribution = _named_color_distribution(sampled)
    if not distribution:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence, upper_region_profile
    dominant_name, dominant_ratio = max(distribution.items(), key=lambda item: item[1])
    non_neutral_ratio = sum(
        ratio
        for name, ratio in distribution.items()
        if name not in {"white", "beige", "gray", "black", "brown"}
    )
    adjusted_profile = dict(upper_region_profile)
    if coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        if non_neutral_ratio <= 0.08:
            coverage_class = "low_variation_surface"
            coverage_note = f"most of the visible shoe surface reads as a relatively uniform {dominant_name} treatment"
            pattern_note = None
            color_note = f"the main visible body reads as {dominant_name}"
            color_confidence = max(0.74, float(color_confidence or 0.0))
            adjusted_profile["note"] = None
            adjusted_profile["component_state"] = "uncertain"
            adjusted_profile["upper_region_color"] = None
        elif non_neutral_ratio <= 0.32:
            coverage_class = "localized_visible_pattern"
            coverage_note = "a limited accent or contrast zone appears on one region of the visible shoe surface"
            pattern_note = "the visible shoe body is mostly neutral with limited accent zones rather than an all-over printed treatment"
            color_note = f"the main visible body reads as {dominant_name}"
            color_confidence = max(0.7, float(color_confidence or 0.0))
    return coverage_class, coverage_note, pattern_note, color_note, color_confidence, adjusted_profile


def infer_material_note(
    *,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    evidence_caption: str | None,
) -> str | None:
    title_text = str(product_title).lower()
    hint_text = " ".join(str(part) for part in hint_phrases if part).lower()
    caption_text = str(evidence_caption or "").lower()
    text = " ".join(part for part in [title_text, hint_text, caption_text] if part)
    notes: list[str] = []
    token_set = set(_tokens(text))
    rigid_structured_categories = {"drinkware", "kitchen appliance", "furniture", "home lighting"}
    rigid_structured_types = STRUCTURED_DISPLAY_CANONICAL_TYPES
    for label, patterns in MATERIAL_TEXT_TOKENS.items():
        if (
            label == "fabric"
            and category in rigid_structured_categories
            and canonical_product_type in rigid_structured_types
        ):
            supporting_text = " ".join(part for part in [hint_text, caption_text] if part)
            supporting_tokens = set(_tokens(supporting_text))
            if not any(pattern in supporting_text for pattern in patterns) and not supporting_tokens.intersection(
                _tokens(" ".join(patterns))
            ):
                continue
        if any(pattern in text for pattern in patterns) or token_set.intersection(_tokens(" ".join(patterns))):
            if label == "woven":
                notes.append("visible material cues suggest a woven or interlaced texture")
            elif label == "braided":
                notes.append("visible material cues suggest a braided or corded component")
            elif label == "textured":
                notes.append("visible material cues suggest a distinct textured surface")
            elif label == "transparent":
                notes.append("visible material cues suggest a transparent or translucent component")
            elif label == "fabric":
                notes.append("visible material cues suggest a fabric or textile surface")
            elif label == "metallic":
                notes.append("visible material cues suggest metallic hardware or reflective accents")
            elif label == "glossy":
                notes.append("visible material cues suggest a glossy finish")
            elif label == "matte":
                notes.append("visible material cues suggest a matte finish")
    return "; ".join(_dedupe_strings(notes)[:2]) or None


def infer_surface_relief_note(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
) -> str | None:
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return None
    if canonical_product_type != "table lamp" and category not in {"home lighting", "product"}:
        return None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    height = max(1, y1 - y0)
    shade_end = y0 + int(round(height * 0.46))
    shade_mask = np.zeros_like(mask, dtype=bool)
    shade_mask[y0:shade_end, :] = mask[y0:shade_end, :]
    if not shade_mask.any():
        return None
    luminance = (
        0.2126 * source[:, :, 0]
        + 0.7152 * source[:, :, 1]
        + 0.0722 * source[:, :, 2]
    )
    shade_ys, shade_xs = np.nonzero(shade_mask)
    if len(shade_xs) == 0:
        return None
    x0, x1 = int(shade_xs.min()), int(shade_xs.max()) + 1
    residual_sum = np.zeros(x1 - x0, dtype=np.float32)
    residual_count = np.zeros(x1 - x0, dtype=np.float32)
    for y in range(y0, shade_end):
        row_mask = shade_mask[y, x0:x1]
        if row_mask.sum() < max(6, int((x1 - x0) * 0.25)):
            continue
        row_values = luminance[y, x0:x1][row_mask]
        row_mean = float(row_values.mean())
        row_residual = luminance[y, x0:x1] - row_mean
        residual_sum[row_mask] += row_residual[row_mask]
        residual_count[row_mask] += 1.0
    valid = residual_count > 0
    if int(valid.sum()) < 8:
        return None
    profile = residual_sum[valid] / residual_count[valid]
    if len(profile) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smoothed = np.convolve(profile, kernel, mode="same")
    else:
        smoothed = profile
    profile_range = float(smoothed.max() - smoothed.min())
    profile_std = float(smoothed.std())
    if profile_range < 5.0 or profile_std < 1.6:
        return None
    derivative = np.diff(smoothed)
    sign_changes = 0
    for left, right in zip(derivative[:-1], derivative[1:], strict=False):
        if abs(left) < 0.55 or abs(right) < 0.55:
            continue
        if left > 0 and right < 0:
            sign_changes += 1
        elif left < 0 and right > 0:
            sign_changes += 1
    if sign_changes < 3:
        return None
    return "the visible structured surface shows repeated vertical ridges or fluted relief rather than a completely smooth finish"


def build_evidence_tags(
    *,
    coverage_class: str | None,
    trim_note: str | None,
    upper_region_note: str | None,
    upper_component_state: str | None,
    lower_region_note: str | None,
    lower_component_state: str | None,
    form_factor_note: str | None,
    material_note: str | None,
    surface_relief_note: str | None,
    edge_profile_note: str | None,
    soft_structure_note: str | None,
) -> list[str]:
    tags: list[str] = []
    if coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        tags.append("broad_surface_treatment")
    elif coverage_class == "localized_visible_pattern":
        tags.append("localized_surface_treatment")
    if trim_note:
        tags.append("distinct_boundary_trim")
    if upper_region_note:
        tags.append("distinct_upper_component")
    elif upper_component_state == "absent":
        tags.append("no_distinct_upper_component")
    if lower_region_note:
        tags.append("distinct_lower_component")
    elif lower_component_state == "absent":
        tags.append("no_distinct_lower_component")
    if form_factor_note:
        tags.extend(_tokens(form_factor_note))
    if material_note:
        tags.extend(_tokens(material_note))
    if surface_relief_note:
        tags.append("structured_surface_relief")
        tags.extend(_tokens(surface_relief_note))
    if edge_profile_note:
        tags.append("structured_edge_profile")
        tags.extend(_tokens(edge_profile_note))
    if soft_structure_note:
        tags.append("soft_structure_profile")
        tags.extend(_tokens(soft_structure_note))
    return _dedupe_strings(tags)


def _caption_candidate_paths(
    *,
    source_image: Path | None,
    cutout_path: Path | None,
    crop_path: Path | None,
    prefer_source_context: bool = False,
) -> list[Path]:
    candidate_paths: list[Path] = []
    if prefer_source_context and source_image is not None and source_image.exists():
        candidate_paths.append(source_image)
    if cutout_path is not None and cutout_path.exists():
        candidate_paths.append(cutout_path)
    if crop_path is not None and crop_path.exists() and crop_path not in candidate_paths:
        candidate_paths.append(crop_path)
    if not candidate_paths and source_image is not None and source_image.exists():
        candidate_paths.append(source_image)
    return candidate_paths


def infer_raw_evidence_caption(
    backbone: VisionBackbone | None,
    *,
    source_image: Path | None,
    cutout_path: Path | None,
    crop_path: Path | None,
    prefer_source_context: bool = False,
) -> str | None:
    if backbone is None:
        return None
    for candidate_path in _caption_candidate_paths(
        source_image=source_image,
        cutout_path=cutout_path,
        crop_path=crop_path,
        prefer_source_context=prefer_source_context,
    ):
        caption = " ".join(backbone.caption_image(candidate_path).split()).strip().lower()
        if not caption or caption == "product":
            continue
        return caption
    return None


def infer_evidence_caption(
    backbone: VisionBackbone | None,
    *,
    source_image: Path | None,
    cutout_path: Path | None,
    crop_path: Path | None,
    canonical_product_type: str,
    category: str,
    prefer_source_context: bool = False,
) -> str | None:
    if backbone is None:
        return None
    for candidate_path in _caption_candidate_paths(
        source_image=source_image,
        cutout_path=cutout_path,
        crop_path=crop_path,
        prefer_source_context=prefer_source_context,
    ):
        caption = " ".join(backbone.caption_image(candidate_path).split()).strip().lower()
        if not caption or caption == "product":
            continue
        if infer_category(caption) != category:
            # Keep captions that still mention the canonical type even if generic category inference differs.
            canonical_tokens = set(_tokens(canonical_product_type))
            if not canonical_tokens.intersection(_tokens(caption)):
                continue
        return caption
    return None


def _strip_human_context_from_caption(
    caption: str,
    *,
    canonical_product_type: str,
    category: str,
) -> str | None:
    cleaned = " ".join(caption.lower().split()).strip(" ,.")
    substitutions = (
        (
            r"\b(?:a|an|the)\s+(?:woman|women|man|men|person|people|model|girl|boy|lady|gentleman|female|male)\s+"
            r"(?:wearing|holding|carrying|in)\s+",
            "a ",
        ),
        (
            r"\b(?:worn by|held by|carried by|on)\s+(?:a|an|the)\s+"
            r"(?:woman|women|man|men|person|people|model|girl|boy|lady|gentleman|female|male)\b.*$",
            "",
        ),
        (
            r"\b(?:and|with)\s+(?:glasses|sunglasses|hat|cap|hair|hairstyle|face|smile|earrings|necklace|bracelet|watch|makeup)\b(?:\s+\w+){0,3}",
            "",
        ),
        (
            r"\b(?:a|an|the)\s+(?:woman|women|man|men|person|people|model|girl|boy|lady|gentleman|female|male)\b",
            "",
        ),
    )
    for pattern, replacement in substitutions:
        cleaned = re.sub(pattern, replacement, cleaned)
    cleaned = re.sub(r"\b(a|an|the)\s+(a|an|the)\b", r"\1", cleaned)
    cleaned = re.sub(r"^(?:'s|s'|’s)\s+", "", cleaned)
    cleaned = re.sub(r"^[\"'`’\s]*s[\"'`’\s]+", "", cleaned)
    cleaned = " ".join(cleaned.replace(" ,", ",").split()).strip(" ,.")
    if not cleaned:
        return None

    cleaned_tokens = set(_tokens(cleaned))
    canonical_tokens = set(_tokens(canonical_product_type))
    category_tokens = set(CATEGORY_CAPTION_HINTS.get(category, ()))
    if (canonical_tokens or category_tokens) and not cleaned_tokens.intersection(canonical_tokens | category_tokens):
        return None
    competing_tokens = COMPETING_CATEGORY_TYPE_TOKENS.get(category, frozenset()) - canonical_tokens
    if competing_tokens and cleaned_tokens.intersection(competing_tokens):
        return None
    if cleaned_tokens.intersection(PERSON_CAPTION_TOKENS | PERSON_ACCESSORY_TOKENS):
        return None
    return cleaned


def sanitize_evidence_caption(
    caption: str | None,
    *,
    canonical_product_type: str,
    category: str,
    requires_human_model: bool = False,
) -> str | None:
    if not caption:
        return None
    if requires_human_model:
        caption = _strip_human_context_from_caption(
            caption,
            canonical_product_type=canonical_product_type,
            category=category,
        )
        if not caption:
            return None
    caption_tokens = set(_tokens(caption))
    canonical_tokens = set(_tokens(canonical_product_type))
    negative_tokens = set(CATEGORY_NEGATIVE_HINTS.get(category, ()))
    if caption_tokens.intersection(negative_tokens):
        return None
    if canonical_product_type == "water bottle" and "bottle" not in caption_tokens and caption_tokens.intersection({"cup", "mug", "glass", "tumbler"}):
        return None
    if canonical_product_type == "mug" and not caption_tokens.intersection({"mug", "cup", "tumbler"}):
        return None
    if category in {"drinkware", "kitchen appliance", "furniture", "home lighting", "bedding", "pet home"}:
        caption_type = infer_canonical_product_type(caption, (), caption)
        if caption_type and caption_type != canonical_product_type:
            structured_types = (
                STRUCTURED_DISPLAY_CANONICAL_TYPES
                | BEDDING_CANONICAL_TYPES
                | {"pet bed"}
            )
            if canonical_product_type in structured_types and caption_type in structured_types:
                return None
    if canonical_tokens and not caption_tokens.intersection(canonical_tokens):
        category_tokens = set(CATEGORY_CAPTION_HINTS.get(category, ()))
        if not category_tokens or not caption_tokens.intersection(category_tokens):
            return None
    ambient_tokens = {
        "background",
        "bed",
        "bedroom",
        "room",
        "floor",
        "wall",
        "grass",
        "tree",
        "counter",
        "countertop",
        "table",
        "chair",
        "sofa",
        "couch",
        "shelf",
        "cabinet",
        "closet",
        "door",
        "window",
        "dog",
        "cat",
    }
    if caption_tokens.intersection(ambient_tokens):
        return None
    caption = repair_sparse_evidence_caption_phrase(caption, canonical_product_type=canonical_product_type)
    return caption


def repair_sparse_evidence_caption_phrase(
    caption: str | None,
    *,
    canonical_product_type: str,
) -> str | None:
    if not caption:
        return None
    cleaned = " ".join(str(caption).split()).strip(" ,.")
    if not cleaned:
        return None
    caption_tokens = set(_tokens(cleaned))
    canonical_tokens = set(_tokens(canonical_product_type))
    sparse_noise_tokens = {"a", "an", "the", "and", "with", "of", "on", "in"}
    if canonical_tokens and caption_tokens and caption_tokens.issubset(canonical_tokens | sparse_noise_tokens):
        article = "an" if canonical_product_type[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"{article} {canonical_product_type}"
    return cleaned


def is_low_information_retrieval_caption(caption: str | None) -> bool:
    if not caption:
        return False
    tokens = _tokens(caption)
    if len(tokens) < 6:
        return False
    counts = Counter(tokens)
    unique_count = len(counts)
    unique_ratio = unique_count / float(len(tokens))
    max_share = max(counts.values()) / float(len(tokens))
    if unique_count <= 2:
        return True
    if max_share >= 0.5 and unique_ratio <= 0.35:
        return True
    return False


def sanitize_retrieval_candidates_for_planning(
    candidates: Sequence[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    sanitized: list[RetrievalCandidate] = []
    for candidate in candidates:
        if not is_low_information_retrieval_caption(candidate.caption):
            sanitized.append(candidate)
            continue
        sanitized.append(
            replace(
                candidate,
                caption="",
                style_atoms=tuple(),
                scenario_slots=tuple(),
            )
        )
    return sanitized


def extract_caption_colors(caption: str | None) -> list[str]:
    if not caption:
        return []
    ordered: list[str] = []
    for token in _tokens(caption):
        normalized = "gray" if token == "grey" else token
        if normalized in EVIDENCE_COLOR_SWATCHES:
            ordered.append(normalized)
    return _dedupe_strings(ordered)


def sanitize_evidence_caption_against_color_evidence(
    caption: str | None,
    *,
    canonical_product_type: str,
    palette: Sequence[str],
    color_note: str | None,
    color_confidence: float | None,
) -> str | None:
    if not caption or color_confidence is None or color_confidence < 0.68:
        return caption
    caption_colors = extract_caption_colors(caption)
    if not caption_colors:
        return caption
    expected_colors = extract_caption_colors(color_note) if color_note else []
    if not expected_colors:
        expected_colors = list(palette[:1])
    if not expected_colors or expected_colors[0] in caption_colors:
        return caption
    filtered_tokens: list[str] = []
    for token in caption.split():
        lowered = token.strip(",.;:!?").lower()
        normalized = "gray" if lowered == "grey" else lowered
        if normalized in EVIDENCE_COLOR_SWATCHES:
            continue
        filtered_tokens.append(token)
    sanitized = " ".join(filtered_tokens).strip(" ,-/")
    return repair_sparse_evidence_caption_phrase(sanitized or None, canonical_product_type=canonical_product_type)


def should_apply_caption_color_override(
    *,
    evidence_caption: str | None,
    canonical_product_type: str,
    category: str,
    color_confidence: float | None,
    coverage_class: str | None,
) -> bool:
    if not evidence_caption:
        return False
    if color_confidence is not None and color_confidence >= 0.55:
        return False
    if coverage_class not in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        return False
    if category in {"bedding", "pet home", "furniture", "kitchen appliance", "drinkware"}:
        return False
    caption_tokens = set(_tokens(evidence_caption))
    ambient_tokens = {
        "background",
        "bed",
        "bedroom",
        "room",
        "floor",
        "wall",
        "grass",
        "tree",
        "counter",
        "countertop",
        "table",
        "chair",
        "sofa",
        "couch",
        "shelf",
        "cabinet",
        "closet",
        "door",
        "window",
        "dog",
        "cat",
    }
    if caption_tokens.intersection(ambient_tokens):
        return False
    canonical_tokens = set(_tokens(canonical_product_type))
    return not canonical_tokens or bool(caption_tokens.intersection(canonical_tokens))


def reconcile_color_note_with_caption(
    color_note: str | None,
    *,
    caption_colors: Sequence[str],
    coverage_class: str | None,
) -> str | None:
    colors = _dedupe_strings(caption_colors)
    if not colors:
        return color_note
    base = colors[0]
    accents = [color for color in colors if color != base][:2]
    if coverage_class not in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        return color_note
    if accents:
        return f"the main visible body reads as {base} with compatible printed accents in {', '.join(accents)}"
    return f"the main visible body reads as {base}"


def infer_low_saturation_cool_body_override(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
    coverage_class: str | None,
    palette: Sequence[str],
) -> tuple[str | None, list[str] | None, float | None]:
    drinkware_candidate = bool(category == "drinkware" and canonical_product_type in {"mug", "water bottle"})
    structured_rigid_candidate = bool(
        category in {"kitchen appliance", "home lighting", "furniture"}
        and canonical_product_type in STRUCTURED_DISPLAY_CANONICAL_TYPES
    )
    if not drinkware_candidate and not structured_rigid_candidate:
        return None, None, None
    if coverage_class not in {
        "low_variation_surface",
        "localized_visible_pattern",
        "full_visible_surface_pattern",
        "broad_visible_surface_pattern",
    }:
        return None, None, None
    if not palette or palette[0] not in {"gray", "beige", "white", "brown", "black"}:
        return None, None, None
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return None, None, None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None, None, None
    core_mask = _extract_core_body_mask(mask)
    if core_mask.sum() < max(64, int(mask.sum() * 0.16)):
        core_mask = _erode_mask(mask, steps=2)
    pixels = source[core_mask] if core_mask.any() else source[mask]
    if len(pixels) < 64:
        return None, None, None
    stride = max(1, int(math.ceil(len(pixels) / 3000.0)))
    sampled = np.clip(pixels[::stride].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled], dtype=np.float32)
    if hsv.size == 0:
        return None, None, None
    value = hsv[:, 2]
    saturation = hsv[:, 1]
    median_value = float(np.quantile(value, 0.5))
    upper_quartile_value = float(np.quantile(value, 0.75))
    median_saturation = float(np.quantile(saturation, 0.5))
    if float(np.quantile(saturation, 0.5)) < 0.1 or float(np.quantile(saturation, 0.75)) < 0.16:
        return None, None, None
    chromatic_mask = saturation >= max(0.11, float(np.quantile(saturation, 0.5)) * 0.8)
    chroma_hsv = hsv[chromatic_mask] if np.count_nonzero(chromatic_mask) >= 24 else hsv
    chroma_pixels = sampled[chromatic_mask] if np.count_nonzero(chromatic_mask) >= 24 else sampled
    hue = float(np.quantile(chroma_hsv[:, 0], 0.5))
    mean_rgb = np.mean(chroma_pixels, axis=0)
    cool_rgb_bias = bool(mean_rgb[2] >= mean_rgb[1] + 0.03 and mean_rgb[1] >= mean_rgb[0] + 0.02)
    if median_value < 0.42:
        if not (
            structured_rigid_candidate
            and median_value >= 0.36
            and upper_quartile_value >= 0.43
            and median_saturation >= 0.18
            and cool_rgb_bias
        ):
            return None, None, None
    base: str | None = None
    if 0.18 <= hue < 0.38 and mean_rgb[1] >= mean_rgb[2] + 0.03:
        base = "green"
    elif 0.38 <= hue <= 0.58:
        base = "teal"
    elif 0.58 < hue <= 0.76:
        base = "blue"
    if base is None:
        return None, None, None
    updated_palette = [base]
    updated_palette.extend(
        color
        for color in _dedupe_strings(palette)
        if color != base and color in {"white", "gray", "beige", "brown", "black"}
    )
    if canonical_product_type == "mug":
        note = f"the main visible body reads as {base} with low-saturation glazed variation"
    elif drinkware_candidate:
        note = f"the main visible body reads as {base} with subtle cool-toned surface variation"
    else:
        note = f"the main visible body reads as {base} with cool-toned reflective variation"
    return note, updated_palette[:3], 0.72


def infer_dark_reflective_body_override(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str | None,
    coverage_class: str | None,
    palette: Sequence[str],
) -> tuple[str | None, list[str] | None, float | None]:
    structured_rigid_candidate = bool(
        category in {"kitchen appliance", "home lighting", "furniture"}
        and canonical_product_type in STRUCTURED_DISPLAY_CANONICAL_TYPES
        and coverage_class in {"low_variation_surface", "localized_visible_pattern"}
    )
    drinkware_candidate = bool(category == "drinkware" and coverage_class == "localized_visible_pattern")
    if not drinkware_candidate and not structured_rigid_candidate:
        return None, None, None
    if structured_rigid_candidate:
        dominant_palette = [color for color in _dedupe_strings(palette[:4]) if color]
        warm_earth_palette = {"brown", "orange", "beige", "gold", "yellow", "red"}
        neutral_palette = {"black", "gray", "white"}
        if dominant_palette:
            leading = dominant_palette[0]
            warm_count = sum(1 for color in dominant_palette if color in warm_earth_palette)
            if leading in warm_earth_palette or (warm_count >= 2 and leading not in neutral_palette):
                return None, None, None
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return None, None, None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None, None, None
    core_mask = _extract_core_body_mask(mask)
    if not core_mask.any():
        return None, None, None
    pixels = source[core_mask]
    if len(pixels) < 64:
        return None, None, None
    stride = max(1, int(math.ceil(len(pixels) / 3000.0)))
    sampled = np.clip(pixels[::stride].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled], dtype=np.float32)
    if hsv.size == 0:
        return None, None, None
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    dark_ratio = float(np.mean(value <= 0.34))
    mid_dark_ratio = float(np.mean(value <= 0.46))
    median_value = float(np.quantile(value, 0.5))
    lower_quartile_value = float(np.quantile(value, 0.25))
    median_saturation = float(np.quantile(saturation, 0.5))
    if drinkware_candidate:
        if mid_dark_ratio < 0.58:
            return None, None, None
        if median_value > 0.42 and dark_ratio < 0.34:
            return None, None, None
        if median_value <= 0.32 or dark_ratio >= 0.5 or lower_quartile_value <= 0.22:
            base = "black"
        else:
            base = "gray"
        if median_saturation <= 0.18 and base == "black" and lower_quartile_value > 0.24:
            base = "gray"
    else:
        if mid_dark_ratio < 0.42:
            return None, None, None
        if median_value > 0.5 and dark_ratio < 0.18:
            return None, None, None
        base = "black" if (
            median_value <= 0.46
            or lower_quartile_value <= 0.28
            or dark_ratio >= 0.24
        ) else "gray"
        if base == "gray" and median_saturation <= 0.22 and dark_ratio >= 0.18:
            base = "black"
    neutral_palette = [base]
    neutral_palette.extend(
        color
        for color in _dedupe_strings(palette)
        if color != base and color in {"white", "gray", "black"}
    )
    return (
        f"the main visible body reads as {base} with reflective highlight variation",
        neutral_palette[:3],
        0.82,
    )


def infer_backpack_structure_note(
    *,
    category: str,
    canonical_product_type: str,
    palette: Sequence[str],
    accent_palette: Sequence[str],
    evidence_caption: str | None,
) -> str | None:
    if category != "bag" or canonical_product_type != "backpack":
        return None
    dominant = palette[0] if palette else None
    if dominant not in {"blue", "teal", "green", "red", "purple", "gray", "black"}:
        return None
    accent_colors = set(_dedupe_strings([*palette[:3], *accent_palette[:4]]))
    caption_tokens = set(_tokens(evidence_caption or ""))
    if "black" not in accent_colors and "black" not in caption_tokens:
        return None
    if dominant == "black":
        return "the visible backpack body includes distinct darker harness, panel, or attachment zones that should remain structurally readable"
    return (
        f"the visible backpack body includes darker harness, panel, or attachment zones against a {dominant} main body"
    )


def infer_neutral_textile_surface_notes(
    *,
    material_note: str | None,
    palette: Sequence[str],
    coverage_class: str | None,
    category: str | None = None,
    canonical_product_type: str | None = None,
    source_image: Path | None = None,
    mask_path: Path | None = None,
) -> tuple[str | None, str | None, str | None]:
    if coverage_class not in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        return None, None, None
    neutral_palette = [color for color in palette[:3] if color in {"beige", "white", "gray", "brown"}]
    if len(neutral_palette) < 2:
        return None, None, None
    lowered_material = (material_note or "").lower()
    if "woven" in lowered_material or "fabric" in lowered_material or "textile" in lowered_material:
        base = neutral_palette[0]
        return (
            f"the main visible body reads as {base} with subtle tonal variation from woven texture",
            "the visible woven or ribbed texture spans most of the observed product surface",
            "the visible surface is defined by woven or ribbed texture rather than a bold printed graphic",
        )
    if category not in {"home decor", "apparel", "bedding", "pet home"}:
        return None, None, None
    if source_image is None or mask_path is None or not source_image.exists() or not mask_path.exists():
        return None, None, None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None, None, None
    core_mask = _extract_core_body_mask(mask)
    if not core_mask.any():
        return None, None, None
    pixels = source[core_mask]
    if len(pixels) < 64:
        return None, None, None
    stride = max(1, int(math.ceil(len(pixels) / 3000.0)))
    sampled = np.clip(pixels[::stride].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled], dtype=np.float32)
    if hsv.size == 0:
        return None, None, None
    saturation = hsv[:, 1]
    if float(np.quantile(saturation, 0.75)) > 0.42:
        return None, None, None
    value = hsv[:, 2]
    base = neutral_palette[0]
    if base in {"gray", "beige", "brown"}:
        median_value = float(np.quantile(value, 0.5))
        lower_quartile_value = float(np.quantile(value, 0.25))
        if median_value <= 0.3 or lower_quartile_value <= 0.22:
            base = "black"
    return (
        f"the main visible body reads as {base} with subtle tonal variation from textured fabric",
        "the visible tonal texture spans most of the observed product surface",
        "the visible surface is defined by tonal textile texture rather than a bold printed graphic",
    )


def infer_soft_textile_chromatic_override(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
    coverage_class: str | None,
    palette: Sequence[str],
) -> tuple[str | None, list[str] | None, float | None]:
    if category not in {"bedding", "pet home", "home decor", "apparel"} and canonical_product_type not in {
        *BEDDING_CANONICAL_TYPES,
        "pet bed",
        "decorative pillow",
        "shirt",
        "dress",
    }:
        return None, None, None
    if coverage_class not in {
        "low_variation_surface",
        "localized_visible_pattern",
        "full_visible_surface_pattern",
        "broad_visible_surface_pattern",
    }:
        return None, None, None
    if not palette or palette[0] not in {"black", "gray", "brown", "beige"}:
        return None, None, None
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return None, None, None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None, None, None
    core_mask = _extract_core_body_mask(mask)
    if core_mask.sum() < max(64, int(mask.sum() * 0.18)):
        core_mask = _erode_mask(mask, steps=2)
    pixels = source[core_mask] if core_mask.any() else source[mask]
    if len(pixels) < 64:
        return None, None, None
    stride = max(1, int(math.ceil(len(pixels) / 3000.0)))
    sampled = np.clip(pixels[::stride].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled], dtype=np.float32)
    if hsv.size == 0:
        return None, None, None
    value = hsv[:, 2]
    bright_mask = value >= np.quantile(value, 0.7)
    bright_pixels = sampled[bright_mask]
    bright_hsv = hsv[bright_mask]
    if len(bright_pixels) < 32:
        bright_pixels = sampled
        bright_hsv = hsv
    chromatic_mask = bright_hsv[:, 1] >= 0.07
    hue_pixels = bright_pixels[chromatic_mask] if np.count_nonzero(chromatic_mask) >= 24 else bright_pixels
    hue_hsv = bright_hsv[chromatic_mask] if np.count_nonzero(chromatic_mask) >= 24 else bright_hsv
    saturation_q75 = float(np.quantile(hue_hsv[:, 1], 0.75))
    saturation_q5 = float(np.quantile(hue_hsv[:, 1], 0.05))
    saturation_q50 = float(np.quantile(hue_hsv[:, 1], 0.5))
    min_q75 = 0.14
    min_q50 = 0.09
    if category in {"bedding", "pet home"} or canonical_product_type in (BEDDING_CANONICAL_TYPES | {"pet bed"}):
        min_q75 = 0.08
        min_q50 = 0.05
    if saturation_q75 < min_q75 or saturation_q50 < min_q50:
        return None, None, None
    mean_rgb = np.mean(hue_pixels, axis=0)
    hue = float(np.quantile(hue_hsv[:, 0], 0.5))
    base: str | None = None
    soft_home_candidate = bool(
        category in {"bedding", "pet home", "home decor"}
        or canonical_product_type in (*BEDDING_CANONICAL_TYPES, "pet bed", "decorative pillow")
    )
    if 0.07 <= hue <= 0.34 and (
        (mean_rgb[1] >= mean_rgb[2] + 0.03 and mean_rgb[1] >= mean_rgb[0] * 0.88)
        or (
            soft_home_candidate
            and 0.1 <= hue <= 0.24
            and mean_rgb[1] >= mean_rgb[0] - 0.02
            and mean_rgb[1] >= mean_rgb[2] + 0.05
        )
    ):
        base = "green"
    elif 0.38 <= hue <= 0.58:
        base = "teal"
    elif 0.58 < hue <= 0.76:
        base = "blue"
    elif 0.76 < hue <= 0.92:
        base = "purple"
    elif hue <= 0.06 or hue >= 0.94:
        base = "red"
    if base is None:
        return None, None, None
    distribution = _weighted_structural_color_distribution((hue_pixels * 255.0).astype(np.float32))
    base_ratio = float(distribution.get(base, 0.0))
    neutral_ratio = float(sum(distribution.get(color, 0.0) for color in ("black", "gray", "brown", "beige", "white")))
    min_base_ratio = 0.26 if category in {"bedding", "pet home", "home decor"} else 0.42
    strong_channel_separation = False
    if base == "green":
        if soft_home_candidate:
            strong_channel_separation = bool(
                mean_rgb[1] >= mean_rgb[0] - 0.02 and mean_rgb[1] >= mean_rgb[2] + 0.04
            )
        else:
            strong_channel_separation = bool(mean_rgb[1] >= mean_rgb[0] + 0.04 and mean_rgb[1] >= mean_rgb[2] + 0.04)
    elif base == "teal":
        strong_channel_separation = bool(mean_rgb[2] >= mean_rgb[0] + 0.04 and mean_rgb[1] >= mean_rgb[0] + 0.04)
    elif base == "blue":
        strong_channel_separation = bool(mean_rgb[2] >= mean_rgb[0] + 0.05 and mean_rgb[2] >= mean_rgb[1] + 0.05)
    elif base == "purple":
        strong_channel_separation = bool(mean_rgb[2] >= mean_rgb[1] + 0.04 and mean_rgb[0] >= mean_rgb[1] + 0.02)
    elif base == "red":
        strong_channel_separation = bool(mean_rgb[0] >= mean_rgb[1] + 0.05 and mean_rgb[0] >= mean_rgb[2] + 0.05)
    if base_ratio < min_base_ratio and not (
        soft_home_candidate
        and strong_channel_separation
        and (
            saturation_q75 >= 0.18
            or (base == "green" and 0.1 <= hue <= 0.24 and saturation_q75 >= 0.14)
        )
    ):
        return None, None, None
    if category == "apparel" and (
        neutral_ratio >= max(0.34, base_ratio + 0.08)
        or not strong_channel_separation
    ):
        return None, None, None
    median_value = float(np.quantile(hue_hsv[:, 2], 0.5))
    tone_prefix = "dark " if median_value <= 0.45 else ""
    note = f"the main visible body reads as {tone_prefix}{base} with low-luster tonal variation from textured fabric"
    updated_palette = [base]
    updated_palette.extend(color for color in _dedupe_strings(palette) if color != base and color in {"black", "gray", "brown", "beige", "white"})
    return note, updated_palette[:3], 0.72


def soften_uncertain_neutral_apparel_color_evidence(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    evidence_caption: str | None,
    coverage_class: str | None,
    color_note: str | None,
    color_confidence: float | None,
    palette: Sequence[str],
) -> tuple[str | None, float | None, list[str] | None]:
    if category != "apparel" or canonical_product_type not in {"shirt", "dress"}:
        return color_note, color_confidence, None
    if coverage_class not in {"low_variation_surface", "localized_visible_pattern"}:
        return color_note, color_confidence, None
    if not color_note or mask_path is None or not mask_path.exists() or not source_image.exists():
        return color_note, color_confidence, None

    color_text = color_note.lower()
    if not any(token in color_text for token in ("beige", "gold", "orange", "brown")):
        return color_note, color_confidence, None

    text_tokens = set(
        _tokens(
            " ".join(
                part
                for part in (
                    product_title,
                    *hint_phrases,
                    evidence_caption or "",
                )
                if part
            )
        )
    )
    if text_tokens.intersection({"beige", "tan", "camel", "khaki", "taupe", "brown", "sand", "stone", "cream", "ivory"}):
        return color_note, color_confidence, None

    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return color_note, color_confidence, None
    core_mask = _extract_core_body_mask(mask)
    if core_mask.sum() < max(64, int(mask.sum() * 0.18)):
        core_mask = _erode_mask(mask, steps=3)
    pixels = source[core_mask] if core_mask.any() else source[mask]
    if len(pixels) < 64:
        return color_note, color_confidence, None

    stride = max(1, int(math.ceil(len(pixels) / 3000.0)))
    sampled = np.clip(pixels[::stride].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled], dtype=np.float32)
    if hsv.size == 0:
        return color_note, color_confidence, None

    value_q10, value_q50 = (float(q) for q in np.quantile(hsv[:, 2], [0.1, 0.5]))
    saturation_q50, saturation_q75 = (float(q) for q in np.quantile(hsv[:, 1], [0.5, 0.75]))
    if value_q10 < 0.58 or value_q50 < 0.8:
        return color_note, color_confidence, None
    if saturation_q75 > 0.26 or saturation_q50 > 0.24:
        return color_note, color_confidence, None

    softened_note = (
        "the main visible body reads as a light neutral fabric with tonal variation from folds, drape, or soft-surface texture"
    )
    softened_confidence = min(float(color_confidence or 0.6), 0.54)
    return softened_note, softened_confidence, ["white", "beige", "gray"]


def correct_soft_textile_surface_inference(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
    coverage_class: str | None,
    coverage_note: str | None,
    pattern_note: str | None,
    color_note: str | None,
    color_confidence: float | None,
) -> tuple[str | None, str | None, str | None, str | None, float | None]:
    if category not in {"bedding", "pet home", "home decor", "apparel"} and canonical_product_type not in {
        *BEDDING_CANONICAL_TYPES,
        "pet bed",
        "decorative pillow",
        "shirt",
        "dress",
    }:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    if coverage_class not in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    if mask_path is None or not mask_path.exists() or not source_image.exists():
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    core_mask = _extract_core_body_mask(mask)
    if core_mask.sum() < max(64, int(mask.sum() * 0.18)):
        core_mask = _erode_mask(mask, steps=2)
    pixels = source[core_mask] if core_mask.any() else source[mask]
    if len(pixels) < 64:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    stride = max(1, int(math.ceil(len(pixels) / 3000.0)))
    sampled = np.clip(pixels[::stride].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled], dtype=np.float32)
    if hsv.size == 0:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    if float(np.quantile(saturation, 0.75)) > 0.34:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    distribution = _weighted_structural_color_distribution(pixels[::stride])
    if not distribution:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    ranked = _reorder_structural_palette(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    dominant_name, dominant_ratio = ranked[0]
    if dominant_ratio < 0.34:
        return coverage_class, coverage_note, pattern_note, color_note, color_confidence
    if canonical_product_type == "pet bed":
        low_detail_note = "the visible soft surface reads as a tonal textile field rather than a printed multicolor body"
    elif canonical_product_type in BEDDING_CANONICAL_TYPES:
        low_detail_note = "the visible bedding surface reads as a tonal textile field rather than a printed multicolor body"
    else:
        low_detail_note = "the visible soft surface reads as tonal textile variation rather than a printed multicolor body"
    tone_detail = "tonal variation from folds, pile, or soft-surface texture"
    if dominant_name == "black" and float(np.quantile(value, 0.5)) > 0.3:
        tone_detail = "low-luster tonal variation from folds, pile, or soft-surface texture"
    return (
        "low_variation_surface",
        low_detail_note,
        None,
        f"the main visible body reads as {dominant_name} with {tone_detail}",
        max(0.72, float(color_confidence or 0.0)),
    )


def correct_supported_soft_surface_inference(
    *,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    evidence_caption: str | None,
    coverage_class: str | None,
    coverage_ratio: float | None,
    coverage_note: str | None,
    pattern_note: str | None,
    color_note: str | None,
    color_confidence: float | None,
    palette: Sequence[str],
) -> tuple[str | None, float | None, str | None, str | None, str | None, float | None]:
    if category not in {"bedding", "pet home"} and canonical_product_type not in (BEDDING_CANONICAL_TYPES | {"pet bed"}):
        return coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence
    if coverage_class != "localized_visible_pattern":
        return coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence
    if coverage_ratio is not None and coverage_ratio > 0.46:
        return coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence

    text_tokens = set(
        _tokens(
            " ".join(
                part
                for part in (
                    product_title,
                    *hint_phrases,
                    evidence_caption or "",
                )
                if part
            )
        )
    )
    if text_tokens.intersection(EXPLICIT_PRINT_TOKENS):
        return coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence

    palette_colors = _dedupe_strings(palette)
    if not palette_colors:
        return coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence
    chromatic_colors = [
        color for color in palette_colors
        if color in {"blue", "green", "teal", "purple", "red", "pink", "yellow", "orange"}
    ]
    if len(chromatic_colors) >= 2:
        return coverage_class, coverage_ratio, coverage_note, pattern_note, color_note, color_confidence

    dominant_color = chromatic_colors[0] if chromatic_colors else palette_colors[0]
    if canonical_product_type in BEDDING_CANONICAL_TYPES or category == "bedding":
        revised_coverage_note = (
            "most of the visible bedding surface reads as a tonal textile field rather than a localized printed or contrast panel"
        )
        revised_color_note = (
            color_note
            if color_note and "printed accents" not in color_note and "black base" not in color_note
            else f"the main visible body reads as {dominant_color} with low-luster tonal variation from textured fabric"
        )
    else:
        revised_coverage_note = (
            "most of the visible pet resting surface reads as a tonal plush field rather than a localized printed or contrast panel"
        )
        revised_color_note = (
            color_note
            if color_note and "printed accents" not in color_note and "black base" not in color_note
            else f"the main visible body reads as {dominant_color}"
        )
    return (
        "low_variation_surface",
        0.0,
        revised_coverage_note,
        None,
        revised_color_note,
        max(0.72, float(color_confidence or 0.0)),
    )


def harmonize_supported_soft_structure(
    *,
    category: str,
    canonical_product_type: str,
    edge_thickness_class: str | None,
    structure_class: str | None,
    note: str | None,
    confidence: float | None,
) -> tuple[str | None, str | None, float | None]:
    if edge_thickness_class != "low_profile_edge" or structure_class != "raised_perimeter_relief":
        return structure_class, note, confidence
    if canonical_product_type in BEDDING_CANONICAL_TYPES or category == "bedding":
        return (
            "low_perimeter_relief",
            "the visible soft product structure shows moderate loft with softly lifted edges",
            0.62,
        )
    if canonical_product_type == "pet bed" or category == "pet home":
        return (
            "low_perimeter_relief",
            "the visible soft product structure shows only a modest perimeter rise around the resting surface",
            0.66,
        )
    if canonical_product_type == "decorative pillow" or category == "home decor":
        return (
            "low_perimeter_relief",
            "the visible soft product structure shows a modest padded edge around the face",
            0.6,
        )
    return structure_class, note, confidence


def infer_translucent_surface_note(
    source_image: Path,
    mask_path: Path | None,
    *,
    category: str,
    canonical_product_type: str,
    aspect_ratio: float | None,
    coverage_class: str | None,
) -> str | None:
    if (
        mask_path is None
        or not mask_path.exists()
        or not source_image.exists()
        or aspect_ratio is None
        or aspect_ratio < 1.6
        or coverage_class != "localized_visible_pattern"
    ):
        return None
    if category not in {"drinkware", "kitchen appliance"} and canonical_product_type not in (DRINKWARE_CANONICAL_TYPES | KITCHEN_APPLIANCE_CANONICAL_TYPES):
        return None
    with Image.open(source_image) as source_handle:
        source = np.asarray(source_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return None
    body_mask = _extract_core_body_mask(mask)
    if not body_mask.any():
        return None
    pixels = np.clip(source[body_mask].astype(np.float32) / 255.0, 0.0, 1.0)
    if len(pixels) < 64:
        return None
    stride = max(1, int(math.ceil(len(pixels) / 4096.0)))
    sampled = pixels[::stride]
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled], dtype=np.float32)
    if hsv.size == 0:
        return None
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    dark_ratio = float(np.mean(value <= 0.42))
    bright_reflection_ratio = float(np.mean((value >= 0.72) & (saturation <= 0.25)))
    neutral_ratio = float(np.mean(saturation <= 0.26))
    if dark_ratio >= 0.45 and bright_reflection_ratio >= 0.1 and neutral_ratio >= 0.18:
        return "the main visible body reads as a dark translucent or smoky neutral surface"
    return None


def infer_surface_scope(
    mask_path: Path | None,
    *,
    weak_shape_evidence: bool,
    stable_base: bool | None,
) -> str:
    if weak_shape_evidence:
        return "partial_or_ambiguous"
    if mask_path is None or not mask_path.exists():
        return "single_photo_limited"
    with Image.open(mask_path) as mask_handle:
        mask_array = np.asarray(mask_handle.convert("L")) > 0
    if not mask_array.any():
        return "single_photo_limited"
    touches_border = bool(mask_array[0].any() or mask_array[-1].any() or mask_array[:, 0].any() or mask_array[:, -1].any())
    if touches_border:
        return "partial_or_occluded"
    if stable_base is False:
        return "limited_surface_evidence"
    return "single_photo_limited"


def infer_evidence_uncertainty(
    *,
    surface_scope: str,
    weak_shape_evidence: bool,
    stable_base: bool | None,
) -> str:
    if weak_shape_evidence or surface_scope in {"partial_or_ambiguous", "partial_or_occluded"}:
        return "high"
    if stable_base is False or surface_scope == "limited_surface_evidence":
        return "medium"
    return "low"


def assess_source_validity(
    *,
    source_image: Path,
    mask_path: Path | None,
    category: str,
    canonical_product_type: str,
    observed_evidence: ObservedEvidenceSpec,
    weak_shape_evidence: bool,
    localization_confidence: float | None = None,
) -> tuple[str, float, list[str]]:
    score = 1.0
    issues: list[str] = []
    artifact_flags = set(observed_evidence.artifact_flags)
    if "border_text_overlay" in artifact_flags:
        issues.append("source_contains_border_text_overlay")
        score -= 0.12
    if "border_human_fragment" in artifact_flags:
        issues.append("source_contains_border_human_fragment")
        score -= 0.16
    if "border_foreground_intrusion" in artifact_flags:
        issues.append("source_contains_border_foreground_intrusion")
        score -= 0.14

    if mask_path is not None and mask_path.exists():
        with Image.open(source_image) as source_handle:
            source_shape = np.asarray(source_handle.convert("RGB")).shape[:2]
        mask = _load_mask_array(mask_path, source_shape=source_shape)
    else:
        mask = None

    if mask is not None and mask.any():
        structural_score = _structure_completeness_score(
            mask,
            category=category,
            canonical_product_type=canonical_product_type,
        )
    else:
        structural_score = 0.0

    aspect_ratio = observed_evidence.aspect_ratio or 0.0
    top_width_ratio = observed_evidence.top_width_ratio
    partial_scope = observed_evidence.surface_scope in {"partial_or_ambiguous", "partial_or_occluded"}

    if canonical_product_type == "blender":
        if structural_score < 0.58 or aspect_ratio < 1.12:
            issues.append("multipart_appliance_structure_incomplete")
            score -= 0.72
    elif canonical_product_type in {"coffee maker", "slow cooker", "food chopper"} or category == "kitchen appliance":
        if structural_score < 0.52:
            issues.append("multipart_appliance_structure_incomplete")
            score -= 0.58
    elif canonical_product_type == "backpack":
        if partial_scope and top_width_ratio is not None and top_width_ratio >= 0.78 and observed_evidence.lower_region_note is None:
            issues.append("portable_product_global_shape_incomplete")
            score -= 0.62
        elif weak_shape_evidence and observed_evidence.upper_region_note is None and observed_evidence.form_factor_note is None:
            issues.append("portable_product_global_shape_incomplete")
            score -= 0.52

    if observed_evidence.uncertainty_level == "high" and not issues:
        score -= 0.08

    raw_caption = (observed_evidence.raw_evidence_caption or "").strip().lower()
    sanitized_caption = (observed_evidence.evidence_caption or "").strip().lower()
    structured_caption_sensitive = category in {"drinkware", "kitchen appliance", "furniture", "home lighting"}
    if structured_caption_sensitive and raw_caption and not sanitized_caption:
        raw_tokens = set(_tokens(raw_caption))
        category_tokens = set(CATEGORY_CAPTION_HINTS.get(category, ()))
        canonical_tokens = set(_tokens(canonical_product_type))
        has_expected_support = bool(raw_tokens.intersection(category_tokens | canonical_tokens))
        competing_subobject_tokens = STRUCTURED_SUBOBJECT_COMPETING_TOKENS.get(category, frozenset())
        canonical_support_ratio = (
            len(raw_tokens.intersection(canonical_tokens)) / len(canonical_tokens)
            if canonical_tokens
            else 0.0
        )
        if (
            competing_subobject_tokens
            and raw_tokens.intersection(competing_subobject_tokens)
            and canonical_support_ratio < 0.75
        ):
            issues.append("localized_crop_visual_type_conflict")
            score -= 0.62
        if not has_expected_support and (
            (localization_confidence is not None and localization_confidence < 0.35)
            or structural_score < 0.58
            or weak_shape_evidence
        ):
            issues.append("localized_crop_visual_type_conflict")
            score -= 0.62
    soft_surface_caption_sensitive = (
        category in {"bedding", "pet home", "home decor"}
        and observed_evidence.coverage_class == "low_variation_surface"
    )
    soft_surface_foreground_conflict = (
        category in {"bedding", "pet home", "home decor"}
        and "border_foreground_intrusion" in artifact_flags
        and observed_evidence.coverage_class == "localized_visible_pattern"
    )
    if soft_surface_foreground_conflict:
        issues.append("localized_crop_visual_type_conflict")
        score -= 0.46
    if soft_surface_caption_sensitive and raw_caption and not sanitized_caption:
        raw_tokens = set(_tokens(raw_caption))
        category_tokens = set(CATEGORY_CAPTION_HINTS.get(category, ()))
        canonical_tokens = set(_tokens(canonical_product_type))
        has_expected_support = bool(raw_tokens.intersection(category_tokens | canonical_tokens))
        if (
            "border_foreground_intrusion" in artifact_flags
            and raw_tokens.intersection(PERSON_CAPTION_TOKENS | PERSON_ACCESSORY_TOKENS | ANIMAL_FRAGMENT_TOKENS)
            and not raw_tokens.intersection(canonical_tokens)
        ):
            issues.append("localized_crop_visual_type_conflict")
            score -= 0.42
        if not has_expected_support and "border_foreground_intrusion" in artifact_flags:
            issues.append("localized_crop_visual_type_conflict")
            score -= 0.42

    score = max(0.0, min(1.0, score))
    validity = "valid" if score >= 0.5 else "invalid"
    return validity, score, _dedupe_strings(issues)


def _lookup_business_prior_palette(
    palette: dict[tuple[str, str], tuple[str, ...]],
    *,
    scene_family: str,
    persona: str,
) -> tuple[str, ...]:
    return (
        palette.get((scene_family, persona))
        or palette.get((scene_family, "default"))
        or palette.get(("default", persona))
        or palette[("default", "default")]
    )


def _stable_signature_value(parts: Sequence[str]) -> int:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _pick_signature_option(options: Sequence[str], signature_value: int, *, offset: int = 0) -> str:
    if not options:
        raise ValueError("options must not be empty")
    return str(options[(signature_value + offset) % len(options)])


def build_business_prior_creative_metadata(
    identity: ProductIdentitySpec,
    top_matches: Sequence[RetrievalCandidate],
    *,
    scene_family: str,
    support_relation: str,
) -> dict[str, Any]:
    persona = identity.style_persona or "refined_neutral"
    signature_parts = [
        identity.category or "product",
        identity.canonical_product_type or "product",
        scene_family,
        support_relation,
        persona,
    ]
    for candidate in top_matches[:5]:
        signature_parts.extend(
            [
                candidate.item_id,
                candidate.image_name,
                candidate.caption,
                str(candidate.page_views),
                str(candidate.clicks),
            ]
        )
    signature_value = _stable_signature_value(signature_parts)
    lighting_options = _lookup_business_prior_palette(
        BUSINESS_PRIOR_LIGHTING_PALETTES,
        scene_family=scene_family,
        persona=persona,
    )
    direction_options = _lookup_business_prior_palette(
        BUSINESS_PRIOR_DIRECTION_PALETTES,
        scene_family=scene_family,
        persona=persona,
    )
    camera_options = (
        BUSINESS_PRIOR_CAMERA_PALETTES.get((support_relation, scene_family))
        or BUSINESS_PRIOR_CAMERA_PALETTES.get((support_relation, "default"))
        or BUSINESS_PRIOR_CAMERA_PALETTES[("default", "default")]
    )
    cast_options = BUSINESS_PRIOR_CAST_PALETTES.get(persona, ())
    if identity_prefers_compact_hand_focus(identity):
        direction_options = COMPACT_HAND_FOCUS_DIRECTION_PALETTES.get(
            persona,
            COMPACT_HAND_FOCUS_DIRECTION_PALETTES["default"],
        )
        camera_options = COMPACT_HAND_FOCUS_CAMERA_PALETTE
        cast_options = COMPACT_HAND_FOCUS_CAST_PALETTES.get(
            persona,
            COMPACT_HAND_FOCUS_CAST_PALETTES["default"],
        )
    if identity_has_chromatic_soft_textile_lock(identity):
        lighting_options = CHROMATIC_SOFT_TEXTILE_LIGHTING_PALETTE
        direction_options = CHROMATIC_SOFT_TEXTILE_DIRECTION_PALETTE
        if support_relation == "resting_on_surface":
            camera_options = CHROMATIC_SOFT_TEXTILE_CAMERA_PALETTE
    if identity_has_low_profile_soft_structure(identity):
        camera_options = LOW_PROFILE_SOFT_SURFACE_CAMERA_PALETTE
        direction_options = LOW_PROFILE_SOFT_SURFACE_DIRECTION_PALETTE
    metadata: dict[str, Any] = {
        "creative_seed": int(signature_value % 1_000_003),
        "lighting_hint": _pick_signature_option(lighting_options, signature_value, offset=1),
        "camera_hint": _pick_signature_option(camera_options, signature_value, offset=2),
        "creative_direction": _pick_signature_option(direction_options, signature_value, offset=3),
    }
    if cast_options and (
        identity.requires_human_model
        or identity.interaction_mode in {"worn", "worn_or_carried", "held_in_hand", "carried_or_resting"}
    ):
        metadata["cast_hint"] = _pick_signature_option(cast_options, signature_value, offset=4)
    return metadata


def build_business_prior(
    seed: ReviewSeedRecord,
    localized: LocalizedProduct,
    record: LocalizationArtifactRecord,
    retrieval_index: Sequence[RetrievalCandidate],
    backbone: VisionBackbone,
    *,
    top_k: int = 5,
) -> CampaignPriorSpec:
    query_path = record.crop_path or record.local_image_path
    query_embedding = np.asarray(backbone.encode_image(query_path), dtype=np.float32)
    canonical_product_type = localized.identity.canonical_product_type or localized.identity.category
    category = localized.identity.category
    support_mode = localized.identity.support_mode or CATEGORY_SUPPORT_DEFAULTS["product"][0]
    evidence = localized.identity.observed_evidence
    query_tokens = set(_tokens(f"{seed.product_title} {' '.join(seed.hint_phrases)} {canonical_product_type}"))
    canonical_type_tokens = set(_tokens(canonical_product_type))
    filtered_candidates = filter_retrieval_candidates(
        seed,
        record,
        retrieval_index,
        category=category,
        canonical_product_type=canonical_product_type,
        support_mode=support_mode,
        query_embedding=query_embedding,
        evidence=evidence,
    )
    if not filtered_candidates:
        relaxed_scene_candidates = filter_scene_retrieval_candidates(
            seed,
            record,
            retrieval_index,
            category=category,
            canonical_product_type=canonical_product_type,
            support_mode=support_mode,
            query_embedding=query_embedding,
            evidence=evidence,
        )
        if relaxed_scene_candidates:
            return build_scene_retrieval_fallback_prior(
                localized.identity,
                relaxed_scene_candidates[:top_k],
                category=category,
                canonical_product_type=canonical_product_type,
                support_mode=support_mode,
                fallback_source="relaxed_scene_retrieval",
            )
        return build_category_fallback_prior(
            category,
            canonical_product_type,
            support_mode=support_mode,
            default_scene_family=localized.identity.default_scene_family,
            identity=localized.identity,
            style_persona=localized.identity.style_persona,
            interaction_mode=localized.identity.interaction_mode,
            requires_human_model=localized.identity.requires_human_model,
        )
    scored: list[tuple[float, RetrievalCandidate, dict[str, float | bool]]] = []
    compatibility_by_key: dict[tuple[str, str], float] = {}
    hybrid_by_key: dict[tuple[str, str], float] = {}
    image_similarity_by_key: dict[tuple[str, str], float] = {}
    for candidate in filtered_candidates:
        metrics = evaluate_retrieval_candidate(
            candidate,
            query_embedding=query_embedding,
            query_tokens=query_tokens,
            evidence=evidence,
            category=category,
            canonical_product_type=canonical_product_type,
            canonical_type_tokens=canonical_type_tokens,
        )
        compatibility_by_key[(candidate.item_id, candidate.image_name)] = float(metrics["combined_compatibility"])
        hybrid_by_key[(candidate.item_id, candidate.image_name)] = float(metrics["final_score"])
        image_similarity_by_key[(candidate.item_id, candidate.image_name)] = float(
            metrics["effective_image_similarity"]
        )
        scored.append((float(metrics["final_score"]), candidate, metrics))
    ranked_matches = sorted(
        scored,
        key=lambda item: (
            item[0],
            compatibility_by_key.get((item[1].item_id, item[1].image_name), 0.0),
            image_similarity_by_key.get((item[1].item_id, item[1].image_name), 0.0),
            item[1].page_views,
            item[1].clicks,
            item[1].item_id,
            item[1].image_name,
        ),
        reverse=True,
    )[:top_k]
    top_matches = [candidate for _, candidate, _ in ranked_matches]
    planning_matches = sanitize_retrieval_candidates_for_planning(top_matches)
    top_compatibility_scores = [
        compatibility_by_key.get((candidate.item_id, candidate.image_name), 0.0)
        for candidate in top_matches
    ]
    top_hybrid_scores = [
        hybrid_by_key.get((candidate.item_id, candidate.image_name), 0.0)
        for candidate in top_matches
    ]
    top_image_similarities = [
        image_similarity_by_key.get((candidate.item_id, candidate.image_name), 0.0)
        for candidate in top_matches
    ]
    evidence_sensitive = bool(
        evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
        or evidence.upper_region_note
        or evidence.upper_component_state == "absent"
        or evidence.material_note
        or "distinct_boundary_trim" in evidence.evidence_tags
    )
    if (
        evidence_sensitive
        and top_matches
        and not any(candidate_has_visual_evidence(candidate) for candidate in top_matches)
    ) or should_fallback_to_category_prior(
        evidence,
        top_compatibility_scores,
        retrieval_scores=top_hybrid_scores,
        image_similarities=top_image_similarities,
    ):
        relaxed_scene_candidates = filter_scene_retrieval_candidates(
            seed,
            record,
            retrieval_index,
            category=category,
            canonical_product_type=canonical_product_type,
            support_mode=support_mode,
            query_embedding=query_embedding,
            evidence=evidence,
        )
        if top_matches and any(candidate_has_visual_evidence(candidate) for candidate in top_matches):
            fallback = build_scene_retrieval_fallback_prior(
                localized.identity,
                top_matches,
                category=category,
                canonical_product_type=canonical_product_type,
                support_mode=support_mode,
                fallback_source="top_match_scene_retrieval",
            )
        elif relaxed_scene_candidates:
            fallback = build_scene_retrieval_fallback_prior(
                localized.identity,
                relaxed_scene_candidates[:top_k],
                category=category,
                canonical_product_type=canonical_product_type,
                support_mode=support_mode,
                fallback_source="relaxed_scene_retrieval",
            )
        else:
            fallback = build_category_fallback_prior(
                category,
                canonical_product_type,
                support_mode=support_mode,
                default_scene_family=localized.identity.default_scene_family,
                identity=localized.identity,
                style_persona=localized.identity.style_persona,
                interaction_mode=localized.identity.interaction_mode,
                requires_human_model=localized.identity.requires_human_model,
            )
        fallback.metadata.update(
            {
                "retrieval_mode": str(fallback.metadata.get("retrieval_mode", "evidence_fallback")),
                "compatibility_scores": [round(float(value), 4) for value in top_compatibility_scores],
                "hybrid_scores": [round(float(value), 4) for value in top_hybrid_scores],
                "image_similarities": [round(float(value), 4) for value in top_image_similarities],
            }
        )
        return fallback
    support_relation = choose_support_relation(localized.identity, planning_matches)
    scene_family = choose_scene_family(localized.identity, planning_matches, support_relation=support_relation)
    style_atoms = build_style_plan(
        localized.identity,
        planning_matches,
        scene_family=scene_family,
        support_relation=support_relation,
    )
    creative_metadata = build_business_prior_creative_metadata(
        localized.identity,
        planning_matches,
        scene_family=scene_family,
        support_relation=support_relation,
    )
    scenario_slots = [scene_family] if scene_family else ["editorial_interior"]
    semantic_constraints = build_semantic_constraints(
        localized.identity,
        scene_family=scene_family,
        support_relation=support_relation,
    )

    return CampaignPriorSpec(
        neighbor_item_ids=[candidate.item_id for candidate in top_matches],
        style_atoms=list(style_atoms),
        scenario_slots=list(scenario_slots),
        scene_family=scene_family,
        support_relation=support_relation,
        semantic_constraints=semantic_constraints,
        banned_identity_edits=[
            "do not copy any retrieved brand marks or logos",
            "do not replace the featured product with a retrieved neighbor product",
            "use retrieval cues to influence styling and scene only, not the invention of unseen product structure",
        ],
        metadata={
            "retrieved_captions": [candidate.caption for candidate in planning_matches if candidate.caption],
            "retrieved_images": [str(candidate.image_path) for candidate in top_matches],
            "retrieval_mode": "retrieval",
            "compatibility_scores": [round(float(value), 4) for value in top_compatibility_scores],
            "hybrid_scores": [round(float(value), 4) for value in top_hybrid_scores],
            "image_similarities": [round(float(value), 4) for value in top_image_similarities],
            "scene_family": scene_family,
            "support_relation": support_relation,
            **creative_metadata,
        },
    )


def filter_retrieval_candidates(
    seed: ReviewSeedRecord,
    record: LocalizationArtifactRecord,
    retrieval_index: Sequence[RetrievalCandidate],
    *,
    category: str,
    canonical_product_type: str,
    support_mode: str,
    query_embedding: np.ndarray | None = None,
    evidence: ObservedEvidenceSpec | None = None,
) -> list[RetrievalCandidate]:
    query_tokens = set(_tokens(f"{seed.product_title} {' '.join(seed.hint_phrases)} {record.selected_phrase or ''}"))
    canonical_type_tokens = set(_tokens(canonical_product_type))
    allowed_support_relations = set(
        allowed_support_relations_for_identity(
            ProductIdentitySpec(
                category=category,
                canonical_product_type=canonical_product_type,
                support_mode=support_mode,
            )
        )
    )
    strict_structural_retrieval = canonical_product_type in (STRUCTURED_DISPLAY_CANONICAL_TYPES | BEDDING_CANONICAL_TYPES | {"pet bed"})
    scored: list[tuple[float, RetrievalCandidate]] = []
    for candidate in retrieval_index:
        if (
            allowed_support_relations
            and candidate.support_relations
            and not set(candidate.support_relations).intersection(allowed_support_relations)
        ):
            continue
        if query_embedding is not None and evidence is not None:
            metrics = evaluate_retrieval_candidate(
                candidate,
                query_embedding=query_embedding,
                query_tokens=query_tokens,
                evidence=evidence,
                category=category,
                canonical_product_type=canonical_product_type,
                canonical_type_tokens=canonical_type_tokens,
            )
            if metrics["hard_conflict"]:
                continue
            if strict_structural_retrieval:
                if not candidate_has_visual_evidence(candidate):
                    continue
                if (
                    float(metrics["visual_compatibility"]) < 0.12
                    or float(metrics["combined_compatibility"]) < 0.14
                ):
                    continue
            if (
                float(metrics["final_score"]) > 0.08
                or float(metrics["effective_image_similarity"]) > 0.18
                or float(metrics["visual_compatibility"]) > 0.08
            ):
                scored.append((float(metrics["final_score"]), candidate))
            continue
        score = score_retrieval_candidate(
            candidate.caption,
            query_tokens=query_tokens,
            category=category,
            canonical_type_tokens=canonical_type_tokens,
        )
        if score > 0:
            scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored]


def filter_scene_retrieval_candidates(
    seed: ReviewSeedRecord,
    record: LocalizationArtifactRecord,
    retrieval_index: Sequence[RetrievalCandidate],
    *,
    category: str,
    canonical_product_type: str,
    support_mode: str,
    query_embedding: np.ndarray,
    evidence: ObservedEvidenceSpec,
) -> list[RetrievalCandidate]:
    query_tokens = set(_tokens(f"{seed.product_title} {' '.join(seed.hint_phrases)} {record.selected_phrase or ''}"))
    canonical_type_tokens = set(_tokens(canonical_product_type))
    allowed_support_relations = set(
        allowed_support_relations_for_identity(
            ProductIdentitySpec(
                category=category,
                canonical_product_type=canonical_product_type,
                support_mode=support_mode,
            )
        )
    )
    strict_structural_retrieval = canonical_product_type in (STRUCTURED_DISPLAY_CANONICAL_TYPES | BEDDING_CANONICAL_TYPES | {"pet bed"})
    scored: list[tuple[float, RetrievalCandidate]] = []
    for candidate in retrieval_index:
        if candidate.category not in {category, "product"}:
            continue
        if (
            allowed_support_relations
            and candidate.support_relations
            and not set(candidate.support_relations).intersection(allowed_support_relations)
        ):
            continue
        metrics = evaluate_retrieval_candidate(
            candidate,
            query_embedding=query_embedding,
            query_tokens=query_tokens,
            evidence=evidence,
            category=category,
            canonical_product_type=canonical_product_type,
            canonical_type_tokens=canonical_type_tokens,
        )
        if metrics["hard_conflict"]:
            continue
        effective_image_similarity = float(metrics["effective_image_similarity"])
        visual_compatibility = float(metrics["visual_compatibility"])
        text_compatibility = float(metrics["text_compatibility"])
        has_visual_evidence = candidate_has_visual_evidence(candidate)
        if strict_structural_retrieval and not has_visual_evidence:
            continue
        if has_visual_evidence:
            if effective_image_similarity < 0.12 and visual_compatibility < 0.04:
                continue
        else:
            if effective_image_similarity < 0.18 or visual_compatibility < -0.02:
                continue
        scene_score = (
            effective_image_similarity
            + 0.32 * max(visual_compatibility, 0.0)
            + 0.08 * max(text_compatibility, 0.0)
            + (0.05 if has_visual_evidence else 0.0)
        )
        scored.append((scene_score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored]


def build_category_fallback_prior(
    category: str,
    canonical_product_type: str,
    *,
    support_mode: str,
    default_scene_family: str | None,
    identity: ProductIdentitySpec | None = None,
    style_persona: str | None = None,
    interaction_mode: str | None = None,
    requires_human_model: bool = False,
) -> CampaignPriorSpec:
    style_atoms = ["clear hero framing"]
    fallback_identity = (
        identity.model_copy(
            update={
                "category": category,
                "canonical_product_type": canonical_product_type,
                "support_mode": support_mode,
                "default_scene_family": default_scene_family,
                "style_persona": style_persona if style_persona is not None else identity.style_persona,
                "interaction_mode": interaction_mode if interaction_mode is not None else identity.interaction_mode,
                "requires_human_model": requires_human_model or identity.requires_human_model,
            }
        )
        if identity is not None
        else ProductIdentitySpec(
            category=category,
            canonical_product_type=canonical_product_type,
            support_mode=support_mode,
            default_scene_family=default_scene_family,
            style_persona=style_persona,
            interaction_mode=interaction_mode,
            requires_human_model=requires_human_model,
        )
    )
    support_relation = default_support_relation_for_identity(fallback_identity)
    scene_family = default_scene_family or SCENE_FAMILY_DEFAULTS_BY_SUPPORT.get(support_relation, "editorial_interior")
    scenario_slots = [scene_family]
    if canonical_product_type:
        style_atoms.append(f"keep the featured {canonical_product_type} unmistakable")
    style_atoms.extend(SUPPORT_RELATION_STYLE_ATOMS.get(support_relation, ()))
    style_atoms.extend(SCENE_FAMILY_STYLE_ATOMS.get(scene_family, ()))
    creative_metadata = build_business_prior_creative_metadata(
        fallback_identity.model_copy(update={"default_scene_family": scene_family}),
        [],
        scene_family=scene_family,
        support_relation=support_relation,
    )
    return CampaignPriorSpec(
        neighbor_item_ids=[],
        style_atoms=_dedupe_strings(style_atoms),
        scenario_slots=_dedupe_strings(scenario_slots),
        scene_family=scene_family,
        support_relation=support_relation,
        semantic_constraints=build_semantic_constraints(
            fallback_identity.model_copy(update={"default_scene_family": scene_family}),
            scene_family=scene_family,
            support_relation=support_relation,
        ),
        banned_identity_edits=[
            "do not copy any retrieved brand marks or logos",
            "do not replace the featured product with a retrieved neighbor product",
            "use retrieval cues to influence styling and scene only, not the invention of unseen product structure",
        ],
        metadata={
            "retrieved_captions": [],
            "retrieved_images": [],
            "retrieval_mode": "category_fallback",
            "fallback_category": category,
            "scene_family": scene_family,
            "support_relation": support_relation,
            **creative_metadata,
        },
    )


def build_scene_retrieval_fallback_prior(
    identity: ProductIdentitySpec,
    top_matches: Sequence[RetrievalCandidate],
    *,
    category: str,
    canonical_product_type: str,
    support_mode: str,
    fallback_source: str | None = None,
) -> CampaignPriorSpec:
    planning_matches = sanitize_retrieval_candidates_for_planning(top_matches)
    support_relation = choose_support_relation(identity, planning_matches)
    scene_family = choose_scene_family(identity, planning_matches, support_relation=support_relation)
    creative_metadata = build_business_prior_creative_metadata(
        identity.model_copy(update={"default_scene_family": scene_family}),
        planning_matches,
        scene_family=scene_family,
        support_relation=support_relation,
    )
    style_atoms = _dedupe_strings(
        [
            "clear hero framing",
            *SUPPORT_RELATION_STYLE_ATOMS.get(support_relation, ()),
            *SCENE_FAMILY_STYLE_ATOMS.get(scene_family, ()),
            f"keep the featured {canonical_product_type} unmistakable" if canonical_product_type else "",
        ]
    )
    metadata = {
        "retrieved_captions": [candidate.caption for candidate in planning_matches if candidate.caption],
        "retrieved_images": [str(candidate.image_path) for candidate in top_matches],
        "retrieval_mode": "scene_retrieval_fallback",
        "scene_family": scene_family,
        "support_relation": support_relation,
        **creative_metadata,
    }
    if fallback_source:
        metadata["scene_fallback_source"] = fallback_source
    return CampaignPriorSpec(
        neighbor_item_ids=[candidate.item_id for candidate in top_matches],
        style_atoms=style_atoms,
        scenario_slots=[scene_family] if scene_family else ["editorial_interior"],
        scene_family=scene_family,
        support_relation=support_relation,
        semantic_constraints=build_semantic_constraints(
            identity.model_copy(update={"default_scene_family": scene_family}),
            scene_family=scene_family,
            support_relation=support_relation,
        ),
        banned_identity_edits=[
            "do not copy any retrieved brand marks or logos",
            "do not replace the featured product with a retrieved neighbor product",
            "use retrieval cues to influence styling and scene only, not the invention of unseen product structure",
        ],
        metadata=metadata,
    )


def score_retrieval_candidate(
    caption: str,
    *,
    query_tokens: set[str],
    category: str,
    canonical_type_tokens: set[str],
) -> float:
    caption_tokens = set(_tokens(caption))
    positive_tokens = set(CATEGORY_CAPTION_HINTS.get(category, ()))
    negative_tokens = set(CATEGORY_NEGATIVE_HINTS.get(category, ()))
    negative_hits = caption_tokens.intersection(negative_tokens)
    noise_hits = caption_tokens.intersection(MULTI_OBJECT_NOISE_TOKENS)
    if category == "drinkware" and ({"sauce", "scissors", "brush", "comb"} & caption_tokens):
        return -1.0
    if category in {"drinkware", "bag", "home decor"} and len(negative_hits) >= 2:
        return -1.0
    score = 0.0
    score += 3.0 * len(caption_tokens.intersection(positive_tokens))
    score += 2.0 * len(caption_tokens.intersection(canonical_type_tokens))
    score += 1.0 * len(caption_tokens.intersection(query_tokens))
    score -= 4.0 * len(negative_hits)
    score -= 2.0 * len(noise_hits)
    return score


def _coverage_classes_are_compatible(
    source_coverage: str | None,
    candidate_coverage: str | None,
) -> bool:
    if source_coverage is None or candidate_coverage is None:
        return True
    if source_coverage == candidate_coverage:
        return True
    broad_classes = {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
    if source_coverage in broad_classes and candidate_coverage in broad_classes:
        return True
    return False


def score_candidate_visual_evidence_compatibility(
    candidate: RetrievalCandidate,
    *,
    source_evidence: ObservedEvidenceSpec,
    category: str,
    canonical_type_tokens: set[str],
) -> float:
    candidate_evidence = candidate.observed_evidence
    score = 0.0

    if candidate.category == category:
        score += 0.35
    elif candidate.category != "product":
        score -= 0.4

    candidate_type_tokens = set(_tokens(candidate.canonical_product_type))
    if candidate_type_tokens.intersection(canonical_type_tokens):
        score += 0.28
    elif candidate.canonical_product_type and canonical_type_tokens:
        score -= 0.18

    source_colors = set(source_evidence.palette[:3] + source_evidence.structural_palette[:2])
    candidate_colors = set(candidate_evidence.palette[:3] + candidate_evidence.structural_palette[:2])
    if source_colors and candidate_colors:
        overlap = len(source_colors.intersection(candidate_colors))
        if overlap:
            score += min(0.12 * overlap, 0.3)
        elif source_evidence.color_confidence is not None and source_evidence.color_confidence >= 0.72:
            score -= 0.22

    if source_evidence.coverage_class:
        if _coverage_classes_are_compatible(source_evidence.coverage_class, candidate_evidence.coverage_class):
            if source_evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
                score += 0.32
            elif source_evidence.coverage_class == "localized_visible_pattern":
                score += 0.18
        elif candidate_evidence.coverage_class is not None:
            score -= 0.32

    if source_evidence.upper_component_state == "absent":
        if candidate_evidence.upper_component_state == "absent":
            score += 0.22
        elif candidate_evidence.upper_component_state == "present" or candidate_evidence.upper_region_note:
            score -= 0.4
    elif source_evidence.upper_region_note or source_evidence.upper_component_state == "present":
        if candidate_evidence.upper_region_note or candidate_evidence.upper_component_state == "present":
            score += 0.18
        elif category in {"bag", "drinkware"}:
            score -= 0.18

    if source_evidence.material_note and candidate_evidence.material_note:
        source_material_tokens = extract_material_tags(source_evidence.material_note)
        candidate_material_tokens = extract_material_tags(candidate_evidence.material_note)
        if source_material_tokens.intersection(candidate_material_tokens):
            score += 0.18
        elif source_material_tokens and candidate_material_tokens:
            score -= 0.2

    if source_evidence.aspect_ratio is not None and candidate_evidence.aspect_ratio is not None:
        aspect_gap = abs(source_evidence.aspect_ratio - candidate_evidence.aspect_ratio)
        if aspect_gap <= 0.28:
            score += 0.16
        elif aspect_gap >= 0.9:
            score -= 0.16

    if source_evidence.top_width_ratio is not None and candidate_evidence.top_width_ratio is not None:
        top_gap = abs(source_evidence.top_width_ratio - candidate_evidence.top_width_ratio)
        if top_gap <= 0.12:
            score += 0.08
        elif top_gap >= 0.3:
            score -= 0.08

    return score


def candidate_has_visual_evidence(candidate: RetrievalCandidate) -> bool:
    evidence = candidate.observed_evidence
    return bool(
        evidence.palette
        or evidence.structural_palette
        or evidence.coverage_class
        or evidence.upper_region_note
        or evidence.upper_component_state
        or evidence.material_note
        or evidence.hard_facts
    )


def candidate_has_hard_evidence_conflict(
    candidate: RetrievalCandidate,
    *,
    source_evidence: ObservedEvidenceSpec,
    category: str,
    source_canonical_product_type: str | None = None,
) -> bool:
    candidate_evidence = candidate.observed_evidence
    has_visual_evidence = candidate_has_visual_evidence(candidate)
    if candidate.category not in {category, "product"}:
        return True

    source_type = "" if source_canonical_product_type is None else str(source_canonical_product_type).strip().lower()
    candidate_type = str(candidate.canonical_product_type or "").strip().lower()
    structured_types = {
        "backpack",
        "wallet",
        "shoe",
        "pet bed",
        *STRUCTURED_DISPLAY_CANONICAL_TYPES,
        *BEDDING_CANONICAL_TYPES,
    }
    if source_type in structured_types and candidate_type and candidate_type != source_type:
        return True
    if source_type in structured_types and not candidate_type:
        source_type_tokens = set(_tokens(source_type))
        caption_tokens = set(_tokens(candidate.caption))
        if source_type_tokens and not source_type_tokens.intersection(caption_tokens):
            return True
    if source_type in structured_types and candidate.caption:
        source_type_tokens = set(_tokens(source_type))
        category_tokens = set(CATEGORY_CAPTION_HINTS.get(category, ()))
        negative_tokens = set(CATEGORY_NEGATIVE_HINTS.get(category, ()))
        competing_tokens = COMPETING_CATEGORY_TYPE_TOKENS.get(category, frozenset()) - source_type_tokens
        caption_tokens = set(_tokens(candidate.caption))
        if caption_tokens.intersection(negative_tokens | competing_tokens):
            return True
        if (
            caption_tokens
            and not is_low_information_retrieval_caption(candidate.caption)
            and not caption_tokens.intersection(source_type_tokens | category_tokens)
        ):
            return True
    if (
        (source_type in structured_types or category in {"furniture", "kitchen appliance", "home lighting"})
        and not has_visual_evidence
    ):
        source_type_tokens = set(_tokens(source_type))
        caption_tokens = set(_tokens(candidate.caption))
        if source_type_tokens and not source_type_tokens.intersection(caption_tokens):
            return True

    if (
        category in {"bag", "drinkware"}
        and source_evidence.upper_component_state == "absent"
        and (candidate_evidence.upper_component_state == "present" or candidate_evidence.upper_region_note)
    ):
        return True

    if (
        category == "drinkware"
        and source_evidence.coverage_class == "localized_visible_pattern"
        and candidate_evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
    ):
        return True

    dominant_source_color = extract_dominant_body_color(source_evidence)
    candidate_caption_colors = set(extract_caption_colors(candidate.caption))
    if (
        dominant_source_color
        and source_evidence.color_confidence is not None
        and source_evidence.color_confidence >= 0.68
        and candidate_caption_colors
    ):
        expected_colors = {dominant_source_color, *source_evidence.palette[:3]}
        if not expected_colors.intersection(candidate_caption_colors):
            compact_direct_grip = bool(
                source_evidence.upper_component_state == "absent"
                and source_evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
                and "compact" in str(source_evidence.form_factor_note or "").lower()
            )
            if compact_direct_grip or category in {"furniture", "drinkware"}:
                return True

    return False


def score_retrieval_evidence_compatibility(
    caption: str,
    *,
    evidence: ObservedEvidenceSpec,
    canonical_type_tokens: set[str],
    category: str,
    candidate_evidence: ObservedEvidenceSpec | None = None,
    candidate_category: str | None = None,
    candidate_canonical_product_type: str | None = None,
) -> float:
    caption_tokens = set(_tokens(caption))
    score = 0.0
    if candidate_category is not None:
        if candidate_category == category:
            score += 0.12
        elif candidate_category != "product":
            score -= 0.16
    if candidate_canonical_product_type:
        candidate_type_tokens = set(_tokens(candidate_canonical_product_type))
        if candidate_type_tokens.intersection(canonical_type_tokens):
            score += 0.18
        elif candidate_type_tokens:
            score -= 0.14
    caption_colors = caption_tokens.intersection(EVIDENCE_COLOR_SWATCHES)
    expected_colors = set(evidence.palette[:3])
    pattern_tokens = caption_tokens.intersection(PATTERN_TEXT_TOKENS)
    material_tokens = extract_material_tags(caption)
    evidence_material_tokens = extract_material_tags(
        " ".join(part for part in [evidence.material_note or "", evidence.evidence_caption or ""] if part)
    )

    if caption_tokens.intersection(canonical_type_tokens):
        score += 0.35
    elif evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        score -= 0.25

    if expected_colors and caption_colors:
        overlap = len(expected_colors & caption_colors)
        if overlap:
            score += 0.18 * overlap
        elif len(caption_colors) >= 2 or not pattern_tokens:
            score -= 0.45

    if evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        if pattern_tokens:
            score += 0.55
        elif caption_colors and len(caption_colors) <= 2:
            score -= 0.55
        elif not caption_tokens.intersection(canonical_type_tokens):
            score -= 0.3
    elif evidence.coverage_class == "localized_visible_pattern" and pattern_tokens:
        score += 0.2

    if evidence_material_tokens and material_tokens:
        if evidence_material_tokens.intersection(material_tokens):
            score += 0.3
        else:
            score -= 0.35

    if evidence.upper_region_note:
        if caption_tokens.intersection(UPPER_COMPONENT_TEXT_TOKENS):
            score += 0.15
        elif category in {"bag", "drinkware"} and not caption_tokens.intersection(canonical_type_tokens):
            score -= 0.15
    elif evidence.upper_component_state == "absent" and caption_tokens.intersection(UPPER_COMPONENT_TEXT_TOKENS):
        score -= 0.3

    if "no_distinct_upper_component" in evidence.evidence_tags and caption_tokens.intersection({"handle", "handles", "strap", "straps"}):
        score -= 0.35

    if "distinct_boundary_trim" in evidence.evidence_tags and len(caption_colors) >= 2 and expected_colors and not (
        expected_colors & caption_colors
    ):
        score -= 0.2

    if candidate_evidence is not None:
        if evidence.coverage_class and candidate_evidence.coverage_class:
            if _coverage_classes_are_compatible(evidence.coverage_class, candidate_evidence.coverage_class):
                score += 0.16
            else:
                score -= 0.18
        if evidence.upper_component_state == "absent" and candidate_evidence.upper_component_state == "absent":
            score += 0.12
        elif (
            evidence.upper_component_state == "absent"
            and (candidate_evidence.upper_component_state == "present" or candidate_evidence.upper_region_note)
        ):
            score -= 0.22

    return score


def evaluate_retrieval_candidate(
    candidate: RetrievalCandidate,
    *,
    query_embedding: np.ndarray,
    query_tokens: set[str],
    evidence: ObservedEvidenceSpec,
    category: str,
    canonical_product_type: str,
    canonical_type_tokens: set[str],
) -> dict[str, float | bool]:
    candidate_embedding = np.asarray(candidate.embedding, dtype=np.float32)
    image_similarity = float(np.dot(query_embedding, candidate_embedding))
    has_visual_evidence = candidate_has_visual_evidence(candidate)
    caption_score = score_retrieval_candidate(
        candidate.caption,
        query_tokens=query_tokens,
        category=category,
        canonical_type_tokens=canonical_type_tokens,
    )
    visual_compatibility = score_candidate_visual_evidence_compatibility(
        candidate,
        source_evidence=evidence,
        category=category,
        canonical_type_tokens=canonical_type_tokens,
    )
    text_compatibility = score_retrieval_evidence_compatibility(
        candidate.caption,
        evidence=evidence,
        canonical_type_tokens=canonical_type_tokens,
        category=category,
        candidate_evidence=candidate.observed_evidence,
        candidate_category=candidate.category,
        candidate_canonical_product_type=candidate.canonical_product_type,
    )
    hard_conflict = candidate_has_hard_evidence_conflict(
        candidate,
        source_evidence=evidence,
        category=category,
        source_canonical_product_type=canonical_product_type,
    )
    evidence_sensitive = bool(
        evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
        or evidence.upper_region_note
        or evidence.upper_component_state == "absent"
        or evidence.material_note
        or "distinct_boundary_trim" in evidence.evidence_tags
    )
    if not has_visual_evidence:
        visual_compatibility -= 0.22 if evidence_sensitive else 0.08
        text_compatibility -= 0.12 if evidence_sensitive else 0.04
        if category in {"furniture", "kitchen appliance", "home lighting"}:
            visual_compatibility -= 0.28
            text_compatibility -= 0.12
    effective_image_similarity = image_similarity
    if not has_visual_evidence:
        effective_image_similarity *= 0.35 if evidence_sensitive else 0.65
    ctr_bonus = 0.04 * math.log1p(max(candidate.page_views, 0)) + 0.03 * math.log1p(max(candidate.clicks, 0))
    final_score = (
        effective_image_similarity
        + 0.45 * visual_compatibility
        + 0.2 * text_compatibility
        + 0.02 * max(caption_score, -1.0)
        + ctr_bonus
    )
    return {
        "image_similarity": image_similarity,
        "effective_image_similarity": float(effective_image_similarity),
        "caption_score": float(caption_score),
        "visual_compatibility": float(visual_compatibility),
        "text_compatibility": float(text_compatibility),
        "combined_compatibility": float(visual_compatibility + 0.6 * text_compatibility),
        "final_score": float(final_score),
        "hard_conflict": hard_conflict,
    }


def should_fallback_to_category_prior(
    evidence: ObservedEvidenceSpec,
    compatibility_scores: Sequence[float],
    *,
    retrieval_scores: Sequence[float] | None = None,
    image_similarities: Sequence[float] | None = None,
) -> bool:
    if not compatibility_scores:
        return True
    mean_score = sum(compatibility_scores) / float(len(compatibility_scores))
    best_score = max(compatibility_scores)
    best_retrieval = None if not retrieval_scores else max(retrieval_scores)
    best_similarity = None if not image_similarities else max(image_similarities)
    compact_direct_grip = bool(
        evidence.upper_component_state == "absent"
        and evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
        and "compact" in str(evidence.form_factor_note or "").lower()
    )
    evidence_sensitive = bool(
        evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
        or evidence.upper_region_note
        or evidence.upper_component_state == "absent"
        or evidence.material_note
        or "distinct_boundary_trim" in evidence.evidence_tags
    )
    if compact_direct_grip:
        retrieval_ok = best_retrieval is not None and best_retrieval >= 0.28
        visual_ok = best_similarity is not None and best_similarity >= 0.45
        compatibility_ok = best_score >= 0.32 and mean_score >= 0.18
        return not (retrieval_ok and visual_ok and compatibility_ok)
    if best_retrieval is not None or best_similarity is not None:
        retrieval_ok = best_retrieval is not None and best_retrieval >= 0.22
        visual_ok = best_similarity is not None and best_similarity >= 0.22
        if evidence_sensitive:
            return not retrieval_ok and not visual_ok and best_score < -0.08 and mean_score < -0.06
        return not retrieval_ok and not visual_ok and best_score < -0.18 and mean_score < -0.14
    if evidence_sensitive:
        return best_score < 0.1 or mean_score < -0.02
    return best_score < -0.15 and mean_score < -0.2


def identity_prefers_compact_hand_focus(identity: ProductIdentitySpec) -> bool:
    if identity.interaction_mode != "held_in_hand":
        return False
    evidence = identity.observed_evidence
    canonical_type = str(identity.canonical_product_type or identity.category or "").strip().lower()
    if canonical_type in {"wallet", "clutch", "wristlet", "card holder"}:
        return True
    observed_notes = " ".join(
        filter(
            None,
            [
                evidence.form_factor_note,
                evidence.silhouette_note,
                *evidence.hard_facts,
            ],
        )
    ).lower()
    if "no visible handles" in observed_notes or "no visible straps" in observed_notes:
        return True
    if "compact" in observed_notes or "hand-held" in observed_notes or "hand held" in observed_notes:
        return True
    aspect_ratio = evidence.aspect_ratio
    if aspect_ratio is not None and aspect_ratio <= 1.7:
        return True
    return evidence.upper_component_state == "absent"


def identity_has_chromatic_soft_textile_lock(identity: ProductIdentitySpec) -> bool:
    evidence = identity.observed_evidence
    if identity.category not in {"bedding", "pet home", "home decor", "apparel"} and (
        identity.canonical_product_type or ""
    ) not in {*BEDDING_CANONICAL_TYPES, "pet bed", "decorative pillow", "shirt", "dress"}:
        return False
    dominant = extract_dominant_body_color(evidence)
    if dominant not in {"green", "blue", "teal", "purple", "red", "pink", "yellow", "orange"}:
        return False
    if evidence.color_confidence is not None and evidence.color_confidence < 0.68:
        return False
    return evidence.coverage_class in {
        "low_variation_surface",
        "localized_visible_pattern",
        "full_visible_surface_pattern",
        "broad_visible_surface_pattern",
    }


def identity_has_low_profile_soft_structure(identity: ProductIdentitySpec) -> bool:
    evidence = identity.observed_evidence
    canonical_type = str(identity.canonical_product_type or "").strip().lower()
    if identity.rigid_vs_soft != "soft" and identity.category not in {"bedding", "pet home", "home decor"}:
        return False
    if evidence.soft_structure_class in {"flat_surface", "low_perimeter_relief"}:
        return True
    if canonical_type == "pet bed" and evidence.edge_thickness_class in {"thin", "medium"}:
        return True
    return False


def identity_requires_people_out_of_frame(identity: ProductIdentitySpec) -> bool:
    canonical_type = str(identity.canonical_product_type or "").strip().lower()
    return bool(
        not identity.requires_human_model
        and identity.interaction_mode == "placed"
        and (
            canonical_type
            in (
                KITCHEN_APPLIANCE_CANONICAL_TYPES
                | BEDDING_CANONICAL_TYPES
                | {"pet bed", "decorative pillow", "table lamp", "office chair", "folding chair", "chair"}
            )
            or identity.category in {"bedding", "pet home", "home decor"}
        )
    )


def should_prefer_source_frame_primary_input(identity: ProductIdentitySpec) -> bool:
    if identity_requires_functional_context(identity):
        return True
    if identity.interaction_mode in {"held_in_hand", "worn", "worn_or_carried", "carried_or_resting"}:
        return False
    if identity_has_low_profile_soft_structure(identity):
        return True
    return identity_has_chromatic_soft_textile_lock(identity)


def _line_seed_offset(line_name: str) -> int:
    return 0 if line_name == "baseline" else 10_007


def generate_review_batch(
    review_manifest_path: str | Path,
    localization_report_path: str | Path,
    retrieval_index_path: str | Path,
    *,
    output_dir: str | Path,
    include_ids: Sequence[str] | None = None,
    model_id: str = "black-forest-labs/FLUX.2-klein-9B",
    width: int = 512,
    height: int = 512,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    device: str = "cuda",
    analysis_device: str | None = None,
    candidate_modes_override: Sequence[str] | None = None,
    skip_analysis: bool = False,
    include_lines: Sequence[str] | None = None,
) -> Path:
    if not skip_analysis:
        from product_campaign_pipeline.localization import (
            ProductPhoto,
            build_model_backed_localization_pipeline,
            save_localization_artifacts,
            select_primary_mask,
        )

    seeds = load_review_seed_manifest(review_manifest_path)
    if include_ids:
        allowed = {str(value) for value in include_ids}
        seeds = [seed for seed in seeds if seed.id in allowed]
    localization_by_id = load_localization_report(localization_report_path)
    retrieval_index = load_retrieval_index(retrieval_index_path)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = target_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = target_dir / "images"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = target_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    backbone = VisionBackbone(device=analysis_device or ("cpu" if device != "cpu" else device))
    generated_localizer = build_model_backed_localization_pipeline(device="cpu") if not skip_analysis else None
    client = Flux2KleinClient(
        model_id=model_id,
        device=device,
        dtype="bfloat16",
        cpu_offload=True,
        sequential_cpu_offload=True,
        attention_slicing=True,
    )
    composer = PromptComposer()
    report_rows: list[dict[str, Any]] = []
    invalid_sources: list[dict[str, Any]] = []
    selected_lines = [str(line) for line in (include_lines or ("baseline", "business_prior"))]

    for seed_index, seed in enumerate(seeds):
        localization = localization_by_id[seed.id]
        localized = build_localized_product(seed, localization, backbone=backbone)
        if localized.identity.observed_evidence.source_validity != "valid":
            invalid_sources.append(
                {
                    "id": seed.id,
                    "product_title": seed.product_title,
                    "source_image_path": str(seed.local_image_path),
                    "canonical_product_type": localized.identity.canonical_product_type,
                    "category": localized.identity.category,
                    "source_validity": localized.identity.observed_evidence.source_validity,
                    "source_validity_score": localized.identity.observed_evidence.source_validity_score,
                    "source_validity_issues": list(localized.identity.observed_evidence.source_validity_issues),
                }
            )
            continue
        prior = build_business_prior(seed, localized, localization, retrieval_index, backbone, top_k=5)

        for line_name in selected_lines:
            support_relation = (
                prior.support_relation
                if line_name == "business_prior"
                else default_support_relation_for_identity(localized.identity)
            )
            scene_family = (
                prior.scene_family if line_name == "business_prior" else (
                    localized.identity.default_scene_family or SCENE_FAMILY_DEFAULTS_BY_SUPPORT.get(
                        support_relation,
                        "editorial_interior",
                    )
                )
            )
            candidate_modes = (
                tuple(candidate_modes_override)
                if candidate_modes_override
                else select_reinvention_candidate_modes_for_line(localized.identity, line_name=line_name)
            )
            candidate_steps = max(num_inference_steps, 6) if len(candidate_modes) > 1 else num_inference_steps
            candidate_guidance_scale = guidance_scale
            if localized.identity.requires_human_model or localized.identity.interaction_mode in {
                "worn",
                "worn_or_carried",
                "held_in_hand",
                "carried_or_resting",
            }:
                candidate_steps = max(candidate_steps, 8)
            if should_strengthen_dominant_body_color_guidance(localized.identity):
                candidate_steps = max(candidate_steps, 8)
                candidate_guidance_scale = max(candidate_guidance_scale, 1.15)
            candidate_rows: list[dict[str, Any]] = []

            for candidate_index, reinvention_mode in enumerate(candidate_modes):
                candidate_seed = (1000 + seed_index + _line_seed_offset(line_name)) + candidate_index * 101
                prompt_spec = (
                    composer.compose_baseline(
                        localized,
                        seed=candidate_seed,
                        reinvention_mode=reinvention_mode,
                    )
                    if line_name == "baseline"
                    else composer.compose_business_prior(
                        localized,
                        prior,
                        seed=candidate_seed,
                        reinvention_mode=reinvention_mode,
                    )
                )
                candidate_output_path = (
                    outputs_dir / f"{seed.id}.{line_name}.png"
                    if len(candidate_modes) == 1
                    else candidates_dir / f"{seed.id}.{line_name}.{candidate_index:02d}.{reinvention_mode}.png"
                )
                request = build_generation_request(
                    client,
                    prompt_spec,
                    source_image=Path(localized.source_image),
                    reference_images=_conditioning_reference_images(localized),
                    primary_input_image=_primary_generation_input_image(localized),
                    allow_reference_only=should_use_reference_only_conditioning(localized.identity),
                    output_path=candidate_output_path,
                    width=width,
                    height=height,
                    num_inference_steps=candidate_steps,
                    guidance_scale=candidate_guidance_scale,
                )
                generation = client.generate(request)
                maybe_repair_generated_dominant_body_color(
                    Path(generation.output_path),
                    localized,
                    generated_localizer=generated_localizer,
                    product_photo_factory=ProductPhoto,
                    save_artifacts=save_localization_artifacts,
                    select_mask=select_primary_mask,
                )
                prompt_readiness = assess_prompt_readiness(
                    localized,
                    prompt_spec,
                    scene_family=scene_family,
                    support_relation=support_relation,
                )
                if skip_analysis:
                    category_consistency = {}
                    semantic_plausibility = {}
                    evidence_consistency = {}
                    candidate_score = float(prompt_readiness.get("score", 0.0))
                    if reinvention_mode == "balanced":
                        candidate_score += 0.02
                else:
                    focus_artifacts = extract_generated_focus_artifacts(
                        Path(generation.output_path),
                        localized,
                        generated_localizer=generated_localizer,
                        product_photo_factory=ProductPhoto,
                        save_artifacts=save_localization_artifacts,
                        select_mask=select_primary_mask,
                    )
                    category_consistency = assess_category_consistency(
                        generation.output_path,
                        expected_category=localized.identity.category,
                        expected_product_type=localized.identity.canonical_product_type or localized.identity.category,
                        backbone=backbone,
                        focus_image_path=None if focus_artifacts is None else focus_artifacts.get("crop_path"),
                    )
                    semantic_plausibility = assess_semantic_plausibility(
                        generation.output_path,
                        localized.identity,
                        prompt_spec=prompt_spec,
                        scene_family=scene_family,
                        support_relation=support_relation,
                        backbone=backbone,
                        generated_localizer=generated_localizer,
                        product_photo_factory=ProductPhoto,
                    )
                    evidence_consistency = assess_evidence_consistency(
                        generation.output_path,
                        localized,
                        backbone=backbone,
                        generated_localizer=generated_localizer,
                        product_photo_factory=ProductPhoto,
                        save_artifacts=save_localization_artifacts,
                        select_mask=select_primary_mask,
                        focus_artifacts=focus_artifacts,
                    )
                    candidate_score = score_generation_candidate(
                        category_consistency=category_consistency,
                        semantic_plausibility=semantic_plausibility,
                        evidence_consistency=evidence_consistency,
                    )
                candidate_rows.append(
                    {
                        "id": seed.id,
                        "line": line_name,
                        "product_title": seed.product_title,
                        "hint_phrases": list(seed.hint_phrases),
                        "selected_phrase": localization.selected_phrase,
                        "expected_category": localized.identity.category,
                        "canonical_product_type": localized.identity.canonical_product_type,
                        "weak_shape_evidence": localized.identity.weak_shape_evidence,
                        "scene_family": scene_family,
                        "support_relation": support_relation,
                        "source_page_url": seed.source_page_url,
                        "source_image_url": seed.source_image_url,
                        "source_image_path": str(seed.local_image_path),
                        "crop_path": None if localized.crop_path is None else localized.crop_path,
                        "mask_path": None if localized.mask_path is None else localized.mask_path,
                        "output_path": generation.output_path,
                        "elapsed_seconds": generation.elapsed_seconds,
                        "prompt": prompt_spec.model_dump(),
                        "observed_evidence": localized.identity.observed_evidence.model_dump(),
                        "retrieval_metadata": prior.metadata if line_name == "business_prior" else {},
                        "category_consistency": category_consistency,
                        "semantic_plausibility": semantic_plausibility,
                        "evidence_consistency": evidence_consistency,
                        "prompt_readiness": prompt_readiness,
                        "candidate_mode": reinvention_mode,
                        "candidate_index": candidate_index,
                        "candidate_score": round(candidate_score, 4),
                    }
                )
                if skip_analysis:
                    client.reset_pipeline()

            selection_pool = list(candidate_rows)
            evidence_consistent_rows = [
                row for row in selection_pool if row["evidence_consistency"].get("is_consistent", True)
            ]
            if evidence_consistent_rows:
                selection_pool = evidence_consistent_rows
            category_consistent_rows = [
                row for row in selection_pool if row["category_consistency"].get("is_consistent", True)
            ]
            if category_consistent_rows:
                selection_pool = category_consistent_rows
            ghost_free_rows = [
                row for row in selection_pool if not row["semantic_plausibility"].get("ghost_composite_flag", False)
            ]
            if ghost_free_rows:
                selection_pool = ghost_free_rows
            background_resolved_rows = [
                row for row in selection_pool if not row["semantic_plausibility"].get("background_collapse_flag", False)
            ]
            if background_resolved_rows:
                selection_pool = background_resolved_rows
            product_only_rows = [
                row
                for row in selection_pool
                if not (
                    row["semantic_plausibility"].get("people_out_of_frame_required", False)
                    and row["semantic_plausibility"].get("person_presence_flag", False)
                )
            ]
            if product_only_rows:
                selection_pool = product_only_rows
            casting_aligned_rows = [
                row
                for row in selection_pool
                if (
                    not row["semantic_plausibility"].get("human_supported", False)
                    or float(row["semantic_plausibility"].get("casting_margin", 0.0)) >= 0.01
                )
            ]
            if casting_aligned_rows:
                selection_pool = casting_aligned_rows
            dress_layering_margin_values = [
                float(row["semantic_plausibility"].get("dress_layering_margin", 0.0)) for row in selection_pool
            ]
            if dress_layering_margin_values:
                best_dress_layering_margin = max(dress_layering_margin_values)
                worst_dress_layering_margin = min(dress_layering_margin_values)
                if best_dress_layering_margin - worst_dress_layering_margin >= 0.01:
                    dress_layering_preferred_rows = [
                        row
                        for row in selection_pool
                        if float(row["semantic_plausibility"].get("dress_layering_margin", 0.0))
                        >= best_dress_layering_margin - 0.004
                    ]
                    if dress_layering_preferred_rows:
                        selection_pool = dress_layering_preferred_rows
            single_model_margin_values = [
                float(row["semantic_plausibility"].get("single_model_margin", 0.0)) for row in selection_pool
            ]
            if single_model_margin_values:
                best_single_model_margin = max(single_model_margin_values)
                worst_single_model_margin = min(single_model_margin_values)
                if best_single_model_margin - worst_single_model_margin >= 0.01:
                    single_model_preferred_rows = [
                        row
                        for row in selection_pool
                        if float(row["semantic_plausibility"].get("single_model_margin", 0.0))
                        >= best_single_model_margin - 0.004
                    ]
                    if single_model_preferred_rows:
                        selection_pool = single_model_preferred_rows
            compact_focus_rows = [
                row
                for row in selection_pool
                if (
                    not row["evidence_consistency"].get("compact_product_focus_required", False)
                    or float(row["evidence_consistency"].get("product_prominence_alignment", 0.5))
                    >= compact_focus_alignment_threshold(row["evidence_consistency"])
                )
            ]
            if compact_focus_rows:
                selection_pool = compact_focus_rows
            semantic_plausible_rows = [
                row for row in selection_pool if row["semantic_plausibility"].get("is_plausible", True)
            ]
            if semantic_plausible_rows:
                selection_pool = semantic_plausible_rows
            selected_row = max(
                selection_pool,
                key=lambda row: (
                    float(row["candidate_score"]),
                    float(row["evidence_consistency"].get("score", 0.0)),
                    float(row["semantic_plausibility"].get("score", 0.0)),
                ),
            )
            if localized.identity.canonical_product_type == "dress" and float(
                selected_row["semantic_plausibility"].get("dress_layering_margin", 0.0)
            ) < 0.0:
                dress_layering_clean_rows = [
                    row
                    for row in candidate_rows
                    if float(row["semantic_plausibility"].get("dress_layering_margin", 0.0)) >= 0.0
                    and row["category_consistency"].get("is_consistent", True)
                    and not row["semantic_plausibility"].get("ghost_composite_flag", False)
                    and not row["semantic_plausibility"].get("background_collapse_flag", False)
                ]
                if dress_layering_clean_rows:
                    selected_row = max(
                        dress_layering_clean_rows,
                        key=lambda row: (
                            float(row["candidate_score"]),
                            float(row["evidence_consistency"].get("score", 0.0)),
                            float(row["semantic_plausibility"].get("score", 0.0)),
                        ),
                    )
            final_output_path = outputs_dir / f"{seed.id}.{line_name}.png"
            if Path(selected_row["output_path"]).resolve() != final_output_path.resolve():
                shutil.copy2(selected_row["output_path"], final_output_path)
                selected_row["output_path"] = str(final_output_path)
            selected_row["candidate_count"] = len(candidate_rows)
            selected_row["candidate_scores"] = [
                {
                    "mode": row["candidate_mode"],
                    "index": row["candidate_index"],
                    "score": row["candidate_score"],
                    "evidence_score": round(float(row["evidence_consistency"].get("score", 0.0)), 4),
                }
                for row in candidate_rows
            ]
            report_rows.append(selected_row)

    report_path = reports_dir / "generation_report.json"
    report_path.write_text(json.dumps(report_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    if invalid_sources:
        (reports_dir / "invalid_sources.json").write_text(
            json.dumps(invalid_sources, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    board_path = render_review_board(report_rows, target_dir / "human_review_board.html")
    return board_path


def generate_upstream_review_batch(
    review_manifest_path: str | Path,
    localization_report_path: str | Path,
    retrieval_index_path: str | Path,
    *,
    output_dir: str | Path,
    include_ids: Sequence[str] | None = None,
    device: str = "cuda",
) -> Path:
    seeds = load_review_seed_manifest(review_manifest_path)
    if include_ids:
        allowed = {str(value) for value in include_ids}
        seeds = [seed for seed in seeds if seed.id in allowed]
    localization_by_id = load_localization_report(localization_report_path)
    retrieval_index = load_retrieval_index(retrieval_index_path)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = target_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    backbone = VisionBackbone(device=device)
    composer = PromptComposer()
    report_rows: list[dict[str, Any]] = []
    invalid_sources: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        localization = localization_by_id[seed.id]
        localized = build_localized_product(seed, localization, backbone=backbone)
        if localized.identity.observed_evidence.source_validity != "valid":
            invalid_sources.append(
                {
                    "id": seed.id,
                    "product_title": seed.product_title,
                    "source_image_path": str(seed.local_image_path),
                    "canonical_product_type": localized.identity.canonical_product_type,
                    "category": localized.identity.category,
                    "source_validity": localized.identity.observed_evidence.source_validity,
                    "source_validity_score": localized.identity.observed_evidence.source_validity_score,
                    "source_validity_issues": list(localized.identity.observed_evidence.source_validity_issues),
                }
            )
            continue
        prior = build_business_prior(seed, localized, localization, retrieval_index, backbone, top_k=5)

        for line_name in ("baseline", "business_prior"):
            support_relation = (
                prior.support_relation
                if line_name == "business_prior"
                else default_support_relation_for_identity(localized.identity)
            )
            scene_family = (
                prior.scene_family if line_name == "business_prior" else (
                    localized.identity.default_scene_family or SCENE_FAMILY_DEFAULTS_BY_SUPPORT.get(
                        support_relation,
                        "editorial_interior",
                    )
                )
            )
            candidate_modes = select_reinvention_candidate_modes_for_line(localized.identity, line_name=line_name)
            candidate_prompts: list[dict[str, Any]] = []
            for candidate_index, reinvention_mode in enumerate(candidate_modes):
                candidate_seed = (1000 + seed_index + _line_seed_offset(line_name)) + candidate_index * 101
                prompt_spec = (
                    composer.compose_baseline(
                        localized,
                        seed=candidate_seed,
                        reinvention_mode=reinvention_mode,
                    )
                    if line_name == "baseline"
                    else composer.compose_business_prior(
                        localized,
                        prior,
                        seed=candidate_seed,
                        reinvention_mode=reinvention_mode,
                    )
                )
                prompt_readiness = assess_prompt_readiness(
                    localized,
                    prompt_spec,
                    scene_family=scene_family,
                    support_relation=support_relation,
                )
                candidate_prompts.append(
                    {
                        "mode": reinvention_mode,
                        "index": candidate_index,
                        "seed": candidate_seed,
                        "scene_family": scene_family,
                        "support_relation": support_relation,
                        "prompt_readiness": prompt_readiness,
                        "prompt": prompt_spec.model_dump(),
                    }
                )

            selected_prompt = max(
                candidate_prompts,
                key=lambda item: (
                    float(item["prompt_readiness"].get("score", 0.0)),
                    -len(item["prompt_readiness"].get("issues", [])),
                ),
            )
            report_rows.append(
                {
                    "id": seed.id,
                    "line": line_name,
                    "product_title": seed.product_title,
                    "hint_phrases": list(seed.hint_phrases),
                    "selected_phrase": localization.selected_phrase,
                    "selected_confidence": localization.selected_confidence,
                    "expected_category": localized.identity.category,
                    "canonical_product_type": localized.identity.canonical_product_type,
                    "support_mode": localized.identity.support_mode,
                    "scene_family": scene_family,
                    "support_relation": support_relation,
                    "interaction_mode": localized.identity.interaction_mode,
                    "style_persona": localized.identity.style_persona,
                    "casting_note": localized.identity.casting_note,
                    "stable_base": localized.identity.stable_base,
                    "rigid_vs_soft": localized.identity.rigid_vs_soft,
                    "weak_shape_evidence": localized.identity.weak_shape_evidence,
                    "source_page_url": seed.source_page_url,
                    "source_image_url": seed.source_image_url,
                    "source_image_path": str(seed.local_image_path),
                    "overlay_path": None if localization.overlay_path is None else str(localization.overlay_path),
                    "crop_path": None if localized.crop_path is None else localized.crop_path,
                    "mask_path": None if localized.mask_path is None else localized.mask_path,
                    "observed_evidence": localized.identity.observed_evidence.model_dump(),
                    "candidate_prompts": candidate_prompts,
                    "selected_candidate_mode": selected_prompt["mode"],
                    "selected_candidate_index": selected_prompt["index"],
                    "retrieval_metadata": prior.metadata if line_name == "business_prior" else {},
                    "style_atoms": [] if line_name == "baseline" else list(prior.style_atoms),
                    "semantic_constraints": [] if line_name == "baseline" else list(prior.semantic_constraints),
                }
            )

    report_path = reports_dir / "upstream_review_report.json"
    report_path.write_text(json.dumps(report_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    if invalid_sources:
        (reports_dir / "invalid_sources.json").write_text(
            json.dumps(invalid_sources, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    board_path = render_upstream_review_board(report_rows, target_dir / "human_review_board.html")
    return board_path


def build_generation_request(
    client: Flux2KleinClient,
    prompt_spec: FluxPromptSpec,
    *,
    source_image: Path,
    reference_images: Sequence[Path],
    primary_input_image: Path | None = None,
    allow_reference_only: bool = False,
    output_path: Path,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
):
    input_image = None if allow_reference_only else (primary_input_image or source_image)
    deduped_references: list[Path] = []
    input_image_resolved = input_image.resolve() if input_image is not None else None
    for reference_image in reference_images:
        if input_image_resolved is not None and reference_image.resolve() == input_image_resolved:
            continue
        if reference_image not in deduped_references:
            deduped_references.append(reference_image)
    prompt_payload = {
        "subject": prompt_spec.subject,
        "action": prompt_spec.action,
        "style": prompt_spec.style,
        "context": prompt_spec.context,
        "preservation_constraints": list(prompt_spec.preservation_constraints),
    }
    return client.build_request(
        prompt=prompt_spec.to_prompt_text(),
        input_image=input_image,
        reference_images=deduped_references,
        width=width,
        height=height,
        seed=prompt_spec.seed,
        output_path=output_path,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=min(prompt_spec.max_sequence_length, 256),
        extra=prompt_payload,
    )


def _conditioning_reference_images(localized: LocalizedProduct) -> list[Path]:
    evidence = localized.identity.observed_evidence
    if should_use_reference_only_conditioning(localized.identity):
        references: list[Path] = []
        for candidate in (
            evidence.reference_crop_path,
            evidence.reference_silhouette_path,
            localized.crop_path,
        ):
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists() and path not in references:
                references.append(path)
        if identity_requires_functional_context(localized.identity):
            source_path = Path(localized.source_image)
            if source_path.exists() and source_path not in references:
                references.append(source_path)
        return references
    if should_prefer_crop_only_color_lock(localized.identity):
        crop_path = evidence.reference_crop_path or localized.crop_path
        if crop_path:
            crop = Path(crop_path)
            if crop.exists():
                return [crop]
    references: list[Path] = []
    for candidate in (
        evidence.reference_crop_path,
        evidence.reference_cutout_path,
        evidence.reference_silhouette_path,
        localized.crop_path,
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and path not in references:
            references.append(path)
    if identity_requires_functional_context(localized.identity):
        source_path = Path(localized.source_image)
        if source_path.exists() and source_path not in references:
            references.append(source_path)
    return references


def _primary_generation_input_image(localized: LocalizedProduct) -> Path:
    evidence = localized.identity.observed_evidence
    if should_prefer_source_frame_primary_input(localized.identity):
        source_path = Path(localized.source_image)
        if source_path.exists():
            return source_path
    for candidate in (
        evidence.reference_crop_path,
        evidence.reference_cutout_path,
        localized.crop_path,
        evidence.reference_silhouette_path,
        localized.source_image,
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return Path(localized.source_image)


def should_strengthen_dominant_body_color_guidance(identity: ProductIdentitySpec) -> bool:
    evidence = identity.observed_evidence
    dominant_body_color = extract_dominant_body_color(evidence)
    if dominant_body_color is None:
        return False
    if evidence.color_confidence is not None and evidence.color_confidence < 0.64:
        return False
    if evidence.coverage_class in {
        "low_variation_surface",
        "full_visible_surface_pattern",
        "broad_visible_surface_pattern",
    }:
        return True
    if (
        evidence.coverage_class == "localized_visible_pattern"
        and identity.category in {"kitchen appliance", "drinkware", "home lighting"}
        and dominant_body_color not in {"black", "white", "gray", "beige", "brown"}
    ):
        return True
    return identity.category in {"bedding", "pet home", "footwear", "home decor"}


def should_use_reference_only_conditioning(identity: ProductIdentitySpec) -> bool:
    evidence = identity.observed_evidence
    canonical_type = str(identity.canonical_product_type or "").strip().lower()
    if identity.interaction_mode != "held_in_hand":
        return False
    if evidence.upper_component_state != "absent":
        return False
    compact_tokens = " ".join(
        filter(
            None,
            [
                evidence.form_factor_note,
                evidence.silhouette_note,
                *evidence.hard_facts,
            ],
        )
    ).lower()
    if canonical_type in {"wallet", "clutch", "wristlet", "card holder"}:
        return True
    if "compact" in compact_tokens or "hand-held" in compact_tokens:
        return True
    return bool(evidence.aspect_ratio is not None and evidence.aspect_ratio <= 1.7)


def should_prefer_crop_only_color_lock(identity: ProductIdentitySpec) -> bool:
    evidence = identity.observed_evidence
    dominant_body_color = extract_dominant_body_color(evidence)
    if identity.rigid_vs_soft == "rigid":
        return bool(
            identity.category in {"kitchen appliance", "drinkware", "home lighting"}
            and evidence.reference_crop_path
            and should_strengthen_dominant_body_color_guidance(identity)
            and dominant_body_color not in {None, "black", "white", "gray", "beige", "brown"}
        )
    if identity.category in {"bedding", "pet home"}:
        return False
    if (identity.canonical_product_type or "") in (BEDDING_CANONICAL_TYPES | {"pet bed"}):
        return False
    if evidence.edge_profile_note or evidence.soft_structure_note or evidence.lower_region_note:
        return False
    return bool(
        should_strengthen_dominant_body_color_guidance(identity)
        and evidence.reference_crop_path
        and evidence.coverage_class in {
            "low_variation_surface",
            "full_visible_surface_pattern",
            "broad_visible_surface_pattern",
        }
    )


def select_reinvention_candidate_modes(identity: ProductIdentitySpec) -> tuple[str, ...]:
    evidence = identity.observed_evidence
    canonical_type = str(identity.canonical_product_type or "").strip().lower()
    if identity_has_low_profile_soft_structure(identity) or identity_has_chromatic_soft_textile_lock(identity):
        return ("balanced", "clarity")
    if identity.interaction_mode == "held_in_hand" and evidence.upper_component_state == "absent":
        compact_handheld = bool(
            canonical_type in {"wallet", "clutch", "wristlet", "card holder"}
            or "compact" in str(evidence.form_factor_note or "").lower()
            or "hand-held" in str(evidence.form_factor_note or "").lower()
            or (evidence.aspect_ratio is not None and evidence.aspect_ratio <= 1.7)
        )
        if compact_handheld:
            return ("clarity", "hero")
        return ("balanced", "hero")
    if identity.requires_human_model or identity.interaction_mode in {
        "worn",
        "worn_or_carried",
        "held_in_hand",
        "carried_or_resting",
    }:
        return HUMAN_REINVENTION_CANDIDATE_MODES
    if (
        identity.stable_base is True
        and evidence.aspect_ratio is not None
        and evidence.aspect_ratio >= 1.5
        and evidence.top_width_ratio is not None
        and evidence.top_width_ratio >= 0.95
    ):
        return REINVENTION_CANDIDATE_MODES
    if (
        identity.stable_base is True
        and identity.rigid_vs_soft == "rigid"
        and not identity.requires_human_model
        and identity.interaction_mode == "placed"
    ):
        return ("balanced", "reveal")
    if identity.weak_shape_evidence:
        return REINVENTION_CANDIDATE_MODES
    if evidence.uncertainty_level in {"medium", "high"}:
        return REINVENTION_CANDIDATE_MODES
    if identity.stable_base is False:
        return REINVENTION_CANDIDATE_MODES
    return ("balanced",)


def select_reinvention_candidate_modes_for_line(
    identity: ProductIdentitySpec,
    *,
    line_name: str,
) -> tuple[str, ...]:
    base_modes = select_reinvention_candidate_modes(identity)
    canonical_type = str(identity.canonical_product_type or "").strip().lower()
    if (
        identity.category == "furniture"
        and canonical_type in {"folding chair", "chair"}
        and identity.stable_base is True
        and identity.rigid_vs_soft == "rigid"
        and not identity.requires_human_model
        and identity.interaction_mode == "placed"
    ):
        return ("clarity", "reveal")
    if identity_requires_functional_context(identity):
        if line_name == "business_prior":
            return ("clarity", "hero")
        return base_modes
    if line_name == "baseline" and identity_has_low_profile_soft_structure(identity):
        return ("clarity",)
    if line_name == "business_prior" and identity_has_low_profile_soft_structure(identity):
        evidence = identity.observed_evidence
        severe_artifact_flags = [
            flag for flag in evidence.artifact_flags if flag not in {"border_foreground_intrusion"}
        ]
        if severe_artifact_flags or evidence.uncertainty_level == "high":
            return ("clarity",)
        return ("clarity", "balanced")
    return base_modes


def _export_evidence_assets(
    *,
    source_image: Image.Image,
    mask_array: np.ndarray,
    display_mask_array: np.ndarray | None = None,
    crop_path: Path,
    cutout_path: Path,
    silhouette_path: Path,
    mask_output_path: Path,
    category: str,
    canonical_product_type: str,
    force_masked_crop: bool = False,
) -> None:
    mask = Image.fromarray((mask_array.astype(np.uint8) * 255))
    display_mask = Image.fromarray(((display_mask_array if display_mask_array is not None else mask_array).astype(np.uint8) * 255))
    bbox = display_mask.getbbox()
    if bbox is None:
        source_image.save(crop_path)
        source_image.save(cutout_path)
        source_image.save(silhouette_path)
        mask.save(mask_output_path)
        return

    crop_box = _expand_crop_box(
        bbox,
        image_size=source_image.size,
        margin_ratio=0.08,
        min_margin=10,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    source_crop = source_image.crop(crop_box)
    display_mask_crop = display_mask.crop(crop_box)
    matte_color = _reference_matte_color(source_crop, display_mask_crop)

    cutout = Image.new("RGB", source_crop.size, matte_color)
    cutout.paste(source_crop, mask=display_mask_crop)

    silhouette = Image.new("RGB", source_crop.size, (247, 245, 241))
    silhouette_fill = Image.new("RGB", source_crop.size, (34, 34, 34))
    silhouette.paste(silhouette_fill, mask=display_mask_crop)

    if force_masked_crop:
        if category in {"bedding", "pet home", "home decor"} or canonical_product_type in (
            BEDDING_CANONICAL_TYPES | {"pet bed", "decorative pillow"}
        ):
            source_crop.save(crop_path)
        else:
            masked_crop = Image.new("RGB", source_crop.size, matte_color)
            masked_crop.paste(source_crop, mask=display_mask_crop)
            masked_crop.save(crop_path)
    else:
        source_crop.save(crop_path)
    cutout.save(cutout_path)
    silhouette.save(silhouette_path)
    mask.save(mask_output_path)


def _reference_matte_color(source_crop: Image.Image, mask_crop: Image.Image) -> tuple[int, int, int]:
    warm_neutral = np.asarray((244, 241, 236), dtype=np.float32)
    source_array = np.asarray(source_crop.convert("RGB"), dtype=np.float32)
    mask_array = np.asarray(mask_crop.convert("L")) > 0
    if not mask_array.any():
        return tuple(int(value) for value in warm_neutral)
    mean_rgb = np.mean(source_array[mask_array], axis=0)
    matte = np.clip(mean_rgb * 0.68 + warm_neutral * 0.32, 0.0, 255.0)
    return tuple(int(round(float(value))) for value in matte)


def _should_prepare_color_anchor_asset(
    *,
    category: str,
    canonical_product_type: str,
    soft_structure_class: str | None = None,
) -> bool:
    if category == "footwear" or canonical_product_type == "shoe":
        return False
    if category in {"bedding", "pet home"} or canonical_product_type in (BEDDING_CANONICAL_TYPES | {"pet bed"}):
        return False
    if soft_structure_class == "flat_surface" and category in {"pet home", "home decor"}:
        return False
    return canonical_product_type == "decorative pillow" or category == "home decor"


def _export_color_anchor_asset(
    *,
    source_image: Image.Image,
    mask_array: np.ndarray,
    dominant_color: str,
    output_path: Path,
) -> None:
    if dominant_color not in EVIDENCE_COLOR_SWATCHES:
        return
    mask = Image.fromarray((mask_array.astype(np.uint8) * 255))
    bbox = mask.getbbox()
    if bbox is None:
        return
    crop_box = _expand_crop_box(
        bbox,
        image_size=source_image.size,
        margin_ratio=0.08,
        min_margin=10,
        category="product",
        canonical_product_type="product",
    )
    source_crop = np.asarray(source_image.crop(crop_box).convert("RGB"), dtype=np.float32)
    mask_crop = np.asarray(mask.crop(crop_box).convert("L"), dtype=np.float32) > 0
    if not mask_crop.any():
        return

    target_rgb = np.asarray(EVIDENCE_COLOR_SWATCHES[dominant_color], dtype=np.float32)
    luma = source_crop.mean(axis=2)
    masked_luma = luma[mask_crop]
    mean_luma = float(masked_luma.mean())
    std_luma = max(float(masked_luma.std()), 1.0)
    detail = np.clip((luma - mean_luma) / (2.0 * std_luma), -1.0, 1.0)
    detail_scale = 18.0 if dominant_color in {"black", "white", "gray", "beige"} else 24.0

    anchor = np.full_like(source_crop, 244.0)
    anchor_rgb = np.clip(target_rgb[None, None, :] + detail[..., None] * detail_scale, 0.0, 255.0)
    anchor[mask_crop] = anchor_rgb[mask_crop]
    Image.fromarray(anchor.astype(np.uint8)).save(output_path)


def _expand_crop_box(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    margin_ratio: float,
    min_margin: int,
    category: str,
    canonical_product_type: str,
) -> tuple[int, int, int, int]:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    adjusted_margin_ratio = margin_ratio
    bottom_boost = 1.0
    right_boost = 1.0
    if canonical_product_type == "blender":
        adjusted_margin_ratio = max(adjusted_margin_ratio, 0.18)
        bottom_boost = 3.0
        right_boost = 2.8
    elif canonical_product_type in {"office chair", "folding chair"} or category == "furniture":
        adjusted_margin_ratio = max(adjusted_margin_ratio, 0.16)
        bottom_boost = 2.15
    elif canonical_product_type in ({"table lamp"} | KITCHEN_APPLIANCE_CANONICAL_TYPES - {"blender"}) or category in {"kitchen appliance", "home lighting"}:
        adjusted_margin_ratio = max(adjusted_margin_ratio, 0.14)
        bottom_boost = 1.85
    elif canonical_product_type in (BEDDING_CANONICAL_TYPES | {"pet bed"}) or category in {"bedding", "pet home"}:
        adjusted_margin_ratio = max(adjusted_margin_ratio, 0.1)
        bottom_boost = 1.15
    margin_x = max(min_margin, int(round(width * adjusted_margin_ratio)))
    margin_y = max(min_margin, int(round(height * adjusted_margin_ratio)))
    image_width, image_height = image_size
    x0 = max(0, bbox[0] - margin_x)
    y0 = max(0, bbox[1] - margin_y)
    x1 = min(image_width, bbox[2] + int(round(margin_x * right_boost)))
    y1 = min(image_height, bbox[3] + int(round(margin_y * bottom_boost)))
    return (x0, y0, x1, y1)


def refine_observed_evidence_mask(
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
    requires_human_model: bool,
) -> np.ndarray:
    refined = mask.astype(bool)
    if not refined.any():
        return refined
    if requires_human_model and category == "apparel":
        refined = _refine_apparel_product_mask(refined)
    elif category == "footwear" or canonical_product_type == "shoe":
        refined = _refine_footwear_product_mask(refined)
    elif canonical_product_type in (BEDDING_CANONICAL_TYPES | {"pet bed"}) or category in {"bedding", "pet home"}:
        refined = _refine_supported_soft_product_mask(
            refined,
            category=category,
            canonical_product_type=canonical_product_type,
        )
    refined = repair_rigid_body_notches(
        refined,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    refined = _remove_small_border_components(refined)
    return refined if refined.any() else mask.astype(bool)


def suppress_border_attached_reference_artifacts(
    source: np.ndarray,
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
    requires_human_model: bool,
) -> tuple[np.ndarray, list[str]]:
    if not mask.any():
        return mask, []
    cleaned = mask.copy()
    artifact_flags: list[str] = []
    soft_nonhuman_product = bool(
        not requires_human_model
        and (
            category in {"bedding", "pet home", "home decor"}
            or canonical_product_type in (BEDDING_CANONICAL_TYPES | {"pet bed", "decorative pillow"})
        )
    )
    if soft_nonhuman_product:
        skin_fragment = _detect_border_skin_fragment_artifact(source, cleaned)
        if skin_fragment.any():
            candidate = cleaned & ~skin_fragment
            if candidate.any() and candidate.sum() >= int(cleaned.sum() * 0.68):
                cleaned = candidate
                artifact_flags.append("border_human_fragment")
        foreign_intrusion = _detect_border_color_intrusion_artifact(source, cleaned)
        if foreign_intrusion.any():
            candidate = cleaned & ~foreign_intrusion
            if candidate.any() and candidate.sum() >= int(cleaned.sum() * 0.62):
                cleaned = candidate
                artifact_flags.append("border_foreground_intrusion")
    text_overlay = _detect_border_text_overlay_artifact(source, cleaned)
    if text_overlay.any():
        candidate = cleaned & ~text_overlay
        if candidate.any() and candidate.sum() >= int(cleaned.sum() * 0.82):
            cleaned = candidate
            artifact_flags.append("border_text_overlay")
    return (cleaned if cleaned.any() else mask, artifact_flags)


def repair_removed_reference_regions(
    source_image: Image.Image,
    *,
    removed_region_mask: np.ndarray,
    keep_mask: np.ndarray,
) -> Image.Image:
    if not removed_region_mask.any():
        return source_image
    repaired = np.asarray(source_image.convert("RGB"), dtype=np.uint8).copy()
    inpaint_mask = _dilate_mask(removed_region_mask.astype(bool), steps=1) & ~keep_mask
    if not inpaint_mask.any():
        inpaint_mask = removed_region_mask.astype(bool)
    if not inpaint_mask.any():
        return source_image
    try:
        import cv2  # type: ignore

        bgr = cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, (inpaint_mask.astype(np.uint8) * 255), 3, cv2.INPAINT_TELEA)
        repaired = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    except Exception:
        reference_mask = keep_mask & ~inpaint_mask
        if not reference_mask.any():
            reference_mask = keep_mask
        if reference_mask.any():
            fill_color = np.asarray(np.mean(repaired[reference_mask], axis=0), dtype=np.uint8)
            repaired[inpaint_mask] = fill_color
    return Image.fromarray(repaired)


def smooth_reference_export_mask(
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
    artifact_flags: Sequence[str],
) -> np.ndarray:
    if not mask.any() or not artifact_flags:
        return mask
    soft_nonhuman = bool(
        category in {"bedding", "pet home", "home decor"}
        or canonical_product_type in (BEDDING_CANONICAL_TYPES | {"pet bed", "decorative pillow"})
    )
    if not soft_nonhuman:
        return mask
    closed = _erode_mask(_dilate_mask(mask, steps=4), steps=4)
    if not closed.any():
        return mask
    if float(closed.sum()) <= float(mask.sum()) * 1.18:
        return closed
    return mask


def _detect_border_text_overlay_artifact(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    core_mask = _extract_core_body_mask(mask)
    if core_mask.sum() < max(64, int(mask.sum() * 0.18)):
        core_mask = _erode_mask(mask, steps=2)
    reference_pixels = source[core_mask] if core_mask.any() else source[mask]
    if len(reference_pixels) < 64:
        return np.zeros_like(mask, dtype=bool)
    body_rgb = np.mean(reference_pixels, axis=0)
    source_unit = np.clip(source / 255.0, 0.0, 1.0)
    hsv = np.asarray(
        [colorsys.rgb_to_hsv(*pixel) for pixel in source_unit.reshape(-1, 3)],
        dtype=np.float32,
    ).reshape(source.shape[0], source.shape[1], 3)
    color_distance = np.sqrt(np.sum((source - body_rgb) ** 2, axis=2))
    candidate_mask = mask & (color_distance >= 58.0) & (hsv[:, :, 2] >= 0.42) & (hsv[:, :, 1] <= 0.36)
    return _collect_border_attached_artifact_components(
        candidate_mask,
        total_product_pixels=float(mask.sum()),
        min_area_ratio=0.0002,
        max_area_ratio=0.045,
        max_thinness=0.58,
        min_aspect=1.3,
    )


def _detect_border_skin_fragment_artifact(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    source_unit = np.clip(source / 255.0, 0.0, 1.0)
    hsv = np.asarray(
        [colorsys.rgb_to_hsv(*pixel) for pixel in source_unit.reshape(-1, 3)],
        dtype=np.float32,
    ).reshape(source.shape[0], source.shape[1], 3)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    skin_like = (
        (((hue <= 0.08) | (hue >= 0.96)) | ((hue >= 0.08) & (hue <= 0.14)))
        & (saturation >= 0.14)
        & (saturation <= 0.72)
        & (value >= 0.28)
        & (value <= 0.97)
    )
    return _collect_border_attached_artifact_components(
        mask & skin_like,
        total_product_pixels=float(mask.sum()),
        min_area_ratio=0.01,
        max_area_ratio=0.22,
        max_thinness=1.0,
        min_aspect=1.0,
    )


def _detect_border_color_intrusion_artifact(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    core_mask = _extract_core_body_mask(mask)
    if core_mask.sum() < max(96, int(mask.sum() * 0.18)):
        core_mask = _erode_mask(mask, steps=2)
    reference_pixels = source[core_mask] if core_mask.any() else source[mask]
    if len(reference_pixels) < 96:
        return np.zeros_like(mask, dtype=bool)
    body_rgb = np.mean(reference_pixels, axis=0)
    color_distance = np.sqrt(np.sum((source - body_rgb) ** 2, axis=2))
    candidate_mask = mask & (color_distance >= 54.0)
    if not candidate_mask.any():
        return np.zeros_like(mask, dtype=bool)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)
    mask_y0 = int(ys.min())
    mask_y1 = int(ys.max()) + 1
    mask_x0 = int(xs.min())
    mask_x1 = int(xs.max()) + 1
    mask_width = float(mask_x1 - mask_x0)
    mask_height = float(mask_y1 - mask_y0)
    total_product_pixels = float(mask.sum())
    collected = np.zeros_like(mask, dtype=bool)
    for component in _mask_components(candidate_mask, min_pixels=max(24, int(total_product_pixels * 0.008))):
        comp_ys, comp_xs = np.nonzero(component)
        if len(comp_xs) == 0:
            continue
        near_outer_extent = bool(
            comp_ys.min() <= mask_y0 + max(8, int(round(mask_height * 0.18)))
            or comp_ys.max() >= (mask_y1 - 1) - max(8, int(round(mask_height * 0.18)))
            or comp_xs.min() <= mask_x0 + max(8, int(round(mask_width * 0.12)))
            or comp_xs.max() >= (mask_x1 - 1) - max(8, int(round(mask_width * 0.12)))
        )
        if not near_outer_extent:
            continue
        width = float(comp_xs.max() - comp_xs.min() + 1)
        height = float(comp_ys.max() - comp_ys.min() + 1)
        area_ratio = float(component.sum()) / max(total_product_pixels, 1.0)
        width_ratio = width / max(mask_width, 1.0)
        height_ratio = height / max(mask_height, 1.0)
        fill_ratio = float(component.sum()) / max(width * height, 1.0)
        if area_ratio < 0.008 or area_ratio > 0.24:
            continue
        if fill_ratio < 0.18:
            continue
        # Preserve broad textile borders or edge bands; suppress compact foreign intrusions.
        if width_ratio >= 0.78 and height_ratio <= 0.16:
            continue
        if height_ratio >= 0.72 and width_ratio <= 0.16:
            continue
        if width_ratio >= 0.82 and height_ratio >= 0.35:
            continue
        collected |= component
    return collected


def _collect_border_attached_artifact_components(
    candidate_mask: np.ndarray,
    *,
    total_product_pixels: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_thinness: float,
    min_aspect: float,
) -> np.ndarray:
    if not candidate_mask.any():
        return np.zeros_like(candidate_mask, dtype=bool)
    collected = np.zeros_like(candidate_mask, dtype=bool)
    min_pixels = max(8, int(total_product_pixels * min_area_ratio))
    for component in _mask_components(candidate_mask, min_pixels=min_pixels):
        ys, xs = np.nonzero(component)
        if len(xs) == 0:
            continue
        area_ratio = float(component.sum()) / max(total_product_pixels, 1.0)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if not _component_touches_image_border(component):
            continue
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        thinness = min(width, height) / max(width, height, 1.0)
        aspect = max(width, height) / max(min(width, height), 1.0)
        if thinness > max_thinness and aspect < min_aspect:
            continue
        collected |= component
    return collected


def _component_touches_image_border(component: np.ndarray) -> bool:
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return False
    border_margin = 8
    return bool(
        ys.min() <= border_margin
        or xs.min() <= border_margin
        or ys.max() >= component.shape[0] - (border_margin + 1)
        or xs.max() >= component.shape[1] - (border_margin + 1)
    )


def _refine_apparel_product_mask(mask: np.ndarray) -> np.ndarray:
    ys, _ = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    shoulder_row = _estimate_body_shoulder_row(mask)
    refined = np.zeros_like(mask, dtype=bool)
    refined[max(y0, shoulder_row) : y1, :] = mask[max(y0, shoulder_row) : y1, :]
    refined = _keep_primary_center_component(refined)
    if refined.sum() >= max(64, int(mask.sum() * 0.38)):
        return refined
    return mask


def _refine_footwear_product_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    width = float(xs.max() - xs.min() + 1)
    height = float(y1 - y0)
    widths: list[int] = []
    for row in range(y0, y1):
        row_xs = np.nonzero(mask[row])[0]
        widths.append(0 if len(row_xs) == 0 else int(row_xs.max() - row_xs.min() + 1))
    if not widths:
        return mask
    max_width = max(widths)
    body_start = y0
    for index, width in enumerate(widths):
        if width < 0.52 * max_width:
            continue
        window = widths[index : min(index + 5, len(widths))]
        if sum(item >= 0.48 * max_width for item in window) >= max(3, len(window) - 1):
            body_start = y0 + index
            break
    if height / max(width, 1.0) <= 0.9:
        body_start = max(body_start, y0 + int(round(height * 0.18)))
    refined = np.zeros_like(mask, dtype=bool)
    refined[body_start:y1, :] = mask[body_start:y1, :]
    refined = _keep_primary_center_component(refined)
    if refined.sum() >= max(64, int(mask.sum() * 0.42)):
        return refined
    return mask


def _build_footwear_surface_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    width = float(xs.max() - xs.min() + 1)
    height = float(y1 - y0)
    trim_ratio = 0.24 if height / max(width, 1.0) <= 0.9 else 0.14
    upper_cut = y0 + int(round(height * trim_ratio))
    surface_mask = np.zeros_like(mask, dtype=bool)
    surface_mask[upper_cut:y1, :] = mask[upper_cut:y1, :]
    surface_mask = _erode_mask(surface_mask, steps=1)
    if surface_mask.any() and surface_mask.sum() >= max(64, int(mask.sum() * 0.46)):
        return surface_mask
    return mask


def _build_rigid_surface_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    height = float(y1 - y0)
    width = float(x1 - x0)
    trim_y = 0.08 if height / max(width, 1.0) >= 1.1 else 0.05
    trim_x = 0.12 if width >= height else 0.08
    inner_mask = np.zeros_like(mask, dtype=bool)
    inner_y0 = y0 + int(round(height * trim_y))
    inner_y1 = y1 - int(round(height * trim_y))
    inner_x0 = x0 + int(round(width * trim_x))
    inner_x1 = x1 - int(round(width * trim_x))
    inner_mask[max(y0, inner_y0) : max(y0 + 1, min(y1, inner_y1)), max(x0, inner_x0) : max(x0 + 1, min(x1, inner_x1))] = (
        mask[max(y0, inner_y0) : max(y0 + 1, min(y1, inner_y1)), max(x0, inner_x0) : max(x0 + 1, min(x1, inner_x1))]
    )
    inner_mask = _keep_primary_center_component(inner_mask)
    inner_mask = _remove_small_border_components(inner_mask)
    if inner_mask.any() and inner_mask.sum() >= max(64, int(mask.sum() * 0.24)):
        return inner_mask
    return mask


def _build_soft_surface_mask(
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    height = float(y1 - y0)
    width = float(x1 - x0)
    horizontal = height / max(width, 1.0) <= 0.85
    trim_y = 0.08 if horizontal else 0.05
    trim_x = 0.04
    lower_start_ratio = 0.0
    if canonical_product_type == "pet bed" or category == "pet home":
        trim_y = max(trim_y, 0.24 if horizontal else 0.12)
        trim_x = 0.06
        lower_start_ratio = 0.24 if horizontal else 0.1
    elif canonical_product_type in BEDDING_CANONICAL_TYPES or category == "bedding":
        trim_y = max(trim_y, 0.14 if horizontal else 0.08)
        trim_x = 0.05
        lower_start_ratio = 0.12 if horizontal else 0.04
    inner_mask = np.zeros_like(mask, dtype=bool)
    inner_y0 = y0 + int(round(height * trim_y))
    inner_y1 = y1 - int(round(height * trim_y))
    inner_x0 = x0 + int(round(width * trim_x))
    inner_x1 = x1 - int(round(width * trim_x))
    inner_mask[max(y0, inner_y0) : max(y0 + 1, min(y1, inner_y1)), max(x0, inner_x0) : max(x0 + 1, min(x1, inner_x1))] = (
        mask[max(y0, inner_y0) : max(y0 + 1, min(y1, inner_y1)), max(x0, inner_x0) : max(x0 + 1, min(x1, inner_x1))]
    )
    if lower_start_ratio > 0.0:
        lower_start = y0 + int(round(height * lower_start_ratio))
        inner_mask[:lower_start, :] = False
    inner_mask = _erode_mask(inner_mask, steps=1)
    if inner_mask.any() and inner_mask.sum() >= max(64, int(mask.sum() * 0.32)):
        return inner_mask
    return mask


def _refine_supported_soft_product_mask(
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    height = float(y1 - y0)
    width = float(x1 - x0)
    if width <= height:
        return mask
    trim_top_ratio = 0.32 if canonical_product_type == "pet bed" or category == "pet home" else 0.14
    trim_side_ratio = 0.04 if canonical_product_type in BEDDING_CANONICAL_TYPES or category == "bedding" else 0.05
    refined = np.zeros_like(mask, dtype=bool)
    inner_y0 = y0 + int(round(height * trim_top_ratio))
    inner_x0 = x0 + int(round(width * trim_side_ratio))
    inner_x1 = x1 - int(round(width * trim_side_ratio))
    refined[max(y0, inner_y0) : y1, max(x0, inner_x0) : max(x0 + 1, min(x1, inner_x1))] = (
        mask[max(y0, inner_y0) : y1, max(x0, inner_x0) : max(x0 + 1, min(x1, inner_x1))]
    )
    refined = _keep_primary_center_component(refined)
    refined = _remove_small_border_components(refined)
    min_ratio = 0.28 if canonical_product_type == "pet bed" or category == "pet home" else 0.5
    if refined.any() and refined.sum() >= max(128, int(mask.sum() * min_ratio)):
        return refined
    return mask


def repair_rigid_body_notches(
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
) -> np.ndarray:
    if not mask.any():
        return mask
    if canonical_product_type not in DRINKWARE_CANONICAL_TYPES and category != "drinkware":
        return mask
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    shoulder_row = _estimate_body_shoulder_row(mask)
    body_mask = np.zeros_like(mask, dtype=bool)
    body_mask[max(y0, shoulder_row) : y1, :] = mask[max(y0, shoulder_row) : y1, :]
    body_mask = _keep_primary_center_component(body_mask)
    if not body_mask.any():
        return mask

    row_widths = []
    dominant_rows: list[tuple[int, int, int]] = []
    for row in range(max(y0, shoulder_row), y1):
        row_xs = np.nonzero(body_mask[row])[0]
        if len(row_xs) >= 2:
            left = int(row_xs.min())
            right = int(row_xs.max())
            width = float(right - left + 1)
            row_widths.append(width)
            dominant_rows.append((row, left, right))
    if not row_widths:
        return mask
    dominant_width = float(np.percentile(np.asarray(row_widths, dtype=np.float32), 70))
    if dominant_width <= 0:
        return mask
    stable_rows = [(row, left, right) for row, left, right in dominant_rows if (right - left + 1) >= dominant_width * 0.72]
    if not stable_rows:
        stable_rows = dominant_rows
    stable_left = int(round(float(np.median([left for _, left, _ in stable_rows]))))
    stable_right = int(round(float(np.median([right for _, _, right in stable_rows]))))

    filled_body = body_mask.copy()
    for row in range(max(y0, shoulder_row), y1):
        row_xs = np.nonzero(body_mask[row])[0]
        if len(row_xs) < 2:
            continue
        row_width = float(row_xs.max() - row_xs.min() + 1)
        if row_width < dominant_width * 0.58:
            continue
        left = min(int(row_xs.min()), stable_left)
        right = max(int(row_xs.max()), stable_right)
        filled_body[row, left : right + 1] = True

    candidate = (mask & ~body_mask) | filled_body
    added = candidate & ~mask
    if not added.any():
        return mask
    if float(added.sum()) > float(mask.sum()) * 0.12:
        return mask
    return candidate


def _remove_small_border_components(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    components = _mask_components(mask, min_pixels=max(24, int(mask.sum() * 0.01)))
    if not components:
        return mask
    kept = np.zeros_like(mask, dtype=bool)
    total_pixels = float(mask.sum())
    for component in components:
        ys, xs = np.nonzero(component)
        if len(xs) == 0:
            continue
        touches_border = bool(
            ys.min() == 0
            or xs.min() == 0
            or ys.max() == mask.shape[0] - 1
            or xs.max() == mask.shape[1] - 1
        )
        area_ratio = float(component.sum()) / max(total_pixels, 1.0)
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        thinness = min(width, height) / max(width, height, 1.0)
        if touches_border and area_ratio <= 0.12 and thinness <= 0.22:
            continue
        kept |= component
    return kept if kept.any() else mask


def _keep_primary_center_component(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    components = _mask_components(mask, min_pixels=max(32, int(mask.sum() * 0.02)))
    if len(components) <= 1:
        return mask
    ys, xs = np.nonzero(mask)
    center_x = float(xs.min() + xs.max()) / 2.0
    center_y = float(ys.min() + ys.max()) / 2.0
    full_width = float(xs.max() - xs.min() + 1)
    full_height = float(ys.max() - ys.min() + 1)
    scored: list[tuple[float, np.ndarray]] = []
    for component in components:
        comp_ys, comp_xs = np.nonzero(component)
        comp_center_x = float(comp_xs.min() + comp_xs.max()) / 2.0
        comp_center_y = float(comp_ys.min() + comp_ys.max()) / 2.0
        normalized_distance = (
            abs(comp_center_x - center_x) / max(full_width, 1.0)
            + abs(comp_center_y - center_y) / max(full_height, 1.0)
        )
        score = float(component.sum()) - normalized_distance * mask.sum() * 0.35
        scored.append((score, component))
    return max(scored, key=lambda item: item[0])[1]


def suppress_interior_cavity_contamination(
    source: np.ndarray,
    mask: np.ndarray,
    *,
    category: str,
    canonical_product_type: str,
) -> np.ndarray:
    if canonical_product_type != "shoe" and category != "footwear":
        return mask
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    width = float(x1 - x0)
    height = float(y1 - y0)
    if height / max(width, 1.0) > 1.0:
        return mask
    lower_start = y0 + int(round(height * 0.28))
    lower_mask = np.zeros_like(mask, dtype=bool)
    lower_mask[lower_start:y1, :] = mask[lower_start:y1, :]
    lower_mask = _erode_mask(lower_mask, steps=2)
    if not lower_mask.any():
        return mask
    body_pixels = source[lower_mask]
    if len(body_pixels) < 64:
        return mask
    body_rgb = np.mean(body_pixels, axis=0)
    interior_mask = _erode_mask(mask, steps=3)
    upper_end = y0 + int(round(height * 0.38))
    upper_band = np.zeros_like(mask, dtype=bool)
    upper_band[y0:upper_end, :] = interior_mask[y0:upper_end, :]
    if not upper_band.any():
        return mask
    color_distance = np.sqrt(np.sum((source - body_rgb) ** 2, axis=2))
    candidate_mask = upper_band & (color_distance >= 48.0)
    components = _mask_components(candidate_mask, min_pixels=max(18, int(mask.sum() * 0.008)))
    if not components:
        return mask
    cleaned = mask.copy()
    removed_pixels = 0
    for component in components:
        comp_ys, comp_xs = np.nonzero(component)
        if len(comp_xs) == 0:
            continue
        component_height = float(comp_ys.max() - comp_ys.min() + 1)
        component_width = float(comp_xs.max() - comp_xs.min() + 1)
        area_ratio = float(component.sum()) / max(float(mask.sum()), 1.0)
        if area_ratio > 0.12:
            continue
        if component_height < max(6.0, height * 0.06):
            continue
        if component_width < max(6.0, width * 0.06):
            continue
        cleaned &= ~component
        removed_pixels += int(component.sum())
    if removed_pixels == 0:
        return mask
    if cleaned.sum() >= int(mask.sum() * 0.72):
        return cleaned
    return mask


def render_review_board(report_rows: Sequence[dict[str, Any]], destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    board_root = destination_path.parent

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in report_rows:
        grouped.setdefault(str(row["id"]), {})[str(row["line"])] = row

    cards: list[str] = []
    for seed_id, rows in grouped.items():
        baseline = rows.get("baseline")
        business = rows.get("business_prior")
        exemplar = baseline or business
        if exemplar is None:
            continue
        baseline_consistency = baseline.get("category_consistency", {}) if baseline else {}
        business_consistency = business.get("category_consistency", {}) if business else {}
        baseline_semantics = baseline.get("semantic_plausibility", {}) if baseline else {}
        business_semantics = business.get("semantic_plausibility", {}) if business else {}
        baseline_evidence = baseline.get("evidence_consistency", {}) if baseline else {}
        business_evidence = business.get("evidence_consistency", {}) if business else {}
        baseline_warning = _consistency_label(baseline_consistency)
        business_warning = _consistency_label(business_consistency)
        baseline_semantic_label = _semantic_label(baseline_semantics)
        business_semantic_label = _semantic_label(business_semantics)
        baseline_evidence_label = _evidence_label(baseline_evidence)
        business_evidence_label = _evidence_label(business_evidence)
        source_src = _board_image_reference(
            board_root,
            exemplar.get("source_image_path"),
            asset_group="source",
            asset_stem=f"{seed_id}.source",
        )
        crop_src = _board_image_reference(
            board_root,
            exemplar.get("crop_path"),
            asset_group="crop",
            asset_stem=f"{seed_id}.crop",
        )
        baseline_src = _board_image_reference(
            board_root,
            None if baseline is None else baseline.get("output_path"),
            asset_group="generated",
            asset_stem=f"{seed_id}.baseline",
        )
        business_src = _board_image_reference(
            board_root,
            None if business is None else business.get("output_path"),
            asset_group="generated",
            asset_stem=f"{seed_id}.business_prior",
        )

        cards.append(
            f"""
            <section class="card">
              <h2>{seed_id}</h2>
              <p>{_html_escape(str(exemplar["product_title"]))}</p>
              <p class="meta">Expected category: {_html_escape(str(exemplar.get("expected_category", "product")))} | Canonical type: {_html_escape(str(exemplar.get("canonical_product_type", "product")))} | Scene family: {_html_escape(str(exemplar.get("scene_family", "editorial_interior")))} | Support relation: {_html_escape(str(exemplar.get("support_relation", "resting_on_surface")))} | Weak shape evidence: {_html_escape(str(exemplar.get("weak_shape_evidence", False)))}</p>
              <div class="grid">
                <figure><figcaption>Source</figcaption><img src="{_html_escape(source_src)}" /></figure>
                <figure><figcaption>Crop</figcaption><img src="{_html_escape(crop_src)}" /></figure>
                <figure><figcaption>Baseline {_html_escape(baseline_warning)} {_html_escape(baseline_semantic_label)} {_html_escape(baseline_evidence_label)}</figcaption><img class="{_combined_flag_classes(baseline_consistency, baseline_semantics, baseline_evidence)}" src="{_html_escape(baseline_src)}" /></figure>
                <figure><figcaption>Business Prior {_html_escape(business_warning)} {_html_escape(business_semantic_label)} {_html_escape(business_evidence_label)}</figcaption><img class="{_combined_flag_classes(business_consistency, business_semantics, business_evidence)}" src="{_html_escape(business_src)}" /></figure>
              </div>
            </section>
            """
        )

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Human Review Board</title>
      <style>
        body {{ font-family: Helvetica, Arial, sans-serif; background: #f3efe8; color: #1b1b1b; margin: 0; padding: 24px; }}
        h1 {{ margin: 0 0 24px; }}
        .card {{ background: #fffdf8; border: 1px solid #d9d0c3; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
        .meta {{ font-size: 13px; color: #4b4b4b; margin: 6px 0 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
        figure {{ margin: 0; }}
        figcaption {{ font-size: 12px; font-weight: 700; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em; }}
        img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #d9d0c3; background: #f4f0e8; }}
        img.flagged {{ border-color: #b42318; box-shadow: 0 0 0 2px rgba(180, 35, 24, 0.14); }}
      </style>
    </head>
    <body>
      <h1>Human Review Board</h1>
      {''.join(cards)}
    </body>
    </html>
    """
    destination_path.write_text(html, encoding="utf-8")
    return destination_path


def sanitize_review_rows_for_bundle(
    report_rows: Sequence[dict[str, Any]],
    destination_dir: str | Path,
) -> list[dict[str, Any]]:
    board_root = Path(destination_dir)
    sanitized_rows: list[dict[str, Any]] = []
    for row in report_rows:
        seed_id = str(row.get("id", "seed"))
        line_name = str(row.get("line", "line"))
        sanitized = copy.deepcopy(row)
        sanitized["source_image_path"] = _board_image_reference(
            board_root,
            sanitized.get("source_image_path"),
            asset_group="source",
            asset_stem=f"{seed_id}.source",
        )
        sanitized["crop_path"] = _board_image_reference(
            board_root,
            sanitized.get("crop_path"),
            asset_group="crop",
            asset_stem=f"{seed_id}.crop",
        )
        sanitized["output_path"] = _board_image_reference(
            board_root,
            sanitized.get("output_path"),
            asset_group="generated",
            asset_stem=f"{seed_id}.{line_name}",
        )
        observed_evidence = sanitized.get("observed_evidence")
        if isinstance(observed_evidence, dict):
            for field_name, asset_group, asset_stem in (
                ("reference_crop_path", "reference", f"{seed_id}.evidence_crop"),
                ("reference_cutout_path", "reference", f"{seed_id}.evidence_cutout"),
                ("reference_silhouette_path", "reference", f"{seed_id}.evidence_silhouette"),
                ("reference_mask_path", "reference", f"{seed_id}.evidence_mask"),
            ):
                observed_evidence[field_name] = _board_image_reference(
                    board_root,
                    observed_evidence.get(field_name),
                    asset_group=asset_group,
                    asset_stem=asset_stem,
                )
        prompt_payload = sanitized.get("prompt")
        if isinstance(prompt_payload, dict):
            sanitized_references: list[dict[str, Any]] = []
            for reference_index, reference in enumerate(prompt_payload.get("reference_images", ())):
                sanitized_reference = dict(reference)
                sanitized_reference["path"] = _board_image_reference(
                    board_root,
                    sanitized_reference.get("path"),
                    asset_group="reference",
                    asset_stem=f"{seed_id}.{line_name}.reference_{reference_index}",
                )
                sanitized_references.append(sanitized_reference)
            prompt_payload["reference_images"] = sanitized_references
        evidence_consistency = sanitized.get("evidence_consistency")
        if isinstance(evidence_consistency, dict):
            evidence_consistency["reference_image_path"] = _board_image_reference(
                board_root,
                evidence_consistency.get("reference_image_path"),
                asset_group="reference",
                asset_stem=f"{seed_id}.{line_name}.reference_image",
            )
            evidence_consistency["focus_crop_path"] = _board_image_reference(
                board_root,
                evidence_consistency.get("focus_crop_path"),
                asset_group="focus",
                asset_stem=f"{seed_id}.{line_name}.focus_crop",
            )
            evidence_consistency["focus_mask_path"] = _board_image_reference(
                board_root,
                evidence_consistency.get("focus_mask_path"),
                asset_group="focus",
                asset_stem=f"{seed_id}.{line_name}.focus_mask",
            )
        sanitized_rows.append(sanitized)
    return sanitized_rows


def render_upstream_review_board(report_rows: Sequence[dict[str, Any]], destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    board_root = destination_path.parent

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in report_rows:
        grouped.setdefault(str(row["id"]), {})[str(row["line"])] = row

    cards: list[str] = []
    for seed_id, rows in grouped.items():
        baseline = rows.get("baseline")
        business = rows.get("business_prior")
        exemplar = baseline or business
        if exemplar is None:
            continue
        source_src = _board_image_reference(
            board_root,
            exemplar.get("source_image_path"),
            asset_group="source",
            asset_stem=f"{seed_id}.source",
        )
        crop_src = _board_image_reference(
            board_root,
            exemplar.get("crop_path"),
            asset_group="crop",
            asset_stem=f"{seed_id}.crop",
        )
        overlay_src = _board_image_reference(
            board_root,
            exemplar.get("overlay_path"),
            asset_group="overlay",
            asset_stem=f"{seed_id}.overlay",
        )
        observed_evidence = _sanitize_upstream_observed_evidence(
            board_root,
            seed_id=seed_id,
            payload=exemplar.get("observed_evidence", {}),
        )
        evidence_payload = {
            "selected_phrase": exemplar.get("selected_phrase"),
            "selected_confidence": exemplar.get("selected_confidence"),
            "category": exemplar.get("expected_category"),
            "canonical_product_type": exemplar.get("canonical_product_type"),
            "support_mode": exemplar.get("support_mode"),
            "interaction_mode": exemplar.get("interaction_mode"),
            "style_persona": exemplar.get("style_persona"),
            "stable_base": exemplar.get("stable_base"),
            "rigid_vs_soft": exemplar.get("rigid_vs_soft"),
            "weak_shape_evidence": exemplar.get("weak_shape_evidence"),
            "observed_evidence": observed_evidence,
        }
        baseline_payload = {
            "scene_family": None if baseline is None else baseline.get("scene_family"),
            "support_relation": None if baseline is None else baseline.get("support_relation"),
            "candidate_prompts": [] if baseline is None else _sanitize_upstream_candidate_prompts(
                board_root,
                seed_id=seed_id,
                line_name="baseline",
                candidate_prompts=baseline.get("candidate_prompts", []),
            ),
        }
        business_payload = {
            "scene_family": None if business is None else business.get("scene_family"),
            "support_relation": None if business is None else business.get("support_relation"),
            "style_atoms": [] if business is None else business.get("style_atoms", []),
            "semantic_constraints": [] if business is None else business.get("semantic_constraints", []),
            "retrieval_metadata": {} if business is None else business.get("retrieval_metadata", {}),
            "candidate_prompts": [] if business is None else _sanitize_upstream_candidate_prompts(
                board_root,
                seed_id=seed_id,
                line_name="business_prior",
                candidate_prompts=business.get("candidate_prompts", []),
            ),
        }

        cards.append(
            f"""
            <section class="card">
              <h2>{seed_id}</h2>
              <p>{_html_escape(str(exemplar["product_title"]))}</p>
              <p class="meta">Expected category: {_html_escape(str(exemplar.get("expected_category", "product")))} | Canonical type: {_html_escape(str(exemplar.get("canonical_product_type", "product")))} | Support mode: {_html_escape(str(exemplar.get("support_mode", "unknown")))} | Weak shape evidence: {_html_escape(str(exemplar.get("weak_shape_evidence", False)))}</p>
              <div class="image-grid">
                <figure><figcaption>Source</figcaption><img src="{_html_escape(source_src)}" /></figure>
                <figure><figcaption>Crop</figcaption><img src="{_html_escape(crop_src)}" /></figure>
                <figure><figcaption>Localization Overlay</figcaption><img src="{_html_escape(overlay_src)}" /></figure>
              </div>
              <div class="text-grid">
                <section>
                  <h3>Observed Evidence</h3>
                  <pre>{_html_escape(json.dumps(evidence_payload, indent=2, ensure_ascii=True))}</pre>
                </section>
                <section>
                  <h3>Baseline Prompt Inputs</h3>
                  <pre>{_html_escape(json.dumps(baseline_payload, indent=2, ensure_ascii=True))}</pre>
                </section>
                <section>
                  <h3>Business-Prior Prompt Inputs</h3>
                  <pre>{_html_escape(json.dumps(business_payload, indent=2, ensure_ascii=True))}</pre>
                </section>
              </div>
            </section>
            """
        )

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Upstream Human Review Board</title>
      <style>
        body {{ font-family: Helvetica, Arial, sans-serif; background: #f3efe8; color: #1b1b1b; margin: 0; padding: 24px; }}
        h1 {{ margin: 0 0 24px; }}
        h3 {{ margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; }}
        .card {{ background: #fffdf8; border: 1px solid #d9d0c3; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
        .meta {{ font-size: 13px; color: #4b4b4b; margin: 6px 0 14px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }}
        .text-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
        figure {{ margin: 0; }}
        figcaption {{ font-size: 12px; font-weight: 700; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em; }}
        img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid #d9d0c3; background: #f4f0e8; }}
        pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f3eb; border: 1px solid #e0d7ca; border-radius: 10px; padding: 12px; font-size: 12px; line-height: 1.4; }}
        @media (max-width: 1100px) {{
          .image-grid, .text-grid {{ grid-template-columns: 1fr; }}
        }}
      </style>
    </head>
    <body>
      <h1>Upstream Human Review Board</h1>
      {''.join(cards)}
    </body>
    </html>
    """
    destination_path.write_text(html, encoding="utf-8")
    return destination_path


def assess_evidence_consistency(
    image_path: str | Path,
    localized: LocalizedProduct,
    *,
    backbone: VisionBackbone,
    generated_localizer: Any,
    product_photo_factory: Any,
    save_artifacts: Any,
    select_mask: Any,
    focus_artifacts: dict[str, Path] | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    evidence = localized.identity.observed_evidence
    if focus_artifacts is None:
        focus_artifacts = extract_generated_focus_artifacts(
            image_path,
            localized,
            generated_localizer=generated_localizer,
            product_photo_factory=product_photo_factory,
            save_artifacts=save_artifacts,
            select_mask=select_mask,
        )
    focus_crop_path = None if focus_artifacts is None else focus_artifacts.get("crop_path")
    focus_mask_path = None if focus_artifacts is None else focus_artifacts.get("mask_path")
    comparison_image = focus_crop_path or image_path
    generated_embedding = np.asarray(backbone.encode_image(comparison_image), dtype=np.float32)

    reference_image = (
        Path(evidence.reference_cutout_path)
        if evidence.reference_cutout_path
        else (Path(localized.crop_path) if localized.crop_path else Path(localized.source_image))
    )
    reference_embedding = np.asarray(backbone.encode_image(reference_image), dtype=np.float32)
    image_similarity = float(np.dot(reference_embedding, generated_embedding))

    reference_profile = build_visual_evidence_profile(
        Path(localized.source_image),
        (
            Path(evidence.reference_mask_path)
            if evidence.reference_mask_path
            else (Path(localized.mask_path) if localized.mask_path else None)
        ),
        category=localized.identity.category,
        canonical_product_type=localized.identity.canonical_product_type,
    )
    generated_profile = build_visual_evidence_profile(
        image_path,
        focus_mask_path,
        category=localized.identity.category,
        canonical_product_type=localized.identity.canonical_product_type,
    )
    generated_palette = list(generated_profile["palette"])
    observed_coverage_class = generated_profile.get("coverage_class")
    color_histogram_similarity = histogram_similarity_score(
        reference_profile["color_histogram"],
        generated_profile["color_histogram"],
    )
    palette_overlap = palette_overlap_score(evidence.palette, generated_palette)
    dominant_body_color_alignment = compare_dominant_body_color_alignment(evidence, generated_profile)
    dominant_body_value_alignment = compare_dominant_body_value_alignment(reference_profile, generated_profile)
    coverage_alignment = compare_coverage_alignment(evidence, generated_profile)
    upper_region_alignment = compare_upper_region_alignment(evidence, generated_profile)
    lower_region_alignment = compare_lower_region_alignment(evidence, generated_profile)
    edge_profile_alignment = compare_edge_profile_alignment(evidence, generated_profile)
    soft_structure_alignment = compare_soft_structure_alignment(evidence, generated_profile)
    trim_alignment = compare_trim_alignment(evidence, generated_profile)
    shape_alignment = compare_shape_alignment(evidence, generated_profile)
    compact_product_focus_required = identity_prefers_compact_hand_focus(localized.identity)
    product_prominence_alignment = (
        compare_compact_product_prominence(generated_profile) if compact_product_focus_required else 0.5
    )
    wardrobe_color_spill_flag = False
    wardrobe_color_spill_metrics: dict[str, float] = {}
    if focus_mask_path is not None:
        wardrobe_color_spill_flag, wardrobe_color_spill_metrics = detect_compact_accessory_wardrobe_color_spill(
            image_path,
            focus_mask_path=Path(focus_mask_path),
            localized=localized,
        )

    canonical_type = str(localized.identity.canonical_product_type or "").strip().lower()
    category = str(localized.identity.category or "").strip().lower()
    upper_structural_types = {"backpack", "wallet", "table lamp", "water bottle", "mug", "blender", "coffee maker", "slow cooker", "food chopper"}
    lower_structural_types = {"table lamp", "shoe", "water bottle", "mug", "blender", "coffee maker", "slow cooker", "food chopper", "office chair"}
    expected_dominant_color = extract_dominant_body_color(evidence)
    soft_textile_color_lock = identity_has_chromatic_soft_textile_lock(localized.identity)
    low_profile_soft_lock = identity_has_low_profile_soft_structure(localized.identity)
    if canonical_type not in upper_structural_types and category not in {"bag", "drinkware", "kitchen appliance"}:
        upper_region_alignment = 0.5
    if canonical_type not in lower_structural_types and category not in {"footwear", "drinkware", "kitchen appliance", "furniture"}:
        lower_region_alignment = 0.5

    positive_prompts = build_evidence_prompts(localized)
    contradiction_prompts = build_contradiction_prompts(localized)
    prompt_alignment = max(
        (
            float(np.dot(generated_embedding, np.asarray(embedding, dtype=np.float32)))
            for embedding in backbone.encode_texts(positive_prompts)
        ),
        default=0.0,
    )
    contradiction_alignment = max(
        (
            float(np.dot(generated_embedding, np.asarray(embedding, dtype=np.float32)))
            for embedding in backbone.encode_texts(contradiction_prompts)
        ),
        default=0.0,
    )

    score = max(
        0.0,
        min(
            1.0,
            0.2 * image_similarity
            + 0.12 * color_histogram_similarity
            + 0.08 * palette_overlap
            + 0.14 * dominant_body_color_alignment
            + 0.1 * dominant_body_value_alignment
            + 0.11 * coverage_alignment
            + 0.1 * upper_region_alignment
            + 0.08 * lower_region_alignment
            + 0.06 * edge_profile_alignment
            + 0.14 * soft_structure_alignment
            + 0.05 * trim_alignment
            + 0.05 * shape_alignment
            + 0.08 * product_prominence_alignment
            + 0.17 * prompt_alignment
            - 0.15 * contradiction_alignment,
        ),
    )
    if soft_textile_color_lock:
        score = max(0.0, min(1.0, score + 0.08 * dominant_body_color_alignment - 0.04))
    if low_profile_soft_lock:
        score = max(0.0, min(1.0, score + 0.08 * edge_profile_alignment + 0.1 * soft_structure_alignment - 0.06))
    if (
        evidence.coverage_class == "low_variation_surface"
        and localized.identity.category in {"bedding", "pet home", "home decor"}
        and observed_coverage_class in {"localized_visible_pattern", "broad_visible_surface_pattern", "full_visible_surface_pattern"}
    ):
        score = max(0.0, score - 0.22)
    if wardrobe_color_spill_flag:
        score = max(0.0, score - 0.22)
    compact_prominence_threshold = 0.45
    if (
        compact_product_focus_required
        and image_similarity >= 0.55
        and dominant_body_color_alignment >= 0.85
        and coverage_alignment >= 0.8
        and shape_alignment >= 0.8
    ):
        compact_prominence_threshold = 0.25
    if compact_product_focus_required and product_prominence_alignment < compact_prominence_threshold:
        score = max(0.0, score - 0.18)
    min_soft_structure_alignment = 0.75 if evidence.soft_structure_class == "flat_surface" else 0.25
    min_edge_profile_alignment = 0.55 if low_profile_soft_lock else 0.2
    soft_textile_color_threshold = 0.72
    if soft_textile_color_lock and palette_overlap >= 0.66 and dominant_body_value_alignment >= 0.8:
        soft_textile_color_threshold = 0.52
    if low_profile_soft_lock and image_similarity >= 0.72 and edge_profile_alignment >= 0.85:
        soft_textile_color_threshold = min(soft_textile_color_threshold, 0.55)
    low_profile_structure_ok = (
        soft_structure_alignment >= min_soft_structure_alignment
        or evidence.soft_structure_note is None
    )
    if (
        low_profile_soft_lock
        and not low_profile_structure_ok
        and edge_profile_alignment >= 0.85
        and shape_alignment >= 0.6
        and image_similarity >= 0.7
    ):
        low_profile_structure_ok = True
    dominant_body_color_ok = dominant_body_color_alignment >= 0.6 or expected_dominant_color is None
    if (
        low_profile_soft_lock
        and not dominant_body_color_ok
        and palette_overlap >= 0.66
        and image_similarity >= 0.7
    ):
        dominant_body_color_ok = dominant_body_color_alignment >= 0.25
    dominant_body_color_ok = soft_surface_color_ok(
        base_ok=dominant_body_color_ok,
        low_profile_soft_lock=low_profile_soft_lock,
        image_similarity=image_similarity,
        dominant_body_color_alignment=dominant_body_color_alignment,
        edge_profile_alignment=edge_profile_alignment,
    )
    coverage_ok = soft_surface_coverage_ok(
        base_ok=(coverage_alignment >= 0.25 or evidence.coverage_class is None),
        soft_textile_color_lock=soft_textile_color_lock,
        low_profile_soft_lock=low_profile_soft_lock,
        image_similarity=image_similarity,
        dominant_body_color_alignment=dominant_body_color_alignment,
        edge_profile_alignment=edge_profile_alignment,
        soft_structure_alignment=soft_structure_alignment,
    )
    dominant_body_value_ok = soft_surface_value_ok(
        base_ok=(dominant_body_value_alignment >= 0.7 or expected_dominant_color is None),
        soft_textile_color_lock=soft_textile_color_lock,
        low_profile_soft_lock=low_profile_soft_lock,
        image_similarity=image_similarity,
        dominant_body_color_alignment=dominant_body_color_alignment,
        dominant_body_value_alignment=dominant_body_value_alignment,
        edge_profile_alignment=edge_profile_alignment,
    )
    is_consistent = bool(
        score >= 0.42
        and contradiction_alignment <= prompt_alignment + 0.02
        and not wardrobe_color_spill_flag
        and coverage_ok
        and dominant_body_color_ok
        and dominant_body_value_ok
        and (dominant_body_color_alignment >= soft_textile_color_threshold or not soft_textile_color_lock)
        and (
            upper_region_alignment >= 0.22
            or evidence.upper_region_note is None
            or (canonical_type not in upper_structural_types and category not in {"bag", "drinkware", "kitchen appliance"})
        )
        and (
            lower_region_alignment >= 0.22
            or evidence.lower_region_note is None
            or (canonical_type not in lower_structural_types and category not in {"footwear", "drinkware", "kitchen appliance", "furniture"})
        )
        and (edge_profile_alignment >= min_edge_profile_alignment or evidence.edge_profile_note is None)
        and low_profile_structure_ok
        and (product_prominence_alignment >= compact_prominence_threshold or not compact_product_focus_required)
    )
    warning = None
    if not is_consistent:
        if wardrobe_color_spill_flag:
            warning = "generated view spreads the product's dominant color onto a large non-product clothing or prop region"
        elif compact_product_focus_required and product_prominence_alignment < compact_prominence_threshold:
            warning = "generated view keeps the compact hand-held product too small or torso-dominated instead of making the product visually primary"
        else:
            warning = "generated view appears weakly aligned with observed source evidence"
    return {
        "surface_scope": evidence.surface_scope,
        "uncertainty_level": evidence.uncertainty_level,
        "reference_image_path": str(reference_image),
        "focus_crop_path": None if focus_crop_path is None else str(focus_crop_path),
        "focus_mask_path": None if focus_mask_path is None else str(focus_mask_path),
        "image_similarity": round(image_similarity, 4),
        "color_histogram_similarity": round(color_histogram_similarity, 4),
        "palette_overlap": round(palette_overlap, 4),
        "dominant_body_color_alignment": round(dominant_body_color_alignment, 4),
        "dominant_body_value_alignment": round(dominant_body_value_alignment, 4),
        "coverage_alignment": round(coverage_alignment, 4),
        "upper_region_alignment": round(upper_region_alignment, 4),
        "lower_region_alignment": round(lower_region_alignment, 4),
        "edge_profile_alignment": round(edge_profile_alignment, 4),
        "soft_structure_alignment": round(soft_structure_alignment, 4),
        "trim_alignment": round(trim_alignment, 4),
        "shape_alignment": round(shape_alignment, 4),
        "compact_product_focus_required": compact_product_focus_required,
        "product_prominence_alignment": round(product_prominence_alignment, 4),
        "wardrobe_color_spill_flag": wardrobe_color_spill_flag,
        "wardrobe_color_spill_metrics": {
            key: round(value, 4) for key, value in wardrobe_color_spill_metrics.items()
        },
        "prompt_alignment": round(prompt_alignment, 4),
        "contradiction_alignment": round(contradiction_alignment, 4),
        "score": round(score, 4),
        "is_consistent": is_consistent,
        "warning": warning,
    }


def extract_generated_focus_artifacts(
    image_path: Path,
    localized: LocalizedProduct,
    *,
    generated_localizer: Any,
    product_photo_factory: Any,
    save_artifacts: Any,
    select_mask: Any,
) -> dict[str, Path] | None:
    cache_dir = image_path.parent / "evidence_focus"
    canonical_type = localized.identity.canonical_product_type or localized.identity.category
    focus_hint_phrases = tuple(
        _dedupe_strings(
            [
                canonical_type,
                localized.identity.category,
            ]
        )
    )
    photo = product_photo_factory(
        image_path=image_path,
        product_id=image_path.stem,
        # Use a product-only query here instead of the merchant title.
        # Full catalog titles often contain brand, demographic, or style tokens
        # that let body- or scene-scale masks outrank the actual product in
        # generated-output relocalization.
        title=canonical_type,
        hint_phrases=focus_hint_phrases,
        metadata={
            "category": localized.identity.category,
            "canonical_product_type": canonical_type,
        },
    )
    result = generated_localizer.localize(photo)
    selected = _select_generated_focus_mask(
        result,
        localized,
        default_selector=select_mask,
    )
    if selected is None:
        return None
    artifacts = save_artifacts(result, cache_dir, selected_mask=selected)
    if artifacts is None:
        return None
    return {
        "crop_path": Path(artifacts.crop_path),
        "mask_path": Path(artifacts.mask_path),
        "overlay_path": Path(artifacts.overlay_path),
    }


def _select_generated_focus_mask(
    result: Any,
    localized: LocalizedProduct,
    *,
    default_selector: Any,
) -> Any:
    selected = default_selector(result)
    masks = list(getattr(result, "masks", ()) or ())
    if not masks:
        return selected
    category = str(localized.identity.category or "").strip().lower()
    canonical_type = str(localized.identity.canonical_product_type or category).strip().lower()
    soft_nonhuman = bool(
        not localized.identity.requires_human_model
        and (
            category in {"bedding", "pet home", "home decor"}
            or canonical_type in (BEDDING_CANONICAL_TYPES | {"pet bed", "decorative pillow"})
        )
    )
    if not soft_nonhuman or len(masks) == 1:
        return selected
    with Image.open(result.photo.image_path) as handle:
        source = np.asarray(handle.convert("RGB"), dtype=np.float32)
        image_width, image_height = handle.size
    image_area = float(max(1, image_width * image_height))
    evidence = localized.identity.observed_evidence
    expected_colors = _dedupe_strings(
        [
            extract_dominant_body_color(evidence) or "",
            *evidence.palette[:3],
            *evidence.structural_palette[:3],
        ]
    )
    scored_masks: list[tuple[float, Any]] = []
    for mask in masks:
        raster = _rasterize_focus_candidate_mask(mask, image_width=image_width, image_height=image_height)
        if raster is None or not raster.any():
            continue
        area_ratio = float(getattr(mask, "area_pixels", 0.0)) / image_area
        fill_ratio = float(getattr(mask, "area_pixels", 0.0)) / max(float(getattr(mask, "box").area), 1.0)
        aspect_ratio = float(getattr(mask, "box").width) / max(float(getattr(mask, "box").height), 1.0)
        candidate_color = _dominant_focus_candidate_color(source, raster)
        score = 1.1 * float(getattr(mask, "confidence", 0.0)) + 1.8 * area_ratio + 0.45 * fill_ratio
        score += _generated_focus_color_alignment_score(candidate_color, expected_colors)
        if evidence.soft_structure_class in {"flat_surface", "low_perimeter_relief"}:
            if area_ratio < 0.1:
                score -= 1.35
            if area_ratio < 0.06:
                score -= 0.65
            if fill_ratio < 0.16:
                score -= 0.55
            if not (0.75 <= aspect_ratio <= 5.5):
                score -= 0.18
        if _focus_candidate_touches_border(getattr(mask, "box"), image_width=image_width, image_height=image_height) and area_ratio < 0.1:
            score -= 0.18
        scored_masks.append((score, mask))
    if not scored_masks:
        return selected
    best_mask = max(scored_masks, key=lambda item: item[0])[1]
    if selected is None:
        return best_mask
    if _generated_focus_candidate_is_artifact_like(
        selected,
        image_width=image_width,
        image_height=image_height,
        image_area=image_area,
        source=source,
        expected_colors=expected_colors,
    ):
        return best_mask
    return selected


def _rasterize_focus_candidate_mask(mask: Any, *, image_width: int, image_height: int) -> np.ndarray | None:
    polygon = getattr(mask, "polygon", None)
    if not polygon:
        return None
    canvas = Image.new("L", (image_width, image_height), 0)
    ImageDraw.Draw(canvas).polygon(polygon, fill=255)
    return np.asarray(canvas, dtype=np.uint8) > 0


def _focus_candidate_touches_border(box: Any, *, image_width: int, image_height: int) -> bool:
    return bool(
        getattr(box, "x0", 1) <= 0
        or getattr(box, "y0", 1) <= 0
        or getattr(box, "x1", image_width - 1) >= image_width
        or getattr(box, "y1", image_height - 1) >= image_height
    )


def _dominant_focus_candidate_color(source: np.ndarray, raster: np.ndarray) -> str | None:
    pixels = source[raster]
    if len(pixels) < 24:
        return None
    stride = max(1, int(math.ceil(len(pixels) / 1024.0)))
    distribution = _weighted_structural_color_distribution(pixels[::stride])
    if not distribution:
        return None
    return max(distribution.items(), key=lambda item: item[1])[0]


def _generated_focus_color_alignment_score(candidate_color: str | None, expected_colors: Sequence[str]) -> float:
    if candidate_color is None or not expected_colors:
        return 0.0
    expected = {str(color).lower() for color in expected_colors if color}
    if candidate_color in expected:
        return 0.42
    neutral_family = {"black", "gray", "white", "beige", "brown"}
    if candidate_color in neutral_family and expected.intersection(neutral_family):
        return 0.16
    return -0.48


def _generated_focus_candidate_is_artifact_like(
    mask: Any,
    *,
    image_width: int,
    image_height: int,
    image_area: float,
    source: np.ndarray,
    expected_colors: Sequence[str],
) -> bool:
    area_ratio = float(getattr(mask, "area_pixels", 0.0)) / max(image_area, 1.0)
    fill_ratio = float(getattr(mask, "area_pixels", 0.0)) / max(float(getattr(mask, "box").area), 1.0)
    raster = _rasterize_focus_candidate_mask(mask, image_width=image_width, image_height=image_height)
    if raster is None or not raster.any():
        return True
    candidate_color = _dominant_focus_candidate_color(source, raster)
    color_alignment = _generated_focus_color_alignment_score(candidate_color, expected_colors)
    if area_ratio < 0.08:
        return True
    if area_ratio < 0.1 and fill_ratio < 0.22:
        return True
    if color_alignment < -0.2 and area_ratio < 0.12:
        return True
    return False


def apply_dominant_body_color_correction(
    image_path: Path,
    *,
    mask_path: Path,
    dominant_color: str,
    coverage_class: str | None,
    category: str,
    canonical_product_type: str,
    rigid_vs_soft: str | None,
) -> None:
    if dominant_color not in EVIDENCE_COLOR_SWATCHES or not image_path.exists() or not mask_path.exists():
        return
    neutral_target = dominant_color in {"black", "white", "gray", "beige"}
    with Image.open(image_path) as image_handle:
        source = np.asarray(image_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return
    refined_mask = refine_observed_evidence_mask(
        mask,
        category=category,
        canonical_product_type=canonical_product_type,
        requires_human_model=False,
    )
    if refined_mask.any():
        mask = refined_mask
    if _mask_has_heavy_border_spill(mask):
        return
    correction_mask = mask
    if (
        rigid_vs_soft == "rigid"
        and coverage_class in {"low_variation_surface", "full_visible_surface_pattern", "broad_visible_surface_pattern"}
        and category not in {"bedding", "pet home", "home decor"}
    ):
        core_mask = _extract_core_body_mask(mask)
        if core_mask.any() and neutral_target:
            correction_mask = core_mask

    target_rgb = np.asarray(EVIDENCE_COLOR_SWATCHES[dominant_color], dtype=np.float32)
    alpha_seed = Image.fromarray((correction_mask.astype(np.uint8) * 255))
    feather_radius = max(1, int(round(min(source.shape[:2]) * 0.012)))
    alpha_map = np.asarray(alpha_seed.filter(ImageFilter.GaussianBlur(radius=feather_radius)), dtype=np.float32) / 255.0
    alpha_map = np.clip(alpha_map * mask.astype(np.float32), 0.0, 1.0)

    corrected = source.copy()
    if neutral_target:
        luma = source.mean(axis=2)
        masked_luma = luma[correction_mask]
        mean_luma = float(masked_luma.mean())
        std_luma = max(float(masked_luma.std()), 1.0)
        detail = np.clip((luma - mean_luma) / (2.0 * std_luma), -1.0, 1.0)
        detail_scale = 14.0
        anchor_rgb = np.clip(target_rgb[None, None, :] + detail[..., None] * detail_scale, 0.0, 255.0)
        blend_strength = 0.96 if dominant_color in {"black", "white"} else 0.82
        alpha = np.clip(alpha_map * blend_strength, 0.0, 1.0)
        active = alpha > 0.0
        corrected[active] = (
            corrected[active] * (1.0 - alpha[active, None])
            + anchor_rgb[active] * alpha[active, None]
        )
    else:
        target_h, target_s, _ = colorsys.rgb_to_hsv(*(target_rgb / 255.0))
        source_pixels = np.clip(source[correction_mask] / 255.0, 0.0, 1.0)
        hsv_pixels = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in source_pixels], dtype=np.float32)
        hsv_pixels[:, 0] = target_h
        hsv_pixels[:, 1] = np.clip(0.35 * target_s + 0.65 * np.maximum(hsv_pixels[:, 1], target_s * 0.45), 0.0, 1.0)
        recolored = np.asarray([colorsys.hsv_to_rgb(*pixel) for pixel in hsv_pixels], dtype=np.float32) * 255.0
        alpha = np.clip(alpha_map[correction_mask] * 0.72, 0.0, 1.0)[:, None]
        corrected[correction_mask] = corrected[correction_mask] * (1.0 - alpha) + recolored * alpha
    Image.fromarray(np.clip(corrected, 0.0, 255.0).astype(np.uint8)).save(image_path)


def maybe_repair_generated_dominant_body_color(
    image_path: Path,
    localized: LocalizedProduct,
    *,
    generated_localizer: Any,
    product_photo_factory: Any,
    save_artifacts: Any,
    select_mask: Any,
) -> None:
    evidence = localized.identity.observed_evidence
    dominant_body_color = extract_dominant_body_color(evidence)
    if dominant_body_color is None or not should_apply_post_generation_color_repair(localized.identity):
        return
    reference_profile = build_visual_evidence_profile(
        Path(localized.source_image),
        Path(evidence.reference_mask_path)
        if evidence.reference_mask_path
        else (Path(localized.mask_path) if localized.mask_path else None),
        category=localized.identity.category,
        canonical_product_type=localized.identity.canonical_product_type,
    )
    focus_artifacts = extract_generated_focus_artifacts(
        image_path,
        localized,
        generated_localizer=generated_localizer,
        product_photo_factory=product_photo_factory,
        save_artifacts=save_artifacts,
        select_mask=select_mask,
    )
    if focus_artifacts is None:
        return
    focus_mask_path = focus_artifacts["mask_path"]
    if not generated_focus_mask_is_safe_for_color_repair(
        focus_mask_path,
        category=localized.identity.category,
        canonical_product_type=localized.identity.canonical_product_type or localized.identity.category,
    ):
        return
    generated_profile = build_visual_evidence_profile(
        image_path,
        focus_mask_path,
        category=localized.identity.category,
        canonical_product_type=localized.identity.canonical_product_type,
    )
    if (
        compare_dominant_body_color_alignment(evidence, generated_profile) >= 0.6
        and compare_dominant_body_value_alignment(reference_profile, generated_profile) >= 0.7
    ):
        return
    apply_dominant_body_color_correction(
        image_path,
        mask_path=focus_mask_path,
        dominant_color=dominant_body_color,
        coverage_class=evidence.coverage_class,
        category=localized.identity.category,
        canonical_product_type=localized.identity.canonical_product_type or localized.identity.category,
        rigid_vs_soft=localized.identity.rigid_vs_soft,
    )


def generated_focus_mask_is_safe_for_color_repair(
    mask_path: Path | None,
    *,
    category: str,
    canonical_product_type: str,
) -> bool:
    if mask_path is None or not mask_path.exists():
        return False
    with Image.open(mask_path) as mask_handle:
        mask = np.asarray(mask_handle.convert("L")) > 0
    if not mask.any():
        return False
    ys, xs = np.nonzero(mask)
    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    if bbox_area <= 0:
        return False
    fill_ratio = float(mask.sum()) / bbox_area
    if (
        (canonical_product_type in STRUCTURED_DISPLAY_CANONICAL_TYPES or category in {"furniture", "home lighting"})
        and fill_ratio < 0.55
    ):
        return False
    return True


def should_apply_post_generation_color_repair(identity: ProductIdentitySpec) -> bool:
    evidence = identity.observed_evidence
    lowered_color_note = str(evidence.color_note or "").lower()
    if not should_strengthen_dominant_body_color_guidance(identity):
        return False
    if identity.category == "bag" and identity_prefers_compact_hand_focus(identity):
        return False
    carried_or_worn_accessory = identity.category in {"bag", "footwear"}
    if (
        not carried_or_worn_accessory
        and (
            identity.requires_human_model
            or identity.interaction_mode in {
                "held_in_hand",
                "worn",
                "worn_or_carried",
                "carried_or_resting",
            }
        )
    ):
        return False
    if "glazed variation" in lowered_color_note:
        return False
    if identity.category == "bedding" or (identity.canonical_product_type or "") in BEDDING_CANONICAL_TYPES:
        return False
    if evidence.soft_structure_class == "flat_surface" and identity.category in {"pet home", "home decor"}:
        return False
    return True


def build_visual_evidence_profile(
    image_path: Path,
    mask_path: Path | None,
    *,
    category: str = "product",
    canonical_product_type: str = "product",
) -> dict[str, Any]:
    core_body_ranked = infer_core_body_palette_ranked(
        image_path,
        mask_path,
        top_k=3,
    )
    structural_palette = infer_named_palette_with_strategy(
        image_path,
        mask_path,
        top_k=3,
        use_smoothed=True,
        erode_steps=2,
    )
    accent_palette = infer_named_palette_with_strategy(
        image_path,
        mask_path,
        top_k=4,
        use_smoothed=False,
        erode_steps=0,
    )
    palette = structural_palette or accent_palette
    coverage_class, coverage_ratio, _ = infer_surface_coverage_profile(
        image_path,
        mask_path,
        palette=accent_palette or palette,
        base_palette=structural_palette or palette,
    )
    boundary_color, interior_color, _, trim_confidence = infer_trim_profile(image_path, mask_path)
    shape_profile = infer_shape_profile(mask_path)
    upper_region_profile = infer_upper_region_profile(image_path, mask_path)
    lower_region_profile = infer_lower_region_profile(
        source_image=image_path,
        mask_path=mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    edge_profile = infer_edge_profile(
        source_image=image_path,
        mask_path=mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
    )
    soft_structure = infer_soft_structure_profile(
        source_image=image_path,
        mask_path=mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        edge_thickness_class=edge_profile["thickness_class"],
    )
    color_histogram = infer_named_color_histogram(image_path, mask_path)
    dominant_body_color = core_body_ranked[0][0] if core_body_ranked else (structural_palette[0] if structural_palette else None)
    soft_textile_color_note, soft_textile_palette, _ = infer_soft_textile_chromatic_override(
        source_image=image_path,
        mask_path=mask_path,
        category=category,
        canonical_product_type=canonical_product_type,
        coverage_class=coverage_class,
        palette=palette,
    )
    if soft_textile_palette:
        structural_palette = soft_textile_palette
        palette = soft_textile_palette
        dominant_body_color = soft_textile_palette[0]
    coverage_class, coverage_ratio, _, _, _, _ = correct_supported_soft_surface_inference(
        category=category,
        canonical_product_type=canonical_product_type,
        product_title="",
        hint_phrases=(),
        evidence_caption=None,
        coverage_class=coverage_class,
        coverage_ratio=coverage_ratio,
        coverage_note=None,
        pattern_note=None,
        color_note=None,
        color_confidence=None,
        palette=palette,
    )
    mean_luma = None
    mask_area_ratio = None
    bbox_area_ratio = None
    if image_path.exists():
        with Image.open(image_path) as handle:
            source = np.asarray(handle.convert("RGB"), dtype=np.float32)
        mask = _load_mask_array(mask_path, source_shape=source.shape[:2]) if mask_path else None
        pixels = source[mask] if mask is not None and mask.any() else source.reshape(-1, 3)
        if pixels.size:
            mean_luma = float(np.mean(np.mean(pixels, axis=1)))
        if mask is not None and mask.any():
            mask_area_ratio = float(mask.sum()) / float(max(mask.size, 1))
            ys, xs = np.nonzero(mask)
            if len(ys) and len(xs):
                bbox_area_ratio = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) / float(
                    max(mask.size, 1)
                )
    return {
        "palette": palette,
        "structural_palette": structural_palette,
        "accent_palette": accent_palette,
        "dominant_body_color": dominant_body_color,
        "mean_luma": mean_luma,
        "mask_area_ratio": mask_area_ratio,
        "bbox_area_ratio": bbox_area_ratio,
        "coverage_class": coverage_class,
        "coverage_ratio": coverage_ratio,
        "boundary_color": boundary_color,
        "interior_color": interior_color,
        "trim_confidence": trim_confidence,
        "aspect_ratio": shape_profile["aspect_ratio"],
        "top_width_ratio": shape_profile["top_width_ratio"],
        "form_factor_note": infer_form_factor_note(
            category=category,
            canonical_product_type=canonical_product_type,
            shape_profile=shape_profile,
            upper_region_profile=upper_region_profile,
        ),
        "upper_region_note": upper_region_profile["note"],
        "upper_region_color": upper_region_profile["upper_region_color"],
        "body_region_color": upper_region_profile["body_region_color"],
        "upper_component_state": upper_region_profile["component_state"],
        "upper_component_count": upper_region_profile["upper_component_count"],
        "lower_region_note": lower_region_profile["note"],
        "lower_region_color": lower_region_profile["lower_region_color"],
        "lower_component_state": lower_region_profile["component_state"],
        "edge_profile_note": edge_profile["note"],
        "edge_thickness_class": edge_profile["thickness_class"],
        "edge_inner_ratio": edge_profile["inner_ratio"],
        "soft_structure_note": soft_structure["note"],
        "soft_structure_class": soft_structure["structure_class"],
        "color_histogram": color_histogram,
    }


def infer_named_color_histogram(image_path: Path, mask_path: Path | None) -> dict[str, float]:
    if not image_path.exists():
        return {}
    with Image.open(image_path) as handle:
        source = np.asarray(handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(mask_path, source_shape=source.shape[:2]) if mask_path else None
    pixels = source[mask] if mask is not None and mask.any() else source.reshape(-1, 3)
    if pixels.size == 0:
        return {}
    stride = max(1, int(math.ceil(len(pixels) / 4000.0)))
    sampled = pixels[::stride]
    counts = {name: 0 for name in EVIDENCE_COLOR_SWATCHES}
    for pixel in sampled:
        counts[_nearest_color_name(pixel)] += 1
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {name: count / total for name, count in counts.items() if count}


def histogram_similarity_score(reference_histogram: dict[str, float], observed_histogram: dict[str, float]) -> float:
    if not reference_histogram or not observed_histogram:
        return 0.0
    names = set(reference_histogram) | set(observed_histogram)
    return sum(min(reference_histogram.get(name, 0.0), observed_histogram.get(name, 0.0)) for name in names)


def compare_coverage_alignment(evidence: ObservedEvidenceSpec, generated_profile: dict[str, Any]) -> float:
    expected_ratio = evidence.coverage_ratio
    observed_ratio = generated_profile.get("coverage_ratio")
    if expected_ratio is None or observed_ratio is None:
        return 0.5
    score = max(0.0, 1.0 - abs(float(expected_ratio) - float(observed_ratio)) / 0.6)
    observed_class = generated_profile.get("coverage_class")
    if evidence.coverage_class == "low_variation_surface" and observed_class in {
        "localized_visible_pattern",
        "broad_visible_surface_pattern",
        "full_visible_surface_pattern",
    }:
        score -= 0.42 if observed_class == "localized_visible_pattern" else 0.55
    if evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"} and observed_class == "low_variation_surface":
        score -= 0.35
    if evidence.coverage_class == "localized_visible_pattern" and observed_class in {
        "full_visible_surface_pattern",
        "broad_visible_surface_pattern",
    }:
        score -= 0.25
    return max(0.0, min(1.0, score))


def compare_color_names(expected_color: str | None, observed_color: str | None) -> float:
    if expected_color is None or observed_color is None:
        return 0.5
    if expected_color == observed_color:
        return 1.0
    distance = _named_color_distance(expected_color, observed_color)
    if distance <= 0:
        return 0.0
    normalized = min(distance / 255.0, 1.0)
    return max(0.0, 1.0 - normalized)


def extract_dominant_body_color(evidence: ObservedEvidenceSpec) -> str | None:
    if (
        evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
        and evidence.color_note
        and (evidence.color_confidence is None or evidence.color_confidence >= 0.64)
    ):
        colors = extract_caption_colors(evidence.color_note)
        if colors:
            return colors[0]
    if evidence.color_note and evidence.color_confidence is not None and evidence.color_confidence >= 0.68:
        colors = extract_caption_colors(evidence.color_note)
        if colors:
            return colors[0]
    if evidence.body_region_color:
        return evidence.body_region_color
    if evidence.color_note:
        colors = extract_caption_colors(evidence.color_note)
        if colors:
            return colors[0]
    if evidence.palette and evidence.color_confidence is not None and evidence.color_confidence >= 0.76:
        return evidence.palette[0]
    return None


def compare_dominant_body_color_alignment(
    evidence: ObservedEvidenceSpec,
    generated_profile: dict[str, Any],
) -> float:
    expected_color = extract_dominant_body_color(evidence)
    if expected_color is None:
        return 0.5
    observed_color = generated_profile.get("dominant_body_color")
    return compare_color_names(expected_color, observed_color)


def compare_dominant_body_value_alignment(
    reference_profile: dict[str, Any],
    generated_profile: dict[str, Any],
) -> float:
    expected_value = reference_profile.get("mean_luma")
    observed_value = generated_profile.get("mean_luma")
    if expected_value is None or observed_value is None:
        return 0.5
    return max(0.0, 1.0 - abs(float(expected_value) - float(observed_value)) / 95.0)


def compare_upper_region_alignment(evidence: ObservedEvidenceSpec, generated_profile: dict[str, Any]) -> float:
    if evidence.upper_component_state == "absent":
        observed_state = generated_profile.get("upper_component_state")
        if observed_state == "present":
            return 0.0
        if observed_state == "absent":
            return 1.0
        return 0.5
    if evidence.upper_region_note is None and evidence.upper_component_count is None:
        return 0.5
    color_score = compare_color_names(evidence.upper_region_color, generated_profile.get("upper_region_color"))
    body_score = compare_color_names(evidence.body_region_color, generated_profile.get("body_region_color"))
    expected_components = evidence.upper_component_count
    observed_components = generated_profile.get("upper_component_count")
    if expected_components is None or observed_components is None:
        component_score = 0.5
    else:
        component_score = max(0.0, 1.0 - min(abs(int(expected_components) - int(observed_components)), 3) / 3.0)
    return max(0.0, min(1.0, 0.45 * color_score + 0.25 * body_score + 0.3 * component_score))


def compare_trim_alignment(evidence: ObservedEvidenceSpec, generated_profile: dict[str, Any]) -> float:
    if evidence.trim_note is None:
        return 0.5
    boundary_score = compare_color_names(evidence.boundary_color, generated_profile.get("boundary_color"))
    interior_score = compare_color_names(evidence.interior_color, generated_profile.get("interior_color"))
    return max(0.0, min(1.0, 0.55 * boundary_score + 0.45 * interior_score))


def compare_lower_region_alignment(evidence: ObservedEvidenceSpec, generated_profile: dict[str, Any]) -> float:
    if evidence.lower_region_note is None:
        return 0.5
    observed_state = generated_profile.get("lower_component_state")
    if observed_state == "present":
        state_score = 1.0
    elif observed_state == "uncertain":
        state_score = 0.5
    else:
        state_score = 0.0
    color_score = compare_color_names(evidence.lower_region_color, generated_profile.get("lower_region_color"))
    return max(0.0, min(1.0, 0.7 * state_score + 0.3 * color_score))


def compare_edge_profile_alignment(evidence: ObservedEvidenceSpec, generated_profile: dict[str, Any]) -> float:
    if evidence.edge_profile_note is None and evidence.edge_thickness_class is None:
        return 0.5
    observed_class = generated_profile.get("edge_thickness_class")
    expected_class = evidence.edge_thickness_class
    expected_inner_ratio = evidence.edge_inner_ratio
    observed_inner_ratio = generated_profile.get("edge_inner_ratio")
    ratio_score = None
    if expected_inner_ratio is not None and observed_inner_ratio is not None:
        ratio_score = max(
            0.0,
            1.0 - abs(float(expected_inner_ratio) - float(observed_inner_ratio)) / 0.22,
        )
    if expected_class is None or observed_class is None:
        return 0.5 if ratio_score is None else ratio_score
    if expected_class == observed_class:
        return 1.0 if ratio_score is None else 0.55 + 0.45 * ratio_score
    neighbors = {
        ("low_profile_edge", "moderate_edge"),
        ("moderate_edge", "low_profile_edge"),
        ("moderate_edge", "thick_raised_edge"),
        ("thick_raised_edge", "moderate_edge"),
    }
    if (expected_class, observed_class) in neighbors:
        class_score = 0.45
        return class_score if ratio_score is None else min(class_score, ratio_score)
    return 0.0 if ratio_score is None else min(0.12, ratio_score)


def compare_compact_product_prominence(generated_profile: dict[str, Any]) -> float:
    mask_area_ratio = generated_profile.get("mask_area_ratio")
    bbox_area_ratio = generated_profile.get("bbox_area_ratio")
    scores: list[float] = []
    if mask_area_ratio is not None:
        scores.append(min(1.0, float(mask_area_ratio) / 0.03))
    if bbox_area_ratio is not None:
        scores.append(min(1.0, float(bbox_area_ratio) / 0.075))
    if not scores:
        return 0.5
    return sum(scores) / float(len(scores))


def compact_focus_alignment_threshold(evidence_consistency: dict[str, Any]) -> float:
    threshold = 0.45
    if (
        float(evidence_consistency.get("image_similarity", 0.0)) >= 0.55
        and float(evidence_consistency.get("dominant_body_color_alignment", 0.0)) >= 0.85
        and float(evidence_consistency.get("coverage_alignment", 0.0)) >= 0.8
        and float(evidence_consistency.get("shape_alignment", 0.0)) >= 0.8
    ):
        threshold = 0.25
    return threshold


def compare_soft_structure_alignment(evidence: ObservedEvidenceSpec, generated_profile: dict[str, Any]) -> float:
    if evidence.soft_structure_note is None and evidence.soft_structure_class is None:
        return 0.5
    observed_class = generated_profile.get("soft_structure_class")
    expected_class = evidence.soft_structure_class
    if expected_class is None or observed_class is None:
        return 0.5
    if expected_class == observed_class:
        return 1.0
    neighbors = {
        ("flat_surface", "low_perimeter_relief"),
        ("low_perimeter_relief", "flat_surface"),
        ("low_perimeter_relief", "raised_perimeter_relief"),
        ("raised_perimeter_relief", "low_perimeter_relief"),
    }
    if (expected_class, observed_class) in neighbors:
        if expected_class == "flat_surface":
            return 0.1
        return 0.4
    return 0.0


def soft_surface_coverage_ok(
    *,
    base_ok: bool,
    soft_textile_color_lock: bool,
    low_profile_soft_lock: bool,
    image_similarity: float,
    dominant_body_color_alignment: float,
    edge_profile_alignment: float,
    soft_structure_alignment: float,
) -> bool:
    if base_ok:
        return True
    if (
        (soft_textile_color_lock or low_profile_soft_lock)
        and image_similarity >= 0.72
        and dominant_body_color_alignment >= 0.55
        and (edge_profile_alignment >= 0.85 or soft_structure_alignment >= 0.4)
    ):
        return True
    return False


def soft_surface_value_ok(
    *,
    base_ok: bool,
    soft_textile_color_lock: bool,
    low_profile_soft_lock: bool,
    image_similarity: float,
    dominant_body_color_alignment: float,
    dominant_body_value_alignment: float,
    edge_profile_alignment: float,
) -> bool:
    if base_ok:
        return True
    if (
        low_profile_soft_lock
        and image_similarity >= 0.72
        and dominant_body_color_alignment >= 0.55
        and edge_profile_alignment >= 0.85
    ):
        return True
    if (
        soft_textile_color_lock
        and image_similarity >= 0.78
        and dominant_body_color_alignment >= 0.85
        and dominant_body_value_alignment >= 0.45
    ):
        return True
    return False


def soft_surface_color_ok(
    *,
    base_ok: bool,
    low_profile_soft_lock: bool,
    image_similarity: float,
    dominant_body_color_alignment: float,
    edge_profile_alignment: float,
) -> bool:
    if base_ok:
        return True
    if (
        low_profile_soft_lock
        and image_similarity >= 0.72
        and dominant_body_color_alignment >= 0.55
        and edge_profile_alignment >= 0.85
    ):
        return True
    return False


def compare_shape_alignment(evidence: ObservedEvidenceSpec, generated_profile: dict[str, Any]) -> float:
    scores: list[float] = []
    expected_aspect = evidence.aspect_ratio
    observed_aspect = generated_profile.get("aspect_ratio")
    if expected_aspect is not None and observed_aspect is not None:
        scores.append(max(0.0, 1.0 - abs(float(expected_aspect) - float(observed_aspect)) / 0.9))
    expected_top_width = evidence.top_width_ratio
    observed_top_width = generated_profile.get("top_width_ratio")
    if expected_top_width is not None and observed_top_width is not None:
        scores.append(max(0.0, 1.0 - abs(float(expected_top_width) - float(observed_top_width)) / 0.5))
    if not scores:
        return 0.5
    return sum(scores) / float(len(scores))


def build_evidence_prompts(localized: LocalizedProduct) -> list[str]:
    evidence = localized.identity.observed_evidence
    product = localized.identity.canonical_product_type or localized.identity.category or "product"
    prompts = [f"a campaign image of a {product} consistent with the observed source evidence"]
    prompts.extend(f"a {product} where {fact}" for fact in evidence.hard_facts[:5])
    dominant_body_color = extract_dominant_body_color(evidence)
    if evidence.color_note:
        prompts.append(f"a {product} where {evidence.color_note}")
    if dominant_body_color and (evidence.color_confidence is None or evidence.color_confidence >= 0.64):
        prompts.append(f"a {product} whose dominant visible body color remains {dominant_body_color}")
    if evidence.coverage_note:
        prompts.append(f"a {product} where {evidence.coverage_note}")
    if evidence.pattern_note:
        prompts.append(f"a {product} where {evidence.pattern_note}")
    if evidence.trim_note:
        prompts.append(f"a {product} with {evidence.trim_note}")
    if evidence.form_factor_note:
        prompts.append(f"a {product} where {evidence.form_factor_note}")
    if evidence.upper_region_note:
        prompts.append(f"a {product} where {evidence.upper_region_note}")
    if evidence.lower_region_note:
        prompts.append(f"a {product} where {evidence.lower_region_note}")
    if evidence.edge_profile_note:
        prompts.append(f"a {product} where {evidence.edge_profile_note}")
    if evidence.soft_structure_note:
        prompts.append(f"a {product} where {evidence.soft_structure_note}")
    if evidence.material_note:
        prompts.append(f"a {product} where {evidence.material_note}")
    if (
        evidence.palette
        and evidence.color_confidence is not None
        and evidence.color_confidence >= 0.72
        and evidence.coverage_class != "localized_visible_pattern"
    ):
        prompts.append(f"a {product} using an observed palette of {', '.join(evidence.palette[:3])}")
    return _dedupe_strings(prompts)


def build_contradiction_prompts(localized: LocalizedProduct) -> list[str]:
    evidence = localized.identity.observed_evidence
    product = localized.identity.canonical_product_type or localized.identity.category or "product"
    prompts = [
        f"a {product} whose invented surfaces contradict the observed source evidence",
        f"a {product} with incompatible unseen panel design or trim",
    ]
    dominant_body_color = extract_dominant_body_color(evidence)
    if dominant_body_color and (evidence.color_confidence is None or evidence.color_confidence >= 0.64):
        prompts.append(
            f"a {product} whose dominant visible body color drifts away from {dominant_body_color}"
        )
    if evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}:
        prompts.append(
            f"a {product} where a broad observed print or multicolor surface treatment is collapsed into a small patch or mostly solid surface"
        )
    if evidence.trim_note:
        prompts.append(f"a {product} where the visible boundary or edging color changes incompatibly")
    if evidence.form_factor_note:
        prompts.append(f"a {product} where the visible form factor changes incompatibly")
    if evidence.upper_region_note:
        prompts.append(f"a {product} where the visible upper component changes color or geometry incompatibly")
    elif evidence.upper_component_state == "absent":
        prompts.append(f"a {product} with invented handles, straps, lids, or top attachments unsupported by the source")
    if evidence.lower_region_note:
        prompts.append(f"a {product} where the visible lower support, base, or lower assembly disappears or changes geometry incompatibly")
    if evidence.edge_profile_note:
        prompts.append(f"a {product} where the observed edge thickness or perimeter profile changes incompatibly")
    if evidence.soft_structure_note:
        prompts.append(
            f"a {product} where the observed soft-surface structure changes incompatibly, such as a flat pad turning into a boxed bolster or a raised rim flattening into a sheet"
        )
    if evidence.material_note:
        prompts.append(f"a {product} with a conflicting material zone or finish")
    if evidence.surface_scope in {"single_photo_limited", "limited_surface_evidence", "partial_or_occluded", "partial_or_ambiguous"}:
        prompts.append(
            f"a {product} that incorrectly mirrors one observed local detail across every unseen surface"
        )
    return _dedupe_strings(prompts)


def palette_overlap_score(expected_palette: Sequence[str], observed_palette: Sequence[str]) -> float:
    if not expected_palette or not observed_palette:
        return 0.0
    expected = set(expected_palette)
    observed = set(observed_palette)
    return len(expected & observed) / float(len(expected))


def score_generation_candidate(
    *,
    category_consistency: dict[str, Any],
    semantic_plausibility: dict[str, Any],
    evidence_consistency: dict[str, Any],
) -> float:
    score = 0.0
    score += float(evidence_consistency.get("score", 0.0)) * 0.65
    score += float(semantic_plausibility.get("score", 0.0)) * 0.18
    score += 0.12 if category_consistency.get("is_consistent", True) else -0.2
    score += 0.14 if evidence_consistency.get("is_consistent", True) else -0.22
    score += 0.06 if semantic_plausibility.get("is_plausible", True) else -0.1
    anatomy_margin = float(semantic_plausibility.get("anatomy_margin", 0.0))
    if bool(semantic_plausibility.get("human_supported")) and anatomy_margin < 0.0:
        score += anatomy_margin * 8.0
        if anatomy_margin <= -0.01:
            score -= 0.25
    casting_margin = float(semantic_plausibility.get("casting_margin", 0.0))
    if bool(semantic_plausibility.get("human_supported")) and casting_margin < 0.0:
        score += casting_margin * 7.0
        if casting_margin <= -0.01:
            score -= 0.18
    if bool(semantic_plausibility.get("people_out_of_frame_required")) and bool(
        semantic_plausibility.get("person_presence_flag")
    ):
        score -= 0.32
    if bool(semantic_plausibility.get("background_collapse_flag")):
        score -= 0.3
    if evidence_consistency.get("is_consistent", True) is False:
        evidence_gap = max(0.0, 0.52 - float(evidence_consistency.get("score", 0.0)))
        score -= evidence_gap * 0.8
    return score


def assess_category_consistency(
    image_path: str | Path,
    *,
    expected_category: str,
    expected_product_type: str,
    backbone: VisionBackbone,
    focus_image_path: str | Path | None = None,
) -> dict[str, Any]:
    embedding_path = Path(focus_image_path) if focus_image_path else Path(image_path)
    if not embedding_path.exists():
        embedding_path = Path(image_path)
    image_embedding = np.asarray(backbone.encode_image(embedding_path), dtype=np.float32)
    category_scores: dict[str, float] = {}
    for category, prompts in CATEGORY_CLASSIFICATION_TEXTS.items():
        text_embeddings = [
            np.asarray(embedding, dtype=np.float32)
            for embedding in backbone.encode_texts(prompts)
        ]
        if not text_embeddings:
            continue
        category_scores[category] = float(max(np.dot(image_embedding, embedding) for embedding in text_embeddings))
    return evaluate_category_scores(
        expected_category=expected_category,
        expected_product_type=expected_product_type,
        category_scores=category_scores,
    )


def evaluate_category_scores(
    *,
    expected_category: str,
    expected_product_type: str,
    category_scores: dict[str, float],
) -> dict[str, Any]:
    if not category_scores:
        return {
            "expected_category": expected_category,
            "expected_product_type": expected_product_type,
            "predicted_category": None,
            "predicted_score": None,
            "expected_score": None,
            "margin": None,
            "is_consistent": True,
            "warning": None,
        }

    predicted_category, predicted_score = max(category_scores.items(), key=lambda item: item[1])
    expected_score = float(category_scores.get(expected_category, min(category_scores.values())))
    margin = float(predicted_score - expected_score)
    is_consistent = predicted_category == expected_category or margin < 0.02
    warning = None
    if not is_consistent:
        warning = (
            f"category drift flagged: expected {expected_category} / {expected_product_type}, "
            f"predicted {predicted_category}"
        )
    return {
        "expected_category": expected_category,
        "expected_product_type": expected_product_type,
        "predicted_category": predicted_category,
        "predicted_score": predicted_score,
        "expected_score": expected_score,
        "margin": margin,
        "is_consistent": is_consistent,
        "warning": warning,
        "scores": {key: round(value, 4) for key, value in sorted(category_scores.items())},
    }


def assess_semantic_plausibility(
    image_path: str | Path,
    identity: ProductIdentitySpec,
    *,
    prompt_spec: FluxPromptSpec,
    scene_family: str,
    support_relation: str,
    backbone: VisionBackbone,
    generated_localizer: Any | None = None,
    product_photo_factory: Any | None = None,
) -> dict[str, Any]:
    image_embedding = np.asarray(backbone.encode_image(image_path), dtype=np.float32)
    canonical_product_type = identity.canonical_product_type or identity.category or "product"
    support_prompts = SUPPORT_RELATION_EVAL_PROMPTS.get(support_relation, {})
    positive_prompts = [
        template.format(product=canonical_product_type)
        for template in support_prompts.get("positive", ())
    ]
    negative_prompts = [
        template.format(product=canonical_product_type)
        for template in support_prompts.get("negative", ())
    ]
    if identity.interaction_mode == "held_in_hand" and identity.observed_evidence.upper_component_state == "absent":
        positive_prompts = [
            f"a {canonical_product_type} held directly in one hand with visible body contact and no carry attachment",
            f"a compact {canonical_product_type} shown close to the hand with direct grip on the product body",
        ]
        negative_prompts.extend(
            [
                f"a {canonical_product_type} hanging from a shoulder strap or crossbody attachment",
                f"a {canonical_product_type} with an invented wrist loop, strap, or dangling handle",
            ]
        )
    scene_prompts = [
        template.format(product=canonical_product_type)
        for template in SCENE_FAMILY_EVAL_PROMPTS.get(scene_family, ())
    ]
    support_positive = max(
        (
            float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
            for embedding in backbone.encode_texts(positive_prompts)
        ),
        default=0.0,
    )
    support_negative = max(
        (
            float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
            for embedding in backbone.encode_texts(negative_prompts)
        ),
        default=0.0,
    )
    scene_alignment = max(
        (
            float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
            for embedding in backbone.encode_texts(scene_prompts)
        ),
        default=0.0,
    )
    human_supported = bool(
        identity.requires_human_model or support_relation in {"carried_by_hand", "worn_on_body"}
    )
    anatomy_positive = 0.0
    anatomy_negative = 0.0
    anatomy_margin = 0.0
    casting_positive = 0.0
    casting_negative = 0.0
    casting_margin = 0.0
    single_model_positive = 0.0
    single_model_negative = 0.0
    single_model_margin = 0.0
    dress_layering_positive = 0.0
    dress_layering_negative = 0.0
    dress_layering_margin = 0.0
    casting_prompts: dict[str, tuple[str, ...]] | None = None
    functional_positive = 0.0
    functional_negative = 0.0
    functional_margin = 0.0
    functional_prompts: dict[str, tuple[str, ...]] | None = None
    ghost_composite_flag = False
    ghost_composite_metrics: dict[str, float] = {}
    background_collapse_flag = False
    background_collapse_metrics: dict[str, float] = {}
    multi_person_flag = False
    multi_person_metrics: dict[str, float] = {}
    people_out_of_frame_required = identity_requires_people_out_of_frame(identity)
    person_presence_flag = False
    person_presence_metrics: dict[str, float] = {}
    background_collapse_flag, background_collapse_metrics = detect_background_collapse_artifact(image_path)
    if human_supported:
        anatomy_positive = max(
            (
                float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                for embedding in backbone.encode_texts(
                    [template.format(product=canonical_product_type) for template in HUMAN_ANATOMY_EVAL_PROMPTS["positive"]]
                )
            ),
            default=0.0,
        )
        anatomy_negative = max(
            (
                float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                for embedding in backbone.encode_texts(
                    [template.format(product=canonical_product_type) for template in HUMAN_ANATOMY_EVAL_PROMPTS["negative"]]
                )
            ),
            default=0.0,
        )
        anatomy_margin = anatomy_positive - anatomy_negative
        casting_prompts = select_casting_alignment_eval_prompts(identity, canonical_product_type)
        if casting_prompts is not None:
            casting_positive = max(
                (
                    float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                    for embedding in backbone.encode_texts(casting_prompts["positive"])
                ),
                default=0.0,
            )
            casting_negative = max(
                (
                    float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                    for embedding in backbone.encode_texts(casting_prompts["negative"])
                ),
                default=0.0,
            )
            casting_margin = casting_positive - casting_negative
        if identity.requires_human_model:
            single_model_positive = max(
                (
                    float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                    for embedding in backbone.encode_texts(
                        [
                            "a single model alone in the scene with no other people visible",
                            "one person only with an empty background and no passersby",
                            "a lone model in a private or controlled location",
                        ]
                    )
                ),
                default=0.0,
            )
            single_model_negative = max(
                (
                    float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                    for embedding in backbone.encode_texts(
                        [
                            "multiple people visible in the background",
                            "a passerby or second person behind the model",
                            "companions, pedestrians, or extra visible figures in frame",
                        ]
                    )
                ),
                default=0.0,
            )
            single_model_margin = single_model_positive - single_model_negative
        if canonical_product_type == "dress":
            dress_layering_positive = max(
                (
                    float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                    for embedding in backbone.encode_texts(
                        [
                            "a dress worn cleanly with bare legs and no pants visible below the hem",
                            "a dress presented without jeans, trousers, or leggings under it",
                            "a single dress with no visible denim pants, trousers, or leggings underneath",
                        ]
                    )
                ),
                default=0.0,
            )
            dress_layering_negative = max(
                (
                    float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                    for embedding in backbone.encode_texts(
                        [
                            "a dress worn over blue jeans visible below the hem",
                            "a dress layered over denim pants that are clearly visible",
                            "visible trousers or leggings underneath a dress",
                            "a tunic-like dress worn over lower-body garments",
                        ]
                    )
                ),
                default=0.0,
            )
            dress_layering_margin = dress_layering_positive - dress_layering_negative
        ghost_composite_flag, ghost_composite_metrics = detect_human_ghost_composite_artifact(image_path)
        if generated_localizer is not None and product_photo_factory is not None:
            multi_person_flag, multi_person_metrics = detect_multiple_people_in_scene(
                image_path,
                generated_localizer=generated_localizer,
                product_photo_factory=product_photo_factory,
            )
    functional_prompts = select_functional_subtype_eval_prompts(identity, canonical_product_type)
    if functional_prompts is not None:
        functional_positive = max(
            (
                float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                for embedding in backbone.encode_texts(functional_prompts["positive"])
            ),
            default=0.0,
        )
        functional_negative = max(
            (
                float(np.dot(image_embedding, np.asarray(embedding, dtype=np.float32)))
                for embedding in backbone.encode_texts(functional_prompts["negative"])
            ),
            default=0.0,
        )
        functional_margin = functional_positive - functional_negative
    if people_out_of_frame_required and generated_localizer is not None and product_photo_factory is not None:
        person_presence_flag, person_presence_metrics = detect_any_person_in_scene(
            image_path,
            generated_localizer=generated_localizer,
            product_photo_factory=product_photo_factory,
        )
    prompt_conflicts = evaluate_prompt_scene_conflicts(
        prompt_spec,
        scene_family=scene_family,
        support_relation=support_relation,
    )
    support_margin = support_positive - support_negative
    functional_weight = 0.32
    if functional_prompts is not None and identity_requires_functional_context(identity):
        functional_weight = 5.0
    raw_score = (
        0.5
        + 1.5 * support_margin
        + 0.5 * max(scene_alignment, 0.0)
        + (0.22 * anatomy_margin if human_supported else 0.0)
        + (0.28 * casting_margin if human_supported else 0.0)
        + (0.18 * single_model_margin if identity.requires_human_model else 0.0)
        + (0.16 * dress_layering_margin if canonical_product_type == "dress" else 0.0)
        + (functional_weight * functional_margin if functional_prompts is not None else 0.0)
        - (0.24 if multi_person_flag else 0.0)
        - (0.3 if ghost_composite_flag else 0.0)
        - (0.28 if background_collapse_flag else 0.0)
        - (0.34 if person_presence_flag else 0.0)
    )
    score = max(
        0.0,
        min(
            1.0,
            raw_score,
        ),
    )
    support_tolerance = semantic_support_margin_threshold(identity, support_relation=support_relation)
    is_plausible = support_margin >= support_tolerance and not prompt_conflicts and (
        not human_supported or anatomy_margin >= -0.015
    )
    if human_supported and casting_prompts is not None and casting_margin < -0.005:
        is_plausible = False
    if identity.requires_human_model and single_model_margin < -0.005:
        is_plausible = False
    if canonical_product_type == "dress" and dress_layering_margin < -0.005:
        is_plausible = False
    if functional_prompts is not None and functional_margin < -0.005:
        is_plausible = False
    if identity_requires_functional_context(identity) and functional_prompts is not None and functional_margin < 0.01:
        is_plausible = False
    if multi_person_flag:
        is_plausible = False
    if ghost_composite_flag:
        is_plausible = False
    if background_collapse_flag:
        is_plausible = False
    if people_out_of_frame_required and person_presence_flag:
        is_plausible = False
    warning = None
    if prompt_conflicts:
        warning = "; ".join(prompt_conflicts)
    elif people_out_of_frame_required and person_presence_flag:
        warning = "human presence detected even though the prompt requires a product-only frame"
    elif multi_person_flag:
        warning = "multiple people are visible in the scene even though the prompt requires exactly one model"
    elif ghost_composite_flag:
        warning = "human composite plausibility weak; subject appears washed out or ghosted against the scene"
    elif background_collapse_flag:
        warning = "background collapses into a low-detail monochrome gray scene instead of a fully resolved environment"
    elif identity.requires_human_model and single_model_margin < -0.005:
        warning = "single-model exclusivity weak; scene appears to include passersby, companions, or other visible people"
    elif human_supported and anatomy_margin < -0.015:
        warning = "human anatomy plausibility weak; output may contain duplicated or impossible limbs"
    elif canonical_product_type == "dress" and dress_layering_margin < -0.005:
        warning = "dress styling plausibility weak; output appears layered over visible pants or another lower-body garment"
    elif human_supported and casting_prompts is not None and casting_margin < -0.005:
        warning = "human casting or styling appears weakly aligned with the product's expected visual language"
    elif identity_requires_functional_context(identity) and functional_prompts is not None and functional_margin < 0.01:
        warning = "functional subtype alignment weak; output does not preserve the context-sensitive subtype strongly enough"
    elif functional_prompts is not None and functional_margin < -0.005:
        warning = "functional subtype alignment weak; output looks closer to a generic category match than the observed product subtype"
    elif not is_plausible:
        warning = (
            f"semantic plausibility weak for support relation {support_relation} in scene {scene_family}"
        )
    return {
        "support_relation": support_relation,
        "scene_family": scene_family,
        "support_positive": round(support_positive, 4),
        "support_negative": round(support_negative, 4),
        "support_margin": round(support_margin, 4),
        "scene_alignment": round(scene_alignment, 4),
        "human_supported": human_supported,
        "anatomy_positive": round(anatomy_positive, 4),
        "anatomy_negative": round(anatomy_negative, 4),
        "anatomy_margin": round(anatomy_margin, 4),
        "casting_positive": round(casting_positive, 4),
        "casting_negative": round(casting_negative, 4),
        "casting_margin": round(casting_margin, 4),
        "single_model_positive": round(single_model_positive, 4),
        "single_model_negative": round(single_model_negative, 4),
        "single_model_margin": round(single_model_margin, 4),
        "dress_layering_positive": round(dress_layering_positive, 4),
        "dress_layering_negative": round(dress_layering_negative, 4),
        "dress_layering_margin": round(dress_layering_margin, 4),
        "functional_positive": round(functional_positive, 4),
        "functional_negative": round(functional_negative, 4),
        "functional_margin": round(functional_margin, 4),
        "people_out_of_frame_required": people_out_of_frame_required,
        "person_presence_flag": person_presence_flag,
        "person_presence_metrics": {key: round(value, 4) for key, value in person_presence_metrics.items()},
        "multi_person_flag": multi_person_flag,
        "multi_person_metrics": {key: round(value, 4) for key, value in multi_person_metrics.items()},
        "ghost_composite_flag": ghost_composite_flag,
        "ghost_composite_metrics": {key: round(value, 4) for key, value in ghost_composite_metrics.items()},
        "background_collapse_flag": background_collapse_flag,
        "background_collapse_metrics": {key: round(value, 4) for key, value in background_collapse_metrics.items()},
        "score": round(score, 4),
        "prompt_conflicts": prompt_conflicts,
        "is_plausible": is_plausible,
        "warning": warning,
    }


def semantic_support_margin_threshold(identity: ProductIdentitySpec, *, support_relation: str) -> float:
    if support_relation == "carried_by_hand" and identity_prefers_compact_hand_focus(identity):
        return -0.03
    return -0.01


def detect_background_collapse_artifact(image_path: str | Path) -> tuple[bool, dict[str, float]]:
    path = Path(image_path)
    with Image.open(path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"), dtype=np.float32) / 255.0
    height, width = image.shape[:2]
    band_y = max(16, int(round(height * 0.12)))
    band_x = max(16, int(round(width * 0.12)))
    border = np.concatenate(
        [
            image[:band_y, :, :].reshape(-1, 3),
            image[-band_y:, :, :].reshape(-1, 3),
            image[:, :band_x, :].reshape(-1, 3),
            image[:, -band_x:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    if border.size == 0:
        return False, {}
    max_channel = border.max(axis=1)
    min_channel = border.min(axis=1)
    saturation = np.where(max_channel == 0.0, 0.0, (max_channel - min_channel) / np.maximum(max_channel, 1e-6))
    value = max_channel
    gray = border.mean(axis=1)
    metrics = {
        "border_saturation_mean": float(np.mean(saturation)),
        "border_saturation_p90": float(np.percentile(saturation, 90)),
        "border_value_std": float(np.std(value)),
        "border_luma_std": float(np.std(gray)),
        "border_neutral_fraction": float(np.mean((saturation <= 0.12) & (gray >= 0.2) & (gray <= 0.9))),
    }
    collapsed = bool(
        (
            metrics["border_saturation_mean"] < 0.085
            and metrics["border_saturation_p90"] < 0.17
            and metrics["border_luma_std"] < 0.085
            and metrics["border_neutral_fraction"] >= 0.78
        )
        or (
            metrics["border_saturation_mean"] < 0.11
            and metrics["border_saturation_p90"] < 0.22
            and metrics["border_value_std"] < 0.075
            and metrics["border_neutral_fraction"] >= 0.84
        )
    )
    return collapsed, metrics


def detect_human_ghost_composite_artifact(image_path: str | Path) -> tuple[bool, dict[str, float]]:
    path = Path(image_path)
    with Image.open(path) as image_handle:
        image = np.asarray(image_handle.convert("RGB"), dtype=np.float32) / 255.0
    height, width = image.shape[:2]
    y0, y1 = int(height * 0.15), int(height * 0.9)
    x0, x1 = int(width * 0.3), int(width * 0.7)
    if y1 <= y0 or x1 <= x0:
        return False, {}
    crop = image[y0:y1, x0:x1]
    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    saturation = np.where(max_channel == 0.0, 0.0, (max_channel - min_channel) / np.maximum(max_channel, 1e-6))
    value = max_channel
    metrics = {
        "central_saturation_mean": float(np.mean(saturation)),
        "central_saturation_p90": float(np.percentile(saturation, 90)),
        "central_value_mean": float(np.mean(value)),
        "central_rgb_std": float(np.std(crop)),
    }
    is_ghosted = bool(
        (
            metrics["central_saturation_mean"] < 0.12
            and metrics["central_saturation_p90"] < 0.22
            and metrics["central_value_mean"] > 0.55
            and metrics["central_rgb_std"] < 0.14
        )
        or (
            metrics["central_saturation_mean"] < 0.18
            and metrics["central_saturation_p90"] < 0.45
            and metrics["central_value_mean"] > 0.48
            and metrics["central_rgb_std"] < 0.2
        )
    )
    return is_ghosted, metrics


def _box_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection_x0 = max(left.x0, right.x0)
    intersection_y0 = max(left.y0, right.y0)
    intersection_x1 = min(left.x1, right.x1)
    intersection_y1 = min(left.y1, right.y1)
    if intersection_x1 <= intersection_x0 or intersection_y1 <= intersection_y0:
        return 0.0
    intersection = (intersection_x1 - intersection_x0) * (intersection_y1 - intersection_y0)
    left_area = max(1, (left.x1 - left.x0) * (left.y1 - left.y0))
    right_area = max(1, (right.x1 - right.x0) * (right.y1 - right.y0))
    union = max(1, left_area + right_area - intersection)
    return intersection / float(union)


def _distinct_person_mask_count(
    masks: Sequence[Any],
    *,
    image_area: float,
    min_confidence: float = 0.38,
    min_area_ratio: float = 0.018,
    dedupe_iou_threshold: float = 0.55,
) -> tuple[int, dict[str, float]]:
    filtered: list[Any] = []
    for mask in masks:
        phrase_text = str(getattr(getattr(mask, "phrase", None), "text", "")).lower()
        if not any(token in phrase_text for token in ("person", "woman", "man", "model")):
            continue
        confidence = float(getattr(mask, "confidence", 0.0))
        area_pixels = float(getattr(mask, "area_pixels", 0.0))
        area_ratio = area_pixels / max(1.0, image_area)
        if confidence < min_confidence or area_ratio < min_area_ratio:
            continue
        filtered.append(mask)
    filtered.sort(
        key=lambda mask: (
            float(getattr(mask, "area_pixels", 0.0)),
            float(getattr(mask, "confidence", 0.0)),
        ),
        reverse=True,
    )
    distinct: list[Any] = []
    for mask in filtered:
        box = getattr(mask, "box", None)
        if box is None:
            continue
        if any(_box_iou(box, getattr(existing, "box", box)) >= dedupe_iou_threshold for existing in distinct):
            continue
        distinct.append(mask)
    metrics = {
        "raw_mask_count": float(len(masks)),
        "person_candidate_count": float(len(filtered)),
        "distinct_person_count": float(len(distinct)),
        "largest_person_area_ratio": (
            max(float(getattr(mask, "area_pixels", 0.0)) for mask in distinct) / max(1.0, image_area)
            if distinct
            else 0.0
        ),
        "second_person_area_ratio": (
            sorted(
                (float(getattr(mask, "area_pixels", 0.0)) / max(1.0, image_area) for mask in distinct),
                reverse=True,
            )[1]
            if len(distinct) >= 2
            else 0.0
        ),
    }
    return len(distinct), metrics


def _detect_person_count_in_scene(
    image_path: str | Path,
    *,
    generated_localizer: Any,
    product_photo_factory: Any,
) -> tuple[int, dict[str, float]]:
    path = Path(image_path)
    if not path.exists():
        return 0, {}
    with Image.open(path) as image_handle:
        width, height = image_handle.size
    image_area = float(max(1, width * height))
    photo = product_photo_factory(
        image_path=path,
        product_id=path.stem,
        title="person",
        hint_phrases=("person", "woman", "man", "model"),
        metadata={"category": "person", "canonical_product_type": "person"},
    )
    result = generated_localizer.localize(photo)
    distinct_count, metrics = _distinct_person_mask_count(result.masks, image_area=image_area)
    return distinct_count, metrics


def detect_multiple_people_in_scene(
    image_path: str | Path,
    *,
    generated_localizer: Any,
    product_photo_factory: Any,
) -> tuple[bool, dict[str, float]]:
    distinct_count, metrics = _detect_person_count_in_scene(
        image_path,
        generated_localizer=generated_localizer,
        product_photo_factory=product_photo_factory,
    )
    return distinct_count >= 2, metrics


def detect_any_person_in_scene(
    image_path: str | Path,
    *,
    generated_localizer: Any,
    product_photo_factory: Any,
) -> tuple[bool, dict[str, float]]:
    distinct_count, metrics = _detect_person_count_in_scene(
        image_path,
        generated_localizer=generated_localizer,
        product_photo_factory=product_photo_factory,
    )
    return distinct_count >= 1, metrics


def detect_compact_accessory_wardrobe_color_spill(
    image_path: str | Path,
    *,
    focus_mask_path: Path,
    localized: LocalizedProduct,
) -> tuple[bool, dict[str, float]]:
    if not identity_prefers_compact_hand_focus(localized.identity):
        return False, {}
    dominant_body_color = extract_dominant_body_color(localized.identity.observed_evidence)
    if dominant_body_color is None or dominant_body_color not in EVIDENCE_COLOR_SWATCHES:
        return False, {}
    image = Path(image_path)
    if not image.exists() or not focus_mask_path.exists():
        return False, {}
    with Image.open(image) as image_handle:
        source = np.asarray(image_handle.convert("RGB"), dtype=np.float32)
    mask = _load_mask_array(focus_mask_path, source_shape=source.shape[:2])
    if mask is None or not mask.any():
        return False, {}
    expanded_mask = _dilate_mask(mask, steps=10)
    target_rgb = np.asarray(EVIDENCE_COLOR_SWATCHES[dominant_body_color], dtype=np.float32)
    color_distance = np.sqrt(np.sum((source - target_rgb[None, None, :]) ** 2, axis=2))
    non_product_color = (color_distance <= 58.0) & ~expanded_mask
    height, width = mask.shape
    torso_region = np.zeros_like(mask, dtype=bool)
    torso_region[: int(round(height * 0.9)), int(round(width * 0.08)) : int(round(width * 0.96))] = True
    candidate_mask = non_product_color & torso_region
    components = _mask_components(candidate_mask, min_pixels=max(180, int(mask.sum() * 0.18)))
    if not components:
        return False, {}
    largest_component = max(components, key=lambda component: int(component.sum()))
    ys, xs = np.nonzero(largest_component)
    if len(xs) == 0:
        return False, {}
    component_area = float(largest_component.sum())
    focus_area = float(mask.sum())
    component_height = float(ys.max() - ys.min() + 1)
    component_width = float(xs.max() - xs.min() + 1)
    area_ratio = component_area / max(focus_area, 1.0)
    height_ratio = component_height / max(float(height), 1.0)
    width_ratio = component_width / max(float(width), 1.0)
    flag = bool(area_ratio >= 1.4 and height_ratio >= 0.28 and width_ratio >= 0.12)
    return flag, {
        "component_area_ratio": area_ratio,
        "component_height_ratio": height_ratio,
        "component_width_ratio": width_ratio,
    }


def evaluate_prompt_scene_conflicts(
    prompt_spec: FluxPromptSpec,
    *,
    scene_family: str,
    support_relation: str,
) -> list[str]:
    text = prompt_spec.to_prompt_text().lower()
    conflicts: list[str] = []
    if support_relation == "resting_with_back_support" and any(
        token in text for token in ("tabletop", "counter display", "kitchen setting")
    ):
        conflicts.append("prompt mixes backed-support placement with tabletop display language")
    if support_relation in {"carried_by_hand", "worn_on_body"} and "tabletop" in text:
        conflicts.append("prompt mixes human-supported placement with tabletop composition language")
    if scene_family == "furnished_interior" and "kitchen setting" in text:
        conflicts.append("prompt mixes furnished interior planning with kitchen/tabletop scene language")
    if scene_family == "fashion_lifestyle" and "sofa setting" in text and support_relation == "worn_on_body":
        conflicts.append("prompt mixes worn-body presentation with unsupported interior prop staging")
    return conflicts


def assess_prompt_readiness(
    localized: LocalizedProduct,
    prompt_spec: FluxPromptSpec,
    *,
    scene_family: str,
    support_relation: str,
) -> dict[str, Any]:
    evidence = localized.identity.observed_evidence
    prompt_text = prompt_spec.to_prompt_text().lower()
    issues: list[str] = []
    score = 1.0

    prompt_conflicts = evaluate_prompt_scene_conflicts(
        prompt_spec,
        scene_family=scene_family,
        support_relation=support_relation,
    )
    if prompt_conflicts:
        issues.extend(prompt_conflicts)
        score -= 0.4

    if evidence.color_note and evidence.color_note.lower() not in prompt_text:
        issues.append("prompt does not surface the current color-evidence note")
        score -= 0.08
    if evidence.color_confidence is not None and evidence.color_confidence < 0.55 and "observed palette includes" in prompt_text:
        issues.append("prompt still exposes exact palette language despite low color confidence")
        score -= 0.12
    if evidence.upper_region_note and evidence.upper_region_note.lower() not in prompt_text:
        issues.append("prompt omits observed upper-component evidence")
        score -= 0.08
    if evidence.lower_region_note and evidence.lower_region_note.lower() not in prompt_text:
        issues.append("prompt omits observed lower-support evidence")
        score -= 0.08
    if evidence.edge_profile_note and evidence.edge_profile_note.lower() not in prompt_text:
        issues.append("prompt omits observed edge-profile evidence")
        score -= 0.06
    if evidence.upper_component_state == "absent" and "do not invent handles, straps, lids, or attached upper structures" not in prompt_text:
        issues.append("prompt is missing the no-attached-structure guardrail")
        score -= 0.1
    if evidence.form_factor_note and evidence.form_factor_note.lower() not in prompt_text:
        issues.append("prompt omits form-factor evidence")
        score -= 0.08
    if localized.identity.casting_note and localized.identity.casting_note.lower() not in prompt_text:
        issues.append("prompt omits casting-compatibility guidance")
        score -= 0.06
    if evidence.material_note and evidence.material_note.lower() not in prompt_text:
        issues.append("prompt omits material evidence")
        score -= 0.05
    if (
        evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
        and not evidence.trim_note
        and evidence.upper_component_state != "present"
        and "keep them within the observed palette family" not in prompt_text
    ):
        issues.append("prompt is missing the palette-compatible structural-zone guardrail")
        score -= 0.08
    if localized.identity.rigid_vs_soft != "rigid" and "do not copy incidental daily-photo wrinkles" not in prompt_text:
        issues.append("prompt is missing the transient-wrinkle guardrail")
        score -= 0.05

    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 4),
        "issues": issues,
        "scene_family": scene_family,
        "support_relation": support_relation,
    }


def infer_affordance_profile(
    category: str,
    *,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
) -> dict[str, Any]:
    text = " ".join([category, canonical_product_type, product_title, *hint_phrases]).lower()
    tokens = set(_tokens(text))
    support_mode, default_scene_family, interaction_mode, stable_base = CATEGORY_SUPPORT_DEFAULTS.get(
        category,
        CATEGORY_SUPPORT_DEFAULTS["product"],
    )
    rigid_vs_soft = "semi-rigid"
    if tokens.intersection(
        {
            "bottle",
            "mug",
            "cup",
            "jar",
            "glass",
            "metal",
            "plastic",
            "lamp",
            "lighting",
            "shade",
            "chair",
            "blender",
            "toaster",
            "coffee",
            "coffeemaker",
            "slow",
            "cooker",
            "chopper",
            "processor",
            "appliance",
        }
    ):
        rigid_vs_soft = "rigid"
    elif tokens.intersection(
        {"pillow", "cushion", "shirt", "dress", "textile", "fabric", "knit", "comforter", "quilt", "duvet", "blanket", "bed"}
    ):
        rigid_vs_soft = "soft"
    elif tokens.intersection({"bag", "tote", "wallet", "purse", "backpack"}):
        rigid_vs_soft = "semi-rigid"

    if tokens.intersection({"wall", "mounted", "hook"}):
        support_mode = "mounted"
        default_scene_family = "retail_display"
        interaction_mode = "mounted"
        stable_base = False
    elif category == "footwear":
        support_mode = "wearable"
        default_scene_family = "fashion_lifestyle"
        interaction_mode = "worn"
        stable_base = False
    elif category == "home lighting":
        support_mode = "self_supporting_display"
        default_scene_family = "furnished_interior"
        interaction_mode = "placed"
        stable_base = True
    elif category == "kitchen appliance":
        support_mode = "self_supporting_display"
        default_scene_family = "tabletop_display"
        interaction_mode = "placed"
        stable_base = True
    elif category == "furniture":
        support_mode = "self_supporting_display"
        default_scene_family = "editorial_interior"
        interaction_mode = "placed"
        stable_base = True
    elif category == "bedding":
        support_mode = "externally_supported_soft"
        default_scene_family = "furnished_interior"
        interaction_mode = "placed"
        stable_base = False
    elif category == "pet home":
        support_mode = "externally_supported_soft"
        default_scene_family = "furnished_interior"
        interaction_mode = "placed"
        stable_base = False
    elif category == "bag" and canonical_product_type == "backpack":
        support_mode = "wearable"
        default_scene_family = (
            "outdoor_lifestyle"
            if {"sport", "sports", "baseball", "athletic", "cooler", "insulated", "lunch", "travel"} & tokens
            else "fashion_lifestyle"
        )
        interaction_mode = "worn_or_carried"
        stable_base = False
    elif tokens.intersection({"shirt", "dress", "jacket", "pants", "wear", "blouse", "hoodie", "tunic", "top"}):
        support_mode = "wearable"
        default_scene_family = "fashion_lifestyle"
        interaction_mode = "worn"
        stable_base = False
    elif rigid_vs_soft == "soft" and category not in {"apparel"}:
        support_mode = "externally_supported_soft"
        default_scene_family = "furnished_interior"
        interaction_mode = "placed"
        stable_base = False
    elif category == "bag":
        support_mode = "portable_flexible"
        default_scene_family = "fashion_lifestyle"
        interaction_mode = "held_in_hand" if canonical_product_type in {"wallet", "clutch"} else "carried_or_resting"
        stable_base = False
    elif rigid_vs_soft == "rigid":
        support_mode = "self_supporting_display"
        default_scene_family = "tabletop_display"
        interaction_mode = "handheld_or_display"
        stable_base = True

    return {
        "support_mode": support_mode,
        "default_scene_family": default_scene_family,
        "interaction_mode": interaction_mode,
        "stable_base": stable_base,
        "rigid_vs_soft": rigid_vs_soft,
    }


def _contains_term(text: str, tokens: set[str], term: str) -> bool:
    normalized = term.lower().strip()
    if " " in normalized or "/" in normalized or "-" in normalized:
        return normalized in text
    return normalized in tokens


def _contains_any_term(text: str, tokens: set[str], terms: Sequence[str]) -> bool:
    return any(_contains_term(text, tokens, term) for term in terms)


def infer_functional_subtype_hint(
    *,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    selected_phrase: str,
) -> str | None:
    text = " ".join([product_title, *hint_phrases, selected_phrase]).lower()
    tokens = set(_tokens(text))
    if (
        category == "bag"
        and canonical_product_type == "backpack"
        and (
            _contains_any_term(text, tokens, ("backpack cooler", "cooler backpack"))
            or ("backpack" in tokens and {"cooler", "insulated", "lunch"} & tokens)
        )
    ):
        return "backpack cooler"
    return None


def infer_functional_subtype_hard_facts(
    *,
    category: str,
    canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    selected_phrase: str,
) -> list[str]:
    subtype_hint = infer_functional_subtype_hint(
        category=category,
        canonical_product_type=canonical_product_type,
        product_title=product_title,
        hint_phrases=hint_phrases,
        selected_phrase=selected_phrase,
    )
    if subtype_hint == "backpack cooler":
        return [
            "the product remains a backpack cooler rather than a generic school or laptop backpack",
            "keep the design compatible with an insulated cooler compartment and zipper opening rather than a plain daypack body",
        ]
    return []


def identity_requires_functional_context(identity: ProductIdentitySpec) -> bool:
    subtype_hint = str(identity.subtype_hint or "").lower()
    evidence = identity.observed_evidence
    if not subtype_hint:
        return False
    if identity.requires_human_model:
        return False
    if "border_human_fragment" in evidence.artifact_flags or "source_contains_border_human_fragment" in evidence.source_validity_issues:
        return False
    return subtype_hint in {"backpack cooler"}


def infer_category(*texts: str) -> str:
    text = " ".join(texts).lower()
    tokens = set(_tokens(text))
    if _contains_any_term(
        text,
        tokens,
        ("blender", "toaster", "coffee maker", "slow cooker", "food chopper", "food processor", "air fryer", "appliance", "kettle", "microwave"),
    ):
        return "kitchen appliance"
    if _contains_any_term(text, tokens, ("bottle", "mug", "cup", "drinkware")):
        return "drinkware"
    if _contains_any_term(text, tokens, ("lamp", "lighting", "light fixture", "table lamp", "desk lamp")):
        return "home lighting"
    if _contains_any_term(text, tokens, ("comforter", "duvet", "blanket", "quilt", "bedding", "coverlet")):
        return "bedding"
    if _contains_any_term(text, tokens, ("pillow", "cushion")):
        return "home decor"
    if _contains_any_term(text, tokens, ("office chair", "desk chair", "folding chair", "event chair", "chair", "stool", "bench")):
        return "furniture"
    if _contains_any_term(text, tokens, ("backpack", "bookbag", "knapsack", "bag", "handbag", "tote", "wallet", "satchel", "purse", "clutch")):
        return "bag"
    if _contains_any_term(text, tokens, ("dog bed", "pet bed", "cat bed")):
        return "pet home"
    if _contains_any_term(text, tokens, ("shoe", "sneaker", "sandal", "boot", "loafer", "trainer")):
        return "footwear"
    if _contains_any_term(
        text,
        tokens,
        ("dress", "shirt", "jacket", "pants", "skirt", "blouse", "hoodie", "tunic", "top"),
    ):
        return "apparel"
    return "product"


def infer_canonical_product_type(product_title: str, hint_phrases: Sequence[str], phrase: str) -> str:
    text = " ".join([product_title, *hint_phrases, phrase]).lower()
    tokens = set(_tokens(text))
    for canonical_type, patterns in PRODUCT_TYPE_PATTERNS:
        if _contains_any_term(text, tokens, patterns):
            return canonical_type
    category = infer_category(product_title, " ".join(hint_phrases), phrase)
    defaults = {
        "drinkware": "water bottle",
        "home decor": "decorative pillow",
        "home lighting": "table lamp",
        "bedding": "comforter",
        "furniture": "office chair",
        "kitchen appliance": "blender",
        "pet home": "pet bed",
        "bag": "tote bag",
        "apparel": "shirt",
        "footwear": "shoe",
        "product": "product",
    }
    return defaults[category]


def refine_canonical_product_type(
    *,
    category: str,
    initial_canonical_product_type: str,
    product_title: str,
    hint_phrases: Sequence[str],
    selected_phrase: str,
    observed_evidence: ObservedEvidenceSpec,
) -> str:
    text = " ".join([product_title, *hint_phrases, selected_phrase]).lower()
    tokens = set(_tokens(text))
    selected_tokens = set(_tokens(selected_phrase))

    if category == "footwear":
        if {"shoe", "sneaker", "sandal", "boot", "loafer", "trainer"} & tokens:
            return "shoe"
        return initial_canonical_product_type

    if category == "home lighting":
        if {"lamp", "lighting", "shade"} & tokens:
            return "table lamp"
        return initial_canonical_product_type

    if category == "apparel":
        scores = {
            "dress": 0.0,
            "shirt": 0.0,
        }
        if {"dress", "maxi", "sundress", "swing", "kaftan"} & tokens:
            scores["dress"] += 1.2
        if {"shirt", "tee", "top", "blouse", "tunic", "hoodie", "sweatshirt"} & tokens:
            scores["shirt"] += 1.0
        if observed_evidence.aspect_ratio is not None and observed_evidence.aspect_ratio >= 1.55:
            scores["dress"] += 0.2
        if observed_evidence.upper_component_state == "present":
            scores["shirt"] += 0.12
        best_type, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score >= 0.75:
            return best_type
        return initial_canonical_product_type

    if category != "bag":
        return initial_canonical_product_type

    upper_state = observed_evidence.upper_component_state
    aspect_ratio = observed_evidence.aspect_ratio
    top_width_ratio = observed_evidence.top_width_ratio
    evidence_text = " ".join(
        filter(
            None,
            [
                observed_evidence.form_factor_note,
                observed_evidence.upper_region_note,
                observed_evidence.lower_region_note,
                observed_evidence.soft_structure_note,
                observed_evidence.silhouette_note,
                *observed_evidence.hard_facts,
            ],
        )
    ).lower()
    backpack_form_supported = any(
        phrase in evidence_text
        for phrase in (
            "backpack body",
            "shoulder or back carry",
            "carry straps",
            "backpack with a main body",
            "harness",
        )
    )
    wallet_text_supported = bool({"wallet", "clutch", "wristlet", "pouch"} & tokens)
    wallet_selected = bool({"wallet", "clutch", "wristlet", "pouch"} & selected_tokens)
    compact_wallet_shape = bool(
        aspect_ratio is not None and (aspect_ratio <= 0.92 or (top_width_ratio is not None and top_width_ratio <= 0.7))
    )

    scores = {
        "wallet": 0.0,
        "backpack": 0.0,
        "tote bag": 0.0,
        "handbag": 0.0,
    }
    if wallet_selected:
        scores["wallet"] += 1.2
    elif wallet_text_supported:
        scores["wallet"] += 0.35
    if {"tote", "shopping", "beach", "carryall"} & tokens:
        scores["tote bag"] += 1.0
    if {"handbag", "purse"} & tokens:
        scores["handbag"] += 0.8
    if {"backpack", "bookbag", "knapsack"} & tokens:
        scores["backpack"] += 1.2
    if backpack_form_supported:
        scores["backpack"] += 1.15
        scores["wallet"] -= 0.35

    if upper_state == "absent":
        if compact_wallet_shape and not backpack_form_supported:
            scores["wallet"] += 0.55
            scores["tote bag"] -= 0.2
            scores["handbag"] -= 0.12
            scores["backpack"] -= 0.25
        elif compact_wallet_shape and backpack_form_supported:
            scores["backpack"] += 0.2
        elif observed_evidence.surface_scope not in {"partial_or_occluded", "partial_or_ambiguous"}:
            scores["wallet"] += 0.2
    elif upper_state == "present":
        scores["tote bag"] += 0.45
        scores["handbag"] += 0.4
        scores["wallet"] -= 0.2
        if observed_evidence.aspect_ratio is not None and observed_evidence.aspect_ratio >= 1.25:
            scores["backpack"] += 0.25

    if aspect_ratio is not None:
        if (
            aspect_ratio <= 0.82
            and (wallet_text_supported or (top_width_ratio is not None and top_width_ratio <= 0.82))
            and not backpack_form_supported
        ):
            scores["wallet"] += 0.6
            scores["tote bag"] -= 0.15
            scores["backpack"] -= 0.15
        elif aspect_ratio >= 1.1:
            scores["tote bag"] += 0.25
            scores["handbag"] += 0.15
            scores["backpack"] += 0.4

    if observed_evidence.form_factor_note and "no visible handles" in observed_evidence.form_factor_note:
        if backpack_form_supported:
            scores["backpack"] += 0.2
            scores["wallet"] -= 0.1
        else:
            scores["wallet"] += 0.8
            scores["tote bag"] -= 0.4
            scores["handbag"] -= 0.25
            scores["backpack"] -= 0.2

    best_type = max(scores.items(), key=lambda item: item[1])[0]
    if scores[best_type] >= scores.get(initial_canonical_product_type, float("-inf")) + 0.25:
        return best_type
    return initial_canonical_product_type


def rewrite_evidence_for_canonical_type(
    observed_evidence: ObservedEvidenceSpec,
    *,
    canonical_product_type: str,
) -> ObservedEvidenceSpec:
    updated_hard_facts = [f"the product remains a {canonical_product_type}"]
    updated_hard_facts.extend(TYPE_STRUCTURAL_FACTS.get(canonical_product_type, ()))
    updated_hard_facts.extend(
        fact
        for fact in observed_evidence.hard_facts
        if not fact.startswith("the product remains a ")
    )
    return observed_evidence.model_copy(update={"hard_facts": _dedupe_strings(updated_hard_facts)})


def _should_suppress_contrast_panel_inference(
    *,
    canonical_product_type: str,
    shape_profile: dict[str, Any],
    contrast_panel_note: str | None,
) -> bool:
    if contrast_panel_note is None:
        return False
    aspect_ratio = shape_profile.get("aspect_ratio")
    top_width_ratio = shape_profile.get("top_width_ratio")
    if canonical_product_type == "table lamp" and aspect_ratio is not None and aspect_ratio >= 1.6:
        if top_width_ratio is None or top_width_ratio >= 0.9:
            return True
    return False


def build_identity_phrase(
    product_title: str,
    hint_phrases: Sequence[str],
    selected_phrase: str,
    *,
    canonical_product_type: str,
) -> str:
    title = " ".join(product_title.lower().split()).strip()
    canonical_tokens = set(_tokens(canonical_product_type))
    if title and (
        canonical_product_type in title or canonical_tokens.intersection(_tokens(title))
    ):
        return title
    if title:
        return f"{title} {canonical_product_type}".strip()

    phrase_tokens = set(_tokens(selected_phrase))
    preferred_hint = next(
        (
            hint
            for hint in hint_phrases
            if phrase_tokens.intersection(_tokens(hint)) or canonical_product_type in hint.lower()
        ),
        hint_phrases[0] if hint_phrases else canonical_product_type,
    )
    preferred_hint = " ".join(preferred_hint.lower().split()).strip()
    if canonical_product_type in preferred_hint:
        return preferred_hint
    return f"{preferred_hint} {canonical_product_type}".strip()


def build_refined_identity_phrase(
    *,
    selected_phrase: str,
    hint_phrases: Sequence[str],
    canonical_product_type: str,
) -> str:
    selected_tokens = set(_tokens(selected_phrase))
    preferred_hint = next(
        (hint for hint in hint_phrases if canonical_product_type in hint.lower()),
        None,
    )
    if preferred_hint is None:
        preferred_hint = next(
        (
            hint
            for hint in hint_phrases
            if selected_tokens.intersection(_tokens(hint))
        ),
        None,
        )
    if preferred_hint is None:
        preferred_hint = selected_phrase or canonical_product_type
    normalized = " ".join(preferred_hint.lower().split()).strip()
    if canonical_product_type in normalized:
        return normalized
    return f"{normalized} {canonical_product_type}".strip()


def has_weak_shape_evidence(
    selected_phrase: str,
    hint_phrases: Sequence[str],
    *,
    canonical_product_type: str,
) -> bool:
    phrase_tokens = set(_tokens(selected_phrase))
    canonical_tokens = set(_tokens(canonical_product_type))
    if phrase_tokens.intersection(canonical_tokens):
        return False
    hint_tokens = set(token for hint in hint_phrases for token in _tokens(hint))
    return bool(hint_tokens.intersection(canonical_tokens))


def infer_scene_families(caption: str) -> tuple[str, ...]:
    tokens = set(_tokens(caption))
    families = [
        label
        for label, keywords in SCENE_FAMILY_KEYWORDS.items()
        if any(keyword in tokens for keyword in keywords)
    ]
    return tuple(families or ("editorial_interior",))


def infer_support_relations(caption: str) -> tuple[str, ...]:
    tokens = set(_tokens(caption))
    relations = [
        label
        for label, keywords in SUPPORT_RELATION_KEYWORDS.items()
        if any(keyword in tokens for keyword in keywords)
    ]
    return tuple(relations)


def choose_support_relation(
    identity: ProductIdentitySpec,
    top_matches: Sequence[RetrievalCandidate],
) -> str:
    support_mode = identity.support_mode or "supported_display"
    allowed = allowed_support_relations_for_identity(identity)
    weighted_votes: dict[str, float] = {relation: 0.0 for relation in allowed}
    for rank, candidate in enumerate(top_matches):
        weight = 1.0 / float(rank + 1)
        for relation in candidate.support_relations:
            if relation in weighted_votes:
                weighted_votes[relation] += weight
    if backpack_harness_face_observed(identity):
        if "carried_by_hand" in weighted_votes:
            weighted_votes["carried_by_hand"] += 0.45
        if "worn_on_body" in weighted_votes:
            weighted_votes["worn_on_body"] -= 0.2
    best_relation, best_score = max(weighted_votes.items(), key=lambda item: item[1])
    if best_score <= 0.0:
        return default_support_relation_for_identity(identity)
    return best_relation


def allowed_support_relations_for_identity(identity: ProductIdentitySpec) -> tuple[str, ...]:
    if identity.canonical_product_type in (STRUCTURED_DISPLAY_CANONICAL_TYPES - DRINKWARE_CANONICAL_TYPES):
        return ("standing_on_surface",)
    if identity.canonical_product_type in (BEDDING_CANONICAL_TYPES | {"pet bed"}):
        return ("resting_on_surface",)
    if identity.canonical_product_type == "backpack":
        return ("worn_on_body", "carried_by_hand")
    if identity.canonical_product_type == "decorative pillow":
        return ("resting_with_back_support", "resting_on_surface")
    if identity.category in {"furniture", "kitchen appliance", "home lighting"}:
        return ("standing_on_surface",)
    if identity.category in {"bedding", "pet home"}:
        return ("resting_on_surface",)
    support_mode = identity.support_mode or "supported_display"
    return SUPPORT_MODE_RELATION_COMPATIBILITY.get(support_mode, ("resting_on_surface",))


def default_support_relation_for_identity(identity: ProductIdentitySpec) -> str:
    allowed = allowed_support_relations_for_identity(identity)
    support_mode = identity.support_mode or "supported_display"
    if backpack_harness_face_observed(identity) and "carried_by_hand" in allowed:
        return "carried_by_hand"
    preferred = SUPPORT_RELATION_DEFAULTS.get(support_mode, "resting_on_surface")
    if preferred in allowed:
        return preferred
    return allowed[0] if allowed else "resting_on_surface"


def backpack_harness_face_observed(identity: ProductIdentitySpec) -> bool:
    if str(identity.canonical_product_type or "").strip().lower() != "backpack":
        return False
    evidence = identity.observed_evidence
    evidence_text = " ".join(
        part
        for part in (
            evidence.upper_region_note or "",
            evidence.evidence_caption or "",
            " ".join(evidence.hard_facts),
        )
        if part
    ).lower()
    if (evidence.upper_component_count or 0) >= 2 and (
        "multiple narrow segments" in evidence_text or "harness" in evidence_text or "back-panel" in evidence_text
    ):
        return True
    return bool("multiple narrow segments" in evidence_text and ("harness" in evidence_text or "back-panel" in evidence_text))


def choose_scene_family(
    identity: ProductIdentitySpec,
    top_matches: Sequence[RetrievalCandidate],
    *,
    support_relation: str,
) -> str:
    allowed = SCENE_SUPPORT_COMPATIBILITY.get(support_relation, ("editorial_interior",))
    default_scene = identity.default_scene_family or SCENE_FAMILY_DEFAULTS_BY_SUPPORT.get(
        support_relation,
        "editorial_interior",
    )
    weighted_votes: dict[str, float] = {
        scene: (0.35 if scene == default_scene else 0.0)
        for scene in allowed
    }
    if identity.style_persona == "sport_utility" and "tabletop_display" in weighted_votes:
        weighted_votes["tabletop_display"] += 0.35
    if identity.canonical_product_type == "backpack" and "outdoor_lifestyle" in weighted_votes:
        weighted_votes["outdoor_lifestyle"] += 0.35
    if identity.category == "home lighting" and "furnished_interior" in weighted_votes:
        weighted_votes["furnished_interior"] += 0.35
    if identity.style_persona == "cozy_home" and "furnished_interior" in weighted_votes:
        weighted_votes["furnished_interior"] += 0.3
    if identity.style_persona == "playful_casual" and "fashion_lifestyle" in weighted_votes:
        weighted_votes["fashion_lifestyle"] += 0.2
    for rank, candidate in enumerate(top_matches):
        weight = 1.0 / float(rank + 1)
        candidate_scene_votes = set(candidate.scene_families)
        candidate_scene_votes.update(candidate.scenario_slots)
        if candidate.default_scene_family:
            candidate_scene_votes.add(candidate.default_scene_family)
        for scene in candidate_scene_votes:
            if scene in weighted_votes:
                weighted_votes[scene] += weight
    best_scene, best_score = max(weighted_votes.items(), key=lambda item: item[1])
    if best_score <= 0.0:
        return default_scene
    if (
        identity.support_mode == "self_supporting_display"
        and best_scene == "editorial_interior"
    ):
        return default_scene
    return best_scene


def build_style_plan(
    identity: ProductIdentitySpec,
    top_matches: Sequence[RetrievalCandidate],
    *,
    scene_family: str,
    support_relation: str,
) -> list[str]:
    evidence = identity.observed_evidence
    evidence_sensitive = bool(
        evidence.coverage_class in {"full_visible_surface_pattern", "broad_visible_surface_pattern"}
        or evidence.upper_region_note
        or evidence.upper_component_state == "absent"
        or evidence.material_note
        or "distinct_boundary_trim" in evidence.evidence_tags
    )
    atoms: list[str] = ["clear hero framing", "commercial product storytelling"]
    atoms.extend(SUPPORT_RELATION_STYLE_ATOMS.get(support_relation, ()))
    atoms.extend(SCENE_FAMILY_STYLE_ATOMS.get(scene_family, ()))
    compatible_from_matches = [
        atom
        for candidate in top_matches
        for atom in candidate.style_atoms
        if is_style_atom_compatible(atom, scene_family=scene_family, support_relation=support_relation)
    ]
    if evidence_sensitive:
        compatible_from_matches = [
            atom
            for atom in compatible_from_matches
            if atom not in {"commercial product storytelling", "clear hero framing"}
        ][:1]
    atoms.extend(compatible_from_matches)
    if identity.style_persona == "playful_casual":
        atoms.append("casual approachable styling with no formal business attire")
    elif identity.style_persona == "sport_utility":
        atoms.append("clean active styling with bright value separation")
    elif identity.style_persona == "cozy_home":
        atoms.append("warm relaxed home styling")
    if identity.category == "apparel":
        atoms.append("editorial apparel styling")
    elif identity.category == "footwear":
        atoms.append("active footwear styling with grounded stance")
    elif identity.canonical_product_type == "backpack":
        atoms.append("wearable utility framing with visible strap support")
    elif identity.category == "home lighting":
        atoms.append("interior product styling with stable surface support")
    max_atoms = 4 if evidence_sensitive else 6
    return _dedupe_strings(atoms)[:max_atoms]


def is_style_atom_compatible(atom: str, *, scene_family: str, support_relation: str) -> bool:
    lowered = atom.lower()
    if any(token in lowered for token in ("tabletop", "support-surface")) and support_relation not in {
        "standing_on_surface",
        "resting_on_surface",
    }:
        return False
    if "furnished" in lowered and scene_family != "furnished_interior":
        return False
    if "human-in-use" in lowered and support_relation not in {"carried_by_hand", "worn_on_body"}:
        return False
    if "outdoor" in lowered and scene_family != "outdoor_lifestyle":
        return False
    return True


def build_semantic_constraints(
    identity: ProductIdentitySpec,
    *,
    scene_family: str,
    support_relation: str,
) -> list[str]:
    constraints = [
        SUPPORT_RELATION_CONSTRAINTS.get(
            support_relation,
            "show the product with visible, physically plausible support and grounded contact",
        ),
        "use one coherent scene and support plan instead of mixing conflicting placements or environments",
    ]
    if identity.rigid_vs_soft == "soft":
        constraints.append(
            "respect gravity and material compliance so soft products show believable compression, sag, or drape"
        )
    if identity.stable_base is False:
        constraints.append(
            "do not depict the product as freestanding unless visible support makes that pose physically plausible"
        )
    constraints.append(f"keep the scene consistent with a {scene_family.replace('_', ' ')} presentation")
    return _dedupe_strings(constraints)


def infer_scenario_slots(caption: str) -> tuple[str, ...]:
    return infer_scene_families(caption)


def infer_style_atoms(caption: str) -> tuple[str, ...]:
    atoms = ["clear hero framing", "commercial product storytelling"]
    support_relations = set(infer_support_relations(caption))
    scene_families = set(infer_scene_families(caption))
    if support_relations.intersection({"carried_by_hand", "worn_on_body"}):
        atoms.append("human-in-use framing")
    if support_relations.intersection({"standing_on_surface", "resting_on_surface"}):
        atoms.append("anchored support-surface composition")
    if "furnished_interior" in scene_families:
        atoms.append("soft furnished-environment context")
    if "outdoor_lifestyle" in scene_families:
        atoms.append("outdoor lifestyle context")
    if "retail_display" in scene_families:
        atoms.append("retail display context")
    return tuple(_dedupe_strings(atoms))


def extract_material_tags(text: str) -> set[str]:
    lowered = text.lower()
    tags: set[str] = set()
    for label, patterns in MATERIAL_TEXT_TOKENS.items():
        if any(pattern in lowered for pattern in patterns):
            tags.add(label)
    return tags


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _load_mask_array(mask_path: Path | None, *, source_shape: tuple[int, int]) -> np.ndarray | None:
    if mask_path is None or not mask_path.exists():
        return None
    with Image.open(mask_path) as mask_handle:
        mask = mask_handle.convert("L")
        if mask.size != (source_shape[1], source_shape[0]):
            mask = mask.resize((source_shape[1], source_shape[0]), Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


def _nearest_color_name(pixel: Sequence[float]) -> str:
    red, green, blue = (max(0.0, min(float(value) / 255.0, 1.0)) for value in pixel[:3])
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    spread = max(red, green, blue) - min(red, green, blue)
    if value <= 0.14:
        return "black"
    if saturation <= 0.1 or spread <= 0.06:
        if value >= 0.92:
            return "white"
        if value >= 0.62:
            return "beige" if (red >= blue or hue < 0.16 or hue > 0.9) else "gray"
        if value >= 0.32:
            return "brown" if (red > blue + 0.03 and hue < 0.16) else "gray"
        return "black"
    if saturation <= 0.22:
        if value >= 0.78:
            return "beige" if hue < 0.2 or hue > 0.9 else "gray"
        if value >= 0.46:
            return "beige" if hue < 0.16 or hue > 0.92 else "gray"
        if value >= 0.2:
            return "brown" if hue < 0.16 else "gray"
        return "black"
    if saturation <= 0.35 and hue < 0.18 and value >= 0.6:
        return "beige"
    if saturation <= 0.35 and value <= 0.42:
        if hue < 0.14:
            return "brown"
        return "gray"
    if hue < 0.03 or hue >= 0.97:
        return "red"
    if hue < 0.09:
        return "brown" if value < 0.45 else "orange"
    if hue < 0.16:
        return "gold" if value < 0.72 else "yellow"
    if hue < 0.28:
        return "green"
    if hue < 0.45:
        return "teal"
    if hue < 0.72:
        return "blue"
    if hue < 0.84:
        return "purple"
    return "pink"


def _named_color_family(name: str) -> str:
    if name in {"black", "gray", "brown"}:
        return "dark-neutral"
    if name in {"white", "beige"}:
        return "light-neutral"
    if name in {"blue", "teal", "purple", "green"}:
        return "cool-toned"
    return "warm-toned"


def _named_color_distribution(pixels: np.ndarray) -> dict[str, float]:
    counts: dict[str, int] = {}
    for pixel in pixels:
        name = _nearest_color_name(pixel)
        counts[name] = counts.get(name, 0) + 1
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {name: count / total for name, count in counts.items()}


def _weighted_structural_color_distribution(pixels: np.ndarray) -> dict[str, float]:
    counts: dict[str, float] = {}
    structural_counts: dict[str, float] = {}
    for pixel in pixels:
        name = _nearest_color_name(pixel)
        red, green, blue = (max(0.0, min(float(value) / 255.0, 1.0)) for value in pixel[:3])
        _, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
        weight = 1.0
        if saturation <= 0.22:
            weight += 0.65
        if value <= 0.42:
            weight += 0.5
        if saturation >= 0.58 and value >= 0.7 and name in {"gold", "yellow", "pink", "orange"}:
            weight -= 0.25
        counts[name] = counts.get(name, 0.0) + weight
        if saturation <= 0.28 or value <= 0.35:
            structural_counts[name] = structural_counts.get(name, 0.0) + 1.0
    if structural_counts:
        structural_name, structural_weight = max(structural_counts.items(), key=lambda item: item[1])
        counts[structural_name] = counts.get(structural_name, 0.0) + max(structural_weight, len(pixels) * 0.18)
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {name: count / total for name, count in counts.items()}


def _select_attached_component_color(pixels: np.ndarray) -> tuple[str, float]:
    dominant_name, dominant_ratio = _dominant_named_color_with_ratio(pixels)
    if pixels.size == 0:
        return dominant_name, dominant_ratio
    stride = max(1, int(math.ceil(len(pixels) / 2000.0)))
    sampled = pixels[::stride]
    distribution = _named_color_distribution(sampled)
    structural_distribution = _weighted_structural_color_distribution(sampled)
    if structural_distribution:
        structural_name, structural_ratio = max(structural_distribution.items(), key=lambda item: item[1])
    else:
        structural_name, structural_ratio = dominant_name, dominant_ratio
    sampled_rgb = np.clip(sampled.astype(np.float32) / 255.0, 0.0, 1.0)
    sampled_hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in sampled_rgb], dtype=np.float32)
    mean_value = float(sampled_hsv[:, 2].mean()) if sampled_hsv.size else 0.0
    if (
        mean_value <= 0.32
        and _named_color_family(structural_name) == "dark-neutral"
        and structural_ratio >= 0.12
    ):
        return structural_name, float(max(structural_ratio, dominant_ratio * 0.85))
    dark_candidates = {
        name: ratio
        for name, ratio in distribution.items()
        if _named_color_family(name) == "dark-neutral"
    }
    if dark_candidates and _named_color_family(dominant_name) != "dark-neutral":
        dark_name, dark_ratio = max(dark_candidates.items(), key=lambda item: item[1])
        if dark_ratio >= max(0.16, dominant_ratio * 0.55):
            return dark_name, dark_ratio
    return dominant_name, dominant_ratio


def _reorder_structural_palette(ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not ranked:
        return ranked
    top_name, top_score = ranked[0]
    if _named_color_family(top_name) == "warm-toned":
        for index, (name, score) in enumerate(ranked[1:], start=1):
            if _named_color_family(name) in {"dark-neutral", "cool-toned"} and score >= top_score * 0.22:
                return [ranked[index], *ranked[:index], *ranked[index + 1 :]]
    return ranked


def _named_color_distance(left: str | None, right: str | None) -> float:
    if left is None or right is None:
        return 0.0
    left_rgb = EVIDENCE_COLOR_SWATCHES.get(left)
    right_rgb = EVIDENCE_COLOR_SWATCHES.get(right)
    if left_rgb is None or right_rgb is None:
        return 0.0
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left_rgb, right_rgb, strict=False))
    )


def _dominant_named_color(pixels: np.ndarray) -> str:
    dominant, _ = _dominant_named_color_with_ratio(pixels)
    return dominant


def _dominant_named_color_with_ratio(pixels: np.ndarray) -> tuple[str, float]:
    if pixels.size == 0:
        return "mixed", 0.0
    stride = max(1, int(math.ceil(len(pixels) / 2000.0)))
    sampled = pixels[::stride]
    distribution = _named_color_distribution(sampled)
    if not distribution:
        return "mixed", 0.0
    dominant_name, dominant_ratio = max(distribution.items(), key=lambda item: item[1])
    return dominant_name, float(dominant_ratio)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    return center & ~neighbors


def _erode_mask(mask: np.ndarray, *, steps: int) -> np.ndarray:
    eroded = mask.astype(bool)
    for _ in range(max(0, steps)):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, :-2]
            & padded[:-2, 2:]
            & padded[2:, :-2]
            & padded[2:, 2:]
        )
        if not eroded.any():
            return mask.astype(bool)
    return eroded


def _dilate_mask(mask: np.ndarray, *, steps: int) -> np.ndarray:
    dilated = mask.astype(bool)
    for _ in range(max(0, steps)):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        dilated = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return dilated


def _mask_band_width(mask: np.ndarray, start_ratio: float, end_ratio: float) -> float:
    ys, _ = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    y0 = ys.min()
    y1 = ys.max() + 1
    start = y0 + int(round((y1 - y0) * start_ratio))
    end = y0 + max(int(round((y1 - y0) * end_ratio)), 1)
    widths: list[int] = []
    for row in range(max(y0, start), min(y1, end)):
        xs = np.nonzero(mask[row])[0]
        if len(xs) >= 2:
            widths.append(int(xs.max() - xs.min() + 1))
    if not widths:
        return 0.0
    return float(sum(widths) / len(widths))


def _mask_row_widths(mask: np.ndarray) -> list[int]:
    ys, _ = np.nonzero(mask)
    if len(ys) == 0:
        return []
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    widths: list[int] = []
    for row in range(y0, y1):
        xs = np.nonzero(mask[row])[0]
        widths.append(0 if len(xs) == 0 else int(xs.max() - xs.min() + 1))
    return widths


def _estimate_body_shoulder_row(mask: np.ndarray) -> int:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    widths = []
    for row in range(y0, y1):
        row_xs = np.nonzero(mask[row])[0]
        widths.append(len(row_xs))
    if not widths:
        return y0
    max_width = max(widths)
    if max_width <= 0:
        return y0
    for index, width in enumerate(widths):
        if width < 0.78 * max_width:
            continue
        window = widths[index : min(index + 4, len(widths))]
        if sum(item >= 0.74 * max_width for item in window) >= max(2, len(window) - 1):
            return y0 + index
    return y0 + max(1, int(round((y1 - y0) * 0.22)))


def _extract_core_body_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros_like(mask, dtype=bool)
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    width = float(x1 - x0)
    height = float(y1 - y0)
    aspect_ratio = height / max(width, 1.0)
    if aspect_ratio >= 1.45:
        start_ratio, end_ratio = 0.56, 0.92
    else:
        start_ratio, end_ratio = 0.18, 0.88
    start = y0 + int(round(height * start_ratio))
    end = y0 + max(int(round(height * end_ratio)), 1)
    core_mask = np.zeros_like(mask, dtype=bool)
    core_mask[max(y0, start) : min(y1, end), :] = mask[max(y0, start) : min(y1, end), :]
    eroded = _erode_mask(core_mask, steps=1)
    if eroded.any() and eroded.sum() >= max(64, int(mask.sum() * 0.12)):
        return eroded
    return core_mask


def _extract_dark_structure_mask(source: np.ndarray, upper_mask: np.ndarray) -> np.ndarray:
    if not upper_mask.any():
        return np.zeros_like(upper_mask, dtype=bool)
    pixels = np.clip(source[upper_mask].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in pixels], dtype=np.float32)
    if hsv.size == 0:
        return np.zeros_like(upper_mask, dtype=bool)
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    channel_spread = np.max(pixels, axis=1) - np.min(pixels, axis=1)
    dark_like = (
        (value <= 0.34)
        | ((value <= 0.46) & (saturation <= 0.38))
        | ((value <= 0.42) & (channel_spread <= 0.18))
    )
    dark_mask = np.zeros_like(upper_mask, dtype=bool)
    ys, xs = np.nonzero(upper_mask)
    dark_mask[ys[dark_like], xs[dark_like]] = True
    return dark_mask


def _component_structural_score(
    component: np.ndarray,
    source: np.ndarray,
    *,
    upper_start: int,
    upper_end: int,
) -> float:
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return -1.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    fill_ratio = float(len(xs)) / max(width * height, 1.0)
    band_height = max(float(upper_end - upper_start), 1.0)
    top_band_end = upper_start + int(round(band_height * 0.32))
    upper_band_end = upper_start + int(round(band_height * 0.42))
    top_anchor = 1.0 if int(ys.min()) <= top_band_end else 0.0
    top_presence = float(np.mean(ys <= upper_band_end))
    pixels = np.clip(source[component].astype(np.float32) / 255.0, 0.0, 1.0)
    hsv = np.asarray([colorsys.rgb_to_hsv(*pixel) for pixel in pixels], dtype=np.float32)
    value = hsv[:, 2] if hsv.size else np.asarray([], dtype=np.float32)
    saturation = hsv[:, 1] if hsv.size else np.asarray([], dtype=np.float32)
    dark_ratio = float(np.mean((value <= 0.38) | ((value <= 0.46) & (saturation <= 0.38)))) if len(value) else 0.0
    mean_value = float(value.mean()) if len(value) else 0.0
    mean_saturation = float(saturation.mean()) if len(saturation) else 0.0
    slenderness = min(height / max(width, 1.0), 3.0) / 3.0
    score = (
        1.2 * top_anchor
        + 0.9 * top_presence
        + 0.9 * dark_ratio
        + 0.35 * slenderness
        + 0.15 * (1.0 if fill_ratio <= 0.78 else 0.0)
    )
    if int(ys.min()) > upper_start + int(round(band_height * 0.46)):
        score -= 0.55
    if mean_value > 0.5 and mean_saturation > 0.45:
        score -= 0.3
    return score


def _prioritize_upper_components(
    components: Sequence[np.ndarray],
    source: np.ndarray,
    *,
    upper_start: int,
    upper_end: int,
) -> list[np.ndarray]:
    scored_components: list[tuple[float, int, int, np.ndarray]] = []
    for component in components:
        ys, _ = np.nonzero(component)
        if len(ys) == 0:
            continue
        score = _component_structural_score(
            component,
            source,
            upper_start=upper_start,
            upper_end=upper_end,
        )
        scored_components.append((score, int(ys.min()), int(component.sum()), component))
    if not scored_components:
        return []
    scored_components.sort(key=lambda item: (-item[0], item[1], -item[2]))
    best_score = scored_components[0][0]
    minimum_score = max(0.9, best_score * 0.62)
    selected = [component for score, _, _, component in scored_components if score >= minimum_score]
    return selected[: max(1, min(2, len(selected)))]


def _component_geometry(component: np.ndarray, *, total_product_pixels: float | None = None) -> dict[str, float]:
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return {
            "width": 0.0,
            "height": 0.0,
            "fill_ratio": 0.0,
            "area_ratio": 0.0,
            "x0": 0.0,
            "x1": 0.0,
        }
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    fill_ratio = len(xs) / max(width * height, 1.0)
    area_ratio = len(xs) / max(float(total_product_pixels or 0.0), 1.0) if total_product_pixels else 0.0
    return {
        "width": width,
        "height": height,
        "fill_ratio": float(fill_ratio),
        "area_ratio": float(area_ratio),
        "x0": float(xs.min()),
        "x1": float(xs.max()),
    }


def _looks_like_omitted_upper_attachment(
    components: Sequence[np.ndarray],
    *,
    main_body_width: float,
    total_product_pixels: float,
) -> bool:
    if len(components) < 2 or main_body_width <= 0 or total_product_pixels <= 0:
        return False
    ranked = sorted(
        (_component_geometry(component, total_product_pixels=total_product_pixels) for component in components),
        key=lambda item: item["area_ratio"],
        reverse=True,
    )[:2]
    if len(ranked) < 2:
        return False
    left, right = sorted(ranked, key=lambda item: item["x0"])
    if left["fill_ratio"] < 0.58 or right["fill_ratio"] < 0.58:
        return False
    if left["width"] < 0.16 * main_body_width or right["width"] < 0.16 * main_body_width:
        return False
    gap = right["x0"] - left["x1"] - 1.0
    if gap < max(10.0, main_body_width * 0.06):
        return False
    combined_area_ratio = left["area_ratio"] + right["area_ratio"]
    wide_shallow_split = (
        left["width"] / max(left["height"], 1.0) >= 2.3
        and right["width"] / max(right["height"], 1.0) >= 2.3
        and combined_area_ratio >= 0.075
    )
    large_filled_split = (
        left["area_ratio"] >= 0.12
        and right["area_ratio"] >= 0.12
        and combined_area_ratio >= 0.32
    )
    return wide_shallow_split or large_filled_split


def _count_mask_components(mask: np.ndarray, *, min_pixels: int) -> int:
    return len(_mask_components(mask, min_pixels=min_pixels))


def _mask_components(mask: np.ndarray, *, min_pixels: int) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component_pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component_pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            if len(component_pixels) >= min_pixels:
                component_mask = np.zeros_like(mask, dtype=bool)
                ys, xs = zip(*component_pixels, strict=False)
                component_mask[ys, xs] = True
                components.append(component_mask)
    return components


def _is_attached_component_candidate(
    component: np.ndarray,
    *,
    main_body_width: float,
    total_product_pixels: float | None = None,
    shape_aspect_ratio: float | None = None,
) -> bool:
    geometry = _component_geometry(component, total_product_pixels=total_product_pixels)
    width = geometry["width"]
    height = geometry["height"]
    fill_ratio = geometry["fill_ratio"]
    area_ratio = geometry["area_ratio"]
    if width <= 0 or height <= 0:
        return False
    if area_ratio >= 0.24 and fill_ratio >= 0.58:
        return False
    if (
        main_body_width > 0
        and area_ratio <= 0.16
        and width >= 0.45 * main_body_width
        and width <= 0.92 * main_body_width
        and height <= max(32.0, width * 0.82)
        and (shape_aspect_ratio is None or shape_aspect_ratio >= 1.35)
    ):
        return True
    if main_body_width > 0 and width <= 0.4 * main_body_width and height >= max(6.0, width * 0.5):
        return True
    if (
        main_body_width > 0
        and width <= 0.82 * main_body_width
        and height >= max(6.0, width * 0.5)
        and height <= max(32.0, width * 1.35)
        and fill_ratio <= 0.72
    ):
        return True
    if fill_ratio < 0.65 and height >= max(6.0, width * 0.45):
        return True
    return False


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        deduped.append(normalized)
        seen.add(key)
    return deduped


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _as_posix(value: str | None) -> str:
    if not value:
        return ""
    return Path(value).as_posix()


def _board_image_reference(
    board_root: Path,
    value: str | None,
    *,
    asset_group: str,
    asset_stem: str,
) -> str:
    if not value:
        return ""

    source_path = Path(value)
    if not source_path.is_absolute():
        staged_relative_path = board_root / source_path
        if staged_relative_path.exists():
            return source_path.as_posix()
    if not source_path.exists():
        return ""

    try:
        return source_path.relative_to(board_root).as_posix()
    except ValueError:
        suffix = source_path.suffix or ".png"
        staged_path = board_root / "board_assets" / asset_group / f"{asset_stem}{suffix}"
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != staged_path.resolve():
            shutil.copy2(source_path, staged_path)
        return staged_path.relative_to(board_root).as_posix()


def _sanitize_upstream_observed_evidence(
    board_root: Path,
    *,
    seed_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    sanitized = dict(payload)
    if sanitized.get("reference_cutout_path"):
        sanitized["reference_cutout_path"] = _board_image_reference(
            board_root,
            sanitized.get("reference_cutout_path"),
            asset_group="reference",
            asset_stem=f"{seed_id}.evidence_cutout",
        )
    if sanitized.get("reference_silhouette_path"):
        sanitized["reference_silhouette_path"] = _board_image_reference(
            board_root,
            sanitized.get("reference_silhouette_path"),
            asset_group="reference",
            asset_stem=f"{seed_id}.evidence_silhouette",
        )
    return sanitized


def _sanitize_upstream_candidate_prompts(
    board_root: Path,
    *,
    seed_id: str,
    line_name: str,
    candidate_prompts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized_candidates: list[dict[str, Any]] = []
    for candidate in candidate_prompts:
        sanitized_candidate = dict(candidate)
        prompt_payload = dict(candidate.get("prompt", {}))
        sanitized_references: list[dict[str, Any]] = []
        for reference_index, reference in enumerate(prompt_payload.get("reference_images", ())):
            sanitized_reference = dict(reference)
            if sanitized_reference.get("path"):
                sanitized_reference["path"] = _board_image_reference(
                    board_root,
                    sanitized_reference.get("path"),
                    asset_group="reference",
                    asset_stem=(
                        f"{seed_id}.{line_name}.{candidate.get('mode', 'candidate')}"
                        f".reference_{reference_index}"
                    ),
                )
            sanitized_references.append(sanitized_reference)
        prompt_payload["reference_images"] = sanitized_references
        sanitized_candidate["prompt"] = prompt_payload
        sanitized_candidates.append(sanitized_candidate)
    return sanitized_candidates


def _html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _consistency_class(payload: dict[str, Any]) -> str:
    if payload and not payload.get("is_consistent", True):
        return "flagged"
    return ""


def _consistency_label(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    if payload.get("is_consistent", True):
        predicted = payload.get("predicted_category")
        if predicted:
            return f"| {predicted}"
        return ""
    predicted = payload.get("predicted_category", "unknown")
    expected = payload.get("expected_category", "unknown")
    return f"| flagged: {predicted} vs {expected}"


def _semantic_label(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    if payload.get("is_plausible", True):
        return f"| semantic {payload.get('score', '')}"
    return f"| semantic flag: {payload.get('warning', 'weak plausibility')}"

def _evidence_label(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    if payload.get("is_consistent", True):
        return f"| evidence {payload.get('score', '')}"
    return f"| evidence flag: {payload.get('warning', 'weak evidence consistency')}"


def _combined_flag_classes(
    category_payload: dict[str, Any],
    semantic_payload: dict[str, Any],
    evidence_payload: dict[str, Any] | None = None,
) -> str:
    classes: list[str] = []
    if category_payload and not category_payload.get("is_consistent", True):
        classes.append("flagged")
    if semantic_payload and not semantic_payload.get("is_plausible", True):
        classes.append("flagged")
    if evidence_payload and not evidence_payload.get("is_consistent", True):
        classes.append("flagged")
    return " ".join(classes)
