from __future__ import annotations

DRINKWARE_CANONICAL_TYPES = frozenset(
    {
        "water bottle",
        "mug",
    }
)

KITCHEN_APPLIANCE_CANONICAL_TYPES = frozenset(
    {
        "blender",
        "toaster",
        "coffee maker",
        "slow cooker",
        "food chopper",
    }
)

BEDDING_CANONICAL_TYPES = frozenset(
    {
        "comforter",
        "quilt",
    }
)

SOFT_HOME_CANONICAL_TYPES = frozenset(
    {
        *BEDDING_CANONICAL_TYPES,
        "pet bed",
        "decorative pillow",
    }
)

FURNITURE_CANONICAL_TYPES = frozenset(
    {
        "office chair",
        "folding chair",
    }
)

STRUCTURED_DISPLAY_CANONICAL_TYPES = frozenset(
    {
        *DRINKWARE_CANONICAL_TYPES,
        *KITCHEN_APPLIANCE_CANONICAL_TYPES,
        *FURNITURE_CANONICAL_TYPES,
        "table lamp",
    }
)

MULTIPART_LOCALIZATION_CANONICAL_TYPES = frozenset(
    {
        *KITCHEN_APPLIANCE_CANONICAL_TYPES,
        *FURNITURE_CANONICAL_TYPES,
        "table lamp",
    }
)
