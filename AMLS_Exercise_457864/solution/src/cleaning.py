"""Deterministic label mapping and cleaning rules."""

from __future__ import annotations

from typing import Any


CLEANING_CONFIG: dict[str, Any] = {
    "version": 1,
    "label_mapping": {
        "0": "real",
        "1": "ai_generated",
        "2": "ai_generated",
        "3": "ai_generated",
        "4": "ai_generated",
        "5": "ai_generated",
    },
    "rules": {
        "require_decodable_image": True,
        "min_width": 32,
        "min_height": 32,
        "min_encoded_bytes": 1,
    },
}


def binary_label(source_class: int) -> int:
    """Map source class 0 to real and all other source classes to AI-generated."""

    return 0 if int(source_class) == 0 else 1


def is_clean_image_record(record: dict[str, Any]) -> bool:
    """Evaluate deterministic cleaning rules for one image metadata record."""

    rules = CLEANING_CONFIG["rules"]
    width = record.get("width")
    height = record.get("height")
    image_bytes = record.get("image_bytes")

    if rules["require_decodable_image"] and not bool(record.get("decode_ok")):
        return False
    if image_bytes is None or int(image_bytes) < int(rules["min_encoded_bytes"]):
        return False
    if width is None or int(width) < int(rules["min_width"]):
        return False
    if height is None or int(height) < int(rules["min_height"]):
        return False
    return True
