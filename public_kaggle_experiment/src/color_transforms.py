"""Fixed, geometry-preserving color transformations for controlled evaluation."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageEnhance


TRANSFORMATION_CONFIG: dict[str, dict[str, Any]] = {
    "identity": {"operation": "identity"},
    "brightness_decrease": {"operation": "brightness", "factor": 0.75},
    "brightness_increase": {"operation": "brightness", "factor": 1.25},
    "saturation_decrease": {"operation": "saturation", "factor": 0.65},
    "saturation_increase": {"operation": "saturation", "factor": 1.35},
    "hue_shift_negative": {
        "operation": "hue_shift",
        "fraction_of_hue_cycle": -0.10,
    },
    "hue_shift_positive": {
        "operation": "hue_shift",
        "fraction_of_hue_cycle": 0.10,
    },
    "contrast_decrease": {"operation": "contrast", "factor": 0.80},
    "contrast_increase": {"operation": "contrast", "factor": 1.20},
    "grayscale": {"operation": "grayscale_luminance_to_rgb"},
    "combined_moderate": {
        "operation": "brightness_saturation_hue_contrast",
        "brightness_factor": 1.10,
        "saturation_factor": 1.15,
        "fraction_of_hue_cycle": 0.05,
        "contrast_factor": 1.10,
        "order": ["brightness", "saturation", "hue", "contrast"],
    },
}
TRANSFORMATION_NAMES = tuple(TRANSFORMATION_CONFIG)


def _shift_hue(image: Image.Image, fraction_of_hue_cycle: float) -> Image.Image:
    hsv = image.convert("HSV")
    hue, saturation, value = hsv.split()
    offset = round(fraction_of_hue_cycle * 255)
    shifted_hue = hue.point(lambda channel: (channel + offset) % 256)
    return Image.merge("HSV", (shifted_hue, saturation, value)).convert("RGB")


def apply_color_transform(image: Image.Image, transformation: str) -> Image.Image:
    """Apply one named deterministic transformation and return a new RGB image."""
    if transformation not in TRANSFORMATION_CONFIG:
        choices = ", ".join(TRANSFORMATION_NAMES)
        raise ValueError(f"Unknown transformation {transformation!r}; choose from: {choices}")

    config = TRANSFORMATION_CONFIG[transformation]
    result = image.convert("RGB").copy()
    operation = config["operation"]
    if operation == "identity":
        return result
    if operation == "brightness":
        return ImageEnhance.Brightness(result).enhance(config["factor"])
    if operation == "saturation":
        return ImageEnhance.Color(result).enhance(config["factor"])
    if operation == "hue_shift":
        return _shift_hue(result, config["fraction_of_hue_cycle"])
    if operation == "contrast":
        return ImageEnhance.Contrast(result).enhance(config["factor"])
    if operation == "grayscale_luminance_to_rgb":
        return result.convert("L").convert("RGB")
    if operation == "brightness_saturation_hue_contrast":
        result = ImageEnhance.Brightness(result).enhance(config["brightness_factor"])
        result = ImageEnhance.Color(result).enhance(config["saturation_factor"])
        result = _shift_hue(result, config["fraction_of_hue_cycle"])
        return ImageEnhance.Contrast(result).enhance(config["contrast_factor"])
    raise RuntimeError(f"Unimplemented transformation operation: {operation}")