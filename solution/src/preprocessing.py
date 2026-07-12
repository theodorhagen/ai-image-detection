"""Deterministic image preprocessing shared by preparation and inference."""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps


_RESAMPLING = getattr(Image, "Resampling", Image)


def decode_image_rgb(image_bytes: bytes) -> Image.Image:
    """Decode encoded image bytes and return an EXIF-corrected RGB image."""
    with Image.open(BytesIO(image_bytes)) as image:
        image.load()
        image = ImageOps.exif_transpose(image)

        if "A" in image.getbands():
            foreground = image.convert("RGBA")
            background = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, foreground).convert("RGB")
        else:
            image = image.convert("RGB")

        return image.copy()


def decode_and_resize(image_bytes: bytes, image_size: int) -> np.ndarray:
    """Decode an image and return a contiguous CHW uint8 array."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    image = decode_image_rgb(image_bytes)
    image = ImageOps.fit(
        image,
        (image_size, image_size),
        method=_RESAMPLING.BILINEAR,
        centering=(0.5, 0.5),
    )

    array = np.asarray(image, dtype=np.uint8)
    array = np.moveaxis(array, -1, 0)
    return np.ascontiguousarray(array)


def normalize_uint8_image(
    image: np.ndarray,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert a CHW uint8 image to float32 and optionally standardize channels."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape (3, height, width)")

    normalized = image.astype(np.float32) / 255.0

    if mean is None and std is None:
        return normalized

    if mean is None or std is None:
        raise ValueError("mean and std must either both be set or both be omitted")

    mean_array = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    if np.any(std_array <= 0):
        raise ValueError("std values must be positive")

    return (normalized - mean_array) / std_array
