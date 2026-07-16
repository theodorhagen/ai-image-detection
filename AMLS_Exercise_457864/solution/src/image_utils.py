"""Image decoding and metadata extraction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageInfo:
    """Metadata extracted from one encoded image."""

    decode_ok: bool
    image_bytes: int
    width: int | None
    height: int | None
    mode: str | None
    image_format: str | None
    error_type: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the metadata as a plain dictionary."""

        return asdict(self)


def coerce_image_bytes(value: object) -> bytes:
    """Convert supported binary values to bytes."""

    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(value)


def read_image_info(value: object) -> ImageInfo:
    """Decode image bytes and return basic image metadata."""

    image_bytes = coerce_image_bytes(value)
    if not image_bytes:
        return ImageInfo(
            decode_ok=False,
            image_bytes=0,
            width=None,
            height=None,
            mode=None,
            image_format=None,
            error_type="empty_image",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
            image.load()
    except UnidentifiedImageError:
        return ImageInfo(
            decode_ok=False,
            image_bytes=len(image_bytes),
            width=None,
            height=None,
            mode=None,
            image_format=None,
            error_type="unidentified_image",
        )
    except OSError:
        return ImageInfo(
            decode_ok=False,
            image_bytes=len(image_bytes),
            width=None,
            height=None,
            mode=None,
            image_format=None,
            error_type="os_error",
        )

    return ImageInfo(
        decode_ok=True,
        image_bytes=len(image_bytes),
        width=int(width),
        height=int(height),
        mode=str(mode),
        image_format=str(image_format) if image_format is not None else None,
        error_type=None,
    )
