"""Pick a short video from a segment and encode each frame as PNG.

Jev 1.14 accepts still PNG and JPEG images. Encoding stays in the standard
library so the PEBBLE environment does not need Pillow.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

# TypeSafe image limits for Jev, inclusive.
MIN_SIDE = 256
MAX_SIDE = 1568
MIN_PIXELS = 78_400
MAX_PIXELS = 2_458_624

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def check_frame_size(width: int, height: int) -> None:
    """Reject sizes Jev 1.14 will refuse. The API does not resize images."""
    if width < MIN_SIDE or height < MIN_SIDE or width > MAX_SIDE or height > MAX_SIDE:
        raise ValueError(
            f"frame size {width}x{height} is outside Jev's {MIN_SIDE}..{MAX_SIDE} "
            "pixels on each side"
        )
    pixels = width * height
    if pixels < MIN_PIXELS or pixels > MAX_PIXELS:
        raise ValueError(
            f"frame size {width}x{height} has {pixels} pixels; Jev requires "
            f"{MIN_PIXELS}..{MAX_PIXELS} (a 280x280 image is the smallest square)"
        )


def trajectory_frame_indices(length: int, count: int) -> list[int]:
    """Return ``count`` increasing indices spread across a clip.

    Index 0 is the earliest frame and the last index is the latest. A clip
    shorter than ``count`` uses every step, in order.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    count = min(length, count)
    if count == 1:
        return [0]
    return [int(round(i * (length - 1) / (count - 1))) for i in range(count)]


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode one ``(H, W, 3)`` uint8 RGB array as a PNG byte string."""
    frame = np.asarray(rgb)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be uint8 RGB with shape (H, W, 3)")
    if not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    check_frame_size(width, height)
    raw = b"".join(b"\x00" + frame[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )


def png_data_url(rgb: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(encode_png(rgb)).decode("ascii")


def image_part(rgb: np.ndarray) -> dict:
    """OpenRouter Decisions image part. Jev reads these inside ``state``."""
    return {"type": "image_url", "image_url": {"url": png_data_url(rgb)}}
