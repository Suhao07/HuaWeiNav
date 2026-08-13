"""Small ROS image encoding helpers used by multimodal runtime adapters.

The module intentionally depends only on the ROS image message shape. It keeps
RGB/mask persistence available in deployments where importing ``cv_bridge`` or
OpenCV in the contract layer would be undesirable.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any, Optional


def encode_ros_image_png(msg: Any) -> Optional[bytes]:
    """Encode common ROS image encodings into a readable PNG byte stream.

    Args:
        msg: ROS-like image message with ``height``, ``width``, ``encoding``,
            ``step`` and ``data`` attributes.

    Returns:
        PNG bytes, or ``None`` when the message shape/encoding is unsupported.
    """

    try:
        height = int(getattr(msg, "height"))
        width = int(getattr(msg, "width"))
        step = int(getattr(msg, "step"))
        data = bytes(getattr(msg, "data"))
    except (AttributeError, TypeError, ValueError):
        return None
    if height <= 0 or width <= 0 or step <= 0:
        return None

    encoding = str(getattr(msg, "encoding", "") or "").lower()
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
        color_type = 2
    elif encoding in {"rgba8", "bgra8"}:
        channels = 4
        color_type = 6
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
        color_type = 0
    else:
        return None

    row_bytes = width * channels
    if len(data) < height * step or step < row_bytes:
        return None
    rows = []
    for row_index in range(height):
        row = bytearray(data[row_index * step : row_index * step + row_bytes])
        if encoding in {"bgr8", "bgra8"}:
            for index in range(0, len(row), channels):
                row[index], row[index + 2] = row[index + 2], row[index]
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)
    return _png_bytes(width, height, color_type, raw)


def write_ros_image_png(msg: Any, path: str | Path) -> Optional[str]:
    """Encode a ROS image and write it to a local PNG path.

    Args:
        msg: ROS-like image message.
        path: Destination PNG path.

    Returns:
        String path when encoding succeeds; otherwise ``None``.
    """

    encoded = encode_ros_image_png(msg)
    if encoded is None:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return str(destination)


def _png_bytes(width: int, height: int, color_type: int, raw: bytes) -> bytes:
    """Build a minimal non-interlaced PNG from scanline bytes."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


__all__ = ["encode_ros_image_png", "write_ros_image_png"]
