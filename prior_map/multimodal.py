"""Multimodal evidence contracts for prior-map reasoning."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class PriorMapMultimodalContext:
    """Describe one prior-map image supplied as soft LVLM context."""

    image_path: str = ""
    image_role: str = ""
    map_frame_id: str = ""
    alignment_status: str = "unknown"
    text_context: str = ""
    image_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Hash a local image so it can version cache entries."""

        if self.image_path and not self.image_sha256:
            object.__setattr__(self, "image_sha256", sha256_file(self.image_path))

    @property
    def available(self) -> bool:
        """Return whether a local image can be sent to an LVLM."""

        return bool(self.image_path and Path(self.image_path).is_file())

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly context metadata."""

        return asdict(self)

    def as_image_content(self) -> Optional[dict[str, Any]]:
        """Encode the local image as an OpenAI-compatible image block."""

        if not self.available:
            return None
        suffix = Path(self.image_path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        data = base64.b64encode(Path(self.image_path).read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file, or an empty string on failure."""

    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError):
        return ""


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible payload for evidence-version cache keys."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["PriorMapMultimodalContext", "sha256_file", "stable_payload_hash"]
