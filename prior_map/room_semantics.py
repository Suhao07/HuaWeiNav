"""Prompt-first online room semantic annotation.

SysNav separates geometric room segmentation from room-type reasoning. This
module keeps that separation: geometry creates RoomEvidence and the LVLM
returns an open-ended label. No fixed room taxonomy or object-to-room table is
used here.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from llm_utils.cognav_llm_adapter import get_client_and_model
from llm_utils.lvlm_call_tracker import record_cache_hit
from prompting.registry import ROOM_SEMANTIC
from prompting.schemas import ParsedRoomSemantic
from prompting.templates import ROOM_SEMANTIC_PROMPT

from .multimodal import PriorMapMultimodalContext, sha256_file, stable_payload_hash


@dataclass(frozen=True)
class RoomEvidence:
    """Evidence bundle used to classify one room.

    Args:
        room_uid: Stable room identity.
        rgb_path: Local RGB evidence path.
        room_mask_path: Optional local room mask path.
        visible_object_summary: Object labels visible in the evidence.
        geometry_summary: Bounded geometric facts.
        neighboring_room_uids: Topological neighbors.
        pose: Pose associated with the evidence.
        source: Evidence source.
        metadata: Additional provenance.
    """

    room_uid: str
    rgb_path: str = ""
    room_mask_path: str = ""
    visible_object_summary: tuple[str, ...] = ()
    geometry_summary: dict[str, Any] = field(default_factory=dict)
    neighboring_room_uids: tuple[str, ...] = ()
    pose: tuple[float, ...] = ()
    source: str = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)

    def evidence_hash(self) -> str:
        """Return the content/version hash for this evidence bundle."""

        payload = {
            "room_uid": self.room_uid,
            "rgb": _file_hash_or_ref(self.rgb_path),
            "mask": _file_hash_or_ref(self.room_mask_path),
            "visible_objects": list(self.visible_object_summary),
            "geometry": self.geometry_summary,
            "neighbors": list(self.neighboring_room_uids),
            "source": self.source,
            "metadata": self.metadata,
        }
        return stable_payload_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly evidence metadata without image bytes."""

        payload = asdict(self)
        payload["visible_object_summary"] = list(self.visible_object_summary)
        payload["neighboring_room_uids"] = list(self.neighboring_room_uids)
        payload["pose"] = list(self.pose)
        payload["evidence_hash"] = self.evidence_hash()
        return payload


@dataclass(frozen=True)
class RoomSemanticResult:
    """Open-ended semantic result for one room evidence version.

    Args:
        room_uid: Room identity.
        label: Open-ended room label.
        description: Evidence-grounded explanation.
        confidence: Label confidence.
        alternatives: Other plausible labels.
        evidence_summary: Model evidence summary.
        uncertainty: Ambiguity statement.
        evidence_hash: Evidence version hash.
        source: Result source.
        raw_response: Auditable structured response.
        prompt_version: Prompt version used.
        latency_ms: Wall-clock request latency in milliseconds.
        request_metadata: Request provenance without image bytes.
    """

    room_uid: str
    label: str = "unknown"
    description: str = ""
    confidence: float = 0.0
    alternatives: tuple[str, ...] = ()
    evidence_summary: str = ""
    uncertainty: str = ""
    evidence_hash: str = ""
    source: str = "fallback"
    raw_response: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = ROOM_SEMANTIC.version
    latency_ms: float = 0.0
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly semantic result."""

        payload = asdict(self)
        payload["alternatives"] = list(self.alternatives)
        return payload


class RoomSemanticCache:
    """Evidence-version cache for room semantic results.

    The cache key is scene, room, evidence hash, model, and prompt version.
    Changed evidence therefore creates a new version instead of reusing a stale
    room label.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Create an in-memory cache with optional JSON persistence."""

        self.path = Path(path) if path else None
        self._items: dict[str, RoomSemanticResult] = {}
        if self.path is not None:
            self._load()

    def get(self, key: str) -> Optional[RoomSemanticResult]:
        """Return one cached result, if present."""

        return self._items.get(str(key))

    def put(self, key: str, result: RoomSemanticResult) -> None:
        """Store one result and persist it when configured."""

        self._items[str(key)] = result
        self._save()

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for key, value in dict(payload).items():
                data = dict(value)
                data["alternatives"] = tuple(_texts(data.get("alternatives")))
                self._items[key] = RoomSemanticResult(**data)
        except (OSError, ValueError, TypeError):
            # 缓存损坏不应阻断导航；下一次成功调用会覆盖为新版本。
            self._items = {}

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({key: value.to_dict() for key, value in self._items.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class RoomSemanticClassifier:
    """Classify room evidence with the shared CogNav/OpenAI-compatible client."""

    def __init__(
        self,
        *,
        vlm: str = "cognav",
        scene_id: str = "",
        cache: Optional[RoomSemanticCache] = None,
        prompt_version: str = ROOM_SEMANTIC.version,
        client: Any = None,
        model: str = "",
    ) -> None:
        """Create a prompt-first room classifier."""

        self.vlm = vlm
        self.scene_id = scene_id
        self.cache = cache or RoomSemanticCache()
        self.prompt_version = prompt_version
        self._client = client
        self._model_override = str(model or "")

    def classify(self, evidence: RoomEvidence, *, force: bool = False) -> RoomSemanticResult:
        """Classify one room, reusing only matching evidence versions.

        Args:
            evidence: RGB/mask/geometry bundle for one room.
            force: Ignore a matching cache entry when true.

        Returns:
            Room semantic result. Missing RGB evidence returns unknown without
            making an external request.
        """

        key = self.cache_key(evidence)
        cached = None if force else self.cache.get(key)
        if cached is not None:
            record_cache_hit(ROOM_SEMANTIC.trace_label)
            return RoomSemanticResult(**{**cached.to_dict(), "source": "cache"})

        rgb = _image_context(evidence.rgb_path, role="room_rgb")
        mask = _image_context(evidence.room_mask_path, role="room_mask")
        if rgb is None:
            result = self._fallback(evidence, "room RGB evidence is unavailable")
            self.cache.put(key, result)
            return result
        if os.getenv("LLM_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}:
            result = self._fallback(evidence, "LLM_OFFLINE is enabled")
            self.cache.put(key, result)
            return result

        payload = {
            "room_uid": evidence.room_uid,
            "visible_objects": list(evidence.visible_object_summary),
            "geometry": evidence.geometry_summary,
            "neighbors": list(evidence.neighboring_room_uids),
            "source": evidence.source,
        }
        content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            rgb,
        ]
        if mask is not None:
            content.append(mask)

        client, model = (self._client, self._model_override) if self._client is not None else get_client_and_model(self.vlm)
        request_metadata = {
            "scene_id": self.scene_id,
            "room_uid": evidence.room_uid,
            "evidence_hash": evidence.evidence_hash(),
            "model": model,
            "prompt_version": self.prompt_version,
            "rgb_path": evidence.rgb_path,
            "rgb_sha256": _file_hash_or_ref(evidence.rgb_path),
            "room_mask_path": evidence.room_mask_path,
            "room_mask_sha256": _file_hash_or_ref(evidence.room_mask_path),
            "image_roles": ["room_rgb"] + (["room_mask"] if mask is not None else []),
        }
        started = time.perf_counter()
        completion = client.beta.chat.completions.parse(
            model=model or self.vlm,
            messages=[
                {"role": "system", "content": ROOM_SEMANTIC_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format=ParsedRoomSemantic,
            temperature=0.0,
            trace_label=ROOM_SEMANTIC.trace_label,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw = _model_dump(completion.choices[0].message.parsed)
        raw_text = str(getattr(completion.choices[0].message, "content", "") or "")
        result = RoomSemanticResult(
            room_uid=evidence.room_uid,
            label=_text(raw.get("label"), "unknown"),
            description=_text(raw.get("description")),
            confidence=_bounded_float(raw.get("confidence")),
            alternatives=tuple(_texts(raw.get("alternatives"))),
            evidence_summary=_text(raw.get("evidence_summary")),
            uncertainty=_text(raw.get("uncertainty")),
            evidence_hash=evidence.evidence_hash(),
            source="vlm",
            raw_response={"text": raw_text, "parsed": raw},
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
            request_metadata=request_metadata,
        )
        self.cache.put(key, result)
        return result

    def cache_key(self, evidence: RoomEvidence) -> str:
        """Return the stable evidence-version cache key."""

        return stable_payload_hash(
            {
                "scene_id": self.scene_id,
                "room_uid": evidence.room_uid,
                "evidence_hash": evidence.evidence_hash(),
                "model": self.vlm,
                "prompt_version": self.prompt_version,
            }
        )

    def _fallback(self, evidence: RoomEvidence, reason: str) -> RoomSemanticResult:
        return RoomSemanticResult(
            room_uid=evidence.room_uid,
            label="unknown",
            uncertainty=reason,
            evidence_hash=evidence.evidence_hash(),
            source="fallback",
            raw_response={"reason": reason},
            prompt_version=self.prompt_version,
            request_metadata={
                "scene_id": self.scene_id,
                "room_uid": evidence.room_uid,
                "evidence_hash": evidence.evidence_hash(),
                "model": self._model_override or self.vlm,
                "prompt_version": self.prompt_version,
                "reason": reason,
            },
        )


def room_evidence_from_record(
    room: Any,
    *,
    fallback_rgb_path: str = "",
    pose: tuple[float, ...] = (),
    source: str = "runtime",
) -> RoomEvidence:
    """Build room evidence from a mapper/SysNav room record.

    The adapter accepts optional ``rgb_image_ref``/``room_mask_ref`` metadata.
    A SysNav mask URI is retained as provenance but is not mistaken for RGB.

    Args:
        room: Room-like record.
        fallback_rgb_path: Current RGB path when the room record has no crop.
        pose: Pose associated with the observation.
        source: Evidence source label.

    Returns:
        Platform-neutral ``RoomEvidence``.
    """

    metadata = dict(getattr(room, "metadata", {}) or {})
    rgb_path = str(
        metadata.get("rgb_image_ref")
        or metadata.get("room_rgb_path")
        or getattr(room, "image_ref", None)
        or fallback_rgb_path
        or ""
    )
    if rgb_path.startswith("ros://"):
        rgb_path = ""
    mask_path = str(metadata.get("room_mask_ref") or metadata.get("room_mask_path") or "")
    if mask_path.startswith("ros://"):
        mask_path = ""
    objects = metadata.get("visible_objects") or getattr(room, "objects", ()) or ()
    geometry = {
        "centroid": list(getattr(room, "centroid", ()) or ()),
        "area": metadata.get("area"),
        "explored": bool(getattr(room, "explored", False)),
        "polygon_point_count": metadata.get("polygon_point_count"),
    }
    return RoomEvidence(
        room_uid=str(getattr(room, "uid", "") or "unknown_room"),
        rgb_path=rgb_path,
        room_mask_path=mask_path,
        visible_object_summary=tuple(str(item) for item in objects if str(item).strip()),
        geometry_summary=geometry,
        neighboring_room_uids=tuple(str(item) for item in getattr(room, "neighbors", ()) or ()),
        pose=tuple(float(item) for item in pose),
        source=source,
        metadata=metadata,
    )


def _image_context(path: str, *, role: str) -> Optional[dict[str, Any]]:
    return PriorMapMultimodalContext(image_path=path, image_role=role).as_image_content()


def _file_hash_or_ref(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    return sha256_file(path) if path.is_file() else str(value)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return {key: getattr(value, key) for key in getattr(value, "__annotations__", {})}


def _text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _texts(value: Any) -> list[str]:
    return [_text(item) for item in list(value or ()) if _text(item)]


def _bounded_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "RoomEvidence",
    "RoomSemanticCache",
    "RoomSemanticClassifier",
    "RoomSemanticResult",
    "room_evidence_from_record",
]
