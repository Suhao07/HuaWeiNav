"""LVLM high-level selection over dynamic prior-map BEV evidence.

The selector follows SysNav's room-level reasoning boundary. It ranks already
generated room/frontier records; it never emits a motion command or grants STOP.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

from llm_utils.cognav_llm_adapter import get_client_and_model
from llm_utils.lvlm_call_tracker import record_cache_hit
from prompting.registry import HIGH_LEVEL_PRIOR_SELECTION
from prompting.schemas import ParsedHighLevelSelection
from prompting.templates import HIGH_LEVEL_PRIOR_MAP_SELECTION_PROMPT

from .multimodal import PriorMapMultimodalContext, stable_payload_hash


@dataclass(frozen=True)
class HighLevelCandidate:
    """Candidate room/frontier shown to the selector.

    Args:
        uid: Runtime candidate identity.
        candidate_type: Candidate family.
        label: Runtime label.
        distance_m: Optional current path estimate.
        room_uid: Related room identity.
        objects: Object labels in the region.
        explored: Whether the region has been explored.
        metadata: Additional factual fields.
    """

    uid: str
    candidate_type: str = "room"
    label: str = ""
    distance_m: Optional[float] = None
    room_uid: Optional[str] = None
    objects: tuple[str, ...] = ()
    explored: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly candidate data."""

        payload = asdict(self)
        payload["objects"] = list(self.objects)
        return payload


@dataclass(frozen=True)
class HighLevelSelectionResult:
    """Validated high-level selection result.

    Args:
        selected_uid: Existing candidate UID selected by the model.
        selected_type: Candidate family.
        decision: Structured selection decision.
        confidence: Model confidence.
        reason: Evidence-grounded explanation.
        alternatives: Alternative candidate UIDs.
        rejected_candidates: Candidate UIDs rejected by the model.
        uncertainty: Ambiguity statement.
        bev_sha256: Dynamic BEV evidence version.
        source: Result source.
        raw_response: Auditable structured response.
        prompt_version: Prompt version used.
        latency_ms: Wall-clock request latency in milliseconds.
        request_metadata: Request provenance without image bytes.
    """

    selected_uid: str = ""
    selected_type: str = ""
    decision: str = "uncertain"
    confidence: float = 0.0
    reason: str = ""
    alternatives: tuple[str, ...] = ()
    rejected_candidates: tuple[str, ...] = ()
    uncertainty: str = ""
    bev_sha256: str = ""
    source: str = "fallback"
    raw_response: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = HIGH_LEVEL_PRIOR_SELECTION.version
    latency_ms: float = 0.0
    request_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly selection result."""

        payload = asdict(self)
        payload["alternatives"] = list(self.alternatives)
        payload["rejected_candidates"] = list(self.rejected_candidates)
        return payload


class HighLevelSelectionCache:
    """Evidence-version cache for room/frontier selection results."""

    def __init__(self) -> None:
        """Create an empty in-memory cache."""

        self._items: dict[str, HighLevelSelectionResult] = {}

    def get(self, key: str) -> Optional[HighLevelSelectionResult]:
        """Return a cached result for one key."""

        return self._items.get(str(key))

    def put(self, key: str, result: HighLevelSelectionResult) -> None:
        """Store a result under one key."""

        self._items[str(key)] = result


class PriorMapHighLevelSelector:
    """Select one existing region using the dynamic BEV and prompt context."""

    def __init__(
        self,
        *,
        vlm: str = "cognav",
        scene_id: str = "",
        cache: Optional[HighLevelSelectionCache] = None,
        prompt_version: str = HIGH_LEVEL_PRIOR_SELECTION.version,
        client: Any = None,
        model: str = "",
    ) -> None:
        """Create a high-level selector."""

        self.vlm = vlm
        self.scene_id = scene_id
        self.cache = cache or HighLevelSelectionCache()
        self.prompt_version = prompt_version
        self._client = client
        self._model_override = str(model or "")

    def select(
        self,
        *,
        instruction: str,
        instruction_plan: Any,
        context: PriorMapMultimodalContext,
        candidates: Iterable[HighLevelCandidate],
        runtime_state: Optional[Mapping[str, Any]] = None,
        force: bool = False,
    ) -> HighLevelSelectionResult:
        """Select an existing candidate using the dynamic BEV.

        Missing image or candidates returns a deterministic fallback. The
        fallback preserves the order supplied by the existing planner.
        """

        candidate_list = tuple(candidates)
        if not candidate_list:
            return HighLevelSelectionResult(decision="uncertain", reason="no candidates")
        candidate_uids = {candidate.uid for candidate in candidate_list if candidate.uid}
        key = self.cache_key(instruction, instruction_plan, context, candidate_list, runtime_state)
        cached = None if force else self.cache.get(key)
        if cached is not None:
            record_cache_hit(HIGH_LEVEL_PRIOR_SELECTION.trace_label)
            return HighLevelSelectionResult(**{**cached.to_dict(), "source": "cache"})
        if not context.available:
            result = self._fallback(candidate_list, context, "dynamic BEV image is unavailable")
            self.cache.put(key, result)
            return result
        if os.getenv("LLM_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}:
            result = self._fallback(candidate_list, context, "LLM_OFFLINE is enabled")
            self.cache.put(key, result)
            return result

        payload = {
            "instruction": instruction,
            "instruction_plan": _json_ready(instruction_plan),
            "runtime_state": dict(runtime_state or {}),
            "candidates": [candidate.to_dict() for candidate in candidate_list],
            "prior_map_context": context.text_context,
            "candidate_uid_contract": sorted(candidate_uids),
        }
        image = context.as_image_content()
        assert image is not None
        client, model = (self._client, self._model_override) if self._client is not None else get_client_and_model(self.vlm)
        request_metadata = {
            "scene_id": self.scene_id,
            "model": model,
            "prompt_version": self.prompt_version,
            "instruction_hash": stable_payload_hash({"instruction": instruction}),
            "bev_sha256": context.image_sha256,
            "candidate_uids": sorted(candidate_uids),
            "candidate_count": len(candidate_list),
        }
        started = time.perf_counter()
        completion = client.beta.chat.completions.parse(
            model=model or self.vlm,
            messages=[
                {"role": "system", "content": HIGH_LEVEL_PRIOR_MAP_SELECTION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
                        image,
                    ],
                },
            ],
            response_format=ParsedHighLevelSelection,
            temperature=0.0,
            trace_label=HIGH_LEVEL_PRIOR_SELECTION.trace_label,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw = _model_dump(completion.choices[0].message.parsed)
        raw_text = str(getattr(completion.choices[0].message, "content", "") or "")
        selected_uid = str(raw.get("selected_uid", "") or "")
        if selected_uid and selected_uid not in candidate_uids:
            # 模型只能选择候选集合中的 UID；非法 UID 不得进入规划器。
            result = self._fallback(
                candidate_list,
                context,
                "model returned an unknown candidate uid",
                raw={"text": raw_text, "parsed": raw},
            )
            result = HighLevelSelectionResult(
                **{**result.to_dict(), "latency_ms": latency_ms, "request_metadata": request_metadata}
            )
        else:
            result = HighLevelSelectionResult(
                selected_uid=selected_uid,
                selected_type=str(raw.get("selected_type", "") or ""),
                decision=str(raw.get("decision", "uncertain") or "uncertain"),
                confidence=_bounded_float(raw.get("confidence")),
                reason=str(raw.get("reason", "") or ""),
                alternatives=tuple(_texts(raw.get("alternatives"))),
                rejected_candidates=tuple(uid for uid in _texts(raw.get("rejected_candidates")) if uid in candidate_uids),
                uncertainty=str(raw.get("uncertainty", "") or ""),
                bev_sha256=context.image_sha256,
                source="vlm",
                raw_response={"text": raw_text, "parsed": raw},
                prompt_version=self.prompt_version,
                latency_ms=latency_ms,
                request_metadata=request_metadata,
            )
        self.cache.put(key, result)
        return result

    def cache_key(
        self,
        instruction: str,
        instruction_plan: Any,
        context: PriorMapMultimodalContext,
        candidates: Iterable[HighLevelCandidate],
        runtime_state: Optional[Mapping[str, Any]],
    ) -> str:
        """Build the evidence-version cache key."""

        return stable_payload_hash(
            {
                "scene_id": self.scene_id,
                "instruction": instruction,
                "instruction_plan": _json_ready(instruction_plan),
                "bev_sha256": context.image_sha256,
                "candidates": [candidate.to_dict() for candidate in candidates],
                "runtime_state": dict(runtime_state or {}),
                "model": self.vlm,
                "prompt_version": self.prompt_version,
            }
        )

    def _fallback(
        self,
        candidates: tuple[HighLevelCandidate, ...],
        context: PriorMapMultimodalContext,
        reason: str,
        *,
        raw: Optional[dict[str, Any]] = None,
    ) -> HighLevelSelectionResult:
        # 候选顺序由已有几何/搜索策略提供；这里不重新发明语义排序。
        first = candidates[0]
        return HighLevelSelectionResult(
            selected_uid=first.uid,
            selected_type=first.candidate_type,
            decision="fallback",
            reason=reason,
            bev_sha256=context.image_sha256,
            raw_response=dict(raw or {}),
            prompt_version=self.prompt_version,
            request_metadata={
                "scene_id": self.scene_id,
                "model": self._model_override or self.vlm,
                "prompt_version": self.prompt_version,
                "bev_sha256": context.image_sha256,
                "candidate_uids": [candidate.uid for candidate in candidates],
                "reason": reason,
            },
        )


def build_runtime_candidates(
    *,
    rooms: Iterable[Any] = (),
    frontiers: Iterable[Any] = (),
    room_rankings: Iterable[Any] = (),
    frontier_biases: Iterable[Any] = (),
) -> tuple[HighLevelCandidate, ...]:
    """Normalize live room and frontier records for high-level reasoning.

    The selector receives only runtime-owned candidate identities. This helper
    keeps vocabulary and geometry extraction out of the LVLM prompt code while
    preserving the identity contract: a model may rank a candidate, but it
    cannot create a new room, frontier, pose, or motion goal.

    Args:
        rooms: Runtime room records.
        frontiers: Runtime frontier records. Mapper nodes are accepted when
            they expose ``has_frontier`` or ``is_frontier``.
        room_rankings: Optional prior-map room ranking records.
        frontier_biases: Optional prior-map frontier ranking records.

    Returns:
        Stable, de-duplicated room/frontier candidates in caller order.
    """

    room_records = tuple(rooms)
    frontier_records = tuple(frontiers)
    room_ranking_records = tuple(room_rankings)
    frontier_bias_records = tuple(frontier_biases)
    candidates: list[HighLevelCandidate] = []
    seen: set[str] = set()
    room_score = {
        _record_uid(item, ("room_uid", "uid")): _safe_float(_record_value(item, ("score",), 0.0))
        for item in room_ranking_records
        if _record_uid(item, ("room_uid", "uid"))
    }
    frontier_score = {
        _record_uid(item, ("frontier_uid", "uid")): _safe_float(_record_value(item, ("score_delta", "score"), 0.0))
        for item in frontier_bias_records
        if _record_uid(item, ("frontier_uid", "uid"))
    }

    for room in room_records:
        uid = _record_uid(room, ("uid", "room_uid", "id"))
        if not uid or uid in seen:
            continue
        seen.add(uid)
        candidates.append(
            HighLevelCandidate(
                uid=uid,
                candidate_type="room",
                label=str(_record_value(room, ("label", "name"), "") or uid),
                room_uid=uid,
                objects=tuple(_texts(_record_value(room, ("objects", "visible_objects"), ()))),
                explored=bool(_record_value(room, ("explored",), False)),
                metadata={"prior_score": room_score.get(uid, 0.0)},
            )
        )

    for frontier in frontier_records:
        frontier_flag = _record_value(frontier, ("has_frontier", "is_frontier"), None)
        if frontier_flag is False:
            continue
        uid = _record_uid(frontier, ("frontier_uid", "uid", "id", "idx"))
        if not uid or uid in seen:
            continue
        seen.add(uid)
        room_uid = _record_uid(frontier, ("room_uid", "room_id", "prior_room_uid")) or None
        position = _record_value(frontier, ("position", "position_xyz", "world_position_xyz"), None)
        candidates.append(
            HighLevelCandidate(
                uid=uid,
                candidate_type="frontier",
                label=str(_record_value(frontier, ("label", "name"), "frontier") or "frontier"),
                room_uid=room_uid,
                explored=bool(_record_value(frontier, ("explored",), False)),
                metadata={
                    "position": _json_ready(position),
                    "prior_score": frontier_score.get(uid, 0.0),
                    "score": _record_value(frontier, ("score",), None),
                },
            )
        )

    # 若运行时尚未发布候选，prior ranking 仍可作为“待验证候选记录”送入
    # prompt；真正执行前仍必须由运行时 UID/可达性检查确认，不能直接生成位姿。
    for ranking in room_ranking_records:
        uid = _record_uid(ranking, ("room_uid", "uid"))
        if not uid or uid in seen:
            continue
        seen.add(uid)
        candidates.append(
            HighLevelCandidate(
                uid=uid,
                candidate_type="room",
                label=str(_record_value(ranking, ("label", "name"), "") or uid),
                room_uid=uid,
                metadata={"source": "prior_ranking", "prior_score": room_score.get(uid, 0.0)},
            )
        )
    for ranking in frontier_bias_records:
        uid = _record_uid(ranking, ("frontier_uid", "uid"))
        if not uid or uid in seen:
            continue
        seen.add(uid)
        metadata = _record_value(ranking, ("metadata",), {}) or {}
        candidates.append(
            HighLevelCandidate(
                uid=uid,
                candidate_type="frontier",
                label="frontier",
                room_uid=_record_uid(ranking, ("prior_room_uid", "target_region_uid")) or None,
                metadata={
                    "source": "prior_ranking",
                    "position": _json_ready(_record_value(metadata, ("frontier_position_xyz", "frontier_prior_xy"), None)),
                    "prior_score": frontier_score.get(uid, 0.0),
                },
            )
        )

    return tuple(candidates)


def runtime_candidate_payloads(candidates: Iterable[HighLevelCandidate]) -> tuple[dict[str, Any], ...]:
    """Return geometry-safe candidate records for BEV overlay diagnostics.

    Args:
        candidates: Normalized runtime room/frontier candidates.

    Returns:
        JSON-friendly records containing only candidate identity, type, room
        association, and optional runtime position metadata.
    """

    return tuple(
        {
            "uid": candidate.uid,
            "candidate_type": candidate.candidate_type,
            "room_uid": candidate.room_uid,
            "position": candidate.metadata.get("position"),
            "explored": candidate.explored,
        }
        for candidate in candidates
    )


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return {key: getattr(value, key) for key in getattr(value, "__annotations__", {})}


def _json_ready(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        # numpy arrays/scalars are common in the Habitat mapper but must not
        # leak into the JSON prompt payload.
        try:
            return _json_ready(value.tolist())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        return {key: _json_ready(item) for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _texts(value: Any) -> list[str]:
    return [str(item).strip() for item in list(value or ()) if str(item).strip()]


def _record_value(record: Any, names: tuple[str, ...], default: Any = None) -> Any:
    """Read the first available field from an object or mapping."""

    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return value
    return default


def _record_uid(record: Any, names: tuple[str, ...]) -> str:
    """Return a normalized non-empty runtime identity from a record."""

    value = _record_value(record, names, "")
    return str(value or "").strip()


def _safe_float(value: Any) -> float:
    """Convert an optional ranking value to a finite float."""

    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bounded_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "build_runtime_candidates",
    "runtime_candidate_payloads",
    "HighLevelCandidate",
    "HighLevelSelectionCache",
    "HighLevelSelectionResult",
    "PriorMapHighLevelSelector",
]
