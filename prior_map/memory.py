"""Runtime memory for STRIVE prior-map mode.

The memory layer keeps monotonic runtime state derived from simulation mappers
or real-robot semantic snapshots. It never mutates the immutable
``PriorMapData`` input and it does not rank frontiers, publish motion goals, or
invoke live model clients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .alignment import PriorMapAlignment
from .contracts import (
    PriorMapData,
    PriorObject,
    PriorObservationRecord,
    PriorRoom,
    Vector3,
)


@dataclass
class PriorRoomRuntimeState:
    """Runtime state for one prior room.

    Args:
        room_uid: Prior room identifier.
        visit_count: Number of explicit visits recorded for this prior room.
        observation_count: Number of mapper/snapshot observations matched to
            this prior room.
        last_visited_step: Last step where the room was marked visited.
        last_observed_step: Last step where runtime evidence matched the room.
        confidence: Runtime confidence estimate.
        confidence_samples: Number of runtime confidence samples integrated.
        matched_runtime_uids: Runtime room ids matched to this prior room.
        metadata: JSON-friendly extension fields.
    """

    room_uid: str
    visit_count: int = 0
    observation_count: int = 0
    last_visited_step: Optional[int] = None
    last_observed_step: Optional[int] = None
    confidence: float = 0.0
    confidence_samples: int = 0
    matched_runtime_uids: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def visited(self) -> bool:
        """Return whether this room has been visited.

        Returns:
            ``True`` when ``visit_count`` is positive.
        """

        return self.visit_count > 0

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly runtime state dictionary.

        Returns:
            Runtime room state payload.
        """

        payload = _json_ready(asdict(self))
        payload["visited"] = self.visited
        return payload


@dataclass
class PriorObjectRuntimeState:
    """Runtime state for one prior object hypothesis.

    Args:
        object_uid: Prior object identifier.
        observation_count: Number of mapper/snapshot observations matched to
            this prior object.
        verification_count: Number of successful verification events.
        rejection_count: Number of rejection events.
        last_observed_step: Last step where runtime evidence matched the object.
        last_verified_step: Last verification step.
        last_rejected_step: Last rejection step.
        confidence: Runtime confidence estimate.
        confidence_samples: Number of runtime confidence samples integrated.
        matched_runtime_uid: Most recent runtime object id matched to this
            prior object.
        matched_runtime_uids: All runtime object ids matched so far.
        rejection_reasons: Recorded rejection reasons.
        metadata: JSON-friendly extension fields.
    """

    object_uid: str
    observation_count: int = 0
    verification_count: int = 0
    rejection_count: int = 0
    last_observed_step: Optional[int] = None
    last_verified_step: Optional[int] = None
    last_rejected_step: Optional[int] = None
    confidence: float = 0.0
    confidence_samples: int = 0
    matched_runtime_uid: Optional[str] = None
    matched_runtime_uids: Tuple[str, ...] = ()
    rejection_reasons: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        """Return whether this prior object has been verified.

        Returns:
            ``True`` when ``verification_count`` is positive and the object has
            not been rejected after the last verification.
        """

        if self.verification_count <= 0:
            return False
        if self.last_rejected_step is None:
            return True
        if self.last_verified_step is None:
            return False
        return self.last_verified_step >= self.last_rejected_step

    @property
    def rejected(self) -> bool:
        """Return whether the latest terminal state is rejected.

        Returns:
            ``True`` when the latest rejection is newer than the latest
            verification.
        """

        if self.rejection_count <= 0:
            return False
        if self.last_verified_step is None:
            return True
        return self.last_rejected_step is not None and self.last_rejected_step > self.last_verified_step

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly runtime state dictionary.

        Returns:
            Runtime object state payload.
        """

        payload = _json_ready(asdict(self))
        payload["verified"] = self.verified
        payload["rejected"] = self.rejected
        return payload


@dataclass
class PriorMapMemory:
    """Runtime memory for a loaded prior map.

    Args:
        base_map: Immutable loaded prior map.
        alignment: Prior-to-runtime coordinate alignment.
        room_states: Runtime state keyed by prior room uid. Missing states are
            initialized from ``base_map`` in ``__post_init__``.
        object_states: Runtime state keyed by prior object uid. Missing states
            are initialized from ``base_map`` in ``__post_init__``.
        observations: Runtime observation records appended monotonically.
        confidence_alpha: Exponential moving-average factor for observation
            confidence updates. ``1`` means replace with the latest sample.
    """

    base_map: PriorMapData
    alignment: PriorMapAlignment
    room_states: Dict[str, PriorRoomRuntimeState] = field(default_factory=dict)
    object_states: Dict[str, PriorObjectRuntimeState] = field(default_factory=dict)
    observations: List[PriorObservationRecord] = field(default_factory=list)
    confidence_alpha: float = 0.5

    def __post_init__(self) -> None:
        """Initialize missing runtime states from the base map.

        Raises:
            ValueError: If ``confidence_alpha`` is outside ``[0, 1]``.
        """

        if not 0.0 <= self.confidence_alpha <= 1.0:
            raise ValueError("confidence_alpha must be in [0, 1]")
        for room in self.base_map.rooms:
            self.room_states.setdefault(
                room.uid,
                PriorRoomRuntimeState(room_uid=room.uid, confidence=room.confidence),
            )
        for obj in self.base_map.objects:
            self.object_states.setdefault(
                obj.uid,
                PriorObjectRuntimeState(object_uid=obj.uid, confidence=obj.confidence),
            )

    def update_from_mapper(self, mapper: Any, step: int) -> PriorObservationRecord:
        """Update memory from a simulation mapper-like object.

        The mapper is consumed by duck typing. The method looks for common
        fields such as ``objects``, ``object_nodes``, ``rooms``,
        ``room_nodes``, ``current_room_uid``, and ``robot_pose``.

        Args:
            mapper: Simulation mapper or a small fake object with compatible
                attributes.
            step: Runtime step index.

        Returns:
            Observation record appended to memory.
        """

        room_hypothesis_uid = _text_or_none(
            _first_attr(mapper, ("current_room_uid", "room_hypothesis_uid", "current_room_id", "room_id"))
        )
        pose_xyz = _pose_xyz(_first_attr(mapper, ("robot_pose", "pose", "current_pose", "agent_pose")))

        observed_object_uids: List[str] = []
        observed_object_labels: List[str] = []
        for runtime_obj in _iter_runtime_records(
            _first_attr(mapper, ("objects", "object_nodes", "detected_objects", "semantic_objects"), default=())
        ):
            runtime_uid = _runtime_uid(runtime_obj)
            label = _runtime_label(runtime_obj)
            confidence = _runtime_confidence(runtime_obj)
            parent_room_uid = _text_or_none(
                _first_attr(runtime_obj, ("room_id", "room_uid", "parent_room_uid", "parent_room"))
            )
            prior_uid = self._match_prior_object(runtime_uid, label, parent_room_uid)
            if prior_uid:
                self._record_object_observation(prior_uid, runtime_uid, confidence, step)
            if runtime_uid:
                observed_object_uids.append(runtime_uid)
            if label:
                observed_object_labels.append(label)

        for runtime_room in _iter_runtime_records(_first_attr(mapper, ("rooms", "room_nodes", "semantic_rooms"), default=())):
            runtime_uid = _runtime_uid(runtime_room)
            label = _runtime_label(runtime_room)
            confidence = _runtime_confidence(runtime_room)
            prior_uid = self._match_prior_room(runtime_uid, label)
            if prior_uid:
                self._record_room_observation(prior_uid, runtime_uid, confidence, step)

        if room_hypothesis_uid:
            matched_room_uid = self._match_prior_room(room_hypothesis_uid, "")
            if matched_room_uid:
                self.mark_room_visited(matched_room_uid, step)

        record = PriorObservationRecord(
            timestamp=float(step),
            pose_xyz=pose_xyz,
            frame_id=_text_or_none(_first_attr(mapper, ("frame_id", "world_frame", "map_frame"))) or "map",
            observed_object_uids=tuple(_dedupe(observed_object_uids)),
            observed_object_labels=tuple(_dedupe(observed_object_labels)),
            room_hypothesis_uid=room_hypothesis_uid,
            source="mapper",
            metadata={"step": step},
        )
        self.observations.append(record)
        return record

    def update_from_snapshot(self, snapshot: Any) -> PriorObservationRecord:
        """Update memory from a real-robot ``SemanticMapSnapshot``-like object.

        The method intentionally uses duck typing instead of importing
        ``real_robot.contracts`` so the prior-map package stays reusable in
        simulation and offline tests.

        Args:
            snapshot: Object exposing ``timestamp``, ``robot_pose``,
                ``objects``, and ``rooms`` attributes.

        Returns:
            Observation record appended to memory.
        """

        step = _safe_int(_first_attr(snapshot, ("step", "timestamp"), default=len(self.observations)))
        pose_xyz = _pose_xyz(_first_attr(snapshot, ("robot_pose", "pose"), default=None))
        observed_object_uids: List[str] = []
        observed_object_labels: List[str] = []

        for runtime_obj in _iter_runtime_records(_first_attr(snapshot, ("objects",), default=())):
            runtime_uid = _runtime_uid(runtime_obj)
            label = _runtime_label(runtime_obj)
            confidence = _runtime_confidence(runtime_obj)
            parent_room_uid = _text_or_none(
                _first_attr(runtime_obj, ("room_id", "room_uid", "parent_room_uid", "parent_room"))
            )
            prior_uid = self._match_prior_object(runtime_uid, label, parent_room_uid)
            if prior_uid:
                self._record_object_observation(prior_uid, runtime_uid, confidence, step)
            if runtime_uid:
                observed_object_uids.append(runtime_uid)
            if label:
                observed_object_labels.append(label)

        room_hypothesis_uid: Optional[str] = None
        for runtime_room in _iter_runtime_records(_first_attr(snapshot, ("rooms",), default=())):
            runtime_uid = _runtime_uid(runtime_room)
            label = _runtime_label(runtime_room)
            confidence = _runtime_confidence(runtime_room)
            prior_uid = self._match_prior_room(runtime_uid, label)
            if prior_uid:
                self._record_room_observation(prior_uid, runtime_uid, confidence, step)
                if bool(_first_attr(runtime_room, ("explored", "visited"), default=False)):
                    self.mark_room_visited(prior_uid, step)
                room_hypothesis_uid = room_hypothesis_uid or prior_uid

        record = PriorObservationRecord(
            timestamp=float(_first_attr(snapshot, ("timestamp",), default=step)),
            pose_xyz=pose_xyz,
            frame_id=_pose_frame_id(_first_attr(snapshot, ("robot_pose", "pose"), default=None)),
            observed_object_uids=tuple(_dedupe(observed_object_uids)),
            observed_object_labels=tuple(_dedupe(observed_object_labels)),
            room_hypothesis_uid=room_hypothesis_uid,
            source=_text_or_none(_first_attr(snapshot, ("source",), default=None)) or "semantic_snapshot",
            metadata={"step": step},
        )
        self.observations.append(record)
        return record

    def mark_room_visited(self, room_uid: str, step: int) -> None:
        """Mark a prior room as visited.

        Args:
            room_uid: Prior room uid.
            step: Runtime step index.

        Raises:
            KeyError: If ``room_uid`` is not part of the prior map.
        """

        state = self._room_state(room_uid)
        state.visit_count += 1
        state.last_visited_step = int(step)
        state.last_observed_step = int(step)
        state.observation_count += 1

    def mark_object_verified(self, prior_uid: str, runtime_uid: str, step: int) -> None:
        """Mark a prior object as verified by runtime evidence.

        Args:
            prior_uid: Prior object uid.
            runtime_uid: Runtime object uid that verified the prior.
            step: Runtime step index.

        Raises:
            KeyError: If ``prior_uid`` is not part of the prior map.
        """

        state = self._object_state(prior_uid)
        state.verification_count += 1
        state.last_verified_step = int(step)
        state.matched_runtime_uid = str(runtime_uid)
        state.matched_runtime_uids = _append_unique(state.matched_runtime_uids, str(runtime_uid))
        state.confidence, state.confidence_samples = _ema_update(
            state.confidence,
            1.0,
            state.confidence_samples,
            self.confidence_alpha,
        )

    def mark_prior_rejected(self, prior_uid: str, reason: str, step: int) -> None:
        """Mark a prior object hypothesis as rejected.

        Args:
            prior_uid: Prior object uid.
            reason: Concrete rejection reason.
            step: Runtime step index.

        Raises:
            KeyError: If ``prior_uid`` is not part of the prior map.
        """

        state = self._object_state(prior_uid)
        state.rejection_count += 1
        state.last_rejected_step = int(step)
        state.rejection_reasons = _append_unique(state.rejection_reasons, reason)
        state.confidence, state.confidence_samples = _ema_update(
            state.confidence,
            0.0,
            state.confidence_samples,
            self.confidence_alpha,
        )

    def current_map(self) -> PriorMapData:
        """Return a read-only merged view of base map and runtime state.

        Returns:
            New ``PriorMapData`` whose rooms, objects, observations, and
            metadata include memory state without mutating ``base_map``.
        """

        rooms = tuple(self._room_with_state(room) for room in self.base_map.rooms)
        objects = tuple(self._object_with_state(obj) for obj in self.base_map.objects)
        metadata = dict(self.base_map.metadata)
        metadata["runtime_memory"] = {
            "room_count": len(self.room_states),
            "object_count": len(self.object_states),
            "observation_count": len(self.observations),
            "alignment": self.alignment.diagnostics_payload(),
        }
        return PriorMapData(
            scene_id=self.base_map.scene_id,
            rooms=rooms,
            objects=objects,
            topology_edges=self.base_map.topology_edges,
            source_format=self.base_map.source_format,
            frame_id=self.base_map.frame_id,
            world_min=self.base_map.world_min,
            world_max=self.base_map.world_max,
            observations=tuple(self.base_map.observations) + tuple(self.observations),
            metadata=metadata,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return memory state for diagnostics and tests.

        Returns:
            JSON-friendly memory state payload.
        """

        return {
            "rooms": {uid: state.to_dict() for uid, state in self.room_states.items()},
            "objects": {uid: state.to_dict() for uid, state in self.object_states.items()},
            "observations": [record.to_dict() for record in self.observations],
            "alignment": self.alignment.diagnostics_payload(),
        }

    def _room_state(self, room_uid: str) -> PriorRoomRuntimeState:
        """Return mutable room state or raise for unknown prior uid.

        Args:
            room_uid: Prior room uid.

        Returns:
            Mutable room state.
        """

        if room_uid not in self.room_states:
            raise KeyError(f"Unknown prior room uid: {room_uid}")
        return self.room_states[room_uid]

    def _object_state(self, object_uid: str) -> PriorObjectRuntimeState:
        """Return mutable object state or raise for unknown prior uid.

        Args:
            object_uid: Prior object uid.

        Returns:
            Mutable object state.
        """

        if object_uid not in self.object_states:
            raise KeyError(f"Unknown prior object uid: {object_uid}")
        return self.object_states[object_uid]

    def _record_room_observation(self, prior_uid: str, runtime_uid: str, confidence: float, step: int) -> None:
        """Integrate one runtime room observation.

        Args:
            prior_uid: Matched prior room uid.
            runtime_uid: Runtime room uid.
            confidence: Runtime observation confidence.
            step: Runtime step index.
        """

        state = self._room_state(prior_uid)
        state.observation_count += 1
        state.last_observed_step = int(step)
        state.matched_runtime_uids = _append_unique(state.matched_runtime_uids, runtime_uid)
        state.confidence, state.confidence_samples = _ema_update(
            state.confidence,
            confidence,
            state.confidence_samples,
            self.confidence_alpha,
        )

    def _record_object_observation(self, prior_uid: str, runtime_uid: str, confidence: float, step: int) -> None:
        """Integrate one runtime object observation.

        Args:
            prior_uid: Matched prior object uid.
            runtime_uid: Runtime object uid.
            confidence: Runtime observation confidence.
            step: Runtime step index.
        """

        state = self._object_state(prior_uid)
        state.observation_count += 1
        state.last_observed_step = int(step)
        state.matched_runtime_uid = runtime_uid
        state.matched_runtime_uids = _append_unique(state.matched_runtime_uids, runtime_uid)
        state.confidence, state.confidence_samples = _ema_update(
            state.confidence,
            confidence,
            state.confidence_samples,
            self.confidence_alpha,
        )

    def _match_prior_room(self, runtime_uid: Optional[str], label: str) -> Optional[str]:
        """Match a runtime room to a prior room.

        Args:
            runtime_uid: Runtime room uid.
            label: Runtime room label.

        Returns:
            Matched prior room uid, or ``None``.
        """

        if runtime_uid and runtime_uid in self.room_states:
            return runtime_uid
        normalized_label = _normalize_text(label)
        if not normalized_label:
            return None
        for room in self.base_map.rooms:
            if _normalize_text(room.label) == normalized_label:
                return room.uid
        return None

    def _match_prior_object(
        self,
        runtime_uid: Optional[str],
        label: str,
        parent_room_uid: Optional[str],
    ) -> Optional[str]:
        """Match a runtime object to a prior object hypothesis.

        Args:
            runtime_uid: Runtime object uid.
            label: Runtime object label.
            parent_room_uid: Optional runtime/prior room uid.

        Returns:
            Matched prior object uid, or ``None``.
        """

        if runtime_uid and runtime_uid in self.object_states:
            return runtime_uid
        normalized_label = _normalize_text(label)
        if not normalized_label:
            return None
        for obj in self.base_map.objects:
            state = self.object_states[obj.uid]
            if state.rejected:
                continue
            terms = {_normalize_text(obj.label), *(_normalize_text(alias) for alias in obj.aliases)}
            if normalized_label not in terms:
                continue
            if parent_room_uid and obj.parent_room_uid and parent_room_uid != obj.parent_room_uid:
                continue
            return obj.uid
        return None

    def _room_with_state(self, room: PriorRoom) -> PriorRoom:
        """Return a copy of a room with runtime metadata.

        Args:
            room: Base prior room.

        Returns:
            Room copy with runtime metadata.
        """

        state = self.room_states[room.uid]
        metadata = dict(room.metadata)
        metadata["runtime_state"] = state.to_dict()
        return PriorRoom(
            uid=room.uid,
            label=room.label,
            boundary_xy=room.boundary_xy,
            centroid_xy=room.centroid_xy,
            neighbors=room.neighbors,
            level=room.level,
            confidence=state.confidence if state.confidence_samples > 0 else room.confidence,
            source=room.source,
            description=room.description,
            metadata=metadata,
        )

    def _object_with_state(self, obj: PriorObject) -> PriorObject:
        """Return a copy of an object with runtime metadata.

        Args:
            obj: Base prior object.

        Returns:
            Object copy with runtime metadata.
        """

        state = self.object_states[obj.uid]
        metadata = dict(obj.metadata)
        metadata["runtime_state"] = state.to_dict()
        return PriorObject(
            uid=obj.uid,
            label=obj.label,
            position_xyz=obj.position_xyz,
            parent_room_uid=obj.parent_room_uid,
            exact=obj.exact,
            confidence=state.confidence if state.confidence_samples > 0 else obj.confidence,
            source=obj.source,
            aliases=obj.aliases,
            metadata=metadata,
        )


def _ema_update(current: float, sample: float, sample_count: int, alpha: float) -> Tuple[float, int]:
    """Update confidence using an exponential moving average.

    Args:
        current: Current confidence.
        sample: New confidence sample.
        sample_count: Number of previous samples.
        alpha: EMA factor in ``[0, 1]``.

    Returns:
        Updated ``(confidence, sample_count)``.
    """

    bounded = max(0.0, min(1.0, float(sample)))
    if sample_count <= 0:
        return bounded, 1
    return (alpha * bounded + (1.0 - alpha) * current, sample_count + 1)


def _iter_runtime_records(value: Any) -> Iterable[Any]:
    """Iterate runtime records from dicts, lists, or tuples.

    Args:
        value: Runtime container.

    Returns:
        Iterable of runtime records.
    """

    if isinstance(value, dict):
        return value.values()
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _first_attr(value: Any, names: Sequence[str], default: Any = None) -> Any:
    """Return the first existing attribute or dictionary value.

    Args:
        value: Runtime object or dictionary.
        names: Candidate field names.
        default: Fallback value.

    Returns:
        First matching field value.
    """

    if value is None:
        return default
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _runtime_uid(value: Any) -> Optional[str]:
    """Extract a runtime object or room uid.

    Args:
        value: Runtime record.

    Returns:
        Runtime uid string or ``None``.
    """

    return _text_or_none(_first_attr(value, ("uid", "id", "object_id", "room_id", "show_id", "track_id")))


def _runtime_label(value: Any) -> str:
    """Extract a runtime semantic label.

    Args:
        value: Runtime record.

    Returns:
        Semantic label string.
    """

    return _text_or_none(_first_attr(value, ("label", "tag", "type", "category", "name", "class_name"))) or ""


def _runtime_confidence(value: Any) -> float:
    """Extract runtime confidence with fallback.

    Args:
        value: Runtime record.

    Returns:
        Confidence in ``[0, 1]``.
    """

    raw = _first_attr(value, ("confidence", "score", "probability"), default=0.5)
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.5


def _pose_xyz(value: Any) -> Optional[Vector3]:
    """Extract a 3-D pose position from common runtime structures.

    Args:
        value: Pose-like object, dict, or tuple/list.

    Returns:
        ``(x, y, z)`` or ``None``.
    """

    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        z = value[2] if len(value) >= 3 else 0.0
        return (_safe_float(value[0]), _safe_float(value[1]), _safe_float(z))
    position = _first_attr(value, ("position",), default=None)
    if position is not None:
        return _pose_xyz(position)
    x = _first_attr(value, ("x",), default=None)
    y = _first_attr(value, ("y",), default=None)
    z = _first_attr(value, ("z",), default=0.0)
    if x is not None and y is not None:
        return (_safe_float(x), _safe_float(y), _safe_float(z))
    return None


def _pose_frame_id(value: Any) -> str:
    """Extract a pose frame id.

    Args:
        value: Pose-like object or dictionary.

    Returns:
        Frame id, defaulting to ``"map"``.
    """

    return _text_or_none(_first_attr(value, ("frame_id",), default=None)) or "map"


def _safe_int(value: Any) -> int:
    """Convert a value to int with fallback.

    Args:
        value: Source value.

    Returns:
        Integer value.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    """Convert a value to float with fallback.

    Args:
        value: Source value.

    Returns:
        Float value.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text_or_none(value: Any) -> Optional[str]:
    """Return stripped text or ``None``.

    Args:
        value: Source value.

    Returns:
        Text or ``None``.
    """

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_text(value: str) -> str:
    """Normalize text for light concept matching.

    Args:
        value: Source text.

    Returns:
        Lowercase normalized text.
    """

    return str(value or "").strip().lower().replace("_", " ")


def _append_unique(values: Tuple[str, ...], value: str) -> Tuple[str, ...]:
    """Append a string if it is not already present.

    Args:
        values: Existing tuple.
        value: Candidate value.

    Returns:
        Tuple with ``value`` appended at most once.
    """

    if not value or value in values:
        return values
    return values + (value,)


def _dedupe(values: Sequence[str]) -> List[str]:
    """Deduplicate strings while preserving order.

    Args:
        values: Source strings.

    Returns:
        Deduplicated list.
    """

    output: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _json_ready(value: Any) -> Any:
    """Convert tuples and dictionaries to JSON-native values.

    Args:
        value: Arbitrary runtime state value.

    Returns:
        JSON-friendly value.
    """

    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
