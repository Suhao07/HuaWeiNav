"""Platform-neutral contracts for STRIVE prior-map mode.

The contracts in this module describe static or slowly changing map priors and
query results. They intentionally avoid ROS, Habitat, OpenCV, detector, mapper,
or live LLM imports so loaders, simulators, real robots, and offline tests can
share the same schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple


Vector2 = Tuple[float, float]
Vector3 = Tuple[float, float, float]
BoundaryXY = Tuple[Vector2, ...]


@dataclass(frozen=True)
class PriorRoom:
    """Room or region hypothesis from a prior map source.

    Args:
        uid: Stable room or region identifier inside the prior map.
        label: Human-readable semantic label, such as ``kitchen``.
        boundary_xy: Optional 2-D polygon in the prior-map frame.
        centroid_xy: Optional 2-D centroid in the prior-map frame.
        neighbors: Adjacent room or region ids from the prior topology.
        level: Floor or vertical level index when available.
        confidence: Source confidence in ``[0, 1]``.
        source: Source component or file that produced this room.
        description: Optional natural-language map note.
        metadata: JSON-friendly extension fields.
    """

    uid: str
    label: str = "unknown"
    boundary_xy: BoundaryXY = ()
    centroid_xy: Optional[Vector2] = None
    neighbors: Tuple[str, ...] = ()
    level: int = 0
    confidence: float = 0.5
    source: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate room identity, geometry, and confidence.

        Raises:
            ValueError: If required identifiers, vector lengths, or confidence
                values are invalid.
        """

        _require_uid(self.uid, "PriorRoom.uid")
        _validate_boundary_xy(self.boundary_xy, "PriorRoom.boundary_xy")
        if self.centroid_xy is not None:
            _validate_vector(self.centroid_xy, 2, "PriorRoom.centroid_xy")
        _validate_confidence(self.confidence, "PriorRoom.confidence")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using lists for tuple fields so it can be serialized by
            ``json.dumps`` without custom encoders.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PriorRoom":
        """Create a room contract from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``PriorRoom``.
        """

        data = dict(payload)
        data["boundary_xy"] = _tuple_of_vectors(data.get("boundary_xy", ()), 2)
        data["centroid_xy"] = _optional_vector(data.get("centroid_xy"), 2)
        data["neighbors"] = tuple(str(v) for v in data.get("neighbors", ()))
        return cls(**data)


@dataclass(frozen=True)
class PriorObject:
    """Object or object-category hypothesis from a prior map source.

    Args:
        uid: Stable prior object id.
        label: Object label or category.
        position_xyz: Optional 3-D position in the prior-map frame.
        parent_room_uid: Optional room id that contains or is likely to contain
            this object.
        exact: Whether this prior names a concrete object instance. ``False``
            means the object is a category or likelihood hint and cannot be
            treated as observed truth.
        confidence: Source confidence in ``[0, 1]``.
        source: Source component or file that produced this object.
        aliases: Alternative labels used for concept matching.
        metadata: JSON-friendly extension fields.
    """

    uid: str
    label: str
    position_xyz: Optional[Vector3] = None
    parent_room_uid: Optional[str] = None
    exact: bool = False
    confidence: float = 0.5
    source: str = ""
    aliases: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate object identity, label, geometry, and confidence.

        Raises:
            ValueError: If required identifiers, labels, vector lengths, or
                confidence values are invalid.
        """

        _require_uid(self.uid, "PriorObject.uid")
        if not str(self.label).strip():
            raise ValueError("PriorObject.label must be non-empty")
        if self.position_xyz is not None:
            _validate_vector(self.position_xyz, 3, "PriorObject.position_xyz")
        _validate_confidence(self.confidence, "PriorObject.confidence")

    @property
    def exactness(self) -> str:
        """Return a stable text view of the object exactness.

        Returns:
            ``"exact"`` for concrete object instances and ``"hypothesis"`` for
            category or room-level priors.
        """

        return "exact" if self.exact else "hypothesis"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using lists for tuple fields.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PriorObject":
        """Create an object contract from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``PriorObject``.
        """

        data = dict(payload)
        data["position_xyz"] = _optional_vector(data.get("position_xyz"), 3)
        data["aliases"] = tuple(str(v) for v in data.get("aliases", ()))
        return cls(**data)


@dataclass(frozen=True)
class PriorTopologyEdge:
    """Topological relation between prior rooms, objects, or regions.

    Args:
        uid: Stable edge identifier.
        source_uid: Source node id.
        target_uid: Target node id.
        edge_type: Edge family, such as ``room-room`` or ``room-object``.
        relation: Relation label, such as ``adjacent`` or ``contains``.
        bidirectional: Whether traversal or adjacency applies both ways.
        confidence: Source confidence in ``[0, 1]``.
        weight: Optional traversal or ranking weight.
        source: Source component or file that produced this edge.
        metadata: JSON-friendly extension fields.
    """

    uid: str
    source_uid: str
    target_uid: str
    edge_type: str = "room-room"
    relation: str = "connected"
    bidirectional: bool = True
    confidence: float = 0.5
    weight: Optional[float] = None
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate edge identity, endpoints, and confidence.

        Raises:
            ValueError: If identifiers, endpoints, or confidence values are
                invalid.
        """

        _require_uid(self.uid, "PriorTopologyEdge.uid")
        _require_uid(self.source_uid, "PriorTopologyEdge.source_uid")
        _require_uid(self.target_uid, "PriorTopologyEdge.target_uid")
        if not str(self.edge_type).strip():
            raise ValueError("PriorTopologyEdge.edge_type must be non-empty")
        if not str(self.relation).strip():
            raise ValueError("PriorTopologyEdge.relation must be non-empty")
        _validate_confidence(self.confidence, "PriorTopologyEdge.confidence")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using JSON-native scalar and container types.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PriorTopologyEdge":
        """Create a topology edge from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``PriorTopologyEdge``.
        """

        return cls(**dict(payload))


@dataclass(frozen=True)
class PriorObservationRecord:
    """Runtime observation summary used to update prior-map memory later.

    Args:
        timestamp: Runtime timestamp in seconds.
        pose_xyz: Optional runtime robot position when the observation was
            recorded.
        frame_id: Coordinate frame for ``pose_xyz``.
        observed_object_uids: Runtime object ids observed at this step.
        observed_object_labels: Runtime object labels observed at this step.
        room_hypothesis_uid: Optional current room hypothesis.
        source: Observation source, such as ``simulator`` or ``real_robot``.
        metadata: JSON-friendly extension fields.
    """

    timestamp: float
    pose_xyz: Optional[Vector3] = None
    frame_id: str = "map"
    observed_object_uids: Tuple[str, ...] = ()
    observed_object_labels: Tuple[str, ...] = ()
    room_hypothesis_uid: Optional[str] = None
    source: str = "runtime"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate observation timestamp and optional pose.

        Raises:
            ValueError: If the timestamp or pose is invalid.
        """

        if not isinstance(self.timestamp, (int, float)):
            raise ValueError("PriorObservationRecord.timestamp must be numeric")
        if self.pose_xyz is not None:
            _validate_vector(self.pose_xyz, 3, "PriorObservationRecord.pose_xyz")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using lists for tuple fields.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PriorObservationRecord":
        """Create an observation record from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``PriorObservationRecord``.
        """

        data = dict(payload)
        data["pose_xyz"] = _optional_vector(data.get("pose_xyz"), 3)
        data["observed_object_uids"] = tuple(str(v) for v in data.get("observed_object_uids", ()))
        data["observed_object_labels"] = tuple(str(v) for v in data.get("observed_object_labels", ()))
        return cls(**data)


@dataclass(frozen=True)
class PriorMapData:
    """Immutable prior-map input view.

    Args:
        scene_id: Dataset scene id, building id, or robot deployment map id.
        rooms: Prior room or region hypotheses.
        objects: Prior object hypotheses.
        topology_edges: Room-room, room-object, or region connectivity edges.
        source_format: Original source format, such as ``json`` or
            ``floorplan_vln``.
        frame_id: Coordinate frame name for prior geometry.
        world_min: Optional map lower bound in ``(x, y)``.
        world_max: Optional map upper bound in ``(x, y)``.
        observations: Optional runtime records stored beside the prior map for
            replay or debugging. Live memory modules may append records without
            mutating this base object.
        metadata: JSON-friendly extension fields.
    """

    scene_id: str
    rooms: Tuple[PriorRoom, ...] = ()
    objects: Tuple[PriorObject, ...] = ()
    topology_edges: Tuple[PriorTopologyEdge, ...] = ()
    source_format: str = "unknown"
    frame_id: str = "prior_map"
    world_min: Optional[Vector2] = None
    world_max: Optional[Vector2] = None
    observations: Tuple[PriorObservationRecord, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate map identity, bounds, and unique child ids.

        Raises:
            ValueError: If the scene id, bounds, or child identifiers are
                invalid.
        """

        _require_uid(self.scene_id, "PriorMapData.scene_id")
        if self.world_min is not None:
            _validate_vector(self.world_min, 2, "PriorMapData.world_min")
        if self.world_max is not None:
            _validate_vector(self.world_max, 2, "PriorMapData.world_max")
        if self.world_min is not None and self.world_max is not None:
            if self.world_min[0] > self.world_max[0] or self.world_min[1] > self.world_max[1]:
                raise ValueError("PriorMapData.world_min must not exceed world_max")
        _require_unique([room.uid for room in self.rooms], "PriorMapData.rooms")
        _require_unique([obj.uid for obj in self.objects], "PriorMapData.objects")
        _require_unique([edge.uid for edge in self.topology_edges], "PriorMapData.topology_edges")

    def room_by_uid(self, uid: str) -> Optional[PriorRoom]:
        """Return a room by uid.

        Args:
            uid: Prior room uid to look up.

        Returns:
            Matching room, or ``None`` when absent.
        """

        return next((room for room in self.rooms if room.uid == uid), None)

    def object_by_uid(self, uid: str) -> Optional[PriorObject]:
        """Return an object by uid.

        Args:
            uid: Prior object uid to look up.

        Returns:
            Matching object, or ``None`` when absent.
        """

        return next((obj for obj in self.objects if obj.uid == uid), None)

    def topology_for_uid(self, uid: str) -> Tuple[PriorTopologyEdge, ...]:
        """Return topology edges touching a node.

        Args:
            uid: Prior node uid.

        Returns:
            Tuple of edges where ``uid`` is either endpoint. Bidirectionality is
            not interpreted here; query code decides traversal semantics.
        """

        return tuple(edge for edge in self.topology_edges if edge.source_uid == uid or edge.target_uid == uid)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Nested dictionary using JSON-native containers.
        """

        return {
            "scene_id": self.scene_id,
            "rooms": [room.to_dict() for room in self.rooms],
            "objects": [obj.to_dict() for obj in self.objects],
            "topology_edges": [edge.to_dict() for edge in self.topology_edges],
            "source_format": self.source_format,
            "frame_id": self.frame_id,
            "world_min": _json_ready(self.world_min),
            "world_max": _json_ready(self.world_max),
            "observations": [record.to_dict() for record in self.observations],
            "metadata": _json_ready(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PriorMapData":
        """Create a prior map from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``PriorMapData``.
        """

        data = dict(payload)
        data["rooms"] = tuple(PriorRoom.from_dict(item) for item in data.get("rooms", ()))
        data["objects"] = tuple(PriorObject.from_dict(item) for item in data.get("objects", ()))
        data["topology_edges"] = tuple(
            PriorTopologyEdge.from_dict(item) for item in data.get("topology_edges", ())
        )
        data["world_min"] = _optional_vector(data.get("world_min"), 2)
        data["world_max"] = _optional_vector(data.get("world_max"), 2)
        data["observations"] = tuple(
            PriorObservationRecord.from_dict(item) for item in data.get("observations", ())
        )
        return cls(**data)


@dataclass(frozen=True)
class RoomPrior:
    """Soft ranking signal for searching a prior room or region.

    Args:
        room_uid: Prior room id.
        label: Room label used for explanations.
        score: Soft ranking score; query code owns the scale.
        reason: Human-readable reason for the ranking.
        visit_state: Runtime visit state, such as ``unvisited``.
        reachable_hint: Optional reachability hint from topology or planner.
        metadata: JSON-friendly extension fields.
    """

    room_uid: str
    label: str = ""
    score: float = 0.0
    reason: str = ""
    visit_state: str = "unknown"
    reachable_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate room-prior identity and score.

        Raises:
            ValueError: If the room id or score is invalid.
        """

        _require_uid(self.room_uid, "RoomPrior.room_uid")
        _validate_number(self.score, "RoomPrior.score")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using JSON-native containers.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RoomPrior":
        """Create a room prior from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``RoomPrior``.
        """

        return cls(**dict(payload))


@dataclass(frozen=True)
class ObjectPrior:
    """Soft ranking signal for searching a prior object hypothesis.

    Args:
        object_uid: Prior object id.
        label: Prior object label.
        score: Soft ranking score; query code owns the scale.
        reason: Human-readable reason for the ranking.
        parent_room_uid: Optional containing room id.
        exact: Whether this prior points to a concrete instance.
        matched_runtime_uid: Optional observed object matched to this prior.
        metadata: JSON-friendly extension fields.
    """

    object_uid: str
    label: str = ""
    score: float = 0.0
    reason: str = ""
    parent_room_uid: Optional[str] = None
    exact: bool = False
    matched_runtime_uid: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate object-prior identity and score.

        Raises:
            ValueError: If the object id or score is invalid.
        """

        _require_uid(self.object_uid, "ObjectPrior.object_uid")
        _validate_number(self.score, "ObjectPrior.score")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using JSON-native containers.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ObjectPrior":
        """Create an object prior from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``ObjectPrior``.
        """

        return cls(**dict(payload))


@dataclass(frozen=True)
class FrontierPrior:
    """Soft ranking signal for a runtime frontier.

    Args:
        frontier_uid: Runtime frontier id.
        score_delta: Ranking delta suggested by the prior map.
        reason: Human-readable reason for the bias.
        prior_room_uid: Optional prior room associated with this frontier.
        target_region_uid: Optional prior region associated with this frontier.
        metadata: JSON-friendly extension fields.
    """

    frontier_uid: str
    score_delta: float = 0.0
    reason: str = ""
    prior_room_uid: Optional[str] = None
    target_region_uid: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate frontier identity and score delta.

        Raises:
            ValueError: If the frontier id or score delta is invalid.
        """

        _require_uid(self.frontier_uid, "FrontierPrior.frontier_uid")
        _validate_number(self.score_delta, "FrontierPrior.score_delta")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using JSON-native containers.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FrontierPrior":
        """Create a frontier prior from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``FrontierPrior``.
        """

        return cls(**dict(payload))


@dataclass(frozen=True)
class SupportRegionPrior:
    """Soft hint for support regions or contextual search areas.

    Args:
        uid: Prior support-region id.
        label: Region or support-object label.
        score: Soft ranking score; query code owns the scale.
        reason: Human-readable reason for the hint.
        room_uid: Optional containing room id.
        boundary_xy: Optional 2-D support region boundary.
        metadata: JSON-friendly extension fields.
    """

    uid: str
    label: str = ""
    score: float = 0.0
    reason: str = ""
    room_uid: Optional[str] = None
    boundary_xy: BoundaryXY = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate support-region identity, geometry, and score.

        Raises:
            ValueError: If identifiers, boundary geometry, or score are
                invalid.
        """

        _require_uid(self.uid, "SupportRegionPrior.uid")
        _validate_number(self.score, "SupportRegionPrior.score")
        _validate_boundary_xy(self.boundary_xy, "SupportRegionPrior.boundary_xy")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Dictionary using lists for tuple fields.
        """

        return _json_ready(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SupportRegionPrior":
        """Create a support-region prior from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``SupportRegionPrior``.
        """

        data = dict(payload)
        data["boundary_xy"] = _tuple_of_vectors(data.get("boundary_xy", ()), 2)
        return cls(**data)


@dataclass(frozen=True)
class SearchPriorResult:
    """Prior-map query result consumed by STRIVE ranking adapters.

    Args:
        room_rankings: Soft room rankings.
        object_rankings: Soft object rankings.
        frontier_biases: Soft runtime frontier score deltas.
        support_regions: Soft region or support-object hints.
        prompt_context: Compact context that can be rendered into prompts.
        diagnostics: Debug data for replay and evaluation.
    """

    room_rankings: Tuple[RoomPrior, ...] = ()
    object_rankings: Tuple[ObjectPrior, ...] = ()
    frontier_biases: Tuple[FrontierPrior, ...] = ()
    support_regions: Tuple[SupportRegionPrior, ...] = ()
    prompt_context: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary.

        Returns:
            Nested dictionary using JSON-native containers.
        """

        return {
            "room_rankings": [prior.to_dict() for prior in self.room_rankings],
            "object_rankings": [prior.to_dict() for prior in self.object_rankings],
            "frontier_biases": [prior.to_dict() for prior in self.frontier_biases],
            "support_regions": [prior.to_dict() for prior in self.support_regions],
            "prompt_context": _json_ready(self.prompt_context),
            "diagnostics": _json_ready(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SearchPriorResult":
        """Create a query result from a JSON-decoded dictionary.

        Args:
            payload: Dictionary decoded from JSON or produced by ``to_dict``.

        Returns:
            Reconstructed ``SearchPriorResult``.
        """

        data = dict(payload)
        data["room_rankings"] = tuple(RoomPrior.from_dict(item) for item in data.get("room_rankings", ()))
        data["object_rankings"] = tuple(ObjectPrior.from_dict(item) for item in data.get("object_rankings", ()))
        data["frontier_biases"] = tuple(FrontierPrior.from_dict(item) for item in data.get("frontier_biases", ()))
        data["support_regions"] = tuple(
            SupportRegionPrior.from_dict(item) for item in data.get("support_regions", ())
        )
        return cls(**data)


def _require_uid(value: str, field_name: str) -> None:
    """Validate that a stable identifier is present.

    Args:
        value: Identifier value to check.
        field_name: Name used in error messages.

    Raises:
        ValueError: If the identifier is empty.
    """

    if not str(value).strip():
        raise ValueError(f"{field_name} must be non-empty")


def _validate_number(value: float, field_name: str) -> None:
    """Validate that a value is numeric.

    Args:
        value: Number to validate.
        field_name: Name used in error messages.

    Raises:
        ValueError: If ``value`` is not numeric.
    """

    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")


def _validate_confidence(value: float, field_name: str) -> None:
    """Validate a confidence score.

    Args:
        value: Confidence score.
        field_name: Name used in error messages.

    Raises:
        ValueError: If ``value`` is not numeric or outside ``[0, 1]``.
    """

    _validate_number(value, field_name)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")


def _validate_vector(value: Tuple[float, ...], size: int, field_name: str) -> None:
    """Validate a fixed-size numeric vector.

    Args:
        value: Vector to validate.
        size: Required vector length.
        field_name: Name used in error messages.

    Raises:
        ValueError: If ``value`` has the wrong length or non-numeric entries.
    """

    if len(value) != size:
        raise ValueError(f"{field_name} must have length {size}")
    for item in value:
        _validate_number(item, field_name)


def _validate_boundary_xy(value: BoundaryXY, field_name: str) -> None:
    """Validate a 2-D boundary polygon representation.

    Args:
        value: Boundary points.
        field_name: Name used in error messages.

    Raises:
        ValueError: If any point is not a 2-D vector.
    """

    for point in value:
        _validate_vector(point, 2, field_name)


def _require_unique(values: Iterable[str], field_name: str) -> None:
    """Validate uniqueness for stable child ids.

    Args:
        values: Identifiers to check.
        field_name: Name used in error messages.

    Raises:
        ValueError: If any identifier appears more than once.
    """

    seen = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{field_name} contains duplicate uid: {value}")
        seen.add(value)


def _optional_vector(value: Any, size: int) -> Optional[Tuple[float, ...]]:
    """Convert an optional JSON vector to a tuple.

    Args:
        value: JSON-decoded vector or ``None``.
        size: Required vector length.

    Returns:
        Tuple vector or ``None``.
    """

    if value is None:
        return None
    result = tuple(float(item) for item in value)
    _validate_vector(result, size, "vector")
    return result


def _tuple_of_vectors(values: Any, size: int) -> Tuple[Tuple[float, ...], ...]:
    """Convert JSON vectors to a tuple of tuples.

    Args:
        values: Iterable of JSON-decoded vectors.
        size: Required vector length for each element.

    Returns:
        Tuple of fixed-size float tuples.
    """

    result = tuple(tuple(float(item) for item in vector) for vector in values or ())
    for vector in result:
        _validate_vector(vector, size, "vector")
    return result


def _json_ready(value: Any) -> Any:
    """Convert dataclasses, tuples, and dictionaries to JSON-native values.

    Args:
        value: Arbitrary value from a contract object.

    Returns:
        JSON-friendly value composed of dictionaries, lists, and scalar values.
    """

    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
