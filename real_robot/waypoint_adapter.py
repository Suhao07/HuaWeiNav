"""Configurable STRIVE waypoint format and frame adapter.

The STRIVE runtime emits a stamped world-frame point.  Some externally owned
local planners consume an un-stamped flat array in the robot/ego frame.  This
module keeps that conversion deterministic and robot-configurable without
owning a chassis controller or publishing velocity commands.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import yaml

from real_robot.contracts import Pose3D


class WaypointAdapterError(ValueError):
    """Raised when an adapter configuration or input waypoint is unsafe."""


@dataclass(frozen=True)
class WaypointAdapterConfig:
    """Per-robot configuration for a PointStamped -> Float32MultiArray bridge."""

    input_topic: str = "/way_point"
    output_topic: str = "/waypoint"
    odom_topic: str = "/aft_mapped_to_init"
    input_frame: str = "map"
    output_frame: str = "base_link"
    coordinate_mode: str = "ego_from_odom"
    output_message_type: str = "std_msgs/msg/Float32MultiArray"
    include_z: bool = False
    max_input_age_s: float = 1.0
    output_enabled: bool = False
    static_translation_xy_m: Tuple[float, float] = (0.0, 0.0)
    static_yaw_rad: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the safety-sensitive parts of the configuration."""

        if not self.input_topic or not self.output_topic:
            raise WaypointAdapterError("input_topic and output_topic are required")
        if self.output_topic.rstrip("/") == "/cmd_vel" or self.output_topic.rstrip("/").endswith("/cmd_vel"):
            raise WaypointAdapterError("waypoint adapter must never publish a direct velocity topic")
        if self.output_message_type != "std_msgs/msg/Float32MultiArray":
            raise WaypointAdapterError(
                "only std_msgs/msg/Float32MultiArray is implemented for the external controller adapter"
            )
        if self.coordinate_mode not in {"identity", "ego_from_odom", "static_se2"}:
            raise WaypointAdapterError(
                "coordinate_mode must be identity, ego_from_odom, or static_se2"
            )
        if len(self.static_translation_xy_m) != 2:
            raise WaypointAdapterError("static_translation_xy_m must contain [x, y]")
        if self.max_input_age_s < 0.0:
            raise WaypointAdapterError("max_input_age_s must be non-negative")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WaypointAdapterConfig":
        """Build a config from either a top-level or ``waypoint_adapter`` map."""

        data = raw.get("waypoint_adapter", raw)
        if not isinstance(data, Mapping):
            raise WaypointAdapterError("waypoint adapter YAML must contain a mapping")
        translation = data.get("static_translation_xy_m", (0.0, 0.0))
        try:
            translation_xy = (float(translation[0]), float(translation[1]))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise WaypointAdapterError("static_translation_xy_m must contain two numbers") from exc
        return cls(
            input_topic=str(data.get("input_topic", cls.input_topic)),
            output_topic=str(data.get("output_topic", cls.output_topic)),
            odom_topic=str(data.get("odom_topic", cls.odom_topic)),
            input_frame=str(data.get("input_frame", cls.input_frame)),
            output_frame=str(data.get("output_frame", cls.output_frame)),
            coordinate_mode=str(data.get("coordinate_mode", cls.coordinate_mode)).strip().lower(),
            output_message_type=str(data.get("output_message_type", cls.output_message_type)),
            include_z=_as_bool(data.get("include_z", cls.include_z)),
            max_input_age_s=float(data.get("max_input_age_s", cls.max_input_age_s)),
            output_enabled=_as_bool(data.get("output_enabled", cls.output_enabled)),
            static_translation_xy_m=translation_xy,
            static_yaw_rad=float(data.get("static_yaw_rad", cls.static_yaw_rad)),
            metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), Mapping) else {},
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WaypointAdapterConfig":
        """Load a robot-specific adapter config without modifying the source project."""

        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise WaypointAdapterError(f"waypoint adapter config does not exist: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls.from_mapping(raw)


@dataclass(frozen=True)
class AdaptedWaypoint:
    """A converted flat waypoint plus audit metadata."""

    values: Tuple[float, ...]
    input_frame: str
    output_frame: str
    coordinate_mode: str
    source_stamp: Optional[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WaypointFormatAdapter:
    """Convert stamped STRIVE points to external-controller array values."""

    def __init__(self, config: WaypointAdapterConfig, now_fn: Callable[[], float] = time.time) -> None:
        self.config = config
        self.now_fn = now_fn
        self.latest_pose: Optional[Pose3D] = None

    def update_pose(self, pose: Pose3D) -> None:
        """Update the read-only robot pose used by ego-frame conversion."""

        self.latest_pose = pose

    def convert(
        self,
        point_xyz: Sequence[float],
        *,
        frame_id: str,
        stamp: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Optional[AdaptedWaypoint]:
        """Convert one point or return ``None`` when it is stale/unavailable."""

        if len(point_xyz) < 2:
            raise WaypointAdapterError("waypoint input must contain at least x and y")
        source_frame = str(frame_id or "").strip()
        if self.config.input_frame and source_frame and source_frame != self.config.input_frame:
            raise WaypointAdapterError(
                f"waypoint frame mismatch: expected {self.config.input_frame}, got {source_frame}"
            )
        source_stamp = None if stamp is None or float(stamp) <= 0.0 else float(stamp)
        if source_stamp is not None and self.config.max_input_age_s > 0.0:
            age = float(now if now is not None else self.now_fn()) - source_stamp
            if age > self.config.max_input_age_s:
                return None
            if age < -self.config.max_input_age_s:
                raise WaypointAdapterError("waypoint timestamp is too far in the future")

        x, y = float(point_xyz[0]), float(point_xyz[1])
        z = float(point_xyz[2]) if len(point_xyz) >= 3 else 0.0
        if self.config.coordinate_mode == "identity":
            out_x, out_y = x, y
        elif self.config.coordinate_mode == "static_se2":
            out_x, out_y = _apply_static_se2(
                x,
                y,
                self.config.static_translation_xy_m,
                self.config.static_yaw_rad,
            )
        else:
            if self.latest_pose is None:
                return None
            if self.config.input_frame and self.latest_pose.frame_id != self.config.input_frame:
                raise WaypointAdapterError(
                    "odometry frame mismatch: "
                    f"expected {self.config.input_frame}, got {self.latest_pose.frame_id}"
                )
            out_x, out_y = _world_to_ego(x, y, self.latest_pose)

        values = (out_x, out_y, z) if self.config.include_z else (out_x, out_y)
        return AdaptedWaypoint(
            values=tuple(float(value) for value in values),
            input_frame=source_frame or self.config.input_frame,
            output_frame=self.config.output_frame,
            coordinate_mode=self.config.coordinate_mode,
            source_stamp=source_stamp,
            metadata={
                "input_topic": self.config.input_topic,
                "output_topic": self.config.output_topic,
                "output_message_type": self.config.output_message_type,
                "max_input_age_s": self.config.max_input_age_s,
                **dict(self.config.metadata),
            },
        )


def _world_to_ego(x: float, y: float, pose: Pose3D) -> Tuple[float, float]:
    """Transform a world-frame point into the robot's planar ego frame."""

    yaw = _yaw_from_quaternion(pose.orientation_xyzw)
    dx, dy = x - pose.position[0], y - pose.position[1]
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def _apply_static_se2(
    x: float,
    y: float,
    translation_xy_m: Sequence[float],
    yaw_rad: float,
) -> Tuple[float, float]:
    """Apply a configured planar rotation then translation."""

    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return c * x - s * y + float(translation_xy_m[0]), s * x + c * y + float(translation_xy_m[1])


def _yaw_from_quaternion(quaternion_xyzw: Sequence[float]) -> float:
    """Return planar yaw from an ``x, y, z, w`` quaternion."""

    x, y, z, w = (float(value) for value in quaternion_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
