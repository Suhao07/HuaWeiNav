"""Observation cache and evidence providers for STRIVE real-robot runtime.

This module keeps ROS sensor payloads at the adapter/runtime boundary. Public
contracts only receive lightweight references such as `image_ref` and
`pointcloud_ref`, so high-level policy and verifier code do not depend on ROS
message objects or image arrays.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from real_robot.contracts import (
    BBox2D,
    CameraFrame,
    CameraModel,
    DetectionFrame,
    EvidenceSource,
    NavigationStatus,
    ObjectNodeSnapshot,
    Pose3D,
    RealObservation,
    SemanticMapSnapshot,
    ViewEvidence,
    ViewpointGoal,
)
from real_robot.image_codec import write_ros_image_png
from real_robot.sysnav_ros_adapters import RosDetectionResultAdapter


@dataclass(frozen=True)
class _CachedImageRecord:
    image_ref: str
    timestamp: float
    frame_id: str
    height: Optional[int]
    width: Optional[int]
    channels: Optional[int]
    encoding: str
    step: Optional[int]
    storage: str
    raw_path: Optional[str] = None
    metadata_path: Optional[str] = None
    visual_path: Optional[str] = None

    def as_camera_frame(
        self,
        camera_model: CameraModel,
        depth_ref: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CameraFrame:
        """Return a contract camera frame with only references and metadata."""

        rgb_shape = None
        if self.height is not None and self.width is not None and self.channels is not None:
            rgb_shape = (self.height, self.width, self.channels)
        return CameraFrame(
            image_ref=self.image_ref,
            camera_model=camera_model,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            rgb_shape=rgb_shape,
            depth_ref=depth_ref,
            metadata={
                "encoding": self.encoding,
                "step": self.step,
                "storage": self.storage,
                "raw_path": self.raw_path,
                "metadata_path": self.metadata_path,
                "visual_path": self.visual_path,
                **dict(metadata or {}),
            },
        )

    @property
    def readable_image_ref(self) -> Optional[str]:
        """Return a local image path when one was encoded for vision clients."""

        return self.visual_path or (self.image_ref if Path(self.image_ref).is_file() else None)


class RosObservationCache:
    """Cache latest ROS sensor messages as STRIVE real-robot observations.

    The cache stores heavy data only as file or ROS URI references. It is safe
    to pass `latest_observation()` into high-level runtime code because the
    returned `RealObservation` contains no ROS message objects.
    """

    def __init__(
        self,
        image_directory: Optional[str | Path] = None,
        persist_images: bool = False,
        camera_model: CameraModel = CameraModel.UNKNOWN,
        max_image_records: int = 5,
        rgb_topic: str = "/camera/image",
        depth_topic: str = "",
        pointcloud_topic: str = "",
        detection_adapter: Optional[RosDetectionResultAdapter] = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the observation cache.

        Args:
            image_directory: Directory used when `persist_images=True`.
            persist_images: Whether raw ROS image bytes should be written to
                disk. When false, image refs are ROS-style URI strings.
            camera_model: Camera projection model recorded in `CameraFrame`.
            max_image_records: Number of latest image records retained in
                memory for debugging.
            rgb_topic: Default RGB topic name used in ROS URI refs.
            depth_topic: Optional depth topic name used in depth URI refs.
            pointcloud_topic: Optional pointcloud topic name used in cloud refs.
            detection_adapter: Adapter for SysNav `DetectionResult` messages.
            now_fn: Time source for ROS-like messages without headers.
        """

        self.image_directory = Path(image_directory) if image_directory else None
        self.persist_images = persist_images
        self.camera_model = camera_model
        self.max_image_records = int(max_image_records)
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.pointcloud_topic = pointcloud_topic
        self.detection_adapter = detection_adapter or RosDetectionResultAdapter()
        self.now_fn = now_fn

        self.latest_pose: Optional[Pose3D] = None
        self.latest_rgb: Optional[_CachedImageRecord] = None
        self.latest_depth_ref: Optional[str] = None
        self.latest_pointcloud_ref: Optional[str] = None
        self.latest_pointcloud_frame_id: Optional[str] = None
        self.latest_detection_frame: Optional[DetectionFrame] = None
        self.image_records: list[_CachedImageRecord] = []
        self._room_mask_paths: dict[str, str] = {}

        if self.persist_images:
            if self.image_directory is None:
                raise ValueError("image_directory is required when persist_images=True")
            self.image_directory.mkdir(parents=True, exist_ok=True)

    def update_pose(self, pose: Pose3D) -> None:
        """Cache the latest robot pose."""

        self.latest_pose = pose

    def update_rgb_image(self, msg: Any, topic: Optional[str] = None) -> _CachedImageRecord:
        """Cache the latest RGB image and return its lightweight record."""

        # 高频相机 topic 只缓存最新引用；默认不把图像字节持续写盘。
        record = self._cache_image(msg, topic or self.rgb_topic, prefix="rgb")
        self.latest_rgb = record
        self.image_records.append(record)
        if len(self.image_records) > self.max_image_records:
            del self.image_records[: len(self.image_records) - self.max_image_records]
        return record

    def update_depth_image(self, msg: Any, topic: Optional[str] = None) -> str:
        """Cache the latest depth image reference and return it."""

        record = self._cache_image(msg, topic or self.depth_topic or "/camera/depth", prefix="depth")
        self.latest_depth_ref = record.image_ref
        return record.image_ref

    def update_pointcloud(self, msg: Any, topic: Optional[str] = None) -> str:
        """Cache the latest pointcloud reference and return it."""

        stamp = _stamp_from_header(getattr(msg, "header", None), default=self.now_fn())
        resolved_topic = topic or self.pointcloud_topic or "/cloud_registered"
        self.latest_pointcloud_ref = f"ros://{resolved_topic}/pointcloud/{stamp:.9f}"
        self.latest_pointcloud_frame_id = _frame_id_from_header(getattr(msg, "header", None))
        return self.latest_pointcloud_ref

    def update_detection_frame(self, frame: DetectionFrame) -> None:
        """Cache a platform-neutral detection frame."""

        self.latest_detection_frame = frame

    def update_detection_result(self, msg: Any, image_ref: Optional[str] = None) -> DetectionFrame:
        """Convert and cache a SysNav `DetectionResult` message."""

        # detection frame 要绑定当前 RGB 引用，后续 evidence 才能追溯 bbox 来自哪一帧。
        frame = self.detection_adapter.from_msg(
            msg,
            image_ref=image_ref or (self.latest_rgb.image_ref if self.latest_rgb is not None else None),
        )
        self.latest_detection_frame = frame
        return frame

    def has_rgb_image(self) -> bool:
        """Return whether an RGB image has arrived."""

        return self.latest_rgb is not None

    def has_pose(self) -> bool:
        """Return whether a robot pose has arrived."""

        return self.latest_pose is not None

    def latest_rgb_visual_path(self) -> str:
        """Return the latest locally readable RGB path, or an empty string.

        Returns:
            PNG/JPEG path suitable for a multimodal client. ROS URI references
            and raw binary transport dumps are intentionally excluded.
        """

        if self.latest_rgb is None:
            return ""
        return str(self.latest_rgb.readable_image_ref or "")

    def persist_room_mask(self, msg: Any, room_id: int) -> str:
        """Persist one SysNav room mask when image persistence is enabled.

        Args:
            msg: SysNav ``RoomNode``-like message containing ``room_mask``.
            room_id: Stable SysNav room id.

        Returns:
            Local mask path, or an empty string when persistence is disabled or
            the message has no encodable mask.
        """

        if not self.persist_images or self.image_directory is None:
            return ""
        image = getattr(msg, "room_mask", None)
        if image is None:
            return ""
        stamp = _stamp_from_header(getattr(image, "header", None), default=self.now_fn())
        key = f"{int(room_id)}:{stamp:.9f}"
        if key in self._room_mask_paths:
            return self._room_mask_paths[key]
        path = self.image_directory / f"room_mask_{int(room_id)}_{_stamp_token(stamp)}.png"
        written = write_ros_image_png(image, path)
        if written:
            self._room_mask_paths[key] = written
        return written or ""

    def latest_observation(self) -> Optional[RealObservation]:
        """Return the latest `RealObservation`, or None when pose/RGB is missing."""

        if self.latest_pose is None or self.latest_rgb is None:
            # RealObservation 必须至少有 pose + RGB；缺任意一个都交给 runtime readiness 等待。
            return None
        camera = self.latest_rgb.as_camera_frame(
            camera_model=self.camera_model,
            depth_ref=self.latest_depth_ref,
            metadata={
                "rgb_topic": self.rgb_topic,
                "depth_topic": self.depth_topic,
                "latest_detection_timestamp": (
                    self.latest_detection_frame.timestamp if self.latest_detection_frame is not None else None
                ),
            },
        )
        timestamp = max(
            value
            for value in (
                self.latest_pose.stamp,
                self.latest_rgb.timestamp,
                self.latest_detection_frame.timestamp if self.latest_detection_frame is not None else None,
            )
            if value is not None
        )
        # observation timestamp 取已缓存输入中的最新时间，用于标记这是一组“最近值”融合快照。
        return RealObservation(
            timestamp=float(timestamp),
            robot_pose=self.latest_pose,
            camera_frames=(camera,),
            pointcloud_ref=self.latest_pointcloud_ref,
            pointcloud_frame_id=self.latest_pointcloud_frame_id,
            odom_frame_id=self.latest_pose.frame_id,
            metadata={
                "has_detection_frame": self.latest_detection_frame is not None,
                "has_depth": self.latest_depth_ref is not None,
                "has_pointcloud": self.latest_pointcloud_ref is not None,
            },
        )

    def _cache_image(self, msg: Any, topic: str, prefix: str) -> _CachedImageRecord:
        """Create a lightweight image record from a ROS-like image message."""

        stamp = _stamp_from_header(getattr(msg, "header", None), default=self.now_fn())
        frame_id = _frame_id_from_header(getattr(msg, "header", None)) or "camera"
        height = _optional_int(getattr(msg, "height", None))
        width = _optional_int(getattr(msg, "width", None))
        encoding = str(getattr(msg, "encoding", "") or "")
        step = _optional_int(getattr(msg, "step", None))
        channels = _channels_from_encoding(encoding)

        raw_path = None
        metadata_path = None
        visual_path = None
        storage = "ros_uri"
        image_ref = f"ros://{topic}/image/{stamp:.9f}"
        if self.persist_images:
            # 只有显式 persist_images 时才写原始 bytes；真机默认避免高频写盘。
            raw_path, metadata_path, visual_path = self._write_image_payload(msg, prefix, stamp, topic)
            image_ref = raw_path
            storage = "file"

        return _CachedImageRecord(
            image_ref=image_ref,
            timestamp=stamp,
            frame_id=frame_id,
            height=height,
            width=width,
            channels=channels,
            encoding=encoding,
            step=step,
            storage=storage,
            raw_path=raw_path,
            metadata_path=metadata_path,
            visual_path=visual_path,
        )

    def _write_image_payload(self, msg: Any, prefix: str, stamp: float, topic: str) -> Tuple[str, str, Optional[str]]:
        """Write raw ROS image bytes and sidecar metadata to disk."""

        assert self.image_directory is not None
        name = f"{prefix}_{_stamp_token(stamp)}"
        raw_path = self.image_directory / f"{name}.bin"
        metadata_path = self.image_directory / f"{name}.json"
        data = getattr(msg, "data", b"")
        raw_path.write_bytes(bytes(data))
        visual_path = self.image_directory / f"{name}.png"
        visual_path = write_ros_image_png(msg, visual_path)
        if visual_path is None:
            visual_path = None
        metadata = {
            "topic": topic,
            "stamp": stamp,
            "frame_id": _frame_id_from_header(getattr(msg, "header", None)),
            "height": getattr(msg, "height", None),
            "width": getattr(msg, "width", None),
            "encoding": getattr(msg, "encoding", None),
            "step": getattr(msg, "step", None),
            "bytes": len(bytes(data)),
            "visual_path": visual_path,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return str(raw_path), str(metadata_path), visual_path


class ObjectCropEvidenceProvider:
    """Build full-image or bbox-crop evidence for reached viewpoint goals."""

    def __init__(
        self,
        observation_provider: Callable[[], Optional[RealObservation]],
        object_provider: Callable[[], Optional[Any]],
        detection_provider: Optional[Callable[[], Optional[DetectionFrame]]] = None,
        mode: str = "auto",
        default_camera_model: CameraModel = CameraModel.UNKNOWN,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        """Initialize the evidence provider.

        Args:
            observation_provider: Callable returning the latest observation.
            object_provider: Callable returning a `SemanticMapSnapshot`, object
                iterable, or None.
            detection_provider: Callable returning the latest detection frame.
            mode: `auto`, `full_image`, or `bbox_crop`.
            default_camera_model: Camera model used when no camera frame exists.
            now_fn: Time source used if observation/status stamps are missing.
        """

        normalized = mode.strip().lower()
        if normalized not in {"auto", "full_image", "bbox_crop"}:
            raise ValueError("mode must be one of: auto, full_image, bbox_crop")
        self.observation_provider = observation_provider
        self.object_provider = object_provider
        self.detection_provider = detection_provider
        self.mode = normalized
        self.default_camera_model = default_camera_model
        self.now_fn = now_fn

    def capture(self, goal: ViewpointGoal, status: NavigationStatus) -> ViewEvidence:
        """Capture evidence for a reached viewpoint goal."""

        # 这里假设 caller 已确认 REACHED；provider 只负责把最新观测包装成 verifier evidence。
        observation = self.observation_provider()
        camera = observation.primary_camera() if observation is not None else None
        objects = _objects_from_provider(self.object_provider())
        target_uid = goal.target_object_uid or goal.anchor_object_uid
        # target uid 可以来自 terminal target，也可以来自 anchor-first 的 anchor 视点。
        target_object = _find_object(objects, target_uid)
        detection = self.detection_provider() if self.detection_provider is not None else None

        bbox, bbox_source = _select_bbox(target_object, detection)
        image_ref, image_source = _select_image_ref(self.mode, camera, target_object, bbox)
        quality = _evidence_quality(
            bbox=bbox,
            camera=camera,
            detection=detection,
            mode=self.mode,
            bbox_source=bbox_source,
            observation=observation,
        )

        return ViewEvidence(
            source=EvidenceSource.VIEWPOINT_CAPTURE,
            timestamp=_evidence_timestamp(observation, status, self.now_fn),
            pose=status.current_pose or (observation.robot_pose if observation is not None else None),
            image_ref=image_ref,
            camera_model=_camera_model(camera, self.default_camera_model),
            bbox_xyxy=bbox if self.mode != "full_image" else None,
            target_object_uid=goal.target_object_uid,
            anchor_object_uid=goal.anchor_object_uid,
            relation_edge_id=goal.relation_edge_id,
            quality=quality,
            verifier_payload={
                "evidence_mode": quality["evidence_mode"],
                "target_object_uid": target_uid,
                "target_object_label": target_object.label if target_object is not None else None,
                "bbox_source": bbox_source,
            },
            metadata={
                "observation_available": observation is not None,
                "camera_available": camera is not None,
                "object_found": target_object is not None,
                "detection_frame_available": detection is not None,
                "image_ref_source": image_source,
                "object_image_ref": target_object.image_ref if target_object is not None else None,
                "motion_status": status.status.value,
                "evidence_requirements": goal.evidence_requirements,
            },
        )


def _objects_from_provider(value: Optional[Any]) -> Tuple[ObjectNodeSnapshot, ...]:
    """Return object snapshots from a snapshot, iterable, or None."""

    if value is None:
        return ()
    if isinstance(value, SemanticMapSnapshot):
        return tuple(value.objects)
    if hasattr(value, "objects"):
        return tuple(getattr(value, "objects") or ())
    return tuple(value)


def _find_object(objects: Iterable[ObjectNodeSnapshot], uid: Optional[str]) -> Optional[ObjectNodeSnapshot]:
    """Return the object matching uid, or None."""

    if uid is None:
        return None
    return next((obj for obj in objects if obj.uid == uid), None)


def _select_bbox(
    target_object: Optional[ObjectNodeSnapshot],
    detection: Optional[DetectionFrame],
) -> Tuple[Optional[BBox2D], Optional[str]]:
    """Return the best bbox and its source."""

    if target_object is not None:
        if target_object.bbox2d_xyxy is not None:
            # object node 自带 bbox 时优先使用，因为它和 mapper uid 绑定最强。
            return target_object.bbox2d_xyxy, "object_node"
        metadata_bbox = target_object.metadata.get("bbox2d_xyxy") if isinstance(target_object.metadata, dict) else None
        if metadata_bbox is not None:
            return _bbox_tuple(metadata_bbox), "object_metadata"
        track_ids = set(str(track_id) for track_id in target_object.track_ids)
        if detection is not None and track_ids and detection.track_ids:
            for idx, track_id in enumerate(detection.track_ids):
                if str(track_id) in track_ids and idx < len(detection.boxes_xyxy):
                    # track id 次优先，能把 detector bbox 和 SysNav object uid 对齐。
                    return detection.boxes_xyxy[idx], "detection_track"
        if detection is not None:
            for idx, label in enumerate(detection.labels):
                if label == target_object.label and idx < len(detection.boxes_xyxy):
                    # label fallback 只用于弱证据；多个同类物体时需要 verifier 再判断。
                    return detection.boxes_xyxy[idx], "detection_label"
    return None, None


def _select_image_ref(
    mode: str,
    camera: Optional[CameraFrame],
    target_object: Optional[ObjectNodeSnapshot],
    bbox: Optional[BBox2D],
) -> Tuple[Optional[str], str]:
    """Return image reference and source label for the evidence mode."""

    if mode in {"auto", "bbox_crop"} and target_object is not None and target_object.image_ref:
        # SysNav object image_ref 通常是对象裁剪或标注图，优先给 verifier 看目标局部。
        return target_object.image_ref, "object_image_ref"
    if camera is not None:
        # 没有对象图像时退回当前整帧 RGB，避免伪造 crop。
        return camera.image_ref, "camera_frame"
    if target_object is not None and target_object.image_ref:
        return target_object.image_ref, "object_image_ref"
    return None, "missing"


def _evidence_quality(
    bbox: Optional[BBox2D],
    camera: Optional[CameraFrame],
    detection: Optional[DetectionFrame],
    mode: str,
    bbox_source: Optional[str],
    observation: Optional[RealObservation],
) -> Dict[str, Any]:
    """Return view quality facts for verifier prompts and logs."""

    width, height = _camera_width_height(camera)
    actual_mode = "full_image" if mode == "full_image" or bbox is None else "bbox_crop"
    quality: Dict[str, Any] = {
        "evidence_mode": actual_mode,
        "requested_mode": mode,
        "bbox_source": bbox_source,
        "source_timestamp": observation.timestamp if observation is not None else None,
        "detection_timestamp": detection.timestamp if detection is not None else None,
        "frame_width": width,
        "frame_height": height,
    }
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        area_px = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        quality["bbox_area_px"] = area_px
        if width and height and width > 0 and height > 0:
            # 这些 view quality facts 只作为 verifier prompt/日志事实，不在这里硬编码成功阈值。
            quality["bbox_area_ratio"] = area_px / float(width * height)
            quality["center_score"] = _center_score(bbox, width, height)
            margin_px = min(x1, y1, width - x2, height - y2)
            quality["border_margin_px"] = margin_px
            quality["border_margin_ratio"] = margin_px / float(min(width, height))
        else:
            quality["bbox_area_ratio"] = None
            quality["center_score"] = None
            quality["border_margin_px"] = None
            quality["border_margin_ratio"] = None
    return quality


def _camera_width_height(camera: Optional[CameraFrame]) -> Tuple[Optional[int], Optional[int]]:
    """Return camera width and height from `rgb_shape`."""

    if camera is None or camera.rgb_shape is None:
        return None, None
    height, width = camera.rgb_shape[:2]
    return int(width), int(height)


def _center_score(bbox: BBox2D, width: int, height: int) -> float:
    """Return 1.0 near image center and 0.0 at the farthest corner."""

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = abs(cx - width / 2.0)
    dy = abs(cy - height / 2.0)
    max_dx = width / 2.0
    max_dy = height / 2.0
    if max_dx <= 0 or max_dy <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((dx / max_dx) ** 2 + (dy / max_dy) ** 2) ** 0.5))


def _evidence_timestamp(
    observation: Optional[RealObservation],
    status: NavigationStatus,
    now_fn: Callable[[], float],
) -> float:
    """Return best timestamp for one evidence record."""

    if observation is not None:
        return float(observation.timestamp)
    if status.stamp is not None:
        return float(status.stamp)
    return float(now_fn())


def _camera_model(camera: Optional[CameraFrame], default: CameraModel) -> CameraModel:
    """Return camera model from frame or default."""

    if camera is None:
        return default
    return camera.camera_model


def _stamp_from_header(header: Any, default: float) -> float:
    """Return a float timestamp from a ROS-like header."""

    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return float(default)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    if sec is None:
        return float(default)
    return float(sec) + float(nanosec) / 1e9


def _frame_id_from_header(header: Any) -> Optional[str]:
    """Return frame id from a ROS-like header."""

    frame_id = getattr(header, "frame_id", None)
    return str(frame_id) if frame_id else None


def _channels_from_encoding(encoding: str) -> Optional[int]:
    """Return likely channel count for common ROS image encodings."""

    normalized = encoding.lower()
    if normalized in {"rgb8", "bgr8", "8uc3", "16uc3", "32fc3"}:
        return 3
    if normalized in {"rgba8", "bgra8", "8uc4", "16uc4", "32fc4"}:
        return 4
    if normalized in {"mono8", "mono16", "8uc1", "16uc1", "32fc1", "32fc", "16uc1"}:
        return 1
    return None


def _optional_int(value: Any) -> Optional[int]:
    """Return int(value), or None when value is missing."""

    if value is None:
        return None
    return int(value)


def _stamp_token(stamp: float) -> str:
    """Return a filesystem-friendly timestamp token."""

    return f"{stamp:.9f}".replace(".", "_")


def _bbox_tuple(value: Sequence[Any]) -> BBox2D:
    """Return a validated bbox tuple."""

    if len(value) != 4:
        raise ValueError("bbox must contain four values")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
