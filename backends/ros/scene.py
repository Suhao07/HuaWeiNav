"""Build immutable scene snapshots from SysNav topic records."""
from __future__ import annotations
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional
from scene.contracts import Pose3D, SemanticMapSnapshot
from real_robot.sysnav_ros_adapters import (
    RosObjectNodeAdapter, RosRoomNodeAdapter, RosViewpointPoseAdapter,
    build_semantic_map_snapshot,
)

@dataclass
class SysNavSemanticMapBridge:
    """Cache SysNav object, room, and viewpoint topics for VLN snapshots.

    SysNav publishes object updates with the viewpoint ID at which an object
    was observed.  The bridge keeps that relation monotonically and combines
    it with the pose records emitted by the viewpoint bridge.  It does not
    change labels, infer semantic relations, or rank viewpoints.
    """

    robot_pose_provider: Callable[[], Pose3D]
    object_adapter: RosObjectNodeAdapter = field(default_factory=RosObjectNodeAdapter)
    room_adapter: RosRoomNodeAdapter = field(default_factory=RosRoomNodeAdapter)
    viewpoint_adapter: RosViewpointPoseAdapter = field(default_factory=RosViewpointPoseAdapter)
    latest_object_list_msg: Optional[Any] = None
    latest_room_list_msg: Optional[Any] = None
    latest_viewpoints: Dict[str, Any] = field(default_factory=dict)
    object_viewpoint_history: Dict[str, set[str]] = field(default_factory=dict)

    def update_object_nodes(self, msg: Any) -> None:
        """Store the latest SysNav ``/object_nodes_list`` message."""

        # 核心：SysNav semantic_mapping_node 是对象图唯一写入方，VLN runtime 只缓存只读消息。
        self.latest_object_list_msg = msg
        for obj in self.object_adapter.from_list_msg(msg):
            if not obj.visible_viewpoints:
                continue
            self.object_viewpoint_history.setdefault(obj.uid, set()).update(obj.visible_viewpoints)
            for viewpoint_uid in obj.visible_viewpoints:
                viewpoint = self.latest_viewpoints.get(viewpoint_uid)
                if viewpoint is None or obj.uid in viewpoint.visible_objects:
                    continue
                # 中文核心约束：对象消息晚到时只补充同一 viewpoint 的观测集合，
                # 不覆盖 viewpoint pose，也不改变 SysNav 分配的对象身份。
                self.latest_viewpoints[viewpoint_uid] = replace(
                    viewpoint,
                    visible_objects=tuple(sorted(set(viewpoint.visible_objects) | {obj.uid})),
                )

    def update_room_nodes(self, msg: Any) -> None:
        """Store the latest SysNav ``/room_nodes_list`` message."""

        self.latest_room_list_msg = msg

    def update_viewpoint_pose(self, msg: Any) -> None:
        """Store one bridge-produced SysNav viewpoint pose record.

        Args:
            msg: ROS-like ``tare_planner/ViewpointPose`` message.

        Raises:
            ValueError: If the bridge message does not contain a stable ID or
                a pose frame.
        """

        viewpoint = self.viewpoint_adapter.from_msg(msg)
        self.latest_viewpoints[viewpoint.uid] = viewpoint
        for object_uid in viewpoint.visible_objects:
            self.object_viewpoint_history.setdefault(object_uid, set()).add(viewpoint.uid)

    def has_object_snapshot(self) -> bool:
        """Return whether at least one object list message has arrived."""

        return self.latest_object_list_msg is not None

    def build_snapshot(self, timestamp: Optional[float] = None) -> Optional[SemanticMapSnapshot]:
        """Build a VLN semantic map snapshot from cached SysNav messages."""

        if self.latest_object_list_msg is None:
            return None
        base_snapshot = build_semantic_map_snapshot(
            object_list_msg=self.latest_object_list_msg,
            room_list_msg=self.latest_room_list_msg,
            robot_pose=self.robot_pose_provider(),
            timestamp=timestamp,
            object_adapter=self.object_adapter,
            room_adapter=self.room_adapter,
        )
        # 中文核心约束：viewpoint 历史只补充“在哪个视角观察到对象”，不覆盖
        # SysNav 当前对象位置、标签或身份；viewpoint pose 仍完全来自下层 bridge。
        objects = tuple(
            replace(
                obj,
                visible_viewpoints=tuple(
                    sorted(
                        set(obj.visible_viewpoints)
                        | self.object_viewpoint_history.get(obj.uid, set())
                    )
                ),
            )
            for obj in base_snapshot.objects
        )
        return replace(
            base_snapshot,
            objects=objects,
            viewpoints=tuple(self.latest_viewpoints.values()),
            metadata={
                **base_snapshot.metadata,
                "viewpoint_count": len(self.latest_viewpoints),
                "viewpoint_topic_connected": bool(self.latest_viewpoints),
            },
        )

    def create_ros_subscriptions(
        self,
        node: Any,
        object_node_list_type: Any,
        room_node_list_type: Any,
        viewpoint_pose_type: Optional[Any] = None,
        object_topic: str = "/object_nodes_list",
        room_topic: str = "/room_nodes_list",
        viewpoint_pose_topic: str = "/strive/sysnav/viewpoint_pose",
        queue_size: int = 10,
    ) -> Dict[str, Any]:
        """Register ROS subscriptions on a provided node and return handles."""

        # ROS 类型由调用方注入，避免非 ROS 环境 import runtime 时失败。
        subscriptions = {
            "object_nodes": node.create_subscription(
                object_node_list_type,
                object_topic,
                self.update_object_nodes,
                queue_size,
            ),
            "room_nodes": node.create_subscription(
                room_node_list_type,
                room_topic,
                self.update_room_nodes,
                queue_size,
            ),
        }
        if viewpoint_pose_type is not None and viewpoint_pose_topic:
            subscriptions["viewpoint_pose"] = node.create_subscription(
                viewpoint_pose_type,
                viewpoint_pose_topic,
                self.update_viewpoint_pose,
                queue_size,
            )
        return subscriptions
