#!/usr/bin/env python3
"""Generate a minimal rosbag2 for the SysNav viewpoint bridge contract.

The bag contains real ROS2 CDR serialization for ``ViewpointRep``,
``ObjectNodeList`` and ``Odometry``.  It validates message compatibility and
timestamp joining only; it is not evidence about a physical sensor or robot.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from tare_planner.msg import ObjectNode, ObjectNodeList, ViewpointRep


def _stamp(seconds: int, nanoseconds: int):
    """Create a ROS time message without converting through float seconds."""

    from builtin_interfaces.msg import Time

    message = Time()
    message.sec = seconds
    message.nanosec = nanoseconds
    return message


def _odometry(frame_id: str, seconds: int, nanoseconds: int) -> Odometry:
    """Create one stationary odometry message."""

    message = Odometry()
    message.header.stamp = _stamp(seconds, nanoseconds)
    message.header.frame_id = frame_id
    message.pose.pose.position.x = 1.25
    message.pose.pose.position.y = -0.75
    message.pose.pose.orientation.w = 1.0
    return message


def _viewpoint(frame_id: str, seconds: int, nanoseconds: int, viewpoint_id: int) -> ViewpointRep:
    """Create one SysNav viewpoint header message."""

    message = ViewpointRep()
    message.header.stamp = _stamp(seconds, nanoseconds)
    message.header.frame_id = frame_id
    message.viewpoint_id = viewpoint_id
    return message


def _object_nodes(frame_id: str, seconds: int, nanoseconds: int, viewpoint_id: int, object_id: int) -> ObjectNodeList:
    """Create one semantic mapping update associated with a viewpoint."""

    node = ObjectNode()
    node.header.stamp = _stamp(seconds, nanoseconds)
    node.header.frame_id = frame_id
    node.object_id = [object_id]
    node.label = "book"
    node.position.x = 2.0
    node.position.y = -0.5
    node.status = True
    node.viewpoint_id = viewpoint_id

    message = ObjectNodeList()
    message.header.stamp = _stamp(seconds, nanoseconds)
    message.header.frame_id = frame_id
    message.nodes = [node]
    return message


def write_bag(
    output: Path,
    *,
    frame_id: str = "map",
    seconds: int = 10,
    nanoseconds: int = 123_456_789,
    viewpoint_id: int = 7,
    object_id: int = 42,
) -> None:
    """Write one timestamp-aligned SysNav viewpoint event to rosbag2.

    Args:
        output: Destination rosbag2 directory. Existing output is removed.
        frame_id: Shared frame ID for the three messages.
        seconds: ROS timestamp seconds component.
        nanoseconds: ROS timestamp nanoseconds component.
        viewpoint_id: SysNav viewpoint ID.
        object_id: SysNav object ID associated with the viewpoint.

    Raises:
        ValueError: If timestamp components or IDs are invalid.
    """

    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("invalid ROS timestamp")
    if viewpoint_id < 0 or object_id < 0:
        raise ValueError("IDs must be non-negative")
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topics = (
        ("/state_estimation", "nav_msgs/msg/Odometry"),
        ("/viewpoint_rep_header", "tare_planner/msg/ViewpointRep"),
        ("/object_nodes_list", "tare_planner/msg/ObjectNodeList"),
    )
    for name, message_type in topics:
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=name,
                type=message_type,
                serialization_format="cdr",
                offered_qos_profiles="",
            )
        )

    timestamp_ns = seconds * 1_000_000_000 + nanoseconds
    messages = (
        ("/state_estimation", _odometry(frame_id, seconds, nanoseconds)),
        ("/viewpoint_rep_header", _viewpoint(frame_id, seconds, nanoseconds, viewpoint_id)),
        (
            "/object_nodes_list",
            _object_nodes(frame_id, seconds, nanoseconds, viewpoint_id, object_id),
        ),
    )
    # 中文核心约束：三条输入共享同一整数时间戳，回放验证的是 ROS2 CDR
    # 消息和 bridge 的时序 join，不在生成器中写任何导航策略。
    for topic, message in messages:
        writer.write(topic, serialize_message(message), timestamp_ns)


def main() -> None:
    """Generate a synthetic SysNav viewpoint bag."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    write_bag(arguments.output)
    print(f"synthetic SysNav viewpoint bag written: {arguments.output}")


if __name__ == "__main__":
    main()
