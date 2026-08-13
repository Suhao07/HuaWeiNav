#!/usr/bin/env python3
"""Generate a small rosbag2 input for the migrated SysNav local planner.

The generated bag is deliberately synthetic. It exercises ROS message
serialization, topic remapping, and the lower-planner replay boundary; it is
not evidence about a real sensor, localization system, or chassis.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import rosbag2_py
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2, PointField


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for synthetic bag generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output rosbag2 directory")
    parser.add_argument("--frames", type=int, default=80, help="Number of sensor frames")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="Synthetic sensor rate")
    parser.add_argument("--frame-id", default="map", help="Frame id for both messages")
    return parser


def _stamp(seconds: float) -> Time:
    """Return a ROS time message for one synthetic timestamp."""

    message = Time()
    whole = int(seconds)
    message.sec = whole
    message.nanosec = int((seconds - whole) * 1e9)
    return message


def _odom(timestamp: float, frame_id: str) -> Odometry:
    """Create a stationary odometry message in the planner world frame."""

    message = Odometry()
    message.header.stamp = _stamp(timestamp)
    message.header.frame_id = frame_id
    message.child_frame_id = "base_link"
    message.pose.pose.orientation.w = 1.0
    return message


def _pointcloud(timestamp: float, frame_id: str) -> PointCloud2:
    """Create an obstacle-free registered scan accepted by SysNav localPlanner."""

    message = PointCloud2()
    message.header.stamp = _stamp(timestamp)
    message.header.frame_id = frame_id
    message.height = 1
    message.width = 1
    message.is_bigendian = False
    message.is_dense = True
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.point_step = 16
    message.row_step = 16
    # 中文说明：点放在规划器局部裁剪范围之外，测试关注的是合法输入和
    # planner 消费链，不为某个障碍布局手写导航答案。
    message.data = list(struct.pack("<ffff", 10.0, 10.0, 0.0, 0.0))
    return message


def write_bag(output: Path, *, frames: int, rate_hz: float, frame_id: str) -> None:
    """Write odometry and registered scan messages into a rosbag2 directory.

    Args:
        output: Destination rosbag2 directory. Existing output is removed.
        frames: Number of paired odometry and scan messages.
        rate_hz: Timestamp frequency used for the synthetic stream.
        frame_id: Shared coordinate frame for generated messages.

    Raises:
        ValueError: If frame count or rate is invalid.
    """

    if frames < 2:
        raise ValueError("frames must be at least 2")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
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
    for name, message_type in (
        ("/aft_mapped_to_init", "nav_msgs/msg/Odometry"),
        ("/cloud_registered", "sensor_msgs/msg/PointCloud2"),
    ):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=name,
                type=message_type,
                serialization_format="cdr",
                offered_qos_profiles="",
            )
        )

    for index in range(frames):
        timestamp = index / rate_hz
        # 中文说明：按时间顺序写入成对观测，避免 replay 依赖任意消息排序。
        writer.write(
            "/aft_mapped_to_init",
            serialize_message(_odom(timestamp, frame_id)),
            int(timestamp * 1e9),
        )
        writer.write(
            "/cloud_registered",
            serialize_message(_pointcloud(timestamp, frame_id)),
            int(timestamp * 1e9),
        )


def main() -> None:
    """Generate one synthetic lower-planner bag and exit."""

    arguments = build_parser().parse_args()
    write_bag(
        arguments.output,
        frames=arguments.frames,
        rate_hz=arguments.rate_hz,
        frame_id=arguments.frame_id,
    )
    print(f"synthetic lower planner bag written: {arguments.output}")


if __name__ == "__main__":
    main()
