#!/usr/bin/env python3
"""Replay SysNav viewpoint topics from a ROS 2 bag through the VLN bridge.

This utility is intentionally an offline observer.  It does not publish ROS
messages, select viewpoints, or command a robot.  It deserializes the three
SysNav-facing topics used by ``SysNavViewpointBridgeModel`` and writes the
resolved ``ViewpointPose``-equivalent records as JSON Lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO

# 让仓库内脚本在直接执行时也能导入平台无关的 real_robot 模块。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from real_robot.sysnav_viewpoint_bridge import (
    SysNavViewpointBridgeModel,
    SysNavViewpointRecord,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ROS 2 bag replay."""

    parser = argparse.ArgumentParser(
        description="Replay SysNav viewpoint/odometry/object topics and export resolved poses."
    )
    parser.add_argument("bag_path", type=Path, help="ROS 2 bag directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("-"),
        help="JSONL output path; use '-' for stdout",
    )
    parser.add_argument("--viewpoint-topic", default="/viewpoint_rep_header")
    parser.add_argument("--object-topic", default="/object_nodes_list")
    parser.add_argument("--odom-topic", default="/state_estimation")
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument("--max-time-offset-s", type=float, default=0.25)
    parser.add_argument("--odom-history-size", type=int, default=400)
    return parser


def _load_rosbag_runtime() -> tuple[Any, Any, Any, Any, Any]:
    """Load ROS 2 bag dependencies only when the replay command is executed.

    Returns:
        ``SequentialReader``, ``StorageOptions``, ``ConverterOptions``,
        ``deserialize_message`` and ``get_message`` constructors.

    Raises:
        RuntimeError: If the command is run outside a sourced ROS 2 setup.
    """

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 rosbag2_py is unavailable; source ROS2 Humble and the STRIVE overlay first."
        ) from exc
    return (
        rosbag2_py.SequentialReader,
        rosbag2_py.StorageOptions,
        rosbag2_py.ConverterOptions,
        deserialize_message,
        get_message,
    )


def _topic_types(reader: Any) -> dict[str, str]:
    """Return the message type advertised for every topic in a bag."""

    return {item.name: item.type for item in reader.get_all_topics_and_types()}


def _record_to_dict(record: SysNavViewpointRecord, bag_timestamp_ns: int) -> dict[str, Any]:
    """Serialize one resolved record using the stable bridge contract."""

    return {
        "bag_timestamp_ns": int(bag_timestamp_ns),
        "viewpoint_id": record.viewpoint_id,
        "timestamp": record.timestamp,
        "timestamp_ns": record.timestamp_ns,
        "frame_id": record.pose.frame_id,
        "pose": {
            "position": list(record.pose.position),
            "orientation_xyzw": list(record.pose.orientation_xyzw),
        },
        "observed_object_ids": list(record.observed_object_ids),
        "source": "sysnav_viewpoint_rep_plus_odom",
    }


def _write_records(
    records: Iterable[SysNavViewpointRecord],
    bag_timestamp_ns: int,
    output: TextIO,
) -> int:
    """Write resolved records and return the number of lines written."""

    count = 0
    for record in records:
        output.write(json.dumps(_record_to_dict(record, bag_timestamp_ns), sort_keys=True) + "\n")
        count += 1
    return count


def replay_bag(args: argparse.Namespace, output: TextIO) -> int:
    """Replay selected ROS 2 bag topics through the pure bridge model.

    Args:
        args: Parsed command-line arguments.
        output: Text stream receiving resolved JSONL records.

    Returns:
        Number of resolved viewpoint records written.

    Raises:
        FileNotFoundError: If the bag path does not exist.
        RuntimeError: If ROS 2 bag dependencies are unavailable or a required
            topic is missing from the bag.
    """

    if not args.bag_path.exists():
        raise FileNotFoundError(f"ROS 2 bag does not exist: {args.bag_path}")

    (
        sequential_reader,
        storage_options,
        converter_options,
        deserialize_message,
        get_message,
    ) = _load_rosbag_runtime()
    reader = sequential_reader()
    reader.open(
        storage_options(uri=str(args.bag_path), storage_id=args.storage_id),
        converter_options(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    types = _topic_types(reader)
    required_topics = (args.viewpoint_topic, args.object_topic, args.odom_topic)
    missing = [topic for topic in required_topics if topic not in types]
    if missing:
        raise RuntimeError(f"Required SysNav topic(s) missing from bag: {', '.join(missing)}")

    message_types = {topic: get_message(types[topic]) for topic in required_topics}
    model = SysNavViewpointBridgeModel(
        max_time_offset_s=args.max_time_offset_s,
        odom_history_size=args.odom_history_size,
    )
    count = 0
    while reader.has_next():
        topic, serialized, bag_timestamp_ns = reader.read_next()
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])
        if topic == args.odom_topic:
            records = model.update_odometry(message)
        elif topic == args.viewpoint_topic:
            records = model.update_viewpoint(message)
        else:
            records = model.update_object_nodes(message)
        # 中文核心约束：回放只验证“原始事件 -> 对齐 viewpoint pose”，不重新
        # 规划路径，也不使用对象中心或房间中心伪造 viewpoint 位姿。
        count += _write_records(records, bag_timestamp_ns, output)
    output.flush()
    return count


def main(argv: list[str] | None = None) -> int:
    """Run the ROS 2 bag replay command."""

    args = build_parser().parse_args(argv)
    output_stream: TextIO
    close_output = False
    if str(args.output) == "-":
        output_stream = sys.stdout
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_stream = args.output.open("w", encoding="utf-8")
        close_output = True
    try:
        count = replay_bag(args, output_stream)
    finally:
        if close_output:
            output_stream.close()
    print(f"resolved viewpoint records: {count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
