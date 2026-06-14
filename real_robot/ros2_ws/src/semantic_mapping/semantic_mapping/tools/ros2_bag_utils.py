import struct

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker


def make_stamp(seconds, nanoseconds):
    stamp = TimeMsg()
    stamp.sec = int(seconds)
    stamp.nanosec = int(nanoseconds)
    return stamp


def make_header(seconds, nanoseconds, frame_id="map"):
    header = Header()
    header.stamp = make_stamp(seconds, nanoseconds)
    header.frame_id = frame_id
    return header


def create_point_cloud(points, seconds, nanoseconds, frame_id="map"):
    points = _as_points(points)
    header = make_header(seconds, nanoseconds, frame_id)
    return point_cloud2.create_cloud_xyz32(header, points[:, :3].tolist())


def create_colored_point_cloud(points, colors, seconds, nanoseconds, frame_id="map"):
    points = _as_points(points)
    colors = _as_colors(colors, len(points))
    header = make_header(seconds, nanoseconds, frame_id)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    cloud = []
    for point, color in zip(points[:, :3], colors):
        rgb_float = _pack_rgb_float(color)
        cloud.append((float(point[0]), float(point[1]), float(point[2]), rgb_float))
    return point_cloud2.create_cloud(header, fields, cloud)


def create_odom_msg(odom, seconds, nanoseconds, frame_id="map", child_frame_id="sensor"):
    position, quat = _pose_from_odom(odom)
    msg = Odometry()
    msg.header = make_header(seconds, nanoseconds, frame_id)
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = float(position[0])
    msg.pose.pose.position.y = float(position[1])
    msg.pose.pose.position.z = float(position[2])
    msg.pose.pose.orientation.x = float(quat[0])
    msg.pose.pose.orientation.y = float(quat[1])
    msg.pose.pose.orientation.z = float(quat[2])
    msg.pose.pose.orientation.w = float(quat[3])
    return msg


def create_tf_msg(odom, seconds, nanoseconds, frame_id="map", child_frame_id="sensor"):
    position, quat = _pose_from_odom(odom)
    transform = TransformStamped()
    transform.header = make_header(seconds, nanoseconds, frame_id)
    transform.child_frame_id = child_frame_id
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    transform.transform.rotation.x = float(quat[0])
    transform.transform.rotation.y = float(quat[1])
    transform.transform.rotation.z = float(quat[2])
    transform.transform.rotation.w = float(quat[3])
    return TFMessage(transforms=[transform])


def create_wireframe_marker(
    center,
    extent,
    yaw,
    ns,
    box_id,
    color,
    seconds,
    nanoseconds,
    frame_id="map",
    line_width=0.03,
):
    corners = _box_corners(center, extent, yaw)
    return create_wireframe_marker_from_corners(
        corners=corners,
        ns=ns,
        box_id=box_id,
        color=color,
        seconds=seconds,
        nanoseconds=nanoseconds,
        frame_id=frame_id,
        line_width=line_width,
    )


def create_wireframe_marker_from_corners(
    corners,
    ns,
    box_id,
    color,
    seconds,
    nanoseconds,
    frame_id="map",
    line_width=0.03,
):
    corners = _as_points(corners)
    marker = Marker()
    marker.header = make_header(seconds, nanoseconds, frame_id)
    marker.ns = str(ns)
    marker.id = int(box_id)
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(line_width)
    marker.color = _make_color(color)

    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for start, end in edges:
        marker.points.append(_make_point(corners[start]))
        marker.points.append(_make_point(corners[end]))
    return marker


def create_text_marker(
    center,
    marker_id,
    text,
    color,
    text_height,
    seconds,
    nanoseconds,
    frame_id="map",
):
    center = np.asarray(center, dtype=float).reshape(-1)
    marker = Marker()
    marker.header = make_header(seconds, nanoseconds, frame_id)
    marker.ns = "text"
    marker.id = int(marker_id)
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position.x = float(center[0])
    marker.pose.position.y = float(center[1])
    marker.pose.position.z = float(center[2] + text_height)
    marker.pose.orientation.w = 1.0
    marker.scale.z = float(text_height)
    marker.color = _make_color(color)
    marker.text = str(text)
    return marker


def _as_points(points):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros((0, 3), dtype=float)
    return points.reshape(-1, points.shape[-1])[:, :3]


def _as_colors(colors, count):
    colors = np.asarray(colors, dtype=float)
    if colors.size == 0:
        return np.zeros((count, 3), dtype=float)
    colors = colors.reshape(-1, colors.shape[-1])[:, :3]
    if len(colors) == 1 and count > 1:
        colors = np.repeat(colors, count, axis=0)
    if len(colors) != count:
        raise ValueError(f"Expected {count} colors, got {len(colors)}")
    if np.nanmax(colors) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def _pack_rgb_float(color):
    r, g, b = [int(channel) for channel in color[:3]]
    rgb_uint32 = (r << 16) | (g << 8) | b
    return struct.unpack("f", struct.pack("I", rgb_uint32))[0]


def _make_color(color, alpha=1.0):
    color = np.asarray(color, dtype=float).reshape(-1)
    if color.size < 3:
        color = np.array([1.0, 1.0, 1.0], dtype=float)
    if np.nanmax(color[:3]) > 1.0:
        color = color / 255.0
    msg = ColorRGBA()
    msg.r = float(np.clip(color[0], 0.0, 1.0))
    msg.g = float(np.clip(color[1], 0.0, 1.0))
    msg.b = float(np.clip(color[2], 0.0, 1.0))
    msg.a = float(alpha)
    return msg


def _make_point(point):
    point = np.asarray(point, dtype=float).reshape(-1)
    msg = Point()
    msg.x = float(point[0])
    msg.y = float(point[1])
    msg.z = float(point[2])
    return msg


def _box_corners(center, extent, yaw):
    center = np.asarray(center, dtype=float).reshape(3)
    half = np.asarray(extent, dtype=float).reshape(3) / 2.0
    corners = np.array([
        [half[0], half[1], half[2]],
        [half[0], -half[1], half[2]],
        [-half[0], -half[1], half[2]],
        [-half[0], half[1], half[2]],
        [half[0], half[1], -half[2]],
        [half[0], -half[1], -half[2]],
        [-half[0], -half[1], -half[2]],
        [-half[0], half[1], -half[2]],
    ])
    rot = Rotation.from_euler("z", float(yaw)).as_matrix()
    return corners @ rot.T + center


def _pose_from_odom(odom):
    if hasattr(odom, "pose") and hasattr(odom.pose, "pose"):
        pose = odom.pose.pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ], dtype=float)
        return position, quat

    data = np.asarray(odom, dtype=float)
    if data.shape == (4, 4):
        position = data[:3, 3]
        quat = Rotation.from_matrix(data[:3, :3]).as_quat()
        return position, quat

    flat = data.reshape(-1)
    if flat.size >= 7:
        return flat[:3], flat[3:7]
    if flat.size >= 6:
        quat = Rotation.from_euler("xyz", flat[3:6]).as_quat()
        return flat[:3], quat
    raise ValueError(f"Unsupported odom shape: {data.shape}")
