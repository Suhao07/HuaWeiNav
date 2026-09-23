#!/usr/bin/env python3
"""Exercise the native viewpoint producer and existing bridge with synthetic ROS input.

Run in a network-isolated container/ROS domain. This program publishes synthetic
sensor and object messages; it starts no chassis, waypoint or velocity component.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_prefix
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
from tare_planner.msg import ObjectNode, ObjectNodeList, ViewpointPose, ViewpointRep
from strive_sysnav_bringup.viewpoint_bridge_node import SysNavViewpointBridgeNode


def main() -> None:
    """Verify native events, stationary duplicate suppression and object association."""
    executable = Path(get_package_prefix('strive_sysnav_viewpoints')) / 'lib/strive_sysnav_viewpoints/sysnav_viewpoint_provider'
    artifact_dir = Path(os.environ.get('STRIVE_HIL_ARTIFACT_DIR', '/tmp'))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / 'sysnav_viewpoint_provider_smoke.log'
    rclpy.init()
    node = Node('sysnav_viewpoint_provider_probe')
    bridge = SysNavViewpointBridgeNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(bridge)
    events, poses = [], []
    node.create_subscription(ViewpointRep, '/viewpoint_rep_header', events.append, 10)
    node.create_subscription(ViewpointPose, '/strive/sysnav/viewpoint_pose', poses.append, 10)
    odom_pub = node.create_publisher(Odometry, '/state_estimation', 10)
    cloud_pub = node.create_publisher(PointCloud2, '/cloud_registered', 5)
    object_pub = node.create_publisher(ObjectNodeList, '/object_nodes_list', 10)
    with log_path.open('w') as log:
        process = subprocess.Popen([str(executable), '--ros-args', '-p', 'odom_topic:=/state_estimation'], stdout=log, stderr=subprocess.STDOUT)
        try:
            startup_deadline = time.monotonic() + 30.0
            while cloud_pub.get_subscription_count() == 0 or odom_pub.get_subscription_count() < 2:
                if process.poll() is not None or time.monotonic() >= startup_deadline:
                    raise RuntimeError(f'provider subscribers unavailable; see {log_path}')
                executor.spin_once(timeout_sec=.05)

            def pump(seconds: float, x: float) -> None:
                deadline = time.monotonic() + seconds
                next_publish = 0.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f'provider exited {process.returncode}; see {log_path}')
                    if time.monotonic() >= next_publish:
                        stamp = node.get_clock().now().to_msg()
                        odom = Odometry()
                        odom.header = Header(stamp=stamp, frame_id='map')
                        odom.pose.pose.position.x = x
                        odom.pose.pose.orientation.w = 1.0
                        odom_pub.publish(odom)
                        points = [(x + 2.0 + i * .2, j * .2, k * .2)
                                  for i in range(10) for j in range(-10, 11) for k in range(-2, 3)]
                        cloud = create_cloud_xyz32(odom.header, points)
                        cloud_pub.publish(cloud)
                        if events:
                            obj = ObjectNode()
                            obj.header = events[0].header
                            obj.object_id = [42]
                            obj.label = 'book'
                            obj.position.x = 2.0
                            obj.status = True
                            obj.viewpoint_id = events[0].viewpoint_id
                            obj.cloud = create_cloud_xyz32(obj.header, [(2.0, 0.0, 0.0)])
                            objects = ObjectNodeList(header=obj.header, nodes=[obj])
                            object_pub.publish(objects)
                        next_publish = time.monotonic() + .1
                    executor.spin_once(timeout_sec=.02)

            pump(8.0, 0.0)
            assert [e.viewpoint_id for e in events] == [0], [e.viewpoint_id for e in events]
            pump(2.0, 0.0)
            assert len(events) == 1, 'stationary observations added a duplicate viewpoint'
            pump(8.0, 10.0)
            assert [e.viewpoint_id for e in events] == [0, 1], [e.viewpoint_id for e in events]
            associated = [p for p in poses if p.viewpoint_id == 0 and 42 in p.observed_object_ids]
            assert associated, 'bridge did not associate object 42 with native viewpoint 0'
            assert all(p.header.frame_id == 'map' for p in poses)
            assert any(p.viewpoint_id == 1 and abs(p.pose.position.x - 10.0) < 1e-6 for p in poses)
            result = {'native_viewpoint_ids': [e.viewpoint_id for e in events],
                      'stationary_duplicate_suppressed': True, 'associated_object_id': 42,
                      'native_provider_bridge_passed': True}
            (artifact_dir / 'sysnav_viewpoint_provider_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result))
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
            executor.shutdown()
            bridge.destroy_node()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
