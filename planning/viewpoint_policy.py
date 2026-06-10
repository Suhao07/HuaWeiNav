from __future__ import annotations

from typing import Any

import numpy as np

from mapping_utils.geometry import project_to_camera
from navigation_core.view_geometry import (
    habitat_path_to_strive_points,
    interpolate_polyline,
    rotation_towards,
    score_projection,
    truncate_at_stop_radius,
)


def build_check_again_viewpoints(
    *,
    object_node: Any,
    camera_intrinsic: Any,
    mapper_initial_position: np.ndarray,
    habitat_path_points: Any,
    target_height: float,
    success_distance: float,
    stop_criterion: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Generate class-agnostic better-view proposals for one object instance.

    The caller owns navigation state and final verifier decisions. This helper
    only projects the current object point cloud from reachable path samples and
    scores how useful each sample is as visual evidence.
    """

    positions = object_node.pcd.point.positions.cpu().numpy()
    bbox = object_node.bbox
    obj_bbox_area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    obj_bbox_length = float(bbox[2] - bbox[0])
    obj_bbox_width = float(bbox[3] - bbox[1])

    path_points = habitat_path_to_strive_points(
        habitat_path_points,
        mapper_initial_position=mapper_initial_position,
        target_height=target_height,
    )
    interpolated_path = interpolate_polyline(path_points, spacing=0.25)
    interpolated_path = truncate_at_stop_radius(
        interpolated_path,
        object_positions=positions,
        success_distance=success_distance,
        stop_criterion=stop_criterion,
    )
    # STRIVE historically evaluates candidates from target-near to current-near.
    interpolated_path = interpolated_path[::-1]

    final_pos = np.asarray(object_node.position[:2], dtype=float)
    proposals: list[dict[str, Any]] = []
    for point in interpolated_path:
        point = np.asarray(point, dtype=float)
        rotation_matrix = rotation_towards(point[:2], final_pos)
        if rotation_matrix is None:
            continue
        camera_points = project_to_camera(object_node.pcd, camera_intrinsic, point, rotation_matrix)
        camera_points = np.asarray(camera_points)
        if camera_points.ndim < 2 or camera_points.shape[1] <= 0:
            continue
        all_pc_number = int(camera_points.shape[1])
        if all_pc_number <= 0:
            continue
        camera_points = camera_points.T
        camera_points = np.asarray(camera_points[:, :2], dtype=np.int32)
        in_frame = (
            (camera_points[:, 0] >= 0)
            & (camera_points[:, 0] < 640)
            & (camera_points[:, 1] >= 0)
            & (camera_points[:, 1] < 480)
        )
        camera_points = camera_points[in_frame]
        quality = score_projection(
            camera_points=camera_points,
            all_point_count=all_pc_number,
            object_positions=positions,
            viewpoint_position=point,
            original_bbox_area=obj_bbox_area,
            original_bbox_length=obj_bbox_length,
            original_bbox_width=obj_bbox_width,
            success_distance=success_distance,
        )
        if quality is None:
            continue
        proposals.append(
            {
                "position": point.copy(),
                "pose": [float(x) for x in point.reshape(-1).tolist()],
                "score": quality.score,
                "visible_ratio": quality.visible_ratio,
                "area_ratio": quality.area_ratio,
                "center_score": quality.center_score,
                "border_score": quality.border_score,
                "distance_score": quality.distance_score,
                "distance_to_target": quality.distance_to_target,
                "predicted_quality": quality.predicted_quality,
                "reason": "generic geometry proposal from visibility, centering, scale, border margin, and path distance",
            }
        )
    return proposals, interpolated_path
