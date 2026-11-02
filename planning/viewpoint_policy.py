from __future__ import annotations

from typing import Any

import numpy as np

from mapping_utils.geometry import project_to_camera
from navigation_core.view_geometry import (
    ProjectionQuality,
    habitat_path_to_strive_points,
    interpolate_polyline,
    rotation_towards,
    score_projection,
    truncate_at_stop_radius,
)


def score_object_view(
    *,
    object_node: Any,
    camera_intrinsic: Any,
    position: np.ndarray,
    rotation: np.ndarray,
    success_distance: float,
    arrival_tolerance: float = 0.0,
    image_width: int = 640,
    image_height: int = 480,
) -> tuple[ProjectionQuality, bool] | None:
    r"""Score one camera pose against the object's projected visible support.

    停止距离与最终验证使用同一可见支撑集合（画面内、光轴前方的实例点）：

    .. math::
        V(v,R) = \{p \in P : z_{cam}(p) > 0,\ \pi(p) \in \Omega_{H \times W}\},\qquad
        d(v,R) = \min_{p \in V} \lVert p_{xy} - v_{xy} \rVert_2

    可停条件为 :math:`d(v,R) + \rho \le d_{stop}`。规划时没有该视角的深度，
    遮挡检验由到达后的验证完成。

    Args:
        object_node: Instance with ``pcd`` and the observed ``bbox``.
        camera_intrinsic: Pinhole intrinsics.
        position: Camera position ``v`` in mapper coordinates.
        rotation: Camera-to-mapper rotation ``R``.
        success_distance: Required stop distance :math:`d_{stop}`.
        arrival_tolerance: Planner arrival radius :math:`\rho`.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
    Returns:
        ``(quality, stop_feasible)``; None when no object point falls in the image.
    """
    positions = object_node.pcd.point.positions.cpu().numpy()
    bbox = object_node.bbox
    camera_points = np.asarray(project_to_camera(object_node.pcd, camera_intrinsic, position, rotation))
    if camera_points.ndim < 2 or camera_points.shape[1] == 0:
        return None
    pixels = np.asarray(camera_points[:2].T, dtype=np.int32)
    in_frame = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image_width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image_height)
    )
    # project_to_camera 只返回光轴前方的点：-R[:,2]^T (p - v) > 0，用同一判据恢复点序号。
    front = (positions - position) @ rotation[:, 2] < 0
    quality = score_projection(
        camera_points=pixels[in_frame],
        all_point_count=int(camera_points.shape[1]),
        object_positions=positions[front][in_frame],
        viewpoint_position=position,
        original_bbox_area=float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
        original_bbox_length=float(bbox[2] - bbox[0]),
        original_bbox_width=float(bbox[3] - bbox[1]),
        success_distance=success_distance,
        image_width=image_width,
        image_height=image_height,
    )
    if quality is None:
        return None
    return quality, quality.distance_to_target + arrival_tolerance <= success_distance


def build_check_again_viewpoints(
    *,
    object_node: Any,
    camera_intrinsic: Any,
    mapper_initial_position: np.ndarray,
    habitat_path_points: Any,
    target_height: float,
    success_distance: float,
    stop_criterion: float,
    arrival_tolerance: float = 0.0,
    image_width: int = 640,
    image_height: int = 480,
    align_pitch: bool = False,
    pitches: tuple[float, ...] = (0.0,),
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    r"""Generate class-agnostic better-view proposals for one object instance.

    The caller owns navigation state and final verifier decisions. This helper
    only projects the current object point cloud from reachable path samples and
    scores how useful each sample is as visual evidence. ``align_pitch`` uses
    a target-facing camera for executors that pitch toward the object center;
    otherwise each sample takes the best executable pitch in ``pitches``:

    .. math::
        \phi^*(v) = \arg\max_{\phi} \big(\mathbb{1}[d(v,R_\phi) + \rho \le d_{stop}],\ s(v,R_\phi)\big)

    The projection measures image bounds, without new-view occlusion testing.
    """

    positions = object_node.pcd.point.positions.cpu().numpy()
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
    # VLN historically evaluates candidates from target-near to current-near.
    interpolated_path = interpolated_path[::-1]

    final_pos = np.asarray(object_node.position[:2], dtype=float)
    proposals: list[dict[str, Any]] = []
    path_origin = path_points[0] if len(path_points) else np.zeros(3)
    for point in interpolated_path:
        point = np.asarray(point, dtype=float)
        # 先验执行器会俯仰对准当前观察中心，评分使用相同目标与相机高度。
        # theta = atan2(z_target - z_camera, ||target_xy - camera_xy||).
        candidate_pitches = ((float(np.arctan2(object_node.position[2] - point[2],
                                               np.linalg.norm(final_pos - point[:2]))),)
                             if align_pitch else pitches)
        scored = []
        for pitch in candidate_pitches:
            rotation_matrix = rotation_towards(point[:2], final_pos, pitch=pitch)
            if rotation_matrix is None:
                break
            view = score_object_view(
                object_node=object_node,
                camera_intrinsic=camera_intrinsic,
                position=point,
                rotation=rotation_matrix,
                success_distance=success_distance,
                arrival_tolerance=arrival_tolerance,
                image_width=image_width,
                image_height=image_height,
            )
            if view is not None:
                scored.append((view[1], view[0].score, float(pitch), view[0]))
        if not scored:
            continue
        stop_feasible, _, pitch, quality = max(scored, key=lambda row: row[:2])
        proposals.append(
            {
                "position": point.copy(),
                "pose": [float(x) for x in point.reshape(-1).tolist()],
                "pitch": pitch,
                "score": quality.score,
                "travel_distance": float(np.linalg.norm(point[:2] - path_origin[:2])),
                "stop_feasible": stop_feasible,
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
