from __future__ import annotations

from typing import Any

import numpy as np

from navigation_core.view_geometry import (
    ProjectionQuality,
    habitat_path_to_strive_points,
    interpolate_polyline,
    optical_rotation_towards,
    rank_view_proposals,
    score_points_view,
    truncate_at_stop_radius,
)


# Habitat's mapper (x, z, y) is reflected. The shared map is RH (x, -z, y).
_MAPPER_TO_MAP = np.diag([1., -1., 1.])
_OPTICAL_TO_HABITAT_CAMERA = np.diag([1., -1., -1.])


def _habitat_intrinsic(camera_intrinsic: Any) -> np.ndarray:
    """Preserve project_to_camera's legacy v = -fy*y/depth + cy - 1."""
    intrinsic = np.array(camera_intrinsic, dtype=float, copy=True)
    intrinsic[1, 2] -= 1.0
    return intrinsic


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
        rotation: Legacy Habitat camera-to-mapper basis ``R`` (+X right,
            +Y up, -Z forward); mapper axes are the reflected (x, z, y).
        success_distance: Required stop distance :math:`d_{stop}`.
        arrival_tolerance: Planner arrival radius :math:`\rho`.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
    Returns:
        ``(quality, stop_feasible)``; None when no object point falls in the image.
    """
    positions = np.asarray(object_node.pcd.point.positions.cpu().numpy())
    camera_to_map = np.eye(4)
    camera_to_map[:3, :3] = _MAPPER_TO_MAP @ rotation @ _OPTICAL_TO_HABITAT_CAMERA
    camera_to_map[:3, 3] = _MAPPER_TO_MAP @ position
    return score_points_view(
        object_points=positions @ _MAPPER_TO_MAP,
        original_bbox=object_node.bbox,
        camera_intrinsic=_habitat_intrinsic(camera_intrinsic),
        camera_to_map=camera_to_map,
        success_distance=success_distance,
        arrival_tolerance=arrival_tolerance,
        image_width=image_width,
        image_height=image_height,
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

    positions = np.asarray(object_node.pcd.point.positions.cpu().numpy())
    path_points = habitat_path_to_strive_points(
        habitat_path_points,
        mapper_initial_position=mapper_initial_position,
        target_height=target_height,
    )
    proposals, path = build_view_proposals(
        object_points=positions @ _MAPPER_TO_MAP,
        object_position=_MAPPER_TO_MAP @ np.asarray(object_node.position),
        original_bbox=object_node.bbox,
        camera_intrinsic=_habitat_intrinsic(camera_intrinsic),
        path_points=path_points @ _MAPPER_TO_MAP,
        target_height=target_height,
        success_distance=success_distance,
        stop_criterion=stop_criterion,
        arrival_tolerance=arrival_tolerance,
        image_width=image_width,
        image_height=image_height,
        pitches=pitches,
        align_pitch=align_pitch,
    )
    # Keep all legacy outputs in mapper coordinates; scores are frame-invariant.
    for proposal in proposals:
        proposal["position"] = _MAPPER_TO_MAP @ proposal["position"]
        proposal["pose"] = proposal["position"].tolist()
    return proposals, [_MAPPER_TO_MAP @ point for point in path]


def build_view_proposals(
    *,
    object_points: np.ndarray,
    object_position: np.ndarray,
    original_bbox: Any,
    camera_intrinsic: np.ndarray,
    path_points: np.ndarray,
    target_height: float,
    success_distance: float,
    stop_criterion: float,
    arrival_tolerance: float = 0.0,
    image_width: int = 640,
    image_height: int = 480,
    pitches: tuple[float, ...] = (0.0,),
    align_pitch: bool = False,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    r"""Build baseline geometry proposals in a right-handed Z-up map using NumPy.

    Interpolate at 0.25 map units, truncate at the last sample within
    success_distance * stop_criterion, then reverse (target-near first).
    At each position choose pitch by (stop_feasible, score), with stop feasibility
    defined by d_visible_xy + arrival_tolerance <= success_distance. Scoring is
    shared with score_object_view; depth/occlusion and final verification remain
    the caller's responsibility. No model or backend is invoked.

    Args:
        object_points: Nx3 instance points in the common right-handed Z-up map.
        object_position: Object center in that same map; used for yaw/pitch.
        original_bbox: Observed (xmin, ymin, xmax, ymax) pixel bounds.
        camera_intrinsic: Standard optical right/down/forward 3x3 intrinsics.
        path_points: Nx3 path in the same map, with no origin conversion.
        target_height: Camera Z in the map, replacing path/base height.
        success_distance: Required XY stop distance in map units.
        stop_criterion: Multiplier used for baseline path truncation.
        arrival_tolerance: Planner arrival radius reserved in the stop budget.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        pitches: Executable camera elevations in radians; positive looks up.
        align_pitch: Use continuous target-center pitch instead of pitches.
            Only enable when the executor can actually align to that pitch.
    Returns:
        (proposals, sampled_path), both in the input map and in baseline sample
        order, not ranked. Proposal fields match build_check_again_viewpoints.
        travel_distance is XY displacement from path origin, NOT geodesic cost;
        backends replace it before rank_view_proposals and, for active rechecks,
        filter travel_distance > arrival_tolerance. Inputs are never mutated.
    Raises:
        ValueError: Nonempty path_points do not have shape (N, 3).
    """
    positions = np.asarray(object_points, dtype=float)
    object_position = np.asarray(object_position, dtype=float)
    path_points = np.array(path_points, dtype=float, copy=True)
    if path_points.size == 0:
        path_points = np.empty((0, 3), dtype=float)
    if path_points.ndim != 2 or path_points.shape[1] != 3:
        raise ValueError("path_points must have shape (N, 3)")
    path_points[:, 2] = float(target_height)
    interpolated_path = interpolate_polyline(path_points, spacing=0.25)
    interpolated_path = truncate_at_stop_radius(
        interpolated_path,
        object_positions=positions,
        success_distance=success_distance,
        stop_criterion=stop_criterion,
    )
    # VLN historically evaluates candidates from target-near to current-near.
    interpolated_path = interpolated_path[::-1]

    final_pos = object_position[:2]
    proposals: list[dict[str, Any]] = []
    path_origin = path_points[0] if len(path_points) else np.zeros(3)
    for point in interpolated_path:
        point = np.asarray(point, dtype=float)
        # 先验执行器会俯仰对准当前观察中心，评分使用相同目标与相机高度。
        # theta = atan2(z_target - z_camera, ||target_xy - camera_xy||).
        candidate_pitches = ((float(np.arctan2(object_position[2] - point[2],
                                               np.linalg.norm(final_pos - point[:2]))),)
                             if align_pitch else pitches)
        scored = []
        for pitch in candidate_pitches:
            rotation_matrix = optical_rotation_towards(point[:2], final_pos, pitch=pitch)
            if rotation_matrix is None:
                break
            camera_to_map = np.eye(4)
            camera_to_map[:3, :3] = rotation_matrix
            camera_to_map[:3, 3] = point
            view = score_points_view(
                object_points=positions,
                original_bbox=original_bbox,
                camera_intrinsic=camera_intrinsic,
                camera_to_map=camera_to_map,
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
