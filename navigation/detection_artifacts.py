"""Debug artifact writers for panoramic object detection.

This module centralizes detection image/point-cloud output paths. It is an IO
adapter only: it must not change mapper state, refine labels, or decide target
success.
"""

from __future__ import annotations

from typing import Any

import cv2
import open3d as o3d

from cv_utils.visualizer import visualize_mask
from artifact_utils.path_builder import detection_step_dir as build_detection_step_dir


def detection_step_dir(agent: Any) -> str:
    """Return and create the directory for one episode-step detection dump."""

    return build_detection_step_dir(agent.save_dir, agent.episode_samples - 1, agent.episode_steps)


def save_combined_view(step_dir: str, index: int, image: Any, depth_vis: Any) -> None:
    """Persist stitched RGB and depth visualization for one panorama triplet."""

    cv2.imwrite(f"{step_dir}/comb_img_{index}.jpg", image)
    cv2.imwrite(f"{step_dir}/comb_depth_{index}.jpg", depth_vis)


def save_detection_overlay(step_dir: str, prefix: str, index: int, image: Any, detection: Any) -> None:
    """Persist a mask/bbox overlay for one filtered detector slice."""

    if not detection.has_boxes:
        return
    visualization = visualize_mask(
        image,
        detection.boxes,
        detection.confidences,
        detection.classes,
        detection.masks,
    )
    cv2.imwrite(f"{step_dir}/{prefix}_dino_result_{index}.jpg", visualization)


def _to_legacy_pointcloud(obj: Any) -> o3d.geometry.PointCloud:
    """Convert an Open3D tensor point cloud on an object entity to legacy format."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj.pcd_all.point.positions.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(obj.pcd_all.point.colors.cpu().numpy())
    return pcd


def save_candidate_object_pointclouds(step_dir: str, candidate_groups: list[list[Any]]) -> None:
    """Persist raw C-object point clouds before cross-view merging."""

    obj_pcd = o3d.geometry.PointCloud()
    for group_idx, group in enumerate(candidate_groups):
        for obj in group:
            new_pcd = _to_legacy_pointcloud(obj)
            o3d.io.write_point_cloud(f"{step_dir}/C_dino_pcd_{group_idx}.ply", new_pcd)
            obj_pcd = obj_pcd + new_pcd
    if len(obj_pcd.points) > 0:
        o3d.io.write_point_cloud(f"{step_dir}/C_dino_pcd.ply", obj_pcd)


def save_real_object_pointcloud(step_dir: str, obj: Any, index: int) -> None:
    """Persist one merged C-object point cloud after clustering."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj.pcd.point.positions.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(obj.pcd.point.colors.cpu().numpy())
    o3d.io.write_point_cloud(f"{step_dir}/real_C_objs_{index}.ply", pcd)


def save_object_view(step_dir: str, index: int, image: Any, bbox_xyxy: Any | None = None) -> None:
    """Persist the RGB evidence image and optional projected object bbox."""

    cv2.imwrite(f"{step_dir}/real_C_obj_image_{index}.jpg", image)
    if bbox_xyxy is None:
        return
    img = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"{step_dir}/real_C_obj_image_bbox_{index}.jpg", img)
