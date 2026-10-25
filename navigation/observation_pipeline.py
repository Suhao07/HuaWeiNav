"""Observation utilities for VLN agent.

The panoramic segmentation loop is still in the agent because it touches many
mutable runtime buffers. This module starts the split by extracting reusable
point-cloud merge and observation-artifact helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d
import quaternion
from loguru import logger

from artifact_utils.path_builder import episode_subdir
from mapping_utils.geometry import gpu_merge_pointcloud, gpu_pointcloud_from_array


def reset_panoramic_buffers(agent: Any) -> None:
    """Reset mutable buffers used by one panoramic observation sweep."""

    agent.temporary_pcd = []
    agent.angles = []
    agent.mapper.current_obj_indices = []
    agent.B_classes, agent.B_boxes, agent.B_masks, agent.B_confidences, agent.B_visualization, \
        agent.C_boxes, agent.C_masks, agent.C_confidences, agent.C_visualization = \
            [], [], [], [], [], [], [], [], []


def collect_panoramic_observations(agent: Any, rotate_times: int = 12) -> tuple[list[Any], list[Any], list[Any]]:
    """Collect panoramic RGB/depth/pose buffers by rotating the Habitat agent.

    The function mutates only observation buffers and simulator pose. It does
    not run segmentation, update mapper graph nodes, or make planning choices.
    """

    reset_panoramic_buffers(agent)
    temporary_images = []
    temporary_positions, temporary_rotations = [], []
    q_identity = quaternion.quaternion(1, 0, 0, 0)

    for _ in range(rotate_times):
        if agent.env.episode_over:
            logger.info(f'Step: {agent.env._elapsed_steps}')
            logger.info(f'Time: {agent.env._elapsed_seconds}')
            return temporary_images, temporary_positions, temporary_rotations

        if agent.mapper.current_navigable_pcd is None and agent.mapper.current_pcd is None:
            temporary_pcd = gpu_pointcloud_from_array(
                np.zeros((0, 3)),
                np.zeros((0, 3)),
                agent.mapper.pcd_device,
            )
        else:
            temporary_pcd = gpu_merge_pointcloud(
                agent.mapper.current_navigable_pcd,
                agent.mapper.current_pcd,
            ).voxel_down_sample(agent.mapper.pcd_resolution)

        # 先保存当前朝向下的局部点云和图像，再旋转到下一视角。
        # 这样 12 帧 RGB/depth/pose 与 mapper-local 角度保持一一对应。
        agent.temporary_pcd.append(temporary_pcd)
        temporary_images.append(agent.rgb_trajectory[-1])
        temporary_positions.append(agent.mapper.current_position)
        temporary_rotations.append(agent.mapper.current_rotation)
        agent.angles.append(2 * np.arccos((q_identity.inverse() * agent.rotation).w))

        agent.obs = agent.execution.step(3)
        agent.update_trajectory()

    return temporary_images, temporary_positions, temporary_rotations


def merge_temporary_pointclouds(agent: Any) -> Any:
    """Merge temporary panoramic point clouds into one current observation cloud."""

    merged_pcd = o3d.t.geometry.PointCloud(agent.mapper.pcd_device)
    for pcd in agent.temporary_pcd:
        merged_pcd = gpu_merge_pointcloud(merged_pcd, pcd)
    return merged_pcd


def save_observation_pointcloud(agent: Any, pcd: Any, *, episode_idx: int, step: int, path_idx: int | None = None) -> None:
    """Persist the current observation point cloud for debugging.

    保存 debug PLY 只依赖当前观测点云，不应该掺入导航策略或 verifier 状态。
    """

    save_pcd = o3d.geometry.PointCloud()
    save_pcd.points = o3d.utility.Vector3dVector(pcd.point.positions.cpu().numpy())
    save_pcd.colors = o3d.utility.Vector3dVector(pcd.point.colors.cpu().numpy())
    obs_dir = episode_subdir(agent.save_dir, episode_idx, "obs")
    if path_idx is None:
        file_path = f"{obs_dir}/obs_{step}.ply"
    else:
        file_path = f"{obs_dir}/obs_{step}_{path_idx}.ply"
    o3d.io.write_point_cloud(file_path, save_pcd)


def collect_adaptive_observations(agent):
    """Observe missing headings in four-view batches, stopping at a target candidate.

    几何更新使用每帧真实 RGB/depth/pose；语义覆盖只在检测完成后提交。
    同一位姿的候选被否决后继续剩余方向；新位置重新观察，避免视差误复用。
    12 个方向只需 11 次转向，最后一帧后直接进入导航，无需恢复起始朝向。
    """
    reset_panoramic_buffers(agent)
    pose_key = (agent.task.spec.revision, *np.round(agent.mapper.current_position, 3))
    covered = agent.semantic_scan_headings.setdefault(pose_key, set())
    if len(covered) == 12:
        agent.scan_complete = True
        return
    start_step = agent.episode_steps
    images, depths, positions, rotations, headings = [], [], [], [], []

    def process_batch():
        agent.rotate_segmentation(images, depths, positions, rotations, partial=True)
        covered.update(headings)
        images.clear(); depths.clear(); positions.clear(); rotations.clear(); headings.clear()
        return any(agent.mapper._is_target_tag(agent.mapper.objects[i].tag)
                   for i in agent.mapper.current_obj_indices)

    for _ in range(12):
        rotation = np.asarray(agent.mapper.current_rotation)
        heading = int(np.round(np.degrees(np.arctan2(-rotation[1, 2], -rotation[0, 2])) / 30)) % 12
        if heading not in covered:
            if agent.mapper.current_navigable_pcd is None and agent.mapper.current_pcd is None:
                pcd = gpu_pointcloud_from_array(np.empty((0, 3)), np.empty((0, 3)), agent.mapper.pcd_device)
            else:
                pcd = gpu_merge_pointcloud(agent.mapper.current_navigable_pcd, agent.mapper.current_pcd).voxel_down_sample(agent.mapper.pcd_resolution)
            agent.temporary_pcd.append(pcd)
            agent.angles.append(2 * np.arccos(agent.rotation.w))
            images.append(agent.rgb_trajectory[-1]); depths.append(agent.temporary_depths[-1])
            positions.append(agent.mapper.current_position.copy()); rotations.append(rotation.copy())
            headings.append(heading)
        if len(images) == 4 or len(covered) + len(headings) == 12:
            candidate_seen = process_batch()
            if candidate_seen or len(covered) == 12:
                break
        agent.obs = agent.execution.step(3)
        agent.update_trajectory()
        if agent.env.episode_over:
            break
    if images and not agent.env.episode_over:
        process_batch()
    agent.scan_complete = len(covered) == 12

    import json
    from pathlib import Path
    path = Path(agent.save_dir) / f"episode-{agent.episode_samples - 1}" / "observation_cycles.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"start_step": start_step, "end_step": agent.episode_steps,
                                 "views": len(agent.temporary_pcd), "covered_headings": sorted(covered),
                                 "complete": agent.scan_complete}) + "\n")
