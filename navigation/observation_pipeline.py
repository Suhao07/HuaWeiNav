"""Observation utilities for STRIVE agent.

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

        agent.obs = agent.env.step(3)
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
