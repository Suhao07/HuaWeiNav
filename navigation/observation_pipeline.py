"""Shared panoramic detection pipeline and observation collection/artifact helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d
import quaternion
import torch
from loguru import logger

from artifact_utils.path_builder import episode_subdir
from mapping_utils.geometry import gpu_cluster_filter, gpu_merge_pointcloud, gpu_pointcloud_from_array
from navigation.detection_artifacts import (
    detection_step_dir,
    save_candidate_object_pointclouds,
    save_combined_view,
    save_detection_overlay,
    save_object_view,
    save_real_object_pointcloud,
)
from navigation.object_cluster_merger import merge_candidate_objects, sort_target_objects_last
from navigation.object_view_projector import project_object_view
from navigation.panoramic_detection import build_triplet, filter_center_panel, pose_triplet


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


def process_panoramic_detections(agent, images, depths, positions, rotations, *, partial=False, project_view=None):
    """Update panoramic object memory with optional view projection and return per-view detection counts."""

    _, w, _ = images[0].shape
    C_objs = []
    step_dir = detection_step_dir(agent)
    view_stats = []

    for i in range(len(images)):
        triplet = build_triplet(images, depths, i, agent.mapper.camera_intrinsic, circular=not partial)
        B_classes, B_boxes, B_masks, B_confidences, \
            C_classes, C_boxes, C_masks, C_confidences = \
                agent.mapper.object_perceiver.perceive(
                    triplet.image,
                    target=agent.mapper.target,
                    target_list=agent.task.detector_terms if agent.task is not None else None,
                    save_dir=agent.save_dir,
                    episode_idx=agent.episode_samples - 1,
                    episode_step=agent.episode_steps,
                )
        save_combined_view(step_dir, i, triplet.image, triplet.depth_vis)

        current_pos, current_rot, depths_list = pose_triplet(positions, rotations, depths, triplet)
        B_detection = filter_center_panel(
            B_classes,
            B_boxes,
            B_masks,
            B_confidences,
            image_width=w,
        )
        if B_detection.has_boxes:
            save_detection_overlay(step_dir, "B", i, triplet.image, B_detection)
            B_objs = agent.mapper.get_object_entities_pano(
                triplet.depth,
                triplet.image,
                current_pos,
                current_rot,
                B_detection.classes,
                B_detection.boxes,
                B_detection.masks,
                B_detection.confidences,
                depths_list,
            )
            agent.mapper.objects, obj_indices = agent.mapper.associate_object_entities(
                agent.mapper.objects,
                B_objs,
            )
            agent.mapper.current_obj_indices += obj_indices
            agent.mapper.object_pcd = agent.mapper.update_object_pcd()

        C_detection = filter_center_panel(
            C_classes,
            C_boxes,
            C_masks,
            C_confidences,
            image_width=w,
        )
        view_stats.append({
            "view_index": i,
            "background_detections": len(B_detection.classes),
            "target_detections": len(C_detection.classes),
        })
        if not C_detection.has_boxes:
            continue

        save_detection_overlay(step_dir, "C", i, triplet.image, C_detection)
        C_objs.append(agent.mapper.get_object_entities_pano(
            triplet.depth,
            triplet.image,
            current_pos,
            current_rot,
            C_detection.classes,
            C_detection.boxes,
            C_detection.masks,
            C_detection.confidences,
            depths_list,
        ))

    save_candidate_object_pointclouds(step_dir, C_objs)

    # C 类对象来自多视角候选，先按几何 overlap 合成物理实例，
    # 再进入 bbox 视觉复核；这样 VLM 不会对重复实例反复调用。
    real_C_objs = merge_candidate_objects(C_objs)
    for i, obj in enumerate(real_C_objs):
        obj.pcd = gpu_cluster_filter(obj.pcd)
        save_real_object_pointcloud(step_dir, obj, i)

    from navigation.bbox_refinement import refine_candidates
    candidates = []
    for i, obj in enumerate(real_C_objs):
        if project_view is None:
            evidence = project_object_view(agent, obj, images, rotations, torch, partial=partial)
        else:
            evidence = project_view(obj)
        if evidence is None:
            continue
        save_object_view(step_dir, i, evidence.image, evidence.bbox_xyxy)
        candidates.append((i, obj, evidence))
    label_records = refine_candidates(agent, candidates)

    real_C_objs = sort_target_objects_last(real_C_objs, agent.mapper.target)

    agent.mapper.objects, obj_indices = agent.mapper.associate_object_entities(
        agent.mapper.objects, real_C_objs)
    agent.mapper.current_obj_indices += obj_indices
    if label_records:
        from navigation.semantic_label_cache import matching_instance, instance_signature
        for obj in real_C_objs:
            if obj.uid in label_records:
                instance = matching_instance(obj.pcd.point.positions.cpu().numpy(), agent.mapper.objects)
                if instance is not None:
                    record = label_records[obj.uid]
                    record.instance_dependency = instance_signature(instance)
                    agent.semantic_label_evidence[instance.uid] = record
    agent.object_pcd = agent.mapper.update_object_pcd()

    agent.mapper.current_obj_indices = list(set(agent.mapper.current_obj_indices))
    return view_stats


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
