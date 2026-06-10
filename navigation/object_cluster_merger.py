"""Object point-cloud merge helpers for panoramic detections.

The detector may produce the same physical object from multiple panoramic
triplets. This module merges those candidates by geometric overlap only. It
does not call any semantic verifier and does not mark a target as found.
"""

from __future__ import annotations

from typing import Any

import torch

from mapping_utils.geometry import gpu_merge_pointcloud, pointcloud_distance


def _overlap_score(eval_pcd_all: Any, prev_pcd_all: Any) -> float:
    """Estimate symmetric point-cloud overlap between two object candidates."""

    cdist1 = pointcloud_distance(eval_pcd_all, prev_pcd_all)
    cdist2 = pointcloud_distance(prev_pcd_all, eval_pcd_all)
    cdist_all = torch.cat([cdist1, cdist2], dim=0)
    overlap_all = (cdist_all < 0.1)
    overlap_1 = (cdist1 < 0.1)
    overlap_2 = (cdist2 < 0.1)
    score_all = (overlap_all.sum() / (overlap_all.shape[0] + 1e-6)).cpu().numpy().item()
    score_1 = (overlap_1.sum() / (overlap_1.shape[0] + 1e-6)).cpu().numpy().item()
    score_2 = (overlap_2.sum() / (overlap_2.shape[0] + 1e-6)).cpu().numpy().item()
    if (score_1 > 0.85 and score_2 < 0.85) or (score_1 < 0.85 and score_2 > 0.85):
        return -1.0
    return float(score_all)


def merge_candidate_objects(candidate_groups: list[list[Any]], *, overlap_threshold: float = 0.25) -> list[Any]:
    """Merge cross-view C-object candidates into stable object instances.

    The asymmetric-overlap guard keeps a small object near a large object from
    being swallowed by the large object's point cloud.
    """

    real_objects: list[Any] = []
    for group in candidate_groups:
        for obj in group:
            overlap_scores = [_overlap_score(obj.pcd_all, prev_obj.pcd_all) for prev_obj in real_objects]
            overlap_flags = [score > overlap_threshold for score in overlap_scores]

            eval_pcd = obj.pcd
            eval_pcd_all = obj.pcd_all
            for idx, should_merge in enumerate(overlap_flags):
                if not should_merge:
                    continue
                # 多视角重复检测合并的是同一个物理实例；置信度取最大值，
                # 点云取并集，避免后续 association 生成多个重复 ObjectNode。
                eval_pcd = gpu_merge_pointcloud(eval_pcd, real_objects[idx].pcd)
                eval_pcd_all = gpu_merge_pointcloud(eval_pcd_all, real_objects[idx].pcd_all)
                obj.confidence = max(obj.confidence, real_objects[idx].confidence)

            obj.pcd = eval_pcd
            obj.pcd_all = eval_pcd_all
            real_objects = [prev for idx, prev in enumerate(real_objects) if idx >= len(overlap_flags) or not overlap_flags[idx]]
            real_objects.append(obj)

    return real_objects


def sort_target_objects_last(objects: list[Any], target: str) -> list[Any]:
    """Move target-tagged objects to the end to preserve legacy association order."""

    non_targets = [obj for obj in objects if obj.tag != target]
    targets = [obj for obj in objects if obj.tag == target]
    return non_targets + targets
