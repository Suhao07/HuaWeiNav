"""Panoramic detection helpers used by observation pipeline.

The helpers in this module prepare detector inputs and normalize detector
outputs from stitched panoramic views. They do not create mapper objects, write
debug files, call VLMs, or decide whether navigation should stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectionSlice:
    """Detector results restricted to the center panel of a stitched panorama."""

    classes: Any
    boxes: Any
    masks: Any
    confidences: Any

    @property
    def has_boxes(self) -> bool:
        """Return whether this slice contains at least one valid detection."""

        return self.boxes is not None and self.boxes.shape[0] > 0


@dataclass(frozen=True)
class PanoramaTriplet:
    """Combined RGB/depth data and source indices for one panorama triplet."""

    index: int
    prev_index: int
    next_index: int
    image: np.ndarray
    depth: np.ndarray
    depth_vis: np.ndarray


def triplet_indices(index: int, total_views: int = 12) -> tuple[int, int, int]:
    """Return previous/current/next indices for a circular panorama."""

    prev = (index - 1) if index > 0 else total_views - 1
    nxt = (index + 1) if index < total_views - 1 else 0
    return prev, index, nxt


def build_triplet(images: list[np.ndarray], depths: list[np.ndarray], index: int, camera_intrinsic: Any, *, circular: bool = True) -> PanoramaTriplet:
    """Stitch one previous/current/next panorama triplet for detection."""

    from cv_utils.stitch import combine_image

    prev, cur, nxt = triplet_indices(index, total_views=len(images))
    # 有限扫描的两端不相邻。空白侧板保持原检测器中心面板坐标合同。
    left_image = images[prev] if circular or index > 0 else np.zeros_like(images[cur])
    right_image = images[nxt] if circular or index + 1 < len(images) else np.zeros_like(images[cur])
    left_depth = depths[prev] if circular or index > 0 else np.zeros_like(depths[cur])
    right_depth = depths[nxt] if circular or index + 1 < len(images) else np.zeros_like(depths[cur])
    comb_img = combine_image(left_image, images[cur], right_image, camera_intrinsic)
    comb_depth = combine_image(left_depth, depths[cur], right_depth, camera_intrinsic)
    depth_vis = np.clip((comb_depth / 5.0 * 255.0), 0, 255).astype(np.uint8)
    return PanoramaTriplet(
        index=index,
        prev_index=prev,
        next_index=nxt,
        image=comb_img,
        depth=comb_depth,
        depth_vis=depth_vis,
    )


def filter_center_panel(
    classes: Any,
    boxes: Any,
    masks: Any,
    confidences: Any,
    *,
    image_width: int,
) -> DetectionSlice:
    """Keep only detections whose center lies in the original middle view.

    The detector runs on a three-view stitched image. Only the middle panel
    corresponds to the current panorama direction, so side-panel detections are
    filtered out before object entities are created.
    """

    if boxes is None or boxes.shape[0] == 0:
        return DetectionSlice(classes, boxes, masks, confidences)

    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    # 中间面板过滤是几何约束，不是语义规则；它保证同一检测不会被相邻
    # stitched view 重复写入 mapper。
    flag = (centers[:, 0] >= image_width) & (centers[:, 0] < 2 * image_width)
    flag_np = flag.cpu().numpy() if hasattr(flag, "cpu") else np.asarray(flag)
    return DetectionSlice(
        classes=classes[flag_np],
        boxes=boxes[flag],
        masks=masks[flag],
        confidences=confidences[flag],
    )


def pose_triplet(positions: list[Any], rotations: list[Any], depths: list[Any], triplet: PanoramaTriplet) -> tuple[list[Any], list[Any], list[Any]]:
    """Return position, rotation, and depth lists aligned with a panorama triplet."""

    indices = [triplet.prev_index, triplet.index, triplet.next_index]
    return (
        [positions[i] for i in indices],
        [rotations[i] for i in indices],
        [depths[i] for i in indices],
    )
