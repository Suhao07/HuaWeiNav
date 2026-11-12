"""Same-frame instance correspondence and raw-depth stop measurements."""
from __future__ import annotations

import cv2
import numpy as np


def measure_instance_surface(projected_uv_depth, depth_m, masks, intrinsic,
                             camera_rotation, camera_position, base_position,
                             *, depth_tolerance=0.2, min_support_pixels=3):
    r"""Bind projected candidate support to one current connected instance region.

    S = {u in M: D(u)>0 and |z_map(u)-D(u)|<=tau}. A current mask must own a strict majority of distinct surface-aligned pixels.
    Duplicate masks describe the same support; disconnected mask fragments do not
    establish competing identities. Ambiguous or missing correspondence
    returns a domain status; malformed arrays raise their original errors.
    Distance is min ||q_xy-base_xy|| over raw-depth backprojections of S.

    Args:
        projected_uv_depth: In-image candidate projections, shape (N, 3).
        depth_m: Unmodified optical-axis depth, shape (H, W).
        masks: Current-frame instance masks, shape (M, H, W).
        intrinsic: Existing STRIVE camera matrix.
        camera_rotation: Camera-to-mapper rotation/reflection matrix.
        camera_position: Camera origin in mapper coordinates.
        base_position: Robot base origin in the same coordinates.
        depth_tolerance: Surface residual tolerance in metres, separate from stop radius.
        min_support_pixels: Minimum distinct pixels; prevents one-point certification.

    Returns:
        Geometry facts, selected region mask, and indices into projected_uv_depth.
    """
    uv = np.asarray(projected_uv_depth[:, :2], dtype=int)
    measured = depth_m[uv[:, 1], uv[:, 0]]
    residual = projected_uv_depth[:, 2] - measured
    aligned = np.isfinite(measured) & (measured > 0) & (np.abs(residual) <= depth_tolerance)
    # 归属按整张实例 mask 的支撑覆盖计算。遮挡可把一个实例分成多个二维连通域。
    # c(M)=|S ∩ M|/|S|；要求严格多数，避免少量偶然投影支撑认证整个历史实例。
    surface_pixels = np.unique(uv[aligned], axis=0)
    matches = []
    for mask_id, mask in enumerate(masks):
        indices = np.flatnonzero(aligned & mask[uv[:, 1], uv[:, 0]])
        count = len(np.unique(uv[indices], axis=0))
        if count >= min_support_pixels:
            matches.append((count, mask_id, mask, indices))
    matches.sort(key=lambda item: item[0], reverse=True)
    majority = [item for item in matches if item[0] > len(surface_pixels) / 2]
    matched = bool(majority) and all(np.array_equal(item[2], majority[0][2])
                                      for item in majority[1:])
    facts = {
        'surface_match': False,
        'surface_status': 'ambiguous_instance' if matches else 'insufficient_surface_support',
        'distance_to_object': None,
        'distance_source': 'current_instance_raw_depth',
        'depth_tolerance_m': depth_tolerance,
        'min_support_pixels': min_support_pixels,
        'matched_region_count': len(matches),
        'surface_aligned_points': int(aligned.sum()),
    }
    if not matched:
        return facts, np.zeros_like(depth_m, dtype=bool), np.empty(0, dtype=int)
    count, mask_id, mask, indices = matches[0]
    pixels = np.unique(uv[indices], axis=0)
    alternatives = [(other_id, other_mask[pixels[:, 1], pixels[:, 0]])
                    for _, other_id, other_mask, _ in matches if other_id != mask_id]
    # 独立检测的互斥支撑若合计占多数，组合 mask 的归属仍有歧义；不推断物体是否物理相连。
    for index, (left_id, left) in enumerate(alternatives):
        if left.sum() < min_support_pixels:
            continue
        for right_id, right in alternatives[index + 1:]:
            if (right.sum() >= min_support_pixels and not np.any(left & right)
                    and np.count_nonzero(left | right) > count / 2):
                facts['competing_mask_support'] = {
                    'combined_mask_id': int(mask_id), 'combined_support_count': count,
                    'alternative_mask_ids': [int(left_id), int(right_id)],
                    'alternative_support_counts': [int(left.sum()), int(right.sum())],
                }
                return facts, np.zeros_like(depth_m, dtype=bool), np.empty(0, dtype=int)
    # 距离取主要连续区域的实测支撑，排除分割碎片中的孤立近点。
    _, components = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8))
    labels = components[uv[indices, 1], uv[indices, 0]]
    region_id = max(np.unique(labels), key=lambda label: len(np.unique(uv[indices[labels == label]], axis=0)))
    region = components == region_id
    indices = indices[labels == region_id]
    if len(np.unique(uv[indices], axis=0)) < min_support_pixels:
        facts['surface_status'] = 'insufficient_surface_support'
        return facts, np.zeros_like(depth_m, dtype=bool), np.empty(0, dtype=int)
    facts['instance_surface_coverage'] = count / len(surface_pixels)
    pixels = np.unique(uv[indices], axis=0)
    z = depth_m[pixels[:, 1], pixels[:, 0]]
    # 与现有 project_to_camera 严格互逆：相机前向为 -Z，v=cy-1-fy*y/z。
    camera = np.column_stack(((pixels[:, 0]-intrinsic[0, 2])*z/intrinsic[0, 0],
                              (intrinsic[1, 2]-1-pixels[:, 1])*z/intrinsic[1, 1], -z))
    world = camera @ camera_rotation.T + camera_position
    y, x = np.nonzero(region)
    facts.update(surface_match=True, surface_status='matched', mask_id=int(mask_id),
                 support_count=len(pixels), bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                 visible_support_span_map_xyz_m=np.ptp(world, axis=0).tolist(),
                 distance_to_object=float(np.linalg.norm(world[:, :2]-base_position[:2], axis=1).min()))
    return facts, region, indices
