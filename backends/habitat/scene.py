"""Export observed mapper objects without exposing simulator ground truth."""
from __future__ import annotations

import numpy as np
from pathlib import Path

from instruction_adapter.verifier import candidate_from_object
from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot, FrontierSnapshot, ViewpointSnapshot


def current_observation(agent):
    from mapping_utils.geometry import project_to_camera

    depth = agent.obs["depth"].squeeze()
    height, width = depth.shape
    boxes = {}
    for obj in agent.mapper.objects:
        projected = np.asarray(project_to_camera(
            obj.pcd, agent.mapper.camera_intrinsic,
            agent.mapper.current_position, agent.mapper.current_rotation)).T
        uv = projected[:, :2].astype(int)
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        uv, z = uv[inside], projected[inside, 2]
        measured = depth[uv[:, 1], uv[:, 0]]
        aligned = np.isfinite(measured) & (measured > 0) & (np.abs(z - measured) <= 0.2)
        pixels = np.unique(uv[aligned], axis=0)
        if len(pixels) < 3:
            continue
        lo, hi = pixels.min(axis=0), pixels.max(axis=0) + 1
        boxes[obj.uid] = [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]
    episode, step = agent.episode_samples - 1, agent.episode_steps
    return {
        "id": f"{episode}:{step}", "step": step,
        "image_ref": str((Path(agent.save_dir) / f"episode-{episode}" / "rgb"
                          / f"monitor-rgb_{step}.jpg").resolve()),
        "position": agent.mapper.current_position.tolist(),
        "rotation": agent.mapper.current_rotation.tolist(),
        "observed_object_uids": list(boxes), "object_boxes": boxes,
        "source": "depth_validated_map_projection", "depth_tolerance_m": 0.2,
    }


def snapshot_from_mapper(mapper, *, step=0, nodes=()):
    """Return native scene records plus a backend-only UID-to-object lookup.

    The lookup is used only by Habitat geometry execution. It is never sent to
    the model, and semantic query functions receive only the frozen records.
    """
    records, lookup = [], {}
    for obj in mapper.objects:
        candidate = candidate_from_object(obj, step=step)
        # Old maps without stable IDs keep their existing geometry key during
        # migration. A duplicate key cannot be counted as a distinct instance.
        if candidate.uid in lookup:
            continue
        lookup[candidate.uid] = obj
        center, extent = None, None
        try:
            points = obj.pcd.point.positions.cpu().numpy()
            if len(points):
                lo, hi = points.min(axis=0)[:3], points.max(axis=0)[:3]
                # 快照是可序列化领域数据；在 NumPy 导出边界转换标量类型。
                center = tuple(((lo + hi) / 2).tolist())
                extent = tuple((hi - lo).tolist())
        except (AttributeError, TypeError, ValueError):
            pass
        box = tuple(candidate.bbox_2d[:4]) if len(candidate.bbox_2d) >= 4 else None
        records.append(ObjectNodeSnapshot(
            uid=candidate.uid, label=candidate.detector_label,
            position=tuple(candidate.centroid) if len(candidate.centroid) == 3 else None,
            confidence=candidate.confidence, bbox2d_xyxy=box,
            bbox3d_center=center, bbox3d_extent=extent,
        ))
    snapshot = SemanticMapSnapshot(
        timestamp=float(step),
        robot_pose=Pose3D(tuple(float(x) for x in np.asarray(mapper.current_position).reshape(-1)[:3]),
                          frame_id="mapper_local", stamp=float(step)),
        objects=tuple(records), source="habitat_observed_mapper",
        viewpoints=tuple(ViewpointSnapshot(
            uid=f"node:{node.idx}", pose=Pose3D(tuple(float(x) for x in node.position), frame_id="mapper_local"),
            room_id=str(node.room_idx), explored=bool(node.state),
        ) for node in nodes),
        frontiers=tuple(FrontierSnapshot(
            uid=f"node:{node.idx}", position=tuple(float(x) for x in node.position),
            room_id=str(node.room_idx), metadata={"frame_id": "mapper_local"},
        ) for node in nodes if node.has_frontier and node.state == 0),
    )
    return snapshot, lookup
