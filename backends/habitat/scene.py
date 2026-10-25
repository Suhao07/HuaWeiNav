"""Export observed mapper objects without exposing simulator ground truth."""
from __future__ import annotations

import numpy as np

from instruction_adapter.verifier import candidate_from_object
from scene.contracts import ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot, FrontierSnapshot, ViewpointSnapshot


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
                center, extent = tuple((lo + hi) / 2), tuple(hi - lo)
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
