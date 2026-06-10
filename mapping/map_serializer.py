"""JSON serializers for STRIVE mapper state.

The mapper owns runtime state; this module owns only the presentation format
used by LLM prompts, debug files, and metrics artifacts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mapping_utils.representation import NodeState


def to_json(mapper: Any) -> dict[str, Any]:
    """Serialize objects and room viewpoints for room-selection prompts."""

    json_data: dict[str, Any] = {}
    json_data["objects"] = _objects_json(mapper, rounded=True, skip_unknown=False)
    json_data["Room"] = _rooms_json(mapper, include_debug=False)
    return json_data


def to_json_wo_some_class(mapper: Any) -> dict[str, Any]:
    """Serialize objects while filtering classes that are useless for target choice."""

    json_data: dict[str, Any] = {}
    json_data["objects"] = _objects_json(mapper, rounded=False, skip_unknown=True)
    return json_data


def to_json_save_node_info(mapper: Any) -> dict[str, Any]:
    """Serialize detailed node graph state for debug artifacts."""

    json_data: dict[str, Any] = {}
    json_data["objects"] = _objects_json(mapper, rounded=True, skip_unknown=False)
    json_data["Room"] = _rooms_json(mapper, include_debug=True)
    return json_data


def _objects_json(mapper: Any, *, rounded: bool, skip_unknown: bool) -> list[dict[str, Any]]:
    out = []
    for index, obj in enumerate(mapper.objects):
        if skip_unknown and obj.tag in ("unknown", "furniture"):
            continue
        size = _object_size(obj)
        confidence = obj.confidence.numpy().item()
        if rounded:
            out.append({
                "index": index,
                "position": [round(p, 3) for p in obj.position.tolist()],
                "class": obj.tag,
                "confidence": round(confidence, 3),
                "size": [round(s, 3) for s in size.tolist()],
            })
        else:
            out.append({
                "index": index,
                "position": obj.position.tolist(),
                "class": obj.tag,
                "confidence": confidence,
                "size": size.tolist(),
            })
    return out


def _rooms_json(mapper: Any, *, include_debug: bool) -> list[dict[str, Any]]:
    rooms = []
    for room_node in mapper.room_nodes:
        viewpoints = []
        for node in room_node.nodes:
            # 已探索节点不能再作为 frontier 参与 room prompt，否则 LLM 会反复选
            # 已经访问过的区域。
            if node.state == NodeState.EXPLORED:
                node.has_frontier = False
                node.has_true_frontier = False
            item = {
                "position": [round(p, 3) for p in node.position.tolist()],
                "has_frontier": node.has_frontier,
                "objects": node.objects,
            }
            if include_debug:
                item.update({
                    "idx": node.idx,
                    "state": node.state,
                    "neighbors": mapper.neighbors[node.idx],
                    "has_true_frontier": node.has_true_frontier,
                })
            viewpoints.append(item)
        rooms.append({
            "room_idx": room_node.room_id,
            "state": room_node.state,
            "distance": round(room_node.distance, 3),
            "viewpoints": viewpoints,
        })
    return rooms


def _object_size(obj: Any) -> np.ndarray:
    pcd = obj.pcd.point.positions.cpu().numpy()
    size = np.zeros(3)
    size[0] = np.max(pcd[:, 0]) - np.min(pcd[:, 0])
    size[1] = np.max(pcd[:, 1]) - np.min(pcd[:, 1])
    size[2] = np.max(pcd[:, 2]) - np.min(pcd[:, 2])
    return size
