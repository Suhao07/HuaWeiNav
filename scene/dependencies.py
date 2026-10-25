"""Observation dependencies used when admitting delayed navigation proposals."""
from hashlib import sha256
import json
from scene.contracts import SemanticMapSnapshot


def object_dependencies(snapshot: SemanticMapSnapshot, *object_uids: str) -> dict[str, str]:
    """Identify the coordinate frame and selected observed object geometry.

    Args:
        snapshot: Immutable observed scene used by a decision or verification.
        object_uids: Only the target and anchors used by that computation.

    Returns:
        Dependency keys and content fingerprints. Missing objects have no key.

    依赖限定为相关对象的几何与标签；其他对象更新、置信度刷新和机器人移动
    保持结论可用。坐标系重置版本由后端通过 frame_revision 提供。
    """
    values = {"frame": snapshot.robot_pose.frame_id}
    if "frame_revision" in snapshot.metadata:
        values["frame_revision"] = str(snapshot.metadata["frame_revision"])
    for uid in object_uids:
        if not uid:
            continue
        obj = snapshot.object_by_uid(uid)
        if obj is not None:
            geometry = [obj.label, obj.position, obj.bbox3d_center, obj.bbox3d_extent, obj.room_id]
            values[f"object:{uid}"] = sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    return values
