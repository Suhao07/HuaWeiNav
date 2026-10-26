"""Reuse instance label evidence while keeping current geometry verification live."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class LabelEvidence:
    """One confirmed instance label and the quality/identity of its source crop."""
    detector_label: str
    label: str
    area: float
    sharpness: float
    task_revision: int
    step: int
    source_id: str = ""
    configuration: str = ""
    instance_dependency: str = ""


def matching_instance(points, objects):
    """Return a unique map instance with >=80% bidirectional 10cm overlap.

    两侧覆盖同时满足才复用实例身份；相邻对象产生多个匹配时重新取证。
    距离单位为米，0.1m 与现有对象关联的点距离判定一致。
    """
    matches = []
    for obj in objects:
        if not obj.alive:
            continue
        reference = obj.pcd.point.positions.cpu().numpy()
        if len(reference) == 0:
            continue
        if np.any(points.max(axis=0) + 0.1 < reference.min(axis=0)) or np.any(reference.max(axis=0) + 0.1 < points.min(axis=0)):
            continue
        a = np.mean(cKDTree(reference).query(points)[0] < 0.1)
        b = np.mean(cKDTree(points).query(reference)[0] < 0.1)
        if min(a, b) >= 0.8:
            matches.append(obj)
    return matches[0] if len(matches) == 1 else None


def reusable_label(record, *, detector_label, mapped_label, area, sharpness, task_revision):
    """Reuse unchanged semantics; better imagery and contradictory labels request a model.

    25% 以上面积或清晰度提升触发重新核验；最终距离/可见性不使用此缓存。
    """
    return (record.task_revision == task_revision
            and record.detector_label == detector_label
            and record.label == mapped_label
            and record.label not in {"unknown", "nothing"}
            and area <= 1.25 * record.area
            and sharpness <= 1.25 * record.sharpness)


def instance_signature(obj):
    """Fingerprint the map support owned by an instance after association.

    同 UID 的支撑点替换、合并或标签冲突均使旧记录失效。严格哈希首先保证
    复用来源正确；点云增长引起的额外失效由实验量化，首版保留这一边界。
    """
    import hashlib
    digest = hashlib.sha256(obj.pcd.point.positions.cpu().numpy().tobytes())
    digest.update(str(obj.tag).encode())
    return digest.hexdigest()
