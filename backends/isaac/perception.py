"""Live Isaac RGB-D observations using STRIVE detection and instance fusion."""
from pathlib import Path

import numpy as np
from PIL import Image

from prior_map.observations import backproject
from scene.contracts import ObjectNodeSnapshot


def visible_projection(points, depth, intrinsic, camera_to_map, tolerance_m=0.15):
    """Project predicted surfaces and retain points consistent with current optical depth."""
    points = np.asarray(points)
    camera = np.linalg.inv(np.asarray(camera_to_map))
    local = points @ camera[:3, :3].T + camera[:3, 3]
    local = local[local[:, 2] > 0.05]
    if not len(local):
        return None, 0
    projected = local @ np.asarray(intrinsic).T
    uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
    height, width = depth.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, local = uv[inside], local[inside]
    measured = depth[uv[:, 1], uv[:, 0]]
    visible = (measured > 0) & np.isfinite(measured) & (np.abs(measured - local[:, 2]) <= tolerance_m)
    uv = uv[visible]
    if len(uv) < 10:
        return None, len(uv)
    lo, hi = uv.min(axis=0), uv.max(axis=0) + 1
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])], len(uv)


class IsaacObservationMap:
    """Maintain predicted instances without simulator semantic labels or target metadata."""

    def __init__(self, classes, *, device="cuda:0", voxel_m=0.05, max_depth_m=6.0):
        import open3d as o3d
        from cv_utils.sam import MMDINO_Grounded_SAM

        self.detector = MMDINO_Grounded_SAM(classes=list(classes), device=device)
        self.detector.initialize(target=None)
        self.device = o3d.core.Device(device.upper())
        self.voxel_m = voxel_m
        self.max_depth_m = max_depth_m
        self.entities = []
        self.latest = {}

    def add(self, packet):
        """Fuse a real rendered frame through the maintained STRIVE overlap policy."""
        from mapping.object_fusion import associate_object_entities
        from mapping_utils.geometry import gpu_cluster_filter, gpu_pointcloud_from_array
        from mapping_utils.representation import ObjectNode

        root, frame = Path(packet["capture_root"]), packet["frame"]
        rgb = np.asarray(Image.open(root / frame["rgb_path"]).convert("RGB"))
        depth = np.load(root / frame["depth_path"], allow_pickle=False)
        valid = np.isfinite(depth) & (depth > 0) & (depth <= self.max_depth_m)
        boxes, scores, labels, masks, _, _, _ = self.detector.segment(rgb, target=None)
        observed, details = [], []
        for index, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
            mask = mask.cpu().numpy().astype(bool)
            bbox = box.cpu().numpy()
            if mask.sum() <= 100 or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.8 * depth.size:
                continue
            points = backproject(np.where(mask & valid, depth, 0),
                                 np.asarray(frame["intrinsics"]), np.asarray(frame["camera_to_map"]))
            if len(points) < 10:
                continue
            cloud = gpu_pointcloud_from_array(points, np.full(points.shape, 160, dtype=np.uint8), self.device)
            cloud = cloud.voxel_down_sample(self.voxel_m)
            if cloud.point.positions.shape[0] < 10:
                continue
            cloud = gpu_cluster_filter(cloud)
            if cloud.point.positions.shape[0] < 10:
                continue
            position = cloud.point.positions.cpu().numpy().mean(axis=0)
            entity = ObjectNode(cloud, cloud, str(label), float(score), position,
                                int(cloud.point.positions.shape[0]), rgb, bbox.tolist())
            entity.uid = f"observed:{frame['frame_id']}:{index}"
            observed.append(entity)
            details.append({"image_ref": str(root / frame["rgb_path"]), "frame_id": frame["frame_id"],
                            "bbox": bbox.tolist(), "detector_label": str(label), "score": float(score)})
        self.entities, indices = associate_object_entities(
            self.entities, observed, target="", is_target_tag=lambda tag: False)
        for index, detail in zip(indices, details):
            self.latest[self.entities[index].uid] = detail
        return {"frame_id": frame["frame_id"], "detected": len(labels),
                "admitted": len(observed), "map_objects": len(self.entities)}

    def objects(self):
        """Publish fused instance identities and their observed image references."""
        from mapping.object_fusion import label_probabilities

        objects = []
        for entity in self.entities:
            if not entity.alive:
                continue
            points = entity.pcd.point.positions.cpu().numpy()
            detail = self.latest[entity.uid]
            objects.append(ObjectNodeSnapshot(
                uid=entity.uid, label=entity.tag, position=tuple(map(float, entity.position)),
                confidence=float(entity.confidence), bbox2d_xyxy=tuple(detail["bbox"]),
                bbox3d_center=tuple(map(float, (points.min(axis=0) + points.max(axis=0)) / 2)),
                bbox3d_extent=tuple(map(float, points.max(axis=0) - points.min(axis=0))),
                image_ref=detail["image_ref"], metadata={"source": "strive_rgbd_fusion",
                    "semantic_gt_used": False, "frame_id": detail["frame_id"],
                    "label_probabilities": label_probabilities(entity)}))
        return tuple(objects)

    def points(self, uid):
        """Read the predicted surface of an observed instance for fresh-view verification."""
        entity = next(item for item in self.entities if item.uid == uid and item.alive)
        return entity.pcd.point.positions.cpu().numpy()
