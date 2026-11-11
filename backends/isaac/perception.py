"""Live Isaac RGB-D observations using STRIVE detection and instance fusion."""
from pathlib import Path

import numpy as np
from PIL import Image

from scene.contracts import ObjectNodeSnapshot


def visible_projection(points, depth, intrinsic, camera_to_map, tolerance_m=0.15, *, return_support=False):
    """Project predicted surfaces and retain points consistent with current optical depth."""
    points = np.asarray(points)
    camera = np.linalg.inv(np.asarray(camera_to_map))
    local = points @ camera[:3, :3].T + camera[:3, 3]
    front = np.isfinite(local).all(axis=1) & (local[:, 2] > 0.05)
    local, points = local[front], points[front]
    projected = local @ np.asarray(intrinsic).T
    uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
    height, width = depth.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, local, points = uv[inside], local[inside], points[inside]
    measured = depth[uv[:, 1], uv[:, 0]]
    visible = (measured > 0) & np.isfinite(measured) & (np.abs(measured - local[:, 2]) <= tolerance_m)
    uv, points = uv[visible], points[visible]
    bbox = None
    if len(uv) >= 10:
        lo, hi = uv.min(axis=0), uv.max(axis=0) + 1
        bbox = [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]
    result = (bbox, len(uv))
    return (*result, points) if return_support else result


class IsaacObservationMap:
    """Maintain predicted instances without simulator semantic labels or target metadata."""

    def __init__(self, classes, packet, client, *, save_dir, device="cuda:0",
                 floor_world_z, ceiling_world_z):
        from backends.isaac.mapper import IsaacMapper
        from cv_utils.image_perceiver import MMDINOSAM_Perceiver

        perceiver = MMDINOSAM_Perceiver(classes=list(classes), device=device)
        perceiver.sam.initialize(target=None)
        self.mapping = IsaacMapper(packet, client, perceiver=perceiver,
            save_dir=save_dir, device=device, floor_world_z=floor_world_z, ceiling_world_z=ceiling_world_z)
        self.mapper = self.mapping.mapper
        self.save_dir = str(save_dir)
        self.episode_samples = 1
        self.episode_steps = 0
        self.semantic_label_evidence = {}
        self.latest = {}

    def add_panorama(self, packets, session):
        from backends.isaac.mapper import camera_inputs
        from navigation.observation_pipeline import process_panoramic_detections

        if not 1 <= len(packets) <= 12:
            raise ValueError("A panoramic batch requires one to twelve adjacent 30-degree views")
        self.task, self.vlm = session, session.vlm
        active = session.progress.active_target()
        target = next(item for item in session.plan.terminal_targets if item.id == active.id)
        self.mapper.target = target.name
        self.mapper.target_list = list(target.detector_terms or [target.name])
        self.mapper.target_aliases = list(target.match_terms)
        self.episode_steps = self.mapping._step
        images, depths, positions, rotations, headings, image_refs = [], [], [], [], [], []
        for packet in packets:
            root, frame = Path(packet["capture_root"]), packet["frame"]
            intrinsic, position, rotation, shape = camera_inputs(frame)
            if not np.array_equal(intrinsic, self.mapping.intrinsic) or shape != self.mapping.shape:
                raise ValueError("Panoramic camera calibration changed")
            image_ref = str((root / frame["rgb_path"]).resolve())
            with Image.open(image_ref) as image:
                images.append(np.array(image.convert("RGB")))
            depth = np.load(root / frame["depth_path"], allow_pickle=False)
            depths.append(np.where(np.isfinite(depth) & (depth > 0), depth, 0))
            positions.append(position - self.mapper.initial_position)
            rotations.append(rotation)
            headings.append(np.arctan2(-rotation[1, 2], -rotation[0, 2]))
            image_refs.append(image_ref)
        if not np.allclose(positions, positions[0], atol=1e-6, rtol=0):
            raise ValueError("Panoramic views must share one camera position")
        delta = np.arctan2(np.sin(np.diff(headings)), np.cos(np.diff(headings)))
        if not np.allclose(delta, -np.pi / 6, atol=1e-5, rtol=0):
            raise ValueError("Panoramic views must follow the original clockwise 30-degree order")

        def project_view(obj):
            import torch
            from navigation.object_view_projector import ObjectViewEvidence

            projections = self._visible_boxes(obj, packets, depths)
            index = max(range(len(projections)), key=lambda i: projections[i][1])
            bbox, support = projections[index]
            if bbox is None:
                return None
            bounds = np.array(bbox) - [0, 0, 1, 1]
            padded = bounds + [-3, -3, 3, 3]
            height, width = images[index].shape[:2]
            padded = np.clip(padded, 0, [width - 1, height - 1, width - 1, height - 1])
            return ObjectViewEvidence(images[index], padded, torch.tensor(padded).unsqueeze(0), bounds)

        detections = process_panoramic_detections(self, images, depths, positions, rotations,
            partial=len(packets) < 12, project_view=project_view)
        boxes_by_view = [{} for _ in packets]
        for entity in self.mapper.objects:
            if not entity.alive:
                continue
            projections = self._visible_boxes(entity, packets, depths)
            for index, (bbox, support) in enumerate(projections):
                if bbox is not None:
                    boxes_by_view[index][entity.uid] = bbox
            index = max(range(len(projections)), key=lambda i: projections[i][1])
            bbox, support = projections[index]
            if bbox is not None:
                self.latest[entity.uid] = {"image_ref": image_refs[index],
                    "frame_id": packets[index]["frame"]["frame_id"], "bbox": bbox}
        self.mapper.object_pcd = self.mapper.update_object_pcd()
        return [{"frame_id": packet["frame"]["frame_id"], **detection,
            "detected": detection["background_detections"] + detection["target_detections"],
            "map_objects": len(self.mapper.objects), "frontend": "original_panoramic_bc",
            "observation_image": {"id": str(packet["frame"]["frame_id"]),
                "image_ref": image_refs[index], "observed_object_uids": list(boxes_by_view[index]),
                "object_boxes": boxes_by_view[index]}}
            for index, (packet, detection) in enumerate(zip(packets, detections))]

    def _visible_boxes(self, entity, packets, depths):
        points = entity.pcd.point.positions.cpu().numpy() + self.mapper.initial_position
        return [visible_projection(points, depth, np.asarray(packet["frame"]["intrinsics"]),
                                   np.asarray(packet["frame"]["camera_to_map"]))
                for packet, depth in zip(packets, depths)]

    def objects(self):
        """Publish fused instance identities and their observed image references."""
        from mapping.object_fusion import label_probabilities

        objects = []
        for entity in self.mapper.objects:
            if not entity.alive:
                continue
            points = entity.pcd.point.positions.cpu().numpy() + self.mapper.initial_position
            detail = self.latest.get(entity.uid, {})
            objects.append(ObjectNodeSnapshot(
                uid=entity.uid, label=entity.tag,
                position=tuple(map(float, entity.position + self.mapper.initial_position)),
                confidence=float(entity.confidence),
                bbox2d_xyxy=tuple(detail["bbox"]) if detail else None,
                bbox3d_center=tuple(map(float, (points.min(axis=0) + points.max(axis=0)) / 2)),
                bbox3d_extent=tuple(map(float, points.max(axis=0) - points.min(axis=0))),
                image_ref=detail.get("image_ref"), metadata={"source": "strive_rgbd_fusion",
                    "semantic_gt_used": False, "frame_id": detail.get("frame_id"),
                    "label_probabilities": label_probabilities(entity)}))
        return tuple(objects)

    def points(self, uid):
        """Read the predicted surface of an observed instance for fresh-view verification."""
        entity = next(item for item in self.mapper.objects if item.uid == uid and item.alive)
        return entity.pcd.point.positions.cpu().numpy() + self.mapper.initial_position
