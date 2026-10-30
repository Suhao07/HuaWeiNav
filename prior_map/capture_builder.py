"""Build a predicted object map from portable RGB-D records, without an episode."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .bundle import write_bundle, new_map_id
from .contracts import PriorMapData, PriorObject
from .observations import COORDINATES, SCHEMA_VERSION, backproject, read_records, write_json, metric_depth_validity


def build_capture_map(capture, output, *, classes, max_frames=None, voxel_m=0.05,
                      max_depth_m=6.0, device="cuda:0"):
    """Detect, fuse and export observed instances with their original evidence.

    Args:
        capture: Directory containing a completed portable RGB-D capture.
        output: New bundle directory; existing artifacts are preserved.
        classes: Explicit open-vocabulary detector prompts, independent of tasks.
        max_frames: Optional prefix size for a bounded integration test.
        voxel_m: Per-observation voxel size before existing instance fusion.
        max_depth_m: Maximum depth retained in each masked cloud, in metres.
        device: Torch device used by the maintained detector stack.

    Returns:
        Build receipt with counts, timings and predicted label distribution.

    Raises:
        ValueError: Capture coordinates or builder settings are incompatible.
        FileExistsError: Output already exists.

    Geometry uses p_map = T_map_camera [z K^-1(u,v,1), 1]. Instance overlap and
    confidence aggregation reuse STRIVE's shared fusion implementation. Object
    identities denote predictions; acceptance still requires live evidence.
    """
    from cv_utils.sam import MMDINO_Grounded_SAM
    from mapping.object_fusion import associate_object_entities
    from mapping_utils.geometry import gpu_pointcloud_from_array, gpu_cluster_filter
    from mapping_utils.representation import ObjectNode
    import open3d as o3d

    root, destination = Path(capture), Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["coordinates"] != COORDINATES or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Capture must use the portable optical-camera protocol")
    if manifest["status"] != "complete":
        raise ValueError("Capture must complete before object fusion")
    if len(classes) < 1 or voxel_m <= 0 or max_depth_m <= 0 or (max_frames is not None and max_frames < 1):
        raise ValueError("Use a nonempty vocabulary and positive build budgets")
    frames = read_records(root / "frames.jsonl")[:max_frames]
    destination.mkdir(parents=True, exist_ok=False)
    evidence = destination / "evidence"
    evidence.mkdir()
    started = time.monotonic()
    detector = MMDINO_Grounded_SAM(classes=classes, device=device)
    detector.initialize(target=None)
    from config_utils import SAM_CHECKPOINT, GROUNDING_DINO_CHECKPOINT, GROUNDING_DINO_CONFIG
    def file_digest(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    source_root = Path(__file__).resolve().parents[1]
    provenance = {"source_sha256": {name: file_digest(source_root / name) for name in (
        "prior_map/capture_builder.py", "prior_map/observations.py", "cv_utils/sam.py",
        "mapping/object_fusion.py", "mapping_utils/geometry.py")},
        "model_sha256": {"sam": file_digest(SAM_CHECKPOINT),
                         "grounding_dino": file_digest(GROUNDING_DINO_CHECKPOINT),
                         "grounding_dino_config": file_digest(GROUNDING_DINO_CONFIG)}}
    cloud_device = o3d.core.Device(device.upper())
    entities, observations, viewpoints, frame_reports = [], [], [], []
    map_id = new_map_id()
    for frame in frames:
        tick = time.monotonic()
        rgb = np.asarray(Image.open(root / frame["rgb_path"]).convert("RGB"))
        depth = np.load(root / frame["depth_path"], allow_pickle=False)
        profile = manifest.get("capture_profile")
        valid_depth = metric_depth_validity(depth, profile) & (depth <= max_depth_m)
        intrinsic = np.asarray(frame["intrinsics"])
        pose = np.asarray(frame["camera_to_map"])
        boxes, scores, labels, masks, _, _, _ = detector.segment(rgb, target=None)
        observed, accepted = [], []
        for index, (box, score, label, mask) in enumerate(zip(boxes, scores, labels, masks)):
            mask = mask.numpy().astype(bool)
            bbox = box.numpy()
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # 沿用在线感知的面积准入；原始深度保留在证据中，距离裁剪只作用于融合输入。
            if mask.sum() <= 100 or area >= 0.8 * depth.size:
                continue
            masked_depth = np.where(mask & valid_depth, depth, 0)
            points = backproject(masked_depth, intrinsic, pose)
            if len(points) < 10:
                continue
            colors = np.full(points.shape, 160, dtype=np.uint8)
            cloud = gpu_pointcloud_from_array(points, colors, cloud_device).voxel_down_sample(voxel_m)
            if cloud.point.positions.shape[0] < 10:
                continue
            cloud = gpu_cluster_filter(cloud)
            if cloud.point.positions.shape[0] < 10:
                continue
            position = cloud.point.positions.cpu().numpy().mean(axis=0)
            entity = ObjectNode(cloud, cloud, str(label), float(score), position,
                                int(cloud.point.positions.shape[0]), rgb, bbox.tolist())
            entity.uid = f"{map_id}:frame{frame['frame_id']}:detection{index}"
            observed.append(entity)
            accepted.append((index, mask, bbox.tolist(), str(label), float(score)))
        entities, indices = associate_object_entities(
            entities, observed, target="", is_target_tag=lambda tag: False,
        )
        frame_observations = []
        if accepted:
            image_ref = f"evidence/{frame['frame_id']:06d}.png"
            depth_ref = f"evidence/{frame['frame_id']:06d}.npy"
            shutil.copyfile(root / frame["rgb_path"], destination / image_ref)
            shutil.copyfile(root / frame["depth_path"], destination / depth_ref)
            confidence_ref = None
            if frame.get("confidence_ref") is not None:
                confidence_ref = f"evidence/{frame['frame_id']:06d}-confidence.png"
                shutil.copyfile(root / frame["confidence_ref"], destination / confidence_ref)
        for (index, mask, bbox, label, score), object_index in zip(accepted, indices):
            obs_id = f"frame{frame['frame_id']}:detection{index}"
            mask_ref = f"evidence/{frame['frame_id']:06d}-{index}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(destination / mask_ref)
            observations.append({"observation_id": obs_id, "frame_id": frame["frame_id"],
                "object_uids": [entities[object_index].uid], "image_ref": image_ref,
                "depth_ref": depth_ref, "mask_ref": mask_ref, "confidence_ref": confidence_ref, "bbox_xyxy": bbox,
                "detector_label": label, "detector_score": score,
                "camera_to_map": frame["camera_to_map"], "intrinsics": frame["intrinsics"]})
            frame_observations.append(obs_id)
        if frame_observations:
            viewpoints.append({"uid": f"view:{frame['frame_id']}", "kind": "observation_pose",
                "pose_in_map": frame["camera_to_map"], "observation_ids": frame_observations})
        row = {"frame_id": frame["frame_id"], "detected": len(labels),
               "accepted": len(accepted), "map_objects": len(entities),
               "elapsed_s": time.monotonic() - tick}
        frame_reports.append(row)
        print(json.dumps(row), flush=True)
    objects = tuple(PriorObject(uid=e.uid, label=e.tag, position_xyz=tuple(map(float, e.position)),
        exact=True, confidence=float(e.confidence), source="strive_rgbd_fusion",
        metadata={"identity_kind": "predicted_instance", "point_count": int(e.pcd.point.positions.shape[0])})
        for e in entities)
    prior = PriorMapData(scene_id=manifest["scene_id"], objects=objects,
        source_format="predicted_rgbd", frame_id="map_z_up",
        metadata={"capture_mode": manifest["mode"], "pose_source": manifest["pose_source"],
                  "semantic_gt_used": False, "navigation_ready": False})
    write_bundle(destination, map_id=map_id, revision="1", prior_map=prior,
                 observations=observations, viewpoints=viewpoints)
    receipt = {"status": "complete", "frame_count": len(frames), "object_count": len(objects),
        "observation_count": len(observations), "elapsed_s": time.monotonic() - started,
        "provenance": provenance, "confidence_policy": "preserved_not_filtered", "classes": classes, "voxel_m": voxel_m, "max_depth_m": max_depth_m,
        "depth_policy": "positive_within_fusion_range_excluding_recorded_sensor_clip_endpoints",
        "capture_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "capture_manifest": manifest, "frames": frame_reports,
        "labels": {label: sum(o.label == label for o in objects) for label in sorted({o.label for o in objects})},
        "scope": "predicted_instances_with_evidence_not_navigation_acceptance"}
    write_json(destination / "build_receipt.json", receipt)
    return receipt
