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
from .observations import (COORDINATES, SCHEMA_VERSION, backproject, camera_to_base_pose, load_depth,
                           read_records, write_json, metric_depth_validity)


def capture_free_cells(capture, *, resolution, robot_height_m, robot_radius_m, stride, max_depth_m,
                       min_depth_confidence=None):
    r"""Derive traversable floor cells from a portable RGB-D capture.

    Args:
        capture: Completed portable capture directory.
        resolution: Map grid cell size in metres.
        robot_height_m: Body height; points below it block the robot.
        robot_radius_m: Footprint radius used to erode free floor.
        stride: Pixel stride for back-projection.
        max_depth_m: Maximum depth used as a geometric measurement.
        min_depth_confidence: Sensor confidence required for floor and obstacle
            points; None keeps every metric depth.

    Returns:
        ``(cells, floor_height, diagnostics)`` with integer map_z_up cells.

    Raises:
        ValueError: Capture protocol, status or settings are incompatible, or the
            trajectory reaches no robot-traversable floor.

    地面高度 z_f 取相机下方点高度直方图(0.05m格)的众数格中位数。
    地面带沿用 mapper 约定 |z-z_f|<0.2，障碍带为 z_f+0.2 <= z < z_f+h：

    .. math::
        F(c) = 1[\exists p\in c: |p_z-z_f|<0.2]\;\wedge\;
               1[\nexists p\in c: z_f+0.2 \le p_z < z_f+h],\qquad
        F_{robot} = F \ominus \mathrm{Disk}(r),\qquad
        F_{reach} = \bigcup\{K \in \mathrm{cc}_4(F_{robot}) : K \cap T \ne \emptyset\}

    未观测格不属于 F；腐蚀后的 F_robot 与按机器人半径生成的 NavMesh 同义。
    T 为相机轨迹投影格：手持采集者走过的地面可达，透过窗口观测到的孤立地面不可达。
    """
    import math
    from scipy import ndimage

    root = Path(capture)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["coordinates"] != COORDINATES or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Capture must use the portable optical-camera protocol")
    if manifest["status"] != "complete":
        raise ValueError("Capture must complete before layout extraction")
    if min(resolution, robot_height_m, max_depth_m) <= 0 or robot_radius_m < 0 or stride < 1:
        raise ValueError("Use positive grid, robot, depth and stride settings")
    profile = manifest.get("capture_profile")
    clouds, below_camera, camera_xy = [], [], []
    for frame in read_records(root / "frames.jsonl"):
        depth = load_depth(root / frame["depth_path"])
        confidence = (None if frame.get("confidence_ref") is None
                      else np.asarray(Image.open(root / frame["confidence_ref"])))
        valid = (metric_depth_validity(depth, profile, confidence, min_depth_confidence)
                 & (depth <= max_depth_m))
        pose = np.asarray(frame["camera_to_map"])
        points = backproject(np.where(valid, depth, 0), np.asarray(frame["intrinsics"]), pose,
                             stride=stride).astype(np.float32)
        clouds.append(points)
        below_camera.append(points[points[:, 2] < pose[2, 3], 2])
        camera_xy.append(pose[:2, 3])
    points, below_camera = np.concatenate(clouds), np.concatenate(below_camera)
    if not len(below_camera):
        raise ValueError("Capture contains no valid depth below the camera")
    bins = np.floor(below_camera / 0.05).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode = values[np.argmax(counts)]
    floor_height = float(np.median(below_camera[bins == mode]))
    cell_xy = np.round(points[:, :2] / resolution).astype(np.int64)
    height = points[:, 2] - floor_height
    floor = {tuple(c) for c in np.unique(cell_xy[np.abs(height) < 0.2], axis=0)}
    blocked = {tuple(c) for c in np.unique(cell_xy[(height >= 0.2) & (height < robot_height_m)], axis=0)}
    free = floor - blocked
    if not free:
        raise ValueError("Capture observed no unobstructed floor cells")
    indices = np.asarray(sorted(free))
    offset = indices.min(axis=0)
    grid = np.zeros(tuple(indices.max(axis=0) - offset + 1), dtype=bool)
    grid[tuple((indices - offset).T)] = True
    radius = robot_radius_m / resolution
    xx, yy = np.mgrid[-math.ceil(radius):math.ceil(radius) + 1, -math.ceil(radius):math.ceil(radius) + 1]
    eroded = ndimage.binary_erosion(grid, structure=xx * xx + yy * yy <= radius ** 2)
    components, _ = ndimage.label(eroded)
    trajectory = np.round(np.asarray(camera_xy) / resolution).astype(np.int64) - offset
    trajectory = trajectory[np.all((trajectory >= 0) & (trajectory < eroded.shape), axis=1)]
    reachable = np.isin(components, np.setdiff1d(components[tuple(trajectory.T)], [0]))
    if not reachable.any():
        raise ValueError("Capture trajectory crosses no robot-traversable floor")
    cells = {tuple(int(v) for v in c + offset) for c in np.argwhere(reachable)}
    diagnostics = {"geometry_source": "rgbd_capture", "frame_count": len(clouds),
                   "point_count": int(len(points)), "floor_height_estimate": floor_height,
                   "floor_mode_point_count": int(counts.max()), "floor_cell_count": len(floor),
                   "blocked_cell_count": len(blocked), "unobstructed_floor_cell_count": len(free),
                   "robot_cell_count": int(eroded.sum()), "reachable_cell_count": len(cells),
                   "robot_height_m": robot_height_m,
                   "robot_radius_m": robot_radius_m, "stride": stride, "max_depth_m": max_depth_m,
                   "min_depth_confidence": min_depth_confidence,
                   "capture_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}
    return cells, floor_height, diagnostics


def resolve_capture_viewpoints(bundle_directory, capture, *, resolution, robot_height_m, robot_radius_m,
                               stride, max_depth_m, object_uid=None, observation_ids=None,
                               min_depth_confidence=None):
    r"""Place historical camera poses on the robot-traversable floor of their capture.

    Args:
        bundle_directory: Map bundle built from ``capture``; read-only.
        capture: Completed portable capture that produced the bundle.
        resolution: Free-space grid cell size in metres.
        robot_height_m: Body height used for the obstacle band.
        robot_radius_m: Footprint radius used to erode free floor.
        stride: Pixel stride for free-space back-projection.
        max_depth_m: Maximum depth used as a geometric measurement.
        object_uid: Optional historical instance restricting the evidence.
        observation_ids: Optional query-specific subset of that evidence.
        min_depth_confidence: Free-space confidence threshold; must match the layout.

    Returns:
        Candidate report with admitted navigation viewpoints and rejections.

    Raises:
        ValueError: The bundle comes from another capture, the instance or
            evidence is unknown, or the capture yields no traversable floor.

    手持相机高度逐帧变化，基座取相机竖直投影到估计地面 z_f：

    .. math::
        p_{base} = (x_{cam},\, y_{cam},\, z_f),\qquad
        \mathrm{admit} \iff \mathrm{round}(p_{base,xy}/s) \in F_{reach}

    F_reach 为 ``capture_free_cells`` 的轨迹连通、按机器人半径腐蚀的可通行格。
    候选表示历史视点落在可通行地面；相对当前起点的路径可达性由运行时检查。
    """
    from .bundle import load_bundle, select_evidence_ids

    root = Path(bundle_directory)
    bundle = load_bundle(root)
    evidence_ids = select_evidence_ids(bundle, object_uid, observation_ids)
    capture_digest = hashlib.sha256((Path(capture) / "manifest.json").read_bytes()).hexdigest()
    if json.loads((root / "build_receipt.json").read_text())["capture_manifest_sha256"] != capture_digest:
        raise ValueError("Bundle was built from a different capture")
    cells, floor_height, free_space = capture_free_cells(
        capture, resolution=resolution, robot_height_m=robot_height_m, robot_radius_m=robot_radius_m,
        stride=stride, max_depth_m=max_depth_m, min_depth_confidence=min_depth_confidence)
    profile = {"source": "declared_robot_dimensions", "radius_m": robot_radius_m, "height_m": robot_height_m}
    viewpoints, rejected = [], []
    for historical in bundle["viewpoints"]:
        if historical["kind"] != "observation_pose":
            continue
        supported = [uid for uid in historical["observation_ids"] if uid in evidence_ids]
        if not supported:
            continue
        camera = np.asarray(historical["pose_in_map"], dtype=float)
        base = camera_to_base_pose(camera, camera[2, 3] - floor_height)
        if tuple(int(v) for v in np.round(base[:2, 3] / resolution)) not in cells:
            rejected.append({"source_viewpoint": historical["uid"], "reason": "base_outside_robot_free_space"})
            continue
        viewpoints.append({"uid": f"navigation:{historical['uid']}", "kind": "navigation_viewpoint",
            "pose_in_map": base.tolist(), "pose_axes": "x_forward_y_left_z_up",
            "observation_ids": supported, "historical_object_uid": object_uid,
            "robot_profile": profile, "reachability_status": "trajectory_connected_free_space",
            "checked_map_revision": capture_digest,
            "source_map_revision": bundle["manifest"]["map_revision"],
            "motion_admission": "requires_runtime_robot_profile_match_and_current_path_check"})
    return {"scope": "historical_viewpoint_candidates_not_motion_execution",
            "map_id": bundle["manifest"]["map_id"], "historical_object_uid": object_uid,
            "historical_observation_ids": sorted(evidence_ids), "capture_manifest_sha256": capture_digest,
            "free_space": free_space, "robot_profile": profile,
            "viewpoints": viewpoints, "rejected": rejected}


def build_capture_map(capture, output, *, classes, max_frames=None, voxel_m=0.05,
                      max_depth_m=6.0, device="cuda:0", association_config=None,
                      min_depth_confidence=None):
    """Detect, fuse and export observed instances with their original evidence.

    Args:
        capture: Directory containing a completed portable RGB-D capture.
        output: New bundle directory; existing artifacts are preserved.
        classes: Explicit open-vocabulary detector prompts, independent of tasks.
        max_frames: Optional prefix size for a bounded integration test.
        voxel_m: Per-observation voxel size before existing instance fusion.
        max_depth_m: Maximum depth retained in each masked cloud, in metres.
        device: Torch device used by the maintained detector stack.
        association_config: Opt-in deferred association configuration; None keeps legacy fusion.
        min_depth_confidence: Sensor confidence level required for fused depth;
            None keeps every metric depth. Raw depth and confidence stay in evidence.

    Returns:
        Build receipt with counts, timings and predicted label distribution.

    Raises:
        ValueError: Capture coordinates or builder settings are incompatible.
        FileExistsError: Output already exists.

    Geometry uses p_map = T_map_camera [z K^-1(u,v,1), 1]. Instance overlap and
    confidence aggregation reuse STRIVE's shared fusion implementation; after all
    frames, instances with mutual surface coverage >0.85 merge into one identity.
    Each object keeps its vote posterior p(l) besides the decided label.
    Object identities denote predictions; acceptance still requires live evidence.
    """
    from cv_utils.sam import MMDINO_Grounded_SAM
    from mapping.object_fusion import associate_object_entities, label_probabilities, merge_duplicate_instances
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
        "mapping/object_fusion.py", "mapping_utils/geometry.py", "prior_map/association.py")},
        "model_sha256": {"sam": file_digest(SAM_CHECKPOINT),
                         "grounding_dino": file_digest(GROUNDING_DINO_CHECKPOINT),
                         "grounding_dino_config": file_digest(GROUNDING_DINO_CONFIG)}}
    cloud_device = o3d.core.Device(device.upper())
    entities, observations, viewpoints, frame_reports = [], [], [], []
    map_id = new_map_id()
    observation_records = {}
    coordinator = None
    if association_config is not None:
        from .association import PriorAssociation
        from .observations import load_fusion_observation
        def load_observation(oid):
            return load_fusion_observation(destination, observation_records[oid],
                voxel_m=voxel_m, max_depth_m=max_depth_m,
                profile=manifest.get("capture_profile"), device=device.upper(), uid=f"{map_id}:{oid}",
                min_confidence=min_depth_confidence)
        coordinator = PriorAssociation(load_observation, association_config)
    for frame in frames:
        tick = time.monotonic()
        rgb = np.asarray(Image.open(root / frame["rgb_path"]).convert("RGB"))
        depth = np.load(root / frame["depth_path"], allow_pickle=False)
        profile = manifest.get("capture_profile")
        confidence = (None if frame.get("confidence_ref") is None
                      else np.asarray(Image.open(root / frame["confidence_ref"])))
        valid_depth = (metric_depth_validity(depth, profile, confidence, min_depth_confidence)
                       & (depth <= max_depth_m))
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
        if coordinator is None:
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
        for offset, (index, mask, bbox, label, score) in enumerate(accepted):
            obs_id = f"frame{frame['frame_id']}:detection{index}"
            mask_ref = f"evidence/{frame['frame_id']:06d}-{index}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(destination / mask_ref)
            observations.append({"observation_id": obs_id, "frame_id": frame["frame_id"],
                "object_uids": [entities[indices[offset]].uid] if coordinator is None else [], "image_ref": image_ref,
                "depth_ref": depth_ref, "mask_ref": mask_ref, "confidence_ref": confidence_ref, "bbox_xyxy": bbox,
                "detector_label": label, "detector_score": score,
                "camera_to_map": frame["camera_to_map"], "intrinsics": frame["intrinsics"]})
            observation_records[obs_id] = observations[-1]
            frame_observations.append(obs_id)
        if coordinator is not None:
            coordinator.add_frame(frame["frame_id"], frame_observations)
            entities = list(coordinator.entities.values())
        if frame_observations:
            viewpoints.append({"uid": f"view:{frame['frame_id']}", "kind": "observation_pose",
                "pose_in_map": frame["camera_to_map"], "observation_ids": frame_observations})
        row = {"frame_id": frame["frame_id"], "detected": len(labels),
               "accepted": len(accepted), "map_objects": len(entities),
               "elapsed_s": time.monotonic() - tick}
        frame_reports.append(row)
        print(json.dumps(row), flush=True)
    duplicate_merges = {}
    if coordinator is not None:
        coordinator.resolve_pending()
        entities = list(coordinator.entities.values())
        for observation in observations:
            uid = coordinator.assignment[observation['observation_id']]
            observation['object_uids'] = [] if uid is None else [uid]
        write_json(destination / "association_trace.json", coordinator.trace())
    else:
        entities, duplicate_merges = merge_duplicate_instances(entities, voxel_m)
        for observation in observations:
            observation['object_uids'] = [duplicate_merges.get(uid, uid) for uid in observation['object_uids']]
    objects = tuple(PriorObject(uid=e.uid, label=e.tag, position_xyz=tuple(map(float, e.position)),
        exact=True, confidence=float(e.confidence), source="strive_rgbd_fusion",
        label_probabilities=label_probabilities(e), metadata={"identity_kind": "predicted_instance", "point_count": int(e.pcd.point.positions.shape[0])})
        for e in entities)
    prior = PriorMapData(scene_id=manifest["scene_id"], objects=objects,
        source_format="predicted_rgbd", frame_id="map_z_up",
        metadata={"capture_mode": manifest["mode"], "pose_source": manifest["pose_source"],
                  "semantic_gt_used": False, "navigation_ready": False})
    write_bundle(destination, map_id=map_id, revision="1", prior_map=prior,
                 observations=observations, viewpoints=viewpoints)
    receipt = {"association_policy": "legacy" if coordinator is None else coordinator.trace()["config"],
        "status": "complete", "frame_count": len(frames), "object_count": len(objects),
        "observation_count": len(observations), "duplicate_instance_merges": duplicate_merges,
        "elapsed_s": time.monotonic() - started,
        "provenance": provenance, "classes": classes, "voxel_m": voxel_m, "max_depth_m": max_depth_m,
        "confidence_policy": ("preserved_not_filtered" if min_depth_confidence is None
                              else f"fused_depth_requires_confidence_ge_{min_depth_confidence}"),
        "depth_policy": "positive_within_fusion_range_excluding_recorded_sensor_clip_endpoints",
        "capture_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "capture_manifest": manifest, "frames": frame_reports,
        "labels": {label: sum(o.label == label for o in objects) for label in sorted({o.label for o in objects})},
        "scope": "predicted_instances_with_evidence_not_navigation_acceptance"}
    write_json(destination / "build_receipt.json", receipt)
    return receipt
