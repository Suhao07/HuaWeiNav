#!/usr/bin/env python3
"""Replay accepted observations and expose the geometry behind each association."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import open3d as o3d

from prior_map.observations import load_depth
from prior_map.bundle import load_bundle
from prior_map.observations import backproject, write_json
from mapping_utils.geometry import gpu_pointcloud_from_array, gpu_cluster_filter
from mapping_utils.representation import ObjectNode
from mapping.object_fusion import associate_object_entities, clouds_may_overlap, nearest_distances


def main():
    """Hold masks/depth fixed; trace shared fusion without detector or GT input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("output")
    parser.add_argument("--max-frames", type=int, default=1)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--min-coview-iou", type=float,
                        help="Experimental constraint: prohibit same-frame low-IoU observations sharing one UID")
    experiment.add_argument("--min-visible-mask-overlap", type=float,
                            help="Experimental projected-visible-reference mask support threshold")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("Use at least one frame")
    if args.min_coview_iou is not None and not 0 <= args.min_coview_iou <= 1:
        parser.error("Co-view IoU must be in [0, 1]")
    if args.min_visible_mask_overlap is not None and not 0 <= args.min_visible_mask_overlap <= 1:
        parser.error("Visible mask overlap must be in [0, 1]")
    root = Path(args.bundle)
    bundle = load_bundle(root)
    receipt = json.loads((root / "build_receipt.json").read_text())
    frame_ids = sorted({o["frame_id"] for o in bundle["observations"]})[:args.max_frames]
    entities, trace = [], []
    profile = receipt["capture_manifest"].get("capture_profile")
    current_frame, coview_masks = None, {}
    for obs in bundle["observations"]:
        if obs["frame_id"] not in frame_ids:
            continue
        depth = load_depth(root / obs["depth_ref"])
        mask = np.asarray(Image.open(root / obs["mask_ref"])) > 0
        if obs["frame_id"] != current_frame:
            current_frame, coview_masks = obs["frame_id"], {}
        valid = (depth > 0) & (depth <= receipt["max_depth_m"])
        if profile is not None:
            valid &= (depth > profile["depth_min_m"]) & (depth < profile["depth_max_m"])
        points = backproject(np.where(mask & valid, depth, 0),
                            np.asarray(obs["intrinsics"]), np.asarray(obs["camera_to_map"]))
        cloud = gpu_pointcloud_from_array(points, np.full(points.shape, 160), o3d.core.Device("CPU:0"))
        cloud = gpu_cluster_filter(cloud.voxel_down_sample(receipt["voxel_m"]))
        xyz = cloud.point.positions.numpy()
        entity = ObjectNode(cloud, cloud, obs["detector_label"], obs["detector_score"],
                            xyz.mean(axis=0), len(xyz), None, obs["bbox_xyxy"])
        entity.uid = obs["observation_id"]
        scores = []
        eligible = entities
        excluded = []
        projected = []
        if args.min_coview_iou is not None:
            eligible = []
            for other in entities:
                previous = coview_masks.get(other.uid, [])
                # 冻结证据上的消融；原关联函数与阈值保持不变，唯一变量为
                # 同帧是否允许明显分离的 mask 共用一个预测身份。
                if any(float((mask & old).sum() / (mask | old).sum()) < args.min_coview_iou for old in previous):
                    excluded.append(other.uid)
                else:
                    eligible.append(other)
        if args.min_visible_mask_overlap is not None:
            eligible = []
            world_to_camera = np.linalg.inv(np.asarray(obs["camera_to_map"]))
            intrinsic = np.asarray(obs["intrinsics"])
            for other in entities:
                if not clouds_may_overlap(cloud, other.pcd):
                    eligible.append(other)
                    continue
                points = other.pcd.point.positions.numpy()
                camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
                camera = camera[camera[:, 2] > 0]
                pixels = camera @ intrinsic.T
                uv = np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(int)
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
                uv, camera = uv[inside], camera[inside]
                # 可见支持只比较当前量程内、距实测表面 0.1 m 内的参考点。
                # rho = sum(visible & current_mask) / sum(visible)。零可见点
                # 保留为 unknown，当前消融仅否决有可见反证的关联。
                visible = valid[uv[:, 1], uv[:, 0]] & (np.abs(camera[:, 2] - depth[uv[:, 1], uv[:, 0]]) < .1)
                count = int(visible.sum())
                fraction = float(mask[uv[visible, 1], uv[visible, 0]].mean()) if count else None
                projected.append({"uid": other.uid, "visible_points": count, "mask_support": fraction})
                if fraction is not None and fraction < args.min_visible_mask_overlap:
                    excluded.append(other.uid)
                else:
                    eligible.append(other)
        for other in eligible:
            if not clouds_may_overlap(cloud, other.pcd):
                continue
            a, b = nearest_distances(cloud, other.pcd), nearest_distances(other.pcd, cloud)
            score = float(((a < .1).sum() + (b < .1).sum()) / (len(a) + len(b) + 1e-6))
            scores.append({"uid": other.uid, "label": other.tag, "points": len(b),
                "overlap": score, "new_to_old": float((a < .1).float().mean()),
                "old_to_new": float((b < .1).float().mean())})
        eligible, indices = associate_object_entities(eligible, [entity], target="", is_target_tag=lambda tag: False)
        assigned = eligible[indices[0]]
        actual_uid = assigned.uid
        if not any(old.uid == actual_uid for old in entities):
            entities.append(assigned)
        coview_masks.setdefault(actual_uid, []).append(mask)
        saved_uid = obs["object_uids"][0].split(":", 1)[1]
        trace.append({"observation_id": obs["observation_id"], "frame_id": obs["frame_id"],
            "label": obs["detector_label"], "points": len(xyz), "assigned_uid": actual_uid,
            "saved_uid": saved_uid, "assignment_matches": actual_uid == saved_uid,
            "excluded_coview_uids": excluded,
            "projection_support": projected,
            "candidates": sorted(scores, key=lambda item: item["overlap"], reverse=True)[:3]})
    write_json(args.output, {"scope": "frozen_accepted_mask_fusion_trace", "device": "CPU:0",
        "experimental_min_coview_iou": args.min_coview_iou, "object_count": len(entities),
        "experimental_min_visible_mask_overlap": args.min_visible_mask_overlap,
        "frame_ids": frame_ids, "assignment_parity": all(r["assignment_matches"] for r in trace),
        "observations": trace})
    print(json.dumps({"observations": len(trace), "assignment_parity": all(r["assignment_matches"] for r in trace)}))


if __name__ == "__main__":
    main()
