"""Version-one portable map bundle around the existing canonical PriorMapData."""
from pathlib import Path
import json
import uuid
import numpy as np
from .loaders import PriorMapLoader
from .observations import COORDINATES, SCHEMA_VERSION, write_json, read_records


def new_map_id():
    """Allocate identity once per new map build, independently of its directory.

    Reloads and storage-only transformations copy the persisted ID. A scene ID
    describes the environment; this ID distinguishes separate maps of that scene.
    """
    return str(uuid.uuid4())


def load_bundle(directory):
    """Load and validate core data and evidence references without image decoding.

    Returns a dict with manifest, prior_map, observations and viewpoints.
    Raises ValueError for unsupported schemas, inconsistent IDs or escaping refs.
    """
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["coordinates"] != COORDINATES:
        raise ValueError("Unsupported bundle schema or coordinate convention")
    prior = PriorMapLoader().load(root / "prior_map.json", source_format="canonical_json")
    if prior.frame_id != "map_z_up":
        raise ValueError("Bundle core must use map_z_up")
    observations = read_records(root / "observations.jsonl")
    viewpoints = json.loads((root / "viewpoints.json").read_text())
    object_ids = {obj.uid for obj in prior.objects}
    obs_ids = set()
    for obs in observations:
        uid = obs["observation_id"]
        if uid in obs_ids:
            raise ValueError(f"Duplicate observation: {uid}")
        obs_ids.add(uid)
        pose = np.asarray(obs["camera_to_map"], dtype=float)
        k = np.asarray(obs["intrinsics"], dtype=float)
        if pose.shape != (4, 4) or k.shape != (3, 3) or not np.isfinite(pose).all() or not np.isfinite(k).all():
            raise ValueError(f"Invalid pose or intrinsics: {uid}")
        if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.isclose(np.linalg.det(pose[:3, :3]), 1):
            raise ValueError(f"Invalid camera_to_map: {uid}")
        if not set(obs["object_uids"]) <= object_ids:
            raise ValueError(f"Unknown objects in observation {uid}")
        for key in ("image_ref", "depth_ref", "mask_ref", "confidence_ref"):
            if obs.get(key) is not None:
                path = (root / obs[key]).resolve()
                if root not in path.parents:
                    raise ValueError(f"Reference escapes bundle: {obs[key]}")
                if not path.is_file():
                    raise FileNotFoundError(path)
    ids = set()
    for view in viewpoints:
        if view["uid"] in ids or view["kind"] not in ("observation_pose", "navigation_viewpoint"):
            raise ValueError("Duplicate viewpoint or unknown viewpoint kind")
        ids.add(view["uid"])
        pose = np.asarray(view["pose_in_map"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Viewpoint pose must be a finite 4x4 matrix")
        if not set(view["observation_ids"]) <= obs_ids:
            raise ValueError("Viewpoint references unknown observation")
        if view["kind"] == "navigation_viewpoint":
            for key in ("robot_profile", "reachability_status", "checked_map_revision"):
                if key not in view:
                    raise ValueError(f"Navigation viewpoint lacks {key}")
    return {"manifest": manifest, "prior_map": prior,
            "observations": observations, "viewpoints": viewpoints}


def write_bundle(directory, *, map_id, revision, prior_map, observations, viewpoints):
    """Write portable indices after the caller copies evidence into directory.

    Map revision and object identity remain explicit; no predicted object is
    promoted to current task success by serialization.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "prior_map.json", prior_map.to_dict())
    (root / "observations.jsonl").write_text("".join(json.dumps(o, allow_nan=False) + "\n" for o in observations))
    write_json(root / "viewpoints.json", viewpoints)
    write_json(root / "manifest.json", {"schema_version": SCHEMA_VERSION,
        "map_id": map_id, "map_revision": revision, "coordinates": COORDINATES,
        "source": prior_map.source_format, "capabilities": {
            "object_positions": any(o.position_xyz is not None for o in prior_map.objects),
            "visual_evidence": bool(observations), "rooms": bool(prior_map.rooms),
            "observation_poses": any(v["kind"] == "observation_pose" for v in viewpoints),
            "navigation_viewpoints": any(v["kind"] == "navigation_viewpoint" for v in viewpoints),
            "visual_features": False},
        "artifacts": ["prior_map.json", "observations.jsonl", "viewpoints.json"]})
    return load_bundle(root)
