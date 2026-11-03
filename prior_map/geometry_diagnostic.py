"""Cross-view metric-depth reprojection independent of semantic predictions."""
from pathlib import Path
import numpy as np
from .observations import read_records, backproject, write_json


def depth_edge_mask(depth, threshold_m):
    """Mark 3x3 depth discontinuities for stratification, preserving all samples."""
    padded = np.pad(depth, 1, mode="edge")
    neighbors = [padded[y:y + depth.shape[0], x:x + depth.shape[1]]
                 for y in range(3) for x in range(3)]
    return np.maximum.reduce(neighbors) - np.minimum.reduce(neighbors) > threshold_m


def diagnose_capture(directory, *, stride=8, tolerance_m=0.05):
    """Compare overlapping rendered depths; occluded points are reported separately.

    For source points transformed into target optical coordinates, compare z
    against the target depth at the projected pixel. Positive residuals beyond
    tolerance are behind the target surface and therefore occlusion candidates.
    This check diagnoses consistency, not absolute physical calibration.
    """
    if stride < 1 or tolerance_m <= 0:
        raise ValueError("Stride and tolerance must be positive")
    root = Path(directory)
    frames = read_records(root / "frames.jsonl")
    reports = []
    depths = [np.load(root / f["depth_path"], allow_pickle=False) for f in frames]
    edges = [depth_edge_mask(d, tolerance_m * 2) for d in depths]
    for i, source in enumerate(frames):
        sd = depths[i]
        sampled = sd[::stride, ::stride]
        source_edges = edges[i][::stride, ::stride][np.isfinite(sampled) & (sampled > 0)]
        world = backproject(sd, np.array(source["intrinsics"]), np.array(source["camera_to_map"]), stride)
        for j in range(i + 1, len(frames)):
            target, td = frames[j], depths[j]
            t = np.linalg.inv(np.array(target["camera_to_map"]))
            camera = world @ t[:3, :3].T + t[:3, 3]
            forward = camera[:, 2] > 0
            source_edge = source_edges[forward]
            camera = camera[forward]
            if len(camera) == 0:
                continue
            projected = camera @ np.array(target["intrinsics"]).T
            uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < td.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < td.shape[0])
            uv, camera = uv[inside], camera[inside]
            source_edge = source_edge[inside]
            target_z = td[uv[:, 1], uv[:, 0]]
            valid = target_z > 0
            error = camera[valid, 2] - target_z[valid]
            if len(error) == 0:
                continue
            edge = source_edge[valid] | edges[j][uv[valid, 1], uv[valid, 0]]
            conflict = error < -tolerance_m
            visible = error <= tolerance_m
            residual = np.abs(error[visible])
            reports.append({"source": source["frame_id"], "target": target["frame_id"],
                            "projected_count": len(error), "occlusion_candidates": int((~visible).sum()),
                            "consistent_fraction_all_projected": float((np.abs(error) <= tolerance_m).mean()),
                            "front_conflict_count": int(conflict.sum()),
                            "depth_edge_count": int(edge.sum()),
                            "front_conflict_near_depth_edge": int((conflict & edge).sum()),
                            "front_conflict_away_from_depth_edge": int((conflict & ~edge).sum()),
                            "visible_count": int(visible.sum()),
                            "visible_median_abs_m": float(np.median(residual)) if len(residual) else None,
                            "visible_p95_abs_m": float(np.quantile(residual, .95)) if len(residual) else None})
    result = {"frame_count": len(frames), "stride": stride, "tolerance_m": tolerance_m,
              "scope": "cross_view_consistency_not_absolute_calibration", "pairs": reports}
    write_json(root / "geometry_diagnostic.json", result)
    return result


def interpolate_inverse_depth(depth, uv):
    """Sample optical depth at subpixel coordinates inside the image interior.

    On a plane, 1/z(u,v) is affine under a pinhole camera. Bilinear inverse-depth
    interpolation therefore isolates nearest-pixel quantization on smooth
    surfaces. It does not resolve occlusions or justify removing discrepancies.
    The caller selects interior pixels with positive neighborhood depths.
    """
    lower = np.floor(uv).astype(int)
    fraction = uv - lower
    x, y = lower[:, 0], lower[:, 1]
    a, b = fraction[:, 0], fraction[:, 1]
    inverse = ((1-a)*(1-b)/depth[y, x] + a*(1-b)/depth[y, x+1]
               + (1-a)*b/depth[y+1, x] + a*b/depth[y+1, x+1])
    return 1 / inverse


def diagnose_interior_conflicts(directory, diagnostic):
    """Retain non-edge conflicts and compare nearest vs subpixel plane sampling.

    Uses exactly the source sampling and target rounding of diagnose_capture.
    Original conflicts remain in the output even if interpolation resolves them.
    Border samples are reported separately; no acceptance threshold is changed.
    """
    import json
    root = Path(directory)
    report = json.loads(Path(diagnostic).read_text())
    frames = {f['frame_id']: f for f in read_records(root / 'frames.jsonl')}
    stride, tolerance = report['stride'], report['tolerance_m']
    depths = {i: np.load(root / f['depth_path'], allow_pickle=False) for i, f in frames.items()}
    edges = {i: depth_edge_mask(d, tolerance * 2) for i, d in depths.items()}
    records = []
    for pair in report['pairs']:
        if pair['front_conflict_away_from_depth_edge'] == 0:
            continue
        i, j = pair['source'], pair['target']
        source, target = frames[i], frames[j]
        sd, td = depths[i], depths[j]
        v, u = np.mgrid[0:sd.shape[0]:stride, 0:sd.shape[1]:stride]
        valid = sd[v, u] > 0
        source_uv = np.column_stack([u[valid], v[valid]])
        source_edge = edges[i][v[valid], u[valid]]
        world = backproject(sd, np.array(source['intrinsics']), np.array(source['camera_to_map']), stride)
        t = np.linalg.inv(np.asarray(target['camera_to_map']))
        camera = world @ t[:3, :3].T + t[:3, 3]
        front = camera[:, 2] > 0
        source_uv, source_edge, camera = source_uv[front], source_edge[front], camera[front]
        projected = camera @ np.asarray(target['intrinsics']).T
        uv = projected[:, :2] / projected[:, 2:3]
        nearest = np.rint(uv).astype(int)
        inside = ((nearest[:, 0] >= 0) & (nearest[:, 0] < td.shape[1]) &
                  (nearest[:, 1] >= 0) & (nearest[:, 1] < td.shape[0]))
        source_uv, source_edge, camera, uv, nearest = [a[inside] for a in (source_uv, source_edge, camera, uv, nearest)]
        z = td[nearest[:, 1], nearest[:, 0]]
        residual = camera[:, 2] - z
        keep = (z > 0) & (residual < -tolerance) & ~source_edge & ~edges[j][nearest[:, 1], nearest[:, 0]]
        for s, q, c, error in zip(source_uv[keep], uv[keep], camera[keep], residual[keep]):
            x, y = np.floor(q).astype(int)
            interior = 0 <= x < td.shape[1]-1 and 0 <= y < td.shape[0]-1
            interpolated_error = None
            if interior and (td[y:y+2, x:x+2] > 0).all():
                interpolated_error = float(c[2] - interpolate_inverse_depth(td, q[None])[0])
            records.append({'source': i, 'target': j, 'source_uv': s.tolist(), 'target_uv': q.tolist(),
                'source_island': source['island_id'], 'target_island': target['island_id'],
                'nearest_residual_m': float(error), 'inverse_depth_residual_m': interpolated_error,
                'interpolation_interior': bool(interior)})
    return {'scope': 'nonedge_conflict_sampling_diagnosis_not_geometry_acceptance',
            'tolerance_m': tolerance, 'stride': stride, 'conflicts': records}
