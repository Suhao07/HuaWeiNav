#!/usr/bin/env python3
"""Compare captured depth with two-sided mesh ray intersections, offline only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prior_map.observations import read_records, write_json


def main():
    """Use unnormalized camera rays so intersection t is optical-axis depth.

    d_world = R_world_camera K^-1 [u,v,1]; the camera direction has z=1,
    hence ray(origin + t*d_world) reports optical depth t, not radial range.
    Geometry diagnostics stay separate from prediction and navigation inputs.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture')
    p.add_argument('conflicts')
    p.add_argument('mesh')
    p.add_argument('--output', required=True)
    p.add_argument('--asset-up', nargs=3, type=float, required=True)
    p.add_argument('--asset-front', nargs=3, type=float, required=True)
    a = p.parse_args()
    root, mesh_path = Path(a.capture), Path(a.mesh)
    manifest = json.loads((root / 'manifest.json').read_text())
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if digest != manifest['scene_sha256']:
        raise ValueError('Diagnostic mesh differs from captured scene')
    frames = {r['frame_id']: r for r in read_records(root / 'frames.jsonl')}
    conflicts = json.loads(Path(a.conflicts).read_text())['conflicts']
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError('Scene mesh contains no triangles')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # Habitat maps asset up/front to world +Y/-Z, hence canonical map +Z/+Y.
    up, front = np.asarray(a.asset_up), np.asarray(a.asset_front)
    asset_from_map = np.column_stack([np.cross(front, up), front, up])
    if not np.allclose(asset_from_map.T @ asset_from_map, np.eye(3)):
        raise ValueError('Asset orientation must be orthonormal')
    rays, samples = [], []
    for c in conflicts:
        for side in ('source', 'target', 'target_pixel'):
            frame_side = 'target' if side == 'target_pixel' else side
            frame = frames[c[frame_side]]
            uv = np.asarray(c[frame_side + '_uv'])
            if side == 'target_pixel':
                uv = np.rint(uv)
            pose = np.asarray(frame['camera_to_map'])
            origin = asset_from_map @ pose[:3, 3]
            direction = asset_from_map @ pose[:3, :3] @ np.linalg.inv(frame['intrinsics']) @ [*uv, 1.]
            rays.append(np.r_[origin, direction])
            depth = np.load(root / frame['depth_path'], allow_pickle=False)
            x, y = np.rint(uv).astype(int)
            samples.append(float(depth[y, x]))
    result = scene.cast_rays(o3d.core.Tensor(np.asarray(rays, dtype=np.float32)))
    hit = result['t_hit'].numpy()
    normals = result['primitive_normals'].numpy()
    rows = []
    for index, c in enumerate(conflicts):
        row = dict(c)
        for offset, side in enumerate(('source', 'target', 'target_pixel')):
            i = 3 * index + offset
            row[side + '_captured_nearest_depth_m'] = samples[i]
            row[side + '_mesh_ray_depth_m'] = float(hit[i]) if np.isfinite(hit[i]) else None
            row[side + '_normal_dot_ray'] = float(normals[i] @ rays[i][3:])
        rows.append(row)
    write_json(a.output, {'scope': 'offline_scene_geometry_diagnostic_only',
        'scene_sha256': digest, 'triangle_count': len(mesh.triangles),
        'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'capture_manifest_sha256': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
        'conflicts_sha256': hashlib.sha256(Path(a.conflicts).read_bytes()).hexdigest(),
        'asset_up': a.asset_up, 'asset_front': a.asset_front, 'asset_scale': [1,1,1],
        'ray_method': 'two_sided_triangle_intersection_optical_z', 'conflicts': rows})
    print(f'Saved {len(rows)} conflict ray checks; {len(mesh.triangles)} triangles')


if __name__ == '__main__':
    main()
