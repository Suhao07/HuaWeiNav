#!/usr/bin/env python3
"""Compare HM3D rendered depth with oriented intersections of its source mesh.

This is an offline asset diagnostic. Mesh rays never enter predicted mapping.
The supplied HM3D GLB must use the same Z-up coordinates as camera_to_map;
sampled front-face depth agreement is reported to test that convention.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import open3d as o3d
from prior_map.observations import load_depth, read_records, write_json


def diagnose(capture, mesh_path, output):
    """Record nearest two-sided and front-facing hits for fixed image probes.

    Optical ray d = R K^-1 [u,v,1] uses z=1, so intersection t is optical
    depth in metres. For triangle winding normal n, n dot d < 0 is front-facing.
    Rays without an oriented intersection remain explicitly unobserved.
    """
    root, mesh_path = Path(capture), Path(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays, records = [], []
    for frame in read_records(root / 'frames.jsonl'):
        depth = load_depth(root / frame['depth_path'])
        pose, k = np.asarray(frame['camera_to_map']), np.asarray(frame['intrinsics'])
        for y in np.linspace(16, depth.shape[0]-17, 5).astype(int):
            for x in np.linspace(16, depth.shape[1]-17, 5).astype(int):
                direction = pose[:3, :3] @ np.linalg.inv(k) @ [x, y, 1]
                rays.append(np.r_[pose[:3, 3], direction])
                records.append({'frame_id': frame['frame_id'], 'uv': [int(x), int(y)],
                    'rendered_depth_m': float(depth[y, x])})
    rays = np.asarray(rays, dtype=np.float32)
    hits = scene.list_intersections(o3d.core.Tensor(rays))
    indices, times, faces = [hits[key].numpy() for key in ('ray_ids', 't_hit', 'primitive_ids')]
    dots = np.einsum('ij,ij->i', normals[faces], rays[indices, 3:])
    for i, record in enumerate(records):
        values, orientation = times[indices == i], dots[indices == i]
        front = values[orientation < 0]
        record.update(two_sided_depth_m=float(values.min()) if len(values) else None,
                      front_depth_m=float(front.min()) if len(front) else None)
    comparable = [r for r in records if r['front_depth_m'] is not None and r['rendered_depth_m'] > 0]
    matches = [r for r in comparable if abs(r['front_depth_m']-r['rendered_depth_m']) < .01]
    backface = [r for r in matches if r['two_sided_depth_m'] < r['rendered_depth_m']-.05]
    result = {'scope': 'source_mesh_visibility_diagnostic_not_predicted_map',
        'mesh_sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        'coordinate_hypothesis': 'source_HM3D_GLB_equals_map_z_up',
        'sampled_ray_count': len(records), 'comparable_ray_count': len(comparable),
        'front_hit_matches_within_1cm': len(matches),
        'nearer_backface_with_matching_front_hit': len(backface), 'rays': records}
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture')
    parser.add_argument('mesh')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = diagnose(args.capture, args.mesh, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'rays'}))
