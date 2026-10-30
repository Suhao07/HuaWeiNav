"""Oracle-only evaluation of saved predictions; never used by map construction."""
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .observations import load_depth
from .bundle import load_bundle
from .observations import write_json
from .geometry_diagnostic import depth_edge_mask


def summarize_semantic_evidence(rows):
    """Count identity conflicts using observations with majority annotated support.

    Each accepted vote requires >50% of the full mask to belong to one annotated
    instance. Distinct votes within one predicted UID flag a merge for review.
    Annotation granularity and detector part/whole semantics still matter; this
    is not an open-vocabulary accuracy metric or a navigation success measure.
    """
    votes = defaultdict(Counter)
    confusion = Counter()
    for row in rows:
        if row['dominant_fraction_full_mask'] > .5:
            for uid in row['object_uids']:
                votes[uid][row['dominant_gt_id']] += 1
            confusion[(row['detector_label'], row['dominant_gt_label'])] += 1
    return {
        'strong_mask_count': sum(confusion.values()),
        'predicted_instances_with_strong_gt_support': len(votes),
        'predicted_instances_with_conflicting_gt_votes': sum(len(v) > 1 for v in votes.values()),
        'instance_votes': {uid: dict(v) for uid, v in votes.items()},
        'label_pairs': [{'predicted': pred, 'annotated': gt, 'mask_count': count}
                        for (pred, gt), count in confusion.most_common()],
    }


def audit_habitat_instances(bundle_directory, scene_dataset_config, output, semantic_cache=None, scene=None):
    """Render annotation IDs at stored camera poses, checking depth alignment.

    Args:
        bundle_directory: Predicted RGB-D map, kept read-only.
        scene_dataset_config: Annotated Habitat scene dataset configuration.
        output: Separate diagnostic JSON path; never a map input.
        semantic_cache: Optional new directory for oracle-only evaluation arrays.
        scene: Explicit local scene path for a relocated capture; SHA256 must match.

    Returns:
        Diagnostic report containing every mask's annotated support.
    Raises:
        ValueError: Rendered depth disagrees with saved evidence on >0.1% of
            pixels by more than 1 cm. The report still records the discrepancy.
    """
    root = Path(bundle_directory)
    bundle = load_bundle(root)
    if not bundle['observations']:
        write_json(output, {'scope': 'oracle_diagnostic_only',
            'status': 'insufficient_evidence', 'alignment_valid': None,
            'frame_count': 0, 'mask_count': 0, 'summary': None,
            'reason': 'Map contains no observations to audit'})
        raise ValueError('Semantic audit requires observed evidence; map is empty')
    import habitat_sim
    import quaternion
    from backends.habitat.capture import WORLD_TO_MAP, OPTICAL_TO_GL
    cache = Path(semantic_cache) if semantic_cache is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=False)
    cache_frames = {}
    receipt = json.loads((root / 'build_receipt.json').read_text())
    capture = receipt['capture_manifest']
    scene_path = Path(scene if scene is not None else capture['scene_path'])
    if hashlib.sha256(scene_path.read_bytes()).hexdigest() != capture['scene_sha256']:
        raise ValueError('Audit scene hash differs from captured scene')
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene_path)
    cfg.scene_dataset_config_file = str(scene_dataset_config)
    cfg.enable_physics = False
    sensors = []
    # HM3D 的语义 ID 编码在纹理中；保留颜色传感器以启用所需纹理加载。
    for name, kind in [('rgb', habitat_sim.SensorType.COLOR),
                       ('depth', habitat_sim.SensorType.DEPTH),
                       ('semantic', habitat_sim.SensorType.SEMANTIC)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, kind
        spec.resolution = [capture['config']['height'], capture['config']['width']]
        spec.hfov = capture['config']['hfov']
        spec.position = [0., 0., 0.]
        sensors.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    rows, alignment = [], []
    by_frame = defaultdict(list)
    for obs in bundle['observations']:
        by_frame[obs['frame_id']].append(obs)
    try:
        labels = {int(obj.id.rsplit('_', 1)[1]): obj.category.name()
                  for obj in sim.semantic_scene.objects if obj is not None}
        for frame_id, observations in by_frame.items():
            pose = np.array(observations[0]['camera_to_map'])
            state = habitat_sim.AgentState()
            state.position = WORLD_TO_MAP.T @ pose[:3, 3]
            state.rotation = quaternion.from_rotation_matrix(
                WORLD_TO_MAP.T @ pose[:3, :3] @ OPTICAL_TO_GL)
            sim.get_agent(0).set_state(state)
            rendered = sim.get_sensor_observations()
            if cache is not None:
                destination = cache / f'{frame_id:06d}.npy'
                np.save(destination, rendered['semantic'], allow_pickle=False)
                cache_frames[str(frame_id)] = {
                    'file': destination.name,
                    'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'image_sha256': hashlib.sha256((root / observations[0]['image_ref']).read_bytes()).hexdigest()}

            depth = load_depth(root / observations[0]['depth_ref'])
            expected_depth = rendered['depth']
            profile = capture.get('capture_profile')
            if profile is not None:
                # 对照传感器实际输出函数 D=clip(Z,z_min,z_max)，不把截断
                # 端点与原始几何的差异错误归因为位姿错位。
                expected_depth = np.clip(expected_depth, profile['depth_min_m'], profile['depth_max_m'])
            error = np.abs(expected_depth - depth)
            edge = depth_edge_mask(expected_depth, .1) | depth_edge_mask(depth, .1)
            alignment.append({'frame_id': frame_id, 'max_abs_depth_error_m': float(error.max()),
                              'fraction_over_1cm': float((error > .01).mean()),
                              'over_1cm_near_depth_edge': int(((error > .01) & edge).sum()),
                              'over_1cm_away_from_depth_edge': int(((error > .01) & ~edge).sum())})
            for obs in observations:
                mask = np.asarray(Image.open(root / obs['mask_ref'])) > 0
                ids, counts = np.unique(rendered['semantic'][mask], return_counts=True)
                histogram = {int(i): int(n) for i, n in zip(ids, counts)}
                # ID 0 在该 HM3D 标注中为 Unknown，计入完整 mask 分母。
                annotated = {i: n for i, n in histogram.items() if i != 0 and i in labels}
                gt_id = max(annotated, key=annotated.get) if annotated else 0
                x1, y1, x2, y2 = obs['bbox_xyxy']
                x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
                x2, y2 = min(mask.shape[1], int(np.ceil(x2))), min(mask.shape[0], int(np.ceil(y2)))
                box_ids, box_counts = np.unique(rendered['semantic'][y1:y2, x1:x2], return_counts=True)
                box_histogram = {int(i): int(n) for i, n in zip(box_ids, box_counts)}
                rows.append({'observation_id': obs['observation_id'], 'frame_id': frame_id,
                    'object_uids': obs['object_uids'], 'detector_label': obs['detector_label'],
                    'dominant_gt_id': gt_id, 'dominant_gt_label': labels.get(gt_id, 'Unknown'),
                    'dominant_fraction_full_mask': annotated.get(gt_id, 0) / int(mask.sum()),
                    'pixel_histogram': histogram, 'bbox_pixel_histogram': box_histogram})
    finally:
        sim.close()
    valid = all(row['fraction_over_1cm'] <= .001 for row in alignment)
    scene_name = scene_path.name.removesuffix('.basis.glb')
    assets = {'scene_dataset_config': Path(scene_dataset_config),
              'semantic_mesh': scene_path.with_name(scene_name + '.semantic.glb'),
              'semantic_labels': scene_path.with_name(scene_name + '.semantic.txt'),
              'predicted_map': root / 'prior_map.json', 'evaluator': Path(__file__)}
    provenance = {}
    for name, path in assets.items():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        provenance[name] = {'path': str(path), 'sha256': digest.hexdigest()}
    report = {'scope': 'oracle_diagnostic_only', 'semantic_gt_used': True,
        'depth_comparison': 'recorded_sensor_clipping' if capture.get('capture_profile') else 'raw_rendered_depth',
        'predicted_map_modified': False, 'alignment_valid': valid,
        'annotation_labels': labels, 'provenance': provenance,
        'frame_count': len(by_frame), 'mask_count': len(rows), 'depth_alignment': alignment,
        'observations': rows, 'summary': summarize_semantic_evidence(rows) if valid else None}
    if cache is not None:
        write_json(cache / 'manifest.json', {
            'scope': 'oracle_diagnostic_only', 'alignment_valid': valid,
            'annotation_labels': labels, 'frames': cache_frames, 'provenance': provenance})
    write_json(output, report)
    if not valid:
        raise ValueError('Semantic audit camera/depth alignment failed; inspect diagnostic report')
    return report
