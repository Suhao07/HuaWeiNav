"""Offline GT audit of accepted predictions; outputs never enter navigation.

Re-render semantic IDs at the saved verifier pose, first checking RGB-D alignment.
Distinguish accepted legal targets outside the goal-viewpoint radius from support
on other instances. Mixed/unannotated evidence remains unresolved.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import quaternion
import habitat_sim
from PIL import Image
from config_utils import hm3d_config
from mapping_utils.preprocess import ori_preprocess_depth, preprocess_depth


def audit(batch, output, indices=None, surface_replay=False):
    """Compare saved visible support with annotated instances and official goals.

    Mapper coordinates use P(x,y,z)=(x,z,y) relative to the initial RGB camera.
    Recover world pose as t_world=P*t_mapper+t_camera0, R_world=P*R_mapper.
    Membership uses the exact saved projected pixels, not box overlap alone.
    """
    batch, output = Path(batch), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dataset = json.load(gzip.open(batch/'dataset/tasks.json.gz', 'rt'))
    cfg = hm3d_config(stage='val', episodes=200, dataset_path=str(batch/'dataset/tasks.json.gz'))
    camera_offset = np.array(cfg.SIMULATOR.RGB_SENSOR.POSITION)
    scene_config = cfg.SIMULATOR.SCENE_DATASET
    swap = np.array([[1,0,0],[0,0,1],[0,1,0]])
    rows = []
    detector = None
    if surface_replay:
        from cv_utils.sam import MMDINO_Grounded_SAM
        from navigation_core.surface_evidence import measure_instance_surface
        detector = MMDINO_Grounded_SAM(classes=[], device="cuda")
    for file in sorted((batch/'receipts').glob('ep-*.json')):
        receipt = json.loads(file.read_text())
        if indices is not None and receipt['index'] not in indices:
            continue
        if receipt['status'] != 'completed':
            if indices is not None:
                raise ValueError(f"Requested episode {receipt['index']} is {receipt['status']}")
            continue
        metrics = receipt['metrics']
        if indices is None and (metrics.get('instruction_success') != 'True' or float(metrics['success']) != 0):
            continue
        idx = receipt['index']; identity = receipt['identity']; step = int(metrics['instruction_accept_step'])
        root = batch/f'results/ep-{idx:03d}'/f'episode-{idx}'
        evidence = json.loads((root/f'final_verifier/evidence_{step}.json').read_text())['evidence']
        projection = root/'final_verifier'/Path(evidence['geometry']['projection_evidence_path']).name
        saved = np.load(projection, allow_pickle=False)
        depth = saved['depth_m'].squeeze(); h,w=depth.shape
        k = saved['camera_intrinsic']; hfov = np.degrees(2*np.arctan(w/(2*k[0,0])))
        simcfg = habitat_sim.SimulatorConfiguration();simcfg.scene_id=identity['scene_id']
        simcfg.scene_dataset_config_file=scene_config;simcfg.enable_physics=False
        sensors=[]
        for name,kind in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH),('semantic',habitat_sim.SensorType.SEMANTIC)]:
            spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind
            spec.resolution=[h,w];spec.hfov=float(hfov);spec.position=[0,0,0];sensors.append(spec)
        agent=habitat_sim.agent.AgentConfiguration();agent.sensor_specifications=sensors
        sim=habitat_sim.Simulator(habitat_sim.Configuration(simcfg,[agent]))
        try:
            state=habitat_sim.AgentState();state.position=swap@saved['camera_to_mapper_translation']+np.array(identity['start_position'])+camera_offset
            state.rotation=quaternion.from_rotation_matrix(swap@saved['camera_to_mapper_rotation'])
            sim.get_agent(0).set_state(state); rendered=sim.get_sensor_observations()
            valid=(depth>cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH+.01)&(depth<cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH-.01)
            mismatch=float(np.mean(np.abs(rendered['depth'][valid]-depth[valid])>.01))
            rgb_mae=float(np.abs(rendered['rgb'][...,:3].astype(float)-saved['rgb'][...,:3]).mean())
            processed = ori_preprocess_depth(rendered['depth'].copy())
            processed = preprocess_depth(processed)
            processed_mismatch = float(np.mean(np.abs(processed-depth)>.01))
            key=Path(identity['scene_id']).name+'_'+identity['object_category']
            goals=dataset['goals_by_category'][key];goal_ids={int(g['object_id']) for g in goals}
            labels={int(o.id.rsplit('_',1)[1]):o.category.name() for o in sim.semantic_scene.objects if o is not None}
            uv=saved['projected_uv_depth'][:,:2].astype(np.int32)[saved['in_image']][saved['passes_occlusion']]
            ids=rendered['semantic'][uv[:,1],uv[:,0]];counts=Counter(int(v) for v in ids)
            legal=float(np.isin(ids,list(goal_ids)).mean())
            annotated=float(np.array([int(v) in labels for v in ids]).mean())
            points=saved['points_xyz'][saved['visible_point_index']]
            dist=np.linalg.norm(points-saved['camera_to_mapper_translation'],axis=1)
            nearest=int(np.argmin(dist));nearest_id=int(ids[nearest])
            visible_z=saved['projected_uv_depth'][:,2][saved['in_image']][saved['passes_occlusion']]
            surface_delta=visible_z-rendered['depth'][uv[:,1],uv[:,0]]
            surface_aligned=np.abs(surface_delta)<=.2
            box=np.array(evidence['geometry']['bbox_xyxy'],dtype=int)
            box_ids=rendered['semantic'][box[1]:box[3]+1,box[0]:box[2]+1]
            body=state.position-camera_offset
            query=habitat_sim.MultiGoalShortestPath();query.requested_start=body
            query.requested_ends=np.array([v['agent_state']['position'] for g in goals for v in g['view_points']],dtype=np.float32)
            sim.pathfinder.find_path(query)
            # RGB同像素对齐用于确认位姿；保存深度经过建图器原地修改，另报其差异。
            if rgb_mae>.1:
                category='unaligned_replay'
            elif legal>.5:
                category=('legal_target_support_stop_outside_viewpoint_radius'
                          if float(metrics['distance_to_goal']) > float(metrics['benchmark_success_distance'])
                          else 'legal_target_support_within_stop_radius')
            elif annotated-legal>.5:
                category='other_instance_support_accepted'
            else:
                category='mixed_or_unannotated_support'
            row=dict(index=idx,object_category=identity['object_category'],classification=category,
                official_distance=float(metrics['distance_to_goal']),replayed_geodesic=float(query.geodesic_distance),
                accepted_distance=float(metrics['accepted_distance_to_target']),depth_mismatch_fraction=mismatch,rgb_mae=rgb_mae,
                processed_depth_mismatch_fraction=processed_mismatch,
                legal_goal_labels={str(i):labels.get(i) for i in sorted(goal_ids)},
                bbox_legal_fraction=float(np.isin(box_ids,list(goal_ids)).mean()),
                support_on_live_surface_fraction=float(surface_aligned.mean()),
                nearest_support_surface_delta_m=float(surface_delta[nearest]),
                legal_goal_ids=sorted(goal_ids),legal_support_fraction=legal,annotated_fraction=annotated,
                nearest_support_id=nearest_id,nearest_support_label=labels.get(nearest_id),nearest_support_is_legal=nearest_id in goal_ids,
                support_histogram=[dict(id=i,label=labels.get(i),count=n,is_legal=i in goal_ids) for i,n in counts.most_common(8)],
                evidence_path=str(projection),camera_world_position=state.position.tolist())
            if surface_replay:
                masks = detector.segment_current_target(saved['rgb'][..., :3], identity['object_category'].replace('_', ' '))
                facts, region, support = measure_instance_surface(
                    saved['projected_uv_depth'][saved['in_image']], rendered['depth'].squeeze(), masks, k,
                    saved['camera_to_mapper_rotation'], saved['camera_to_mapper_translation'],
                    saved['camera_to_mapper_translation'] - swap@camera_offset)
                row['surface_replay'] = facts
                row['baseline_success'] = float(metrics['success'])
                np.savez_compressed(output/f'ep-{idx:03d}-surface.npz', masks=masks, selected_mask=region,
                                    support_indices=support, raw_depth=rendered['depth'])
            Image.fromarray(saved['rgb'][...,:3]).save(output/f'ep-{idx:03d}.png')
            np.savez_compressed(output/f'ep-{idx:03d}-oracle.npz', semantic=rendered['semantic'], depth=rendered['depth'])
            rows.append(row); (output/'audit.json').write_text(json.dumps(dict(scope='oracle_only_saved_acceptance',rows=rows,counts=dict(Counter(r['classification'] for r in rows))),indent=2))
            print(json.dumps(row),flush=True)
        finally:sim.close()
    return rows


def replay_surface_cache(batch, cache, output):
    """Re-evaluate geometry on frozen masks/depth, without another detector request."""
    from navigation_core.surface_evidence import measure_instance_surface
    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = hm3d_config(stage='val', episodes=200, dataset_path=str(Path(batch)/'dataset/tasks.json.gz'))
    camera_offset = np.asarray(cfg.SIMULATOR.RGB_SENSOR.POSITION)[[0, 2, 1]]
    rows = []
    for row in json.loads((cache/'audit.json').read_text())['rows']:
        idx = row['index']
        saved = np.load(row['evidence_path'])
        current = np.load(cache/f'ep-{idx:03d}-surface.npz')
        facts, mask, support = measure_instance_surface(
            saved['projected_uv_depth'][saved['in_image']], current['raw_depth'].squeeze(),
            current['masks'], saved['camera_intrinsic'], saved['camera_to_mapper_rotation'],
            saved['camera_to_mapper_translation'], saved['camera_to_mapper_translation']-camera_offset)
        rows.append(dict(index=idx, baseline_success=row['baseline_success'],
                         baseline_distance=row['accepted_distance'], surface=facts))
        np.savez_compressed(output/f'ep-{idx:03d}-selection.npz', selected_mask=mask, support_indices=support)
    (output/'results.json').write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--surface-replay', action='store_true')
    parser.add_argument('--surface-cache')
    args = parser.parse_args()
    if args.surface_cache:
        replay_surface_cache(args.batch, args.surface_cache, args.output)
    else:
        audit(args.batch, args.output, args.indices, args.surface_replay)
