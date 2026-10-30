#!/usr/bin/env python3
"""Check fresh arrival evidence with the existing final verifier and reducer.

This diagnostic consumes one current-frame detection bundle. It does not match
historical and current instance identities or send STOP to the simulator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instruction.contracts import TaskSpec
from instruction_adapter.task_compat import runtime_plan
from instruction_adapter.verifier import FinalInstructionVerifier
from prior_map.observations import load_depth
from prior_map.bundle import load_bundle
from prior_map.observations import backproject, write_json
from task.state import TaskProgress
from task.reducer import apply_verification
from verification.contracts import CandidateInstance


def main():
    """Bind verifier decisions to current masks, depth and compiled task revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('current_bundle', 'compiled_query', 'motion', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--required-stop-distance', type=float, required=True)
    parser.add_argument('--vlm', default='ark')
    parser.add_argument('--prepare-only', action='store_true', help='Save reviewable inputs without a model request')
    args = parser.parse_args()
    bundle = load_bundle(args.current_bundle)
    build = json.loads((args.current_bundle / 'build_receipt.json').read_text())
    capture = build['capture_manifest']
    motion_bytes = (args.motion / 'receipt.json').read_bytes()
    if hashlib.sha256(motion_bytes).hexdigest() != capture['source_motion_receipt_sha256']:
        raise ValueError('Current evidence belongs to a different motion receipt')
    if build['frame_count'] != 1:
        raise ValueError('Arrival diagnostic requires one current frame')
    task = TaskSpec.from_dict(json.loads(args.compiled_query.read_text())['task_spec'])
    terminals = [target for target in task.targets if target.terminal]
    if len(terminals) != 1 or terminals[0].min_count != 1 or task.constraints:
        raise ValueError('This arrival diagnostic supports one unconstrained object target')
    progress = TaskProgress(task)
    target = progress.active_target()
    plan = runtime_plan(task)
    args.output.mkdir(parents=True, exist_ok=False)
    verifier = FinalInstructionVerifier(args.vlm, str(args.output / 'image_evidence'))
    objects = {o.uid: o for o in bundle['prior_map'].objects}
    records = []
    for obs in bundle['observations']:
        uid = obs['object_uids'][0]
        obj = objects[uid]
        image = Image.open(args.current_bundle / obs['image_ref']).convert('RGB')
        ImageDraw.Draw(image).rectangle(obs['bbox_xyxy'], outline='red', width=2)
        image_path = (args.output / f"candidate-{len(records)}.png").resolve()
        image.save(image_path)
        mask = np.asarray(Image.open(args.current_bundle / obs['mask_ref'])) > 0
        depth = load_depth(args.current_bundle / obs['depth_ref'])
        pose = np.asarray(obs['camera_to_map'])
        points = backproject(np.where(mask & (depth <= build['max_depth_m']), depth, 0),
                             np.asarray(obs['intrinsics']), pose)
        # 与 baseline 相同的平面最近点距离；支持区域来自当前 mask 的深度。
        distance = float(np.linalg.norm(points[:, :2] - pose[:2, 3], axis=1).min())
        candidate = CandidateInstance(uid, obs['detector_label'], centroid=list(obj.position_xyz),
            bbox_2d=obs['bbox_xyxy'], confidence=obs['detector_score'], step=obs['frame_id'])
        evidence = dict(target_id=target.id, current_rgb_with_bbox_path=str(image_path),
            observation_identity=dict(frame_id=obs['frame_id'], candidate_uid=uid,
                                      image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()),
            geometry=dict(bbox_xyxy=obs['bbox_xyxy'], distance_to_object=distance,
                required_stop_distance=args.required_stop_distance, visible_projected_points=len(points),
                distance_source='current_mask_depth_planar_nearest_point',
                projection_failed_in_final_view=False))
        if args.prepare_only:
            records.append(dict(candidate=candidate.as_dict(), evidence=evidence))
            write_json(args.output / 'prepared_inputs.json', dict(task=task.as_dict(),
                records=records, verification_performed=False, task_stop_sent=False))
            continue
        result = verifier.verify(task.raw_instruction, plan, candidate, evidence)
        apply_verification(progress, target, candidate, result, evidence)
        row = dict(candidate=candidate.as_dict(), evidence=evidence, result=result.as_dict())
        records.append(row)
        write_json(args.output / 'verification.json', dict(
            scope='current_evidence_verifier_reducer_diagnostic_not_episode_success',
            task=task.as_dict(), records=records, progress=progress.as_dict(), task_stop_sent=False))
    print(json.dumps(dict(candidate_count=len(records), progress=progress.as_dict()), ensure_ascii=False))


if __name__ == '__main__':
    main()
