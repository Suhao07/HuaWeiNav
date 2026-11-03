"""Apply existing final verification to motion-bound current evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from instruction.contracts import TaskSpec
from instruction_adapter.task_compat import runtime_plan
from prior_map.observations import load_depth, metric_depth_validity
from prior_map.bundle import load_bundle
from prior_map.observations import backproject, write_json
from task.state import TaskProgress
from task.reducer import apply_verification
from verification.contracts import CandidateInstance



def verify_arrival(current_bundle, task, motion, output, *, required_stop_distance,
                   candidate_uid=None, observation_id=None, vlm='ark', prepare_only=False, verifier=None):
    """Prepare or verify one unconstrained target using current mask geometry.

    Args:
        current_bundle: Single-frame current perception bundle.
        task: Immutable TaskSpec for this verification.
        motion: Execution directory whose terminal receipt is bound to evidence.
        output: New directory for inputs, model decisions and reducer progress.
        required_stop_distance: Current planar stop-distance requirement, metres.
        candidate_uid: Optional explicit current candidate to verify.
        observation_id: Optional raw observation bound to the current candidate.
        vlm: Provider used by the existing FinalInstructionVerifier.
        prepare_only: Save inputs without requesting a model decision.
        verifier: Injected verifier for deterministic protocol tests.
    Returns:
        Evidence records and progress from the existing domain reducer.
    Raises:
        ValueError: Task scope, motion binding or current candidate is invalid.

    This function advances task progress; the caller owns motion and STOP.
    """
    current_bundle, motion, output = map(Path, (current_bundle, motion, output))
    bundle = load_bundle(current_bundle)
    build = json.loads((current_bundle / 'build_receipt.json').read_text())
    capture = build['capture_manifest']
    motion_bytes = (motion / 'receipt.json').read_bytes()
    if hashlib.sha256(motion_bytes).hexdigest() != capture['source_motion_receipt_sha256']:
        raise ValueError('Current evidence belongs to a different motion receipt')
    if build['frame_count'] != 1:
        raise ValueError('Arrival diagnostic requires one current frame')
    terminals = [target for target in task.targets if target.terminal]
    if len(task.targets) != 1 or len(terminals) != 1 or terminals[0].min_count != 1 or task.constraints:
        raise ValueError('This arrival diagnostic supports one unconstrained object target')
    progress = TaskProgress(task)
    target = progress.active_target()
    plan = runtime_plan(task)
    output.mkdir(parents=True, exist_ok=False)
    if not prepare_only and verifier is None:
        from instruction_adapter.verifier import FinalInstructionVerifier
        verifier = FinalInstructionVerifier(vlm, str(output / 'image_evidence'))
    records = []
    observations = [obs for obs in bundle['observations']
                    if (candidate_uid is None or candidate_uid in obs['object_uids'])
                    and (observation_id is None or obs['observation_id'] == observation_id)]
    if (candidate_uid is not None or observation_id is not None) and not observations:
        raise ValueError('Selected current candidate has no observations')
    for obs in observations:
        uid = obs['object_uids'][0]
        image = Image.open(current_bundle / obs['image_ref']).convert('RGB')
        ImageDraw.Draw(image).rectangle(obs['bbox_xyxy'], outline='red', width=2)
        image_path = (output / f"candidate-{len(records)}.png").resolve()
        image.save(image_path)
        mask = np.asarray(Image.open(current_bundle / obs['mask_ref'])) > 0
        depth = load_depth(current_bundle / obs['depth_ref'])
        pose = np.asarray(obs['camera_to_map'])
        valid = metric_depth_validity(depth, capture.get('capture_profile')) & (depth <= build['max_depth_m'])
        points = backproject(np.where(mask & valid, depth, 0),
                             np.asarray(obs['intrinsics']), pose)
        if len(points) == 0:
            raise ValueError('Current mask has no valid depth support')
        # 沿用建图的量程规则：裁剪端点不是测得的物体表面。
        # 与 baseline 相同的平面最近点距离；支持区域来自当前 mask 的有效深度。
        distance = float(np.linalg.norm(points[:, :2] - pose[:2, 3], axis=1).min())
        candidate = CandidateInstance(uid, obs['detector_label'], centroid=points.mean(axis=0).tolist(),
            bbox_2d=obs['bbox_xyxy'], confidence=obs['detector_score'], step=obs['frame_id'])
        evidence = dict(target_id=target.id, current_rgb_with_bbox_path=str(image_path),
            observation_identity=dict(observation_id=obs['observation_id'], frame_id=obs['frame_id'], candidate_uid=uid,
                                      mask_sha256=hashlib.sha256((current_bundle / obs['mask_ref']).read_bytes()).hexdigest(),
                                      depth_sha256=hashlib.sha256((current_bundle / obs['depth_ref']).read_bytes()).hexdigest(),
                                      image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()),
            geometry=dict(bbox_xyxy=obs['bbox_xyxy'], distance_to_object=distance,
                required_stop_distance=required_stop_distance, visible_projected_points=len(points),
                distance_source='current_mask_depth_planar_nearest_point',
                projection_failed_in_final_view=False))
        if prepare_only:
            records.append(dict(candidate=candidate.as_dict(), evidence=evidence))
            write_json(output / 'prepared_inputs.json', dict(task=task.as_dict(),
                records=records, verification_performed=False, task_stop_sent=False))
            continue
        result = verifier.verify(task.raw_instruction, plan, candidate, evidence)
        apply_verification(progress, target, candidate, result, evidence)
        row = dict(candidate=candidate.as_dict(), evidence=evidence, result=result.as_dict())
        records.append(row)
        write_json(output / 'verification.json', dict(
            scope='current_evidence_verifier_reducer_diagnostic_not_episode_success',
            task=task.as_dict(), records=records, progress=progress.as_dict(), task_stop_sent=False))
    report = dict(candidate_count=len(records), progress=progress.as_dict(), records=records,
                  verification_performed=not prepare_only, task_stop_sent=False)
    write_json(output / 'receipt.json', report)
    return report
