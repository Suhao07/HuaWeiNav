"""Export live DualMap instances before upstream pickle drops observations."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .bundle import write_bundle, new_map_id
from .contracts import PriorMapData, PriorObject
from .dualmap_input import selected_capture_frames


def export_dualmap_bundle(capture, output, objects, classes, *, scene_id, max_frames=None, frame_stride=1):
    """Preserve upstream instance IDs and their available frame/mask evidence.

    DualMap observation.idx indexes the selected input sequence, while source
    frame_id can be non-contiguous. Resolve through that sequence explicitly.
    Objects with no retained observations stay hypotheses with no visual proof.
    No class, identity, pose or association is inferred during serialization.
    """
    source, root = Path(capture), Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'evidence').mkdir()
    frames = selected_capture_frames(source, max_frames, frame_stride)
    priors, observations, views, copied = [], [], {}, set()
    for obj in objects:
        uid = str(obj.uid)
        points = np.asarray(obj.pcd.points)
        priors.append(PriorObject(uid=uid, label=classes[int(obj.class_id)],
            position_xyz=tuple(map(float, points.mean(axis=0))), exact=True,
            confidence=float(obj.max_prob), source='dualmap_rgbd_fusion',
            metadata={'identity_kind': 'predicted_instance', 'point_count': len(points)}))
        for index, obs in enumerate(obj.observations):
            frame = frames[obs.idx]
            frame_id = frame['frame_id']
            obs_id = f'{uid}:observation{index}'
            image_ref, depth_ref = f'evidence/{frame_id}.png', f'evidence/{frame_id}.npy'
            confidence_ref = None
            if frame.get('confidence_ref') is not None:
                confidence_ref = f'evidence/{frame_id}-confidence.png'
            if frame_id not in copied:
                shutil.copyfile(source / frame['rgb_path'], root / image_ref)
                shutil.copyfile(source / frame['depth_path'], root / depth_ref)
                if confidence_ref is not None:
                    shutil.copyfile(source / frame['confidence_ref'], root / confidence_ref)
                copied.add(frame_id)
            mask_ref = f'evidence/{uid}-{index}.png'
            Image.fromarray(np.asarray(obs.mask, dtype=np.uint8) * 255).save(root / mask_ref)
            observations.append({'observation_id': obs_id, 'object_uids': [uid],
                'frame_id': frame_id, 'image_ref': image_ref, 'depth_ref': depth_ref,
                'mask_ref': mask_ref, 'confidence_ref': confidence_ref,
                'bbox_xyxy': np.asarray(obs.xyxy).tolist(), 'detector_label': classes[int(obs.class_id)],
                'detector_score': float(obs.conf), 'camera_to_map': frame['camera_to_map'],
                'intrinsics': frame['intrinsics']})
            if frame_id not in views:
                views[frame_id] = {'uid': f'view:{frame_id}', 'kind': 'observation_pose',
                    'pose_in_map': frame['camera_to_map'], 'observation_ids': []}
            views[frame_id]['observation_ids'].append(obs_id)
    prior = PriorMapData(scene_id=scene_id, frame_id='map_z_up', objects=tuple(priors),
        source_format='dualmap_predicted_rgbd', metadata={'navigation_ready': False,
            'semantic_gt_used': False, 'evidence_scope': 'observations_retained_by_upstream_mapper'})
    return write_bundle(root, map_id=new_map_id(), revision='1', prior_map=prior,
                        observations=observations, viewpoints=list(views.values()))
