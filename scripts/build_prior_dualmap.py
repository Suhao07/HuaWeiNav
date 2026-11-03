#!/usr/bin/env python3
"""Run official DualMap on the exact portable RGB-D capture used by STRIVE."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
STRIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRIVE))
from prior_map.dualmap_input import iter_dualmap_inputs, dualmap_vocabulary
from prior_map.observations import write_json
from prior_map.dualmap_export import export_dualmap_bundle


def main():
    """Use upstream mapping with isolated outputs and explicit model paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture')
    parser.add_argument('output')
    parser.add_argument('--dualmap-root', required=True)
    parser.add_argument('--yolo', required=True)
    parser.add_argument('--sam', required=True)
    parser.add_argument('--fastsam', required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--classes', required=True, help='Frozen text vocabulary file')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--frame-stride', type=int, default=1,
                        help='Use every Nth source frame; preserve original evidence IDs')
    parser.add_argument('--trace-matching', action='store_true')
    parser.add_argument('--pcd-sample-ratio', type=float,
                        help='Explicit sampling ablation; omitted uses upstream config')
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error('--frame-stride must be positive')
    capture, output, upstream = [Path(p).resolve() for p in (args.capture, args.output, args.dualmap_root)]
    paths = {name: Path(getattr(args, name)).resolve() for name in ('yolo', 'sam', 'fastsam', 'clip', 'classes')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from dualmap.core import Dualmap
    from utils.types import DataInput
    from utils.logging_helper import setup_logging
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with initialize_config_dir(config_dir=str(upstream / 'config'), version_base=None):
        cfg = compose(config_name='runner_dataset')
    cfg.output_path, cfg.dataset_name, cfg.scene_id = str(output), 'portable_capture', capture.name
    cfg.use_rerun, cfg.use_parallel = False, False
    cfg.visualize_detection, cfg.use_stride, cfg.stride = False, True, 1
    cfg.run_local_mapping_only, cfg.save_local_map = True, True
    cfg.save_detection = True
    if args.pcd_sample_ratio is not None:
        if not 0 < args.pcd_sample_ratio <= 1:
            raise ValueError('pcd_sample_ratio must be in (0, 1]')
        cfg.pcd_sample_ratio = args.pcd_sample_ratio
    # 共享地图的原点不保证位于地面；上游演示的固定 2.5 m 天花板高度不适用。
    cfg.filter_ceiling = False
    cfg.config_file_path = str(output / 'actions.yaml')
    (output / 'actions.yaml').write_text('calculate_path: false\n')
    cfg.yolo.model_path, cfg.sam.model_path = str(paths['yolo']), str(paths['sam'])
    cfg.fastsam.model_path, cfg.clip.pretrained = str(paths['fastsam']), str(paths['clip'])
    concepts = [line.strip() for line in paths['classes'].read_text().splitlines() if line.strip()]
    effective_classes = dualmap_vocabulary(concepts)
    effective_path = output / 'classes.txt'
    effective_path.write_text('\n'.join(effective_classes) + '\n')
    cfg.yolo.use_given_classes, cfg.yolo.given_classes_path = True, str(effective_path)
    OmegaConf.save(cfg, output / 'resolved_config.yaml', resolve=True)
    setup_logging(output_path=str(output), config_path=cfg.logging_config)
    if args.trace_matching:
        tracker_logger = logging.getLogger('utils.tracker')
        tracker_logger.setLevel(logging.DEBUG)
        tracker_handler = logging.FileHandler(output / 'matching.log')
        tracker_handler.setLevel(logging.DEBUG)
        tracker_logger.addHandler(tracker_handler)
    receipt = {'status': 'running', 'upstream_commit': subprocess.check_output(
        ['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        'input_manifest_sha256': hashlib.sha256((capture / 'manifest.json').read_bytes()).hexdigest(),
        'asset_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
        'scope': 'independent_builder_same_observations_not_navigation_acceptance',
        'processing': 'sequential_all_selected_frames', 'frame_count': 0,
        'frame_stride': args.frame_stride, 'max_frames': args.max_frames,
        'source_frames_sha256': hashlib.sha256((capture / 'frames.jsonl').read_bytes()).hexdigest(),
        'depth_policy': 'positive_excluding_recorded_sensor_clip_endpoints',
        'random_seed': args.seed, 'pcd_sample_ratio': cfg.pcd_sample_ratio,
        'effective_classes': effective_classes,
        'unknown_class_id': len(effective_classes) - 1,
        'adapter_sha256': {name: hashlib.sha256((STRIVE / name).read_bytes()).hexdigest()
                           for name in ('scripts/build_prior_dualmap.py', 'prior_map/dualmap_input.py', 'prior_map/dualmap_export.py')},
        'mobileclip_commit': subprocess.check_output(['git', '-C', str(upstream / '3rdparty/mobileclip'), 'rev-parse', 'HEAD'], text=True).strip(),
        'capture_manifest': json.loads((capture / 'manifest.json').read_text())}
    diff = subprocess.check_output(['git', '-C', str(upstream), 'diff'])
    (output / 'upstream.patch').write_bytes(diff)
    receipt['upstream_patch_sha256'] = hashlib.sha256(diff).hexdigest()
    write_json(output / 'build_receipt.json', receipt)
    model = None
    started = time.monotonic()
    try:
        model = Dualmap(cfg)
        for fields in iter_dualmap_inputs(capture, args.max_frames, args.frame_stride):
            model.sequential_process(DataInput(**fields))
            receipt['frame_count'] += 1
        before_finalize = [{'uid': str(obj.uid), 'observed_num': obj.observed_num,
            'is_stable': bool(obj.is_stable), 'status': obj.status.value,
            'class_id': int(obj.class_id)} for obj in model.local_map_manager.local_map]
        write_json(output / 'before_finalize.json', {'stable_num': cfg.stable_num,
            'active_window_size': cfg.active_window_size, 'objects': before_finalize})
        model.end_process()
        bundle = export_dualmap_bundle(capture, output / 'bundle', model.local_map_manager.local_map,
            model.detector.obj_classes.get_classes_arr(),
            scene_id=json.loads((capture / 'manifest.json').read_text())['scene_id'],
            max_frames=args.max_frames, frame_stride=args.frame_stride)
        receipt['object_count'] = len(bundle['prior_map'].objects)
        receipt['evidence_count'] = len(bundle['observations'])
        receipt.update(status='complete', elapsed_s=time.monotonic() - started)
        write_json(output / 'bundle' / 'build_receipt.json', receipt)
        write_json(output / 'build_receipt.json', receipt)
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'build_receipt.json', receipt)
        raise
    finally:
        if model is not None:
            model.stop_threading()


if __name__ == '__main__':
    main()
