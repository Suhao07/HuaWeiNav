#!/usr/bin/env python3
"""Fixed-image contrast of context vocabulary; oracle data is scoring-only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms, box_iou
from cv_utils.sam import MMDINO_Grounded_SAM
from config_utils import SAM_CHECKPOINT, GROUNDING_DINO_CHECKPOINT, GROUNDING_DINO_CONFIG
from prior_map.bundle import load_bundle
from prior_map.observations import write_json


def main():
    """Keep image, weights, threshold and primary prompt fixed across conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    parser.add_argument('oracle_cache')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, cache = Path(args.bundle), Path(args.oracle_cache)
    bundle = load_bundle(root)
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['scope'] != 'oracle_diagnostic_only' or not manifest['alignment_valid']:
        raise ValueError('Oracle cache must pass camera/depth alignment')
    if hashlib.sha256((root / 'prior_map.json').read_bytes()).hexdigest() != manifest['provenance']['predicted_map']['sha256']:
        raise ValueError('Cache and predicted map identity differ')
    by_frame = defaultdict(list)
    for obs in bundle['observations']:
        by_frame[obs['frame_id']].append(obs)
    conditions = {'base': ['door', 'window', 'stairs'],
                  'structure_context': ['door', 'window', 'stairs', 'wall', 'floor', 'ceiling']}
    receipt = json.loads((root / 'build_receipt.json').read_text())
    model_digests = {}
    for name, path in [('sam', SAM_CHECKPOINT), ('grounding_dino', GROUNDING_DINO_CHECKPOINT),
                       ('grounding_dino_config', GROUNDING_DINO_CONFIG)]:
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        model_digests[name] = digest.hexdigest()
    if model_digests != receipt['provenance']['model_sha256']:
        raise ValueError('Detector weights/config differ from the frozen map build')
    detector = MMDINO_Grounded_SAM(classes=conditions['base'], device='cuda:0')
    rows = []
    started = time.monotonic()
    labels = manifest['annotation_labels']
    # 评价词表固定为标注名称包含 door，含 closet door；不参与模型提示词生成。
    door_ids = {int(i) for i, label in labels.items() if 'door' in label.lower()}
    for frame_id, observations in by_frame.items():
        item = manifest['frames'][str(frame_id)]
        image_path = root / observations[0]['image_ref']
        semantic_path = cache / item['file']
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != item['image_sha256'] or hashlib.sha256(semantic_path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Frame {frame_id} differs from frozen oracle cache')
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        semantic = np.load(semantic_path, allow_pickle=False)
        gt_boxes, gt_ids = [], []
        for uid in sorted(door_ids & set(np.unique(semantic).tolist())):
            y, x = np.where(semantic == uid)
            if len(x) >= 100:
                gt_ids.append(uid)
                gt_boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
        for condition, vocabulary in conditions.items():
            prediction = detector._run_grounding(rgb, ' . '.join(vocabulary) + ' .')
            boxes = torch.tensor(prediction['bboxes'], dtype=torch.float32).reshape(-1, 4)
            scores = torch.tensor(prediction['scores'])
            classes = np.asarray(prediction['labels'])
            keep = scores > detector.BOX_THRESHOLD
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep.numpy()]
            raw_door_count = int((classes == 0).sum())
            selected = nms(boxes, scores, detector.NMS_IOU_THRESHOLD)
            boxes, scores, classes = boxes[selected], scores[selected], classes[selected.numpy()]
            door_boxes, purities = [], []
            for box, cls in zip(boxes, classes):
                if cls != 0:
                    continue
                x1, y1, x2, y2 = box.tolist()
                if (x2-x1) * (y2-y1) >= .8 * semantic.size:
                    continue
                x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
                x2, y2 = min(semantic.shape[1], int(np.ceil(x2))), min(semantic.shape[0], int(np.ceil(y2)))
                pixels = semantic[y1:y2, x1:x2]
                purities.append(float(np.isin(pixels, list(door_ids)).mean()))
                door_boxes.append(box.tolist())
            matched = set()
            if door_boxes and gt_boxes:
                overlaps = box_iou(torch.tensor(door_boxes), torch.tensor(gt_boxes, dtype=torch.float32)).numpy()
                # NMS 保持置信度降序；IoU >= 0.5 的一对一匹配用于本诊断召回计数。
                for overlaps_row in overlaps:
                    candidates = [i for i in range(len(gt_ids)) if i not in matched and overlaps_row[i] >= .5]
                    if candidates:
                        matched.add(max(candidates, key=lambda i: overlaps_row[i]))
            rows.append({'frame_id': frame_id, 'condition': condition, 'raw_door_count': raw_door_count,
                'door_box_count': len(door_boxes), 'less_than_10pct_door': sum(p < .1 for p in purities),
                'gt_count': len(gt_ids), 'matched_gt_count': len(matched),
                'boxes': door_boxes, 'door_pixel_fractions': purities})
        print(f'frame {frame_id} complete', flush=True)
    summaries = {}
    for condition in conditions:
        group = [row for row in rows if row['condition'] == condition]
        summaries[condition] = {key: sum(r[key] for r in group) for key in (
            'raw_door_count', 'door_box_count', 'less_than_10pct_door', 'gt_count', 'matched_gt_count')}
    write_json(args.output, {'scope': 'single_prompt_group_diagnostic_not_full_mapper',
        'conditions': conditions, 'score_threshold': detector.BOX_THRESHOLD,
        'nms_iou': detector.NMS_IOU_THRESHOLD, 'frame_count': len(by_frame),
        'model_sha256': model_digests,
        'oracle_manifest_sha256': hashlib.sha256((cache / 'manifest.json').read_bytes()).hexdigest(),
        'elapsed_s': time.monotonic()-started, 'summary': summaries, 'frames': rows})
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
