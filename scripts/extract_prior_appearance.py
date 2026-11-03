#!/usr/bin/env python3
"""Extract DualMap appearance features on frozen STRIVE observations, locally."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from prior_map.bundle import load_bundle
from prior_map.observations import write_json


def main():
    """Reuse upstream padded crops and encoding without loading its detectors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--dualmap-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, required=True)
    parser.add_argument('--mask-background', action='store_true',
                        help='Experimental: replace pixels outside each saved mask with RGB 128')
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error('Frame count must be positive')
    upstream = args.dualmap_root.resolve()
    sys.path.insert(0, str(upstream))
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model
    from utils.object_detector import Detector
    bundle = load_bundle(args.bundle)
    frame_ids = sorted({o['frame_id'] for o in bundle['observations']})[:args.max_frames]
    groups = defaultdict(list)
    for observation in bundle['observations']:
        if observation['frame_id'] in frame_ids:
            groups[observation['frame_id']].append(observation)
    args.output.mkdir(parents=True, exist_ok=False)
    model, _, preprocess = open_clip.create_model_and_transforms(
        'MobileCLIP-S2', pretrained=str(args.checkpoint.resolve()))
    model = reparameterize_model(model.to('cuda').eval())
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')
    # 本实验使用保存的显式检测类别；没有DualMap的unknown类别及其均值替换。
    context = SimpleNamespace(cfg=SimpleNamespace(use_avg_feat_for_unknown=True), unknown_class_id=-1)
    records, image_features, text_features, image_hashes, mask_hashes = [], [], [], {}, {}
    for observations in groups.values():
        image_path = args.bundle / observations[0]['image_ref']
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        image_hashes[str(image_path)] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        batches = [[o] for o in observations] if args.mask_background else [observations]
        for batch in batches:
            image = rgb
            if args.mask_background:
                mask_path = args.bundle / batch[0]['mask_ref']
                mask = np.asarray(Image.open(mask_path)) > 0
                image = np.where(mask[..., None], rgb, np.uint8(128))
                mask_hashes[str(mask_path)] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            boxes = np.asarray([o['bbox_xyxy'] for o in batch], dtype=np.float32)
            labels = [o['detector_label'] for o in batch]
            detections = SimpleNamespace(xyxy=boxes, class_id=np.arange(len(batch)))
            _, images, texts = Detector.compute_clip_features_batched(
                context, image, detections, model, tokenizer, preprocess, 'cuda', labels)
            image_features.extend(images)
            text_features.extend(texts)
            records.extend({'observation_id': o['observation_id'], 'frame_id': o['frame_id'],
                            'label': o['detector_label'], 'bbox_xyxy': o['bbox_xyxy']} for o in batch)
    images, texts = np.asarray(image_features), np.asarray(text_features)
    weighted = .7 * images + .3 * texts
    weighted /= np.linalg.norm(weighted, axis=1, keepdims=True)
    np.savez_compressed(args.output / 'features.npz', image=images, text=texts, weighted=weighted)
    write_json(args.output / 'receipt.json', {'scope': 'frozen_observation_features_not_identity_decisions',
        'model': 'MobileCLIP-S2', 'crop': 'upstream_bbox_padding20_rgb', 'image_weight': .7,
        'background': 'mask_outside_rgb128' if args.mask_background else 'original',
        'observations': records, 'frame_count': len(groups),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'upstream_detector_sha256': hashlib.sha256((upstream / 'utils/object_detector.py').read_bytes()).hexdigest(),
        'observations_sha256': hashlib.sha256((args.bundle / 'observations.jsonl').read_bytes()).hexdigest(),
        'features_sha256': hashlib.sha256((args.output / 'features.npz').read_bytes()).hexdigest(),
        'image_sha256': image_hashes, 'mask_sha256': mask_hashes})
    print(f'{len(records)} observations encoded locally; no map or association changes')


if __name__ == '__main__':
    main()
