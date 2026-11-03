"""Metric-depth input bridge from portable captures to DualMap DataInput fields."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .observations import read_records, metric_depth_validity


def dualmap_vocabulary(concepts):
    """Reserve the final class ID for upstream FastSAM's unknown observations.

    DualMap assigns len(classes)-1 to every unlabeled FastSAM mask. Keep this
    sentinel separate from concrete concepts so the final object category is
    never silently used as the unknown class.
    """
    return [concept for concept in concepts if concept != 'unknown'] + ['unknown']


def selected_capture_frames(capture, max_frames=None, frame_stride=1):
    """Select original frame records in order; cap applies after striding.

    Input and evidence export share this sequence so upstream observation.idx
    always resolves to the same original frame, including non-contiguous IDs.
    """
    if frame_stride < 1:
        raise ValueError('Frame stride must be positive')
    return read_records(Path(capture) / 'frames.jsonl')[::frame_stride][:max_frames]


def iter_dualmap_inputs(capture, max_frames=None, frame_stride=1):
    """Yield RGB, optical-z depth HxWx1, K and T_map_camera without resampling.

    DualMap's dataset runner consumes optical-camera poses in world coordinates.
    Our world is the shared right-handed Z-up map. Keeping this transform avoids
    a first-frame alignment or a second camera-axis flip. Depth remains float32
    metres; recorded clip endpoints become invalid zero pixels, matching STRIVE
    fusion's metric-support rule. Original files remain unchanged.
    Frame index timestamps are not fabricated when capture time is absent.
    """
    root = Path(capture)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Capture is incomplete')
    frames = selected_capture_frames(root, max_frames, frame_stride)
    for index, frame in enumerate(frames):
        depth = np.load(root / frame['depth_path'], allow_pickle=False)
        depth = np.where(metric_depth_validity(depth, manifest.get('capture_profile')), depth, 0)
        result = {'idx': index, 'color': np.array(Image.open(root / frame['rgb_path']).convert('RGB')),
                  'depth': depth[..., None],
                  'color_name': str(frame['frame_id']), 'intrinsics': np.asarray(frame['intrinsics']),
                  'pose': np.asarray(frame['camera_to_map'])}
        if frame.get('timestamp') is not None:
            result['time_stamp'] = frame['timestamp']
        yield result
