"""Lossless portable depth storage, keeping every observation and map value."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from .bundle import load_bundle
from .observations import write_json


def compress_bundle_depth(source, output):
    """Copy a complete map bundle and gzip its referenced NPY depth files.

    Args:
        source: Valid bundle with NPY depth evidence; left unchanged.
        output: New self-contained directory, which must not already exist.
    Returns:
        Receipt containing source indices and byte-exact depth roundtrip hashes.
    Raises:
        ValueError: A depth reference uses an encoding other than NPY.

    This changes storage encoding only. Identity, semantic revision, calibration,
    observation IDs and viewpoint reachability revisions retain their values.
    """
    source, output = Path(source), Path(output)
    bundle = load_bundle(source)
    refs = sorted({obs['depth_ref'] for obs in bundle['observations'] if obs.get('depth_ref')})
    if any(Path(ref).suffix != '.npy' for ref in refs):
        raise ValueError('Depth compression requires source NPY files')
    # 完整复制包含少数反例；只改变深度容器编码，不执行证据筛选。
    shutil.copytree(source, output)
    records = []
    for ref in refs:
        raw = (source / ref).read_bytes()
        encoded = gzip.compress(raw, compresslevel=1, mtime=0)
        if gzip.decompress(encoded) != raw:
            raise ValueError(f'Lossless roundtrip failed: {ref}')
        target = output / (ref + '.gz')
        target.write_bytes(encoded)
        (output / ref).unlink()
        records.append({'source_ref': ref, 'depth_ref': ref + '.gz',
            'source_bytes': len(raw), 'stored_bytes': len(encoded),
            'decoded_sha256': hashlib.sha256(raw).hexdigest(),
            'stored_sha256': hashlib.sha256(encoded).hexdigest()})
    observations = [{**obs, 'depth_ref': obs['depth_ref'] + '.gz'}
                    if obs.get('depth_ref') else obs for obs in bundle['observations']]
    (output / 'observations.jsonl').write_text(''.join(
        json.dumps(obs, allow_nan=False) + '\n' for obs in observations))
    manifest = {**bundle['manifest'], 'depth_storage': {
        'encoding': 'npy_gzip', 'lossless': True, 'receipt': 'depth_storage_receipt.json'}}
    write_json(output / 'manifest.json', manifest)
    receipt = {'scope': 'lossless_storage_encoding', 'observation_count': len(observations),
        'depth_file_count': len(refs), 'source_bytes': sum(r['source_bytes'] for r in records),
        'stored_bytes': sum(r['stored_bytes'] for r in records),
        'source_index_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ('manifest.json', 'prior_map.json', 'observations.jsonl', 'viewpoints.json')},
        'files': records}
    write_json(output / 'depth_storage_receipt.json', receipt)
    load_bundle(output)
    return receipt
