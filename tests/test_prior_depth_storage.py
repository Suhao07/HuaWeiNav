"""Complete evidence survives portable, bit-exact depth compression."""
import hashlib

import numpy as np
import pytest

from prior_map.bundle import load_bundle
from prior_map.depth_storage import compress_bundle_depth
from prior_map.observations import load_depth
from test_prior_compaction import _source


def test_complete_bundle_roundtrip_after_relocation(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'compressed'
    _source(source)
    before = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    original = load_bundle(source)
    receipt = compress_bundle_depth(source, output)
    moved = tmp_path / 'relocated'
    output.rename(moved)
    restored = load_bundle(moved)
    assert restored['prior_map'].to_dict() == original['prior_map'].to_dict()
    assert restored['viewpoints'] == original['viewpoints']
    assert restored['manifest']['map_revision'] == original['manifest']['map_revision']
    assert restored['manifest']['map_id'] == original['manifest']['map_id']
    assert receipt['observation_count'] == 5
    assert receipt['depth_file_count'] == 4  # Shared frames remain shared.
    for a, b in zip(original['observations'], restored['observations']):
        assert {**b, 'depth_ref': a['depth_ref']} == a
        first, second = load_depth(source / a['depth_ref']), load_depth(moved / b['depth_ref'])
        assert first.dtype == second.dtype
        assert first.shape == second.shape
        assert first.tobytes() == second.tobytes()
    assert before == {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob('*') if p.is_file()}


def test_gzip_decoding_errors_propagate(tmp_path):
    broken = tmp_path / 'depth.npy.gz'
    broken.write_bytes(b'broken')
    with pytest.raises(OSError):
        load_depth(broken)


def test_depth_float_bits_are_preserved(tmp_path):
    import gzip
    path = tmp_path / 'depth.npy'
    values = np.array([[0., .5, 1.0000001, 4.9999995, 5.]], dtype=np.float32)
    np.save(path, values, allow_pickle=False)
    encoded = tmp_path / 'depth.npy.gz'
    encoded.write_bytes(gzip.compress(path.read_bytes()))
    assert load_depth(encoded).tobytes() == values.tobytes()
