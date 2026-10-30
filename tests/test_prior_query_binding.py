"""Retrieval identity is checked before a historical object reaches motion."""
import hashlib
import pytest
from prior_map.bundle import load_bundle
from prior_map.query_binding import queried_object_uid
from test_prior_compaction import _source


def _receipt(root):
    manifest = load_bundle(root)['manifest']
    return {'map_id': manifest['map_id'], 'map_revision': manifest['map_revision'],
        'prior_map_sha256': hashlib.sha256((root / 'prior_map.json').read_bytes()).hexdigest(),
        'matches': [{'ranking': {'object_uid': 'instance', 'score': 1.}}]}


def test_query_survives_bundle_relocation(tmp_path):
    root = tmp_path / 'source'
    _source(root)
    receipt = _receipt(root)
    moved = tmp_path / 'moved'
    root.rename(moved)
    assert queried_object_uid(receipt, moved) == 'instance'


@pytest.mark.parametrize('field', ['map_id', 'map_revision', 'prior_map_sha256'])
def test_query_rejects_mismatched_map_before_motion(tmp_path, field):
    root = tmp_path / 'source'
    _source(root)
    receipt = _receipt(root)
    receipt[field] = 'different'
    with pytest.raises(ValueError, match=field):
        queried_object_uid(receipt, root)


def test_query_requires_a_known_hypothesis(tmp_path):
    root = tmp_path / 'source'
    _source(root)
    receipt = _receipt(root)
    receipt['matches'][0]['ranking']['object_uid'] = 'missing'
    with pytest.raises(ValueError, match='unknown historical object'):
        queried_object_uid(receipt, root)
    receipt['matches'] = []
    with pytest.raises(ValueError, match='no historical object'):
        queried_object_uid(receipt, root)
