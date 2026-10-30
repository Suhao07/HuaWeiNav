"""Bind a saved retrieval result to the map used for historical-view execution."""
import hashlib
from pathlib import Path
from .bundle import load_bundle


def queried_object_uid(query, directory):
    """Resolve the top ranked hypothesis after checking identity and revision.

    A legacy map can share its map_id with another bundle. The canonical map
    hash also binds the receipt to its actual object identities and geometry.
    This is hypothesis selection, with no task-progress or verification effect.
    Raises ValueError for stale/map-mismatched receipts or an empty retrieval.
    """
    root = Path(directory)
    bundle = load_bundle(root)
    expected = {'map_id': bundle['manifest']['map_id'],
                'map_revision': bundle['manifest']['map_revision'],
                'prior_map_sha256': hashlib.sha256((root / 'prior_map.json').read_bytes()).hexdigest()}
    for field, value in expected.items():
        if query[field] != value:
            raise ValueError(f'Query receipt differs from current map: {field}')
    if not query['matches']:
        raise ValueError('Query returned no historical object hypothesis')
    best = max(query['matches'], key=lambda item: item['ranking']['score'])
    uid = best['ranking']['object_uid']
    if uid not in {obj.uid for obj in bundle['prior_map'].objects}:
        raise ValueError(f'Query references unknown historical object: {uid}')
    return uid
