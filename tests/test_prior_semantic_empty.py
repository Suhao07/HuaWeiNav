"""An empty map provides no evidence for a successful semantic audit."""
import json
import pytest
from prior_map.bundle import write_bundle
from prior_map.contracts import PriorMapData
from prior_map.semantic_audit import audit_habitat_instances


def test_empty_map_reports_insufficient_evidence(tmp_path):
    bundle = tmp_path / 'bundle'
    write_bundle(bundle, map_id='empty', revision='1',
        prior_map=PriorMapData(scene_id='test', frame_id='map_z_up'),
        observations=[], viewpoints=[])
    report = tmp_path / 'audit.json'
    with pytest.raises(ValueError, match='requires observed evidence'):
        audit_habitat_instances(bundle, 'unused', report)
    result = json.loads(report.read_text())
    assert result['status'] == 'insufficient_evidence'
    assert result['alignment_valid'] is None
    assert result['summary'] is None
