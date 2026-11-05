"""Controlled fixtures test correspondence admission, not physical identity."""
import json
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
import base64
import hashlib

import pytest
from PIL import Image

from prior_map.instance_correspondence import PriorInstanceCorrespondenceVerifier, load_correspondence_package
from prior_map.multimodal import sha256_file
from prior_map.real_robot import PriorMapRealRobotRuntime
from prior_map.contracts import PriorMapData, PriorObject
from prior_map.alignment import PriorMapAlignment
from real_robot.contracts import ObjectNodeSnapshot, SemanticMapSnapshot, Pose3D


def test_masked_rgb_payload_preserves_pixels_holes_and_provenance(tmp_path):
    root, payload = package(tmp_path)
    rgb = Image.new('RGB', (8, 8), (10, 20, 30))
    rgb.putpixel((5, 6), (200, 40, 50))
    mask = Image.new('L', (8, 8), 0)
    mask.putpixel((2, 3), 1)  # Nonzero membership, not a >127 threshold.
    mask.putpixel((5, 6), 255)
    rgb.save(root / 'prior/rgb.png'); mask.save(root / 'prior/mask.png')
    payload['files_sha256'] = {ref: sha256_file(root / ref) for ref in payload['files_sha256']}
    save(root, payload)
    _, _, content = load_correspondence_package(root)
    metadata = json.loads(content[2]['text'])
    data = base64.b64decode(content[4]['image_url']['url'].split(',', 1)[1])
    crop = Image.open(BytesIO(data))
    assert metadata['mask_crop_xyxy'] == [2, 3, 6, 7]
    assert metadata['masked_rgb_sha256'] == hashlib.sha256(data).hexdigest()
    assert crop.size == (4, 4)
    assert crop.getpixel((0, 0)) == (10, 20, 30)
    assert crop.getpixel((3, 3)) == (200, 40, 50)
    assert crop.getpixel((1, 1)) == (127, 127, 127)


@pytest.mark.parametrize('size', [(8, 8), (7, 8)])
def test_empty_or_misaligned_mask_never_reaches_model(tmp_path, monkeypatch, size):
    root, payload = package(tmp_path)
    Image.new('L', size, 0).save(root / 'prior/mask.png')
    payload['files_sha256']['prior/mask.png'] = sha256_file(root / 'prior/mask.png')
    save(root, payload)
    calls = model(monkeypatch)
    with pytest.raises(ValueError, match='Historical mask'):
        PriorInstanceCorrespondenceVerifier().verify(root)
    assert calls == []


def package(tmp_path):
    root = tmp_path / 'package'
    refs = ['current/frame.png', 'prior/rgb.png', 'prior/mask.png']
    for ref in refs:
        path = root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), 'white').save(path)
    payload = dict(map_id='map', map_revision='r1', prior_map_sha256='map-hash',
        runtime_session={'session_id': 'session'},
        current_evidence=dict(timestamp=10., target_object_uid='live', target_tracks=['2'],
            image_ref='/original/frame.png', bbox_xyxy=[0, 0, 8, 8],
            quality=dict(bbox_source='detection_track', detection_timestamp=10.)),
        historical_candidates=[dict(prior_uid='prior', observations=[dict(
            observation_id='obs', object_uids=['prior'], image_ref='rgb.png', mask_ref='mask.png',
            detector_label='chair', bbox_xyxy=[0, 0, 8, 8])])],
        files_sha256={ref: sha256_file(root / ref) for ref in refs})
    save(root, payload)
    return root, payload


def save(root, payload):
    (root / 'inputs.json').write_text(json.dumps(payload))


def model(monkeypatch, outcome='unique', prior_uid='prior', support=None, conflicts=None):
    calls = []
    def parse(**kwargs):
        calls.append(kwargs)
        parsed = kwargs['response_format'](outcome=outcome, prior_uid=prior_uid,
            supporting_observation_ids=['obs'] if support is None else support,
            conflicting_observation_ids=conflicts or [], evidence_summary='Controlled fixture judgment')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, content='fixture'))])
    client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))))
    monkeypatch.setattr('prior_map.instance_correspondence.get_client_and_model', lambda _: (client, 'fixture-model'))
    monkeypatch.setenv('LLM_OFFLINE', '0')
    return calls


def runtime(tmp_path):
    base = PriorMapData(scene_id='lab', objects=(PriorObject(uid='prior', label='chair', exact=True),))
    return PriorMapRealRobotRuntime(base_map=base, alignment=PriorMapAlignment.identity(),
        run_directory=tmp_path / 'runtime', map_identity=dict(map_id='map', map_revision='r1', prior_map_sha256='map-hash'))


def snapshot(objects=True, stamp=11.):
    return SemanticMapSnapshot(timestamp=stamp, robot_pose=Pose3D(position=(0., 0., 0.)),
        objects=(ObjectNodeSnapshot(uid='live', label='chair', position=(0., 0., 0.), track_ids=('2',)),) if objects else ())


def test_corrupted_image_never_reaches_model(tmp_path, monkeypatch):
    root, _ = package(tmp_path); calls = model(monkeypatch)
    (root / 'prior/mask.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        PriorInstanceCorrespondenceVerifier().verify(root)
    assert calls == []


def test_offline_has_no_model_call_or_positive_verdict(tmp_path, monkeypatch):
    root, _ = package(tmp_path); calls = model(monkeypatch)
    monkeypatch.setenv('LLM_OFFLINE', '1')
    report = PriorInstanceCorrespondenceVerifier().verify(root)
    assert report['outcome'] == 'insufficient' and report['source'] == 'offline'
    assert calls == []


@pytest.mark.parametrize('outcome', ['fragments', 'ambiguous', 'no_match', 'insufficient'])
def test_non_unique_never_binds(tmp_path, monkeypatch, outcome):
    root, payload = package(tmp_path); rt = runtime(tmp_path)
    payload['runtime_session'] = {'session_id': rt.session_id}; save(root, payload)
    model(monkeypatch, outcome=outcome)
    report = rt.process_correspondence_package(root, snapshot=snapshot(), step=1)
    assert not report['binding_written']
    assert rt.memory.object_states['prior'].verification_count == 0


@pytest.mark.parametrize('support,conflicts', [(['unknown'], []), ([], []), (['obs'], ['obs'])])
def test_inconsistent_unique_response_is_not_accepted(tmp_path, monkeypatch, support, conflicts):
    root, _ = package(tmp_path); model(monkeypatch, support=support, conflicts=conflicts)
    assert PriorInstanceCorrespondenceVerifier().verify(root)['outcome'] == 'insufficient'


def test_unique_binding_uses_production_memory_path_once(tmp_path, monkeypatch):
    root, payload = package(tmp_path); rt = runtime(tmp_path)
    payload['runtime_session'] = {'session_id': rt.session_id}; save(root, payload)
    calls = model(monkeypatch)
    for step in [1, 2]:
        assert rt.process_correspondence_package(root, snapshot=snapshot(), step=step)['binding_written']
    assert len(calls) == 1
    state = rt.memory.object_states['prior']
    assert state.matched_runtime_uid == 'live' and state.verification_count == 1
    assert (rt.artifact_dir / 'instance_correspondence.jsonl').is_file()
    plan = SimpleNamespace(target_detector_prompts=('chair',), target_match_terms=('chair',),
                           search_priors=SimpleNamespace(room_hints=(), support_objects=(), affordances=()))
    context = rt.update_and_query(snapshot=snapshot(), plan=plan, step=3)
    assert context['prior_result'].object_rankings[0].matched_runtime_uid == 'live'


@pytest.mark.parametrize('change', ['session', 'map', 'track', 'old'])
def test_stale_context_rejected_before_model(tmp_path, monkeypatch, change):
    root, payload = package(tmp_path); rt = runtime(tmp_path); calls = model(monkeypatch)
    payload['runtime_session'] = {'session_id': rt.session_id}
    if change == 'session': payload['runtime_session'] = {'session_id': 'older-process'}
    if change == 'map': payload['map_revision'] = 'old-revision'
    if change == 'track': payload['current_evidence']['target_tracks'] = ['3']
    save(root, payload)
    with pytest.raises(ValueError):
        rt.process_correspondence_package(root, snapshot=snapshot(stamp=200. if change == 'old' else 11.), step=1)
    assert calls == []


def test_target_disappearing_while_model_runs_does_not_bind(tmp_path, monkeypatch):
    root, payload = package(tmp_path); rt = runtime(tmp_path); model(monkeypatch)
    payload['runtime_session'] = {'session_id': rt.session_id}; save(root, payload)
    report = rt.process_correspondence_package(root, snapshot=snapshot(), step=1,
        latest_snapshot_provider=lambda: snapshot(objects=False))
    assert not report['binding_written']


def test_service_error_is_a_recorded_nonbinding_result(tmp_path, monkeypatch):
    root, _ = package(tmp_path)
    monkeypatch.setenv('LLM_OFFLINE', '0')
    def unavailable(_):
        raise RuntimeError('sensitive provider error must not be copied')
    monkeypatch.setattr('prior_map.instance_correspondence.get_client_and_model', unavailable)
    report = PriorInstanceCorrespondenceVerifier().verify(root)
    assert report['source'] == 'error' and report['outcome'] == 'insufficient'
    assert report['evidence_summary'] == 'RuntimeError'
    assert 'sensitive' not in json.dumps(report)


def test_injected_client_result_cannot_be_used_as_production_evidence(tmp_path, monkeypatch):
    root, payload = package(tmp_path); rt = runtime(tmp_path)
    payload['runtime_session'] = {'session_id': rt.session_id}; save(root, payload)
    model(monkeypatch)
    from prior_map.instance_correspondence import get_client_and_model
    client, name = get_client_and_model('ark')
    verifier = PriorInstanceCorrespondenceVerifier(client=client, model=name)
    report = rt.process_correspondence_package(root, snapshot=snapshot(), step=1, verifier=verifier)
    assert report['source'] == 'injected_client' and not report['binding_written']


def test_nonfinite_detection_timestamp_is_not_paired(tmp_path):
    root, payload = package(tmp_path)
    payload['current_evidence']['quality']['detection_timestamp'] = float('nan')
    save(root, payload)
    with pytest.raises(ValueError, match='paired'):
        load_correspondence_package(root)


def test_model_bytes_are_the_exact_validated_bytes_even_if_file_changes(tmp_path, monkeypatch):
    import base64
    import hashlib
    root, payload = package(tmp_path)
    target = (root / 'current/frame.png').resolve()
    read_bytes = Path.read_bytes
    reads = []
    def changing_read(path):
        data = read_bytes(path)
        if path == target:
            reads.append(path)
            path.write_bytes(b'replaced after read')
        return data
    monkeypatch.setattr(Path, 'read_bytes', changing_read)
    _, _, content = load_correspondence_package(root)
    encoded = content[1]['image_url']['url'].split(',', 1)[1]
    assert hashlib.sha256(base64.b64decode(encoded)).hexdigest() == payload['files_sha256']['current/frame.png']
    assert len(reads) == 1
