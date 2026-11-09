"""Test the data boundary and subprocess plumbing, not fake navigation metrics."""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from benchmark import intentionnav
from benchmark.intentionnav import IntentionNavEpisode, load_episode, score_trajectory


@pytest.fixture
def dataset(tmp_path):
    """Create independent episode and intent records with deliberately private fields."""
    meta = {
        "selection_id": "SEL_001",
        "scene_id": "kujiale_0262",
        "start_position": [1.0, 2.0, 1.5],
        "start_rotation_quat_wxyz": [0.6, 0.0, 0.0, 0.8],
        "target_category": "private-category",
        "target_object_id": "private-object/Meshes",
        "target_position": [8.0, 9.0, 1.0],
        "target_room": "private-room",
        "start_room": "private-start-room",
        "geodesic_to_target": 10.5,
        "_resampled": {"orig_start_position": [4.0, 5.0, 1.5], "orig_geodesic": 20.0},
        "unknown_future_field": {"object_ids": ["private-id"], "photo": "private-photo.png"},
    }
    intent = {
        "selection_id": "SEL_001",
        "scene_id": "kujiale_0262",
        "formal_en": "Please find somewhere to rest.",
        "natural_en": "I need somewhere to rest.",
        "casual_en": "Gotta take a break.",
        "emotional_en": "I'm exhausted and need a break!",
        "target_category": "private-intent-category",
        "room": "private-intent-room",
        "photo": "private-intent-photo.png",
        "sequence_id": 99,
        "instruction": "private-unselected-instruction",
    }
    (tmp_path / "dataset").mkdir()
    _write_dataset(tmp_path, [meta], [intent])
    return tmp_path, meta, intent


def _write_dataset(root, episodes, intents):
    """Write fixture records without reusing the loader's serialization logic."""
    for name, records in (("episodes.jsonl", episodes), ("selected_500_intents.jsonl", intents)):
        (root / "dataset" / name).write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )


@pytest.fixture
def fake_official(tmp_path):
    """Provide an echo-only scorer with official-layout imports and noisy stdout."""
    root = tmp_path / "official with spaces"
    aggregate = root / "eval" / "aggregate"
    agents = root / "eval" / "agents"
    aggregate.mkdir(parents=True)
    agents.mkdir()
    (aggregate / "visibility.py").write_text('TOKEN = "aggregate-import"\n', encoding="utf-8")
    (agents / "common.py").write_text('TOKEN = "agents-import"\n', encoding="utf-8")
    (aggregate / "compute_metrics.py").write_text(
        'import os\n'
        'from visibility import TOKEN as aggregate_token\n'
        'from common import TOKEN as agents_token\n'
        'print("noisy scorer import")\n'
        'def nav_metrics(record, radius_m=2.0):\n'
        '    print("noisy scorer call")\n'
        '    return {"received": record, "tokens": [aggregate_token, agents_token], "pid": os.getpid()}\n',
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("style", ["formal", "natural", "casual", "emotional"])
def test_loader_whitelist_and_provenance(dataset, style):
    """Expose exactly six public fields while preserving every evaluator field."""
    root, meta, intent = dataset
    episode = load_episode(root, style=style)
    assert isinstance(episode, IntentionNavEpisode)
    assert episode.policy_input() == {
        "selection_id": "SEL_001",
        "scene_id": meta["scene_id"],
        "style": style,
        "instruction": intent[style + "_en"],
        "start_position": (1.0, 2.0, 1.5),
        "start_rotation_quat_wxyz": (0.6, 0.0, 0.0, 0.8),
    }
    assert episode._evaluation_meta == meta
    assert "private-" not in json.dumps(episode.policy_input())
    assert "private-" not in repr(episode)
    assert episode.source_sha256 == {
        f"dataset/{name}": hashlib.sha256((root / "dataset" / name).read_bytes()).hexdigest()
        for name in ("episodes.jsonl", "selected_500_intents.jsonl")
    }
    episode._evaluation_meta["extra_gt"] = {"position": [50, 50, 50]}
    episode.source_sha256["private-diagnostic"] = "private-value"
    policy = episode.policy_input()
    policy["instruction"] = "changed by policy"
    assert episode.instruction == intent[style + "_en"]
    assert "private-" not in json.dumps(episode.policy_input())
    assert "extra_gt" not in episode.policy_input()
    assert "source_sha256" not in episode.policy_input()


def test_defaults_and_exact_instruction(dataset):
    """Retain selected instruction text instead of rewriting it."""
    root, meta, intent = dataset
    intent["formal_en"] = "  Please rest.\n"
    _write_dataset(root, [meta], [intent])
    episode = load_episode(root)
    assert episode.selection_id == "SEL_001"
    assert episode.style == "formal"
    assert episode.instruction == intent["formal_en"]


@pytest.mark.parametrize("selection_id", [None, "", "  ", 1])
def test_invalid_selection_id(dataset, selection_id):
    """Reject invalid selection identifiers before reading records."""
    with pytest.raises(ValueError, match="selection_id"):
        load_episode(dataset[0], selection_id=selection_id)


@pytest.mark.parametrize("style", ["Formal", "formal_en", "target_category", "", None])
def test_invalid_style(dataset, style):
    """Prevent style selectors from accessing arbitrary intent fields."""
    with pytest.raises(ValueError, match="style"):
        load_episode(dataset[0], style=style)


@pytest.mark.parametrize("instruction", ["", " \n\t", None, 3, []])
def test_invalid_instruction(dataset, instruction):
    """Require a nonempty textual instruction."""
    root, meta, intent = dataset
    intent["formal_en"] = instruction
    _write_dataset(root, [meta], [intent])
    with pytest.raises(ValueError, match="nonempty instruction"):
        load_episode(root)


@pytest.mark.parametrize("source", ["episode", "intent"])
@pytest.mark.parametrize("count", [0, 2])
def test_requested_id_must_be_present_and_unique(dataset, source, count):
    """Reject missing or duplicated joins on either side."""
    root, meta, intent = dataset
    _write_dataset(root, [meta] * (count if source == "episode" else 1),
                   [intent] * (count if source == "intent" else 1))
    with pytest.raises(ValueError, match=f"expected exactly one SEL_001, found {count}"):
        load_episode(root)


def test_join_uses_id_not_record_order(dataset):
    """Ignore unrelated duplicate IDs and preserve the requested record's provenance."""
    root, meta, intent = dataset
    other_meta = dict(meta, selection_id="SEL_OTHER", scene_id="other_scene")
    other_intent = dict(intent, selection_id="SEL_OTHER", formal_en="Other instruction")
    _write_dataset(root, [other_meta, meta, other_meta], [intent, other_intent, other_intent])
    episode = load_episode(root)
    assert episode._evaluation_meta == meta
    assert episode.instruction == intent["formal_en"]
    with pytest.raises(ValueError, match="found 0"):
        load_episode(root, selection_id="SEL_UNKNOWN")


def test_scene_mismatch(dataset):
    """Reject instruction and geometry from different scenes."""
    root, meta, intent = dataset
    intent["scene_id"] = "another_scene"
    _write_dataset(root, [meta], [intent])
    with pytest.raises(ValueError, match="scene_id mismatch"):
        load_episode(root)


@pytest.mark.parametrize("scene_id", [None, "", ".", "..", "../scene", "/scene", "a/b", "a\\b", "C:scene", "a\n"])
def test_unsafe_scene_id(dataset, scene_id):
    """Reject scene identifiers that could encode filesystem paths."""
    root, meta, intent = dataset
    meta["scene_id"] = intent["scene_id"] = scene_id
    _write_dataset(root, [meta], [intent])
    with pytest.raises(ValueError, match="safe scene name"):
        load_episode(root)


@pytest.mark.parametrize("field,value", [
    ("start_position", None),
    ("start_position", [1, 2]),
    ("start_position", [1, 2, 3, 4]),
    ("start_position", "123"),
    ("start_position", [True, 2, 3]),
    ("start_position", ["1", 2, 3]),
    ("start_position", [float("nan"), 2, 3]),
    ("start_position", [1, float("inf"), 3]),
    ("start_position", [1, 2, -float("inf")]),
    ("start_position", [10 ** 400, 2, 3]),
    ("start_rotation_quat_wxyz", None),
    ("start_rotation_quat_wxyz", [1, 0, 0]),
    ("start_rotation_quat_wxyz", [0, 0, 0, 0]),
    ("start_rotation_quat_wxyz", [2, 0, 0, 0]),
    ("start_rotation_quat_wxyz", [1.001, 0, 0, 0]),
    ("start_rotation_quat_wxyz", [float("nan"), 0, 0, 0]),
    ("start_rotation_quat_wxyz", [1, 0, float("inf"), 0]),
    ("start_rotation_quat_wxyz", [True, 0, 0, 0]),
])
def test_invalid_start_pose(dataset, field, value):
    """Require finite three-position coordinates and a unit wxyz quaternion."""
    root, meta, intent = dataset
    meta[field] = value
    _write_dataset(root, [meta], [intent])
    with pytest.raises(ValueError, match=field):
        load_episode(root)


@pytest.mark.parametrize("source", ["episodes.jsonl", "selected_500_intents.jsonl"])
@pytest.mark.parametrize("contents", ["{bad json", "[]\n", "null\n"])
def test_malformed_jsonl(dataset, source, contents):
    """Fail explicitly on malformed JSON and nonobject rows."""
    root, _, _ = dataset
    (root / "dataset" / source).write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        load_episode(root)


@pytest.mark.parametrize("source,field", [("episode", "start_position"), ("episode", "start_rotation_quat_wxyz"),
                                          ("episode", "scene_id"), ("intent", "scene_id"), ("intent", "formal_en")])
def test_missing_fields(dataset, source, field):
    """Reject selected records with required fields absent."""
    root, meta, intent = dataset
    del (meta if source == "episode" else intent)[field]
    _write_dataset(root, [meta], [intent])
    with pytest.raises(ValueError):
        load_episode(root)


def test_missing_source(dataset):
    """Propagate missing external data files without fallback data."""
    root, _, _ = dataset
    (root / "dataset" / "selected_500_intents.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        load_episode(root)


def test_subprocess_plumbing_preserves_record(dataset, fake_official, monkeypatch):
    """Roundtrip the complete record through a real subprocess using a fake scorer."""
    root, meta, _ = dataset
    episode = load_episode(root)
    trajectory = [
        {"position": list(episode.start_position), "yaw": 0.5},
        {"position": [1.5, 2.0, 1.5], "yaw": 0.75},
    ]
    original = copy.deepcopy(trajectory)
    calls = []
    real_run = subprocess.run

    def capture_run(command, **kwargs):
        calls.append((command, kwargs))
        return real_run(command, **kwargs)

    monkeypatch.setattr(intentionnav.subprocess, "run", capture_run)
    result = score_trajectory(episode, trajectory, fake_official)
    scorer = fake_official / "eval" / "aggregate" / "compute_metrics.py"
    assert result["scorer_sha256"] == hashlib.sha256(scorer.read_bytes()).hexdigest()
    assert result["metrics"]["received"] == result["record"]
    assert result["record"]["trajectory"] == trajectory
    assert result["record"]["episode_meta"] == meta
    assert result["record"]["prediction"] == {"target": None}
    assert result["metrics"]["IM"] is None
    assert result["metrics"]["G"] is None
    assert not result["aggregate_ready"]
    assert result["metrics"]["tokens"] == ["aggregate-import", "agents-import"]
    assert result["metrics"]["pid"] != os.getpid()
    assert trajectory == original
    assert episode._evaluation_meta == meta
    command, kwargs = calls[0]
    assert len(calls) == 1
    assert command[0] == sys.executable
    assert Path(command[-1]) == scorer
    assert kwargs["timeout"] == 30
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    payload = json.loads(kwargs["input"])
    assert payload["record"] == result["metrics"]["received"]
    assert payload["options"]["radius_m"] == 2.0
    assert payload["options"]["dataset_root"] == str(fake_official / "results" / "dataset")


@pytest.mark.parametrize("trajectory", [None, [], {}, [None], [{"position": [1, 2, 1.5]}],
    [{"yaw": 0}], [{"position": [1, 2], "yaw": 0}],
    [{"position": [1, 2, float("nan")], "yaw": 0}],
    [{"position": [1, 2, float("inf")], "yaw": 0}],
    [{"position": [1, 2, 1.5], "yaw": float("nan")}],
    [{"position": [1, 2, 1.5], "yaw": float("inf")}],
    [{"position": [1, 2, 1.5], "yaw": "0"}],
    [{"position": [1, 2, 1.5], "yaw": True}],
    [{"position": [1, 2, 1.5], "yaw": 0}, {"position": [2, 3, 1.5], "yaw": -float("inf")}],
    [{"position": [1, 2, 1.5], "yaw": 0}, {"position": [2, float("nan"), 1.5], "yaw": 0}],
])
def test_invalid_trajectory_fails_before_scorer(dataset, trajectory):
    """Validate every row before attempting to open an official scorer."""
    root, _, _ = dataset
    with pytest.raises(ValueError, match="trajectory"):
        score_trajectory(load_episode(root), trajectory, root / "missing-official")


@pytest.mark.parametrize("position", [[2, 2, 1.5], [1, 3, 1.5], [1, 2, 1.6]])
def test_start_mismatch_uses_original_metadata(dataset, position):
    """Reject mismatched starts even if a caller alters the public start position."""
    root, _, _ = dataset
    episode = load_episode(root)
    episode.start_position = tuple(position)
    with pytest.raises(ValueError, match="official start"):
        score_trajectory(episode, [{"position": position, "yaw": 0}], root / "missing-official")


def test_scorer_failure_propagates(dataset, fake_official):
    """Propagate the scorer's nonzero exit with its original stderr diagnostics."""
    (fake_official / "eval" / "aggregate" / "compute_metrics.py").write_text(
        'def nav_metrics(record, radius_m=2.0):\n    raise RuntimeError("official failure")\n', encoding="utf-8"
    )
    episode = load_episode(dataset[0])
    with pytest.raises(subprocess.CalledProcessError) as error:
        score_trajectory(episode, [{"position": episode.start_position, "yaw": 0}], fake_official)
    assert "official failure" in error.value.stderr


def test_scorer_timeout_propagates(dataset, fake_official, monkeypatch):
    """Propagate subprocess timeouts without waiting for a real stalled scorer."""
    def time_out(command, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(intentionnav.subprocess, "run", time_out)
    episode = load_episode(dataset[0])
    with pytest.raises(subprocess.TimeoutExpired):
        score_trajectory(episode, [{"position": episode.start_position, "yaw": 0}], fake_official)


def test_missing_official_scorer(dataset):
    """Fail rather than silently substituting locally implemented metrics."""
    episode = load_episode(dataset[0])
    with pytest.raises(FileNotFoundError):
        score_trajectory(episode, [{"position": episode.start_position, "yaw": 0}], dataset[0])


def test_nonobject_scorer_result(dataset, fake_official):
    """Reject a scorer result that violates the metrics dictionary interface."""
    (fake_official / "eval" / "aggregate" / "compute_metrics.py").write_text(
        'def nav_metrics(record, radius_m=2.0):\n    return []\n', encoding="utf-8"
    )
    episode = load_episode(dataset[0])
    with pytest.raises(ValueError, match="JSON object"):
        score_trajectory(episode, [{"position": episode.start_position, "yaw": 0}], fake_official)


@pytest.fixture
def official_case(dataset, tmp_path):
    """Use unchanged local upstream code with controlled geometry, never fake metrics."""
    source = Path(os.environ.get(
        "INTENTIONNAV_OFFICIAL_ROOT",
        str(Path(__file__).resolve().parents[2] / "IntentionNav-deploy-105" / "source"),
    ))
    if not (source / "eval/aggregate/compute_metrics.py").is_file():
        pytest.skip("fixed upstream checkout not available; set INTENTIONNAV_OFFICIAL_ROOT")
    pytest.importorskip("numpy")
    pytest.importorskip("yaml")
    official = tmp_path / "fixed official"
    for relative in ("eval/aggregate/compute_metrics.py", "eval/aggregate/visibility.py",
                     "eval/agents/common.py", "eval/vocab/category_synonyms.yaml"):
        destination = official / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    _, meta, intent = dataset
    meta = dict(meta, target_category="couch", target_position=[4.0, 2.0, 1.5],
                geodesic_to_target=3.0)
    intent = dict(intent, target_category="couch", target_representative="couch_0000")
    data_root = official / "results"
    (data_root / "dataset").mkdir(parents=True)
    _write_dataset(data_root, [meta], [intent])
    scene = data_root / "dataset" / meta["scene_id"]
    scene.mkdir()
    bbox = {"min": [3.5, 1.5, 1.0], "max": [4.5, 2.5, 2.0]}
    # Two category matches require the official representative-id lookup.
    (scene / "manifest.json").write_text(json.dumps([
        {"target_category": "couch", "target_representative": "other_couch"},
        {"target_category": "couch", "target_representative": "couch_0000",
         "target_details": [{"bbox": bbox}]},
    ]), encoding="utf-8")
    summary = tmp_path / "scene summary"
    (summary / meta["scene_id"]).mkdir(parents=True)
    objects = summary / meta["scene_id"] / "object_dict.json"
    objects.write_text(json.dumps({"couch": {
        "position": [4.0, 2.0, 1.5], "min_points": bbox["min"], "max_points": bbox["max"],
    }}), encoding="utf-8")
    episode = load_episode(data_root)
    trajectory = [
        {"step": 0, "position": list(episode.start_position), "yaw": 0.0},
        {"step": 1, "position": [2.5, 2.0, 1.5], "yaw": 0.0},
    ]
    options = dict(prediction_target="the sofa", tier="goat", model="local-policy",
                   dataset_root=data_root / "dataset", scene_summary_root=summary)
    return official, episode, trajectory, options


@pytest.mark.parametrize("yaw,occluded,radius,sr,g", [
    (0.0, False, 2.0, 1, 1),
    (math.pi, False, 2.0, 1, 0),
    (0.0, True, 2.0, 1, 0),
    (0.0, False, 1.0, 0, 1),
])
def test_real_official_active_metrics(official_case, yaw, occluded, radius, sr, g):
    """Exercise synonym IM, geometric SR, final-only G and occlusion with upstream code."""
    official, episode, trajectory, options = official_case
    trajectory[-1]["yaw"] = yaw
    if occluded:
        objects = options["scene_summary_root"] / episode.scene_id / "object_dict.json"
        data = json.loads(objects.read_text())
        data["blocker"] = {"position": [3.0, 2.0, 1.5],
                           "min_points": [2.9, 1.0, 0.0], "max_points": [3.1, 3.0, 3.0]}
        objects.write_text(json.dumps(data), encoding="utf-8")
    policy_before = episode.policy_input()
    result = score_trajectory(episode, trajectory, official, radius_m=radius, **options)
    metrics = result["metrics"]
    assert {key: metrics[key] for key in ("IM", "SR_hit", "OSR_hit", "SPL", "G", "GSR_hit")} == {
        "IM": 1, "SR_hit": sr, "OSR_hit": sr, "SPL": float(sr), "G": g, "GSR_hit": sr and g,
    }
    assert metrics["d_final"] == metrics["d_min"] == metrics["path_length"] == 1.5
    assert result["visibility"]["G_seen"] == g
    assert result["unavailable_metrics"] == {}
    assert result["aggregate_ready"] is True
    assert episode.policy_input() == policy_before
    assert len(policy_before) == 6
    assert "target_category" not in policy_before
    expected_dependencies = [
        official / "eval/aggregate/compute_metrics.py", official / "eval/aggregate/visibility.py",
        official / "eval/agents/common.py", official / "eval/vocab/category_synonyms.yaml",
        options["dataset_root"] / "selected_500_intents.jsonl",
        options["dataset_root"] / episode.scene_id / "manifest.json",
        options["scene_summary_root"] / episode.scene_id / "object_dict.json",
    ]
    assert result["scorer_dependencies_sha256"] == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in expected_dependencies
    }
    assert result["source_sha256"] == episode.source_sha256


@pytest.mark.parametrize("prediction,im", [("bed", 0), ("", 0), (None, None)])
def test_real_prediction_is_not_inferred_from_gt(official_case, prediction, im):
    """A genuine empty/wrong prediction fails IM; absent prediction is not a scored failure."""
    official, episode, trajectory, options = official_case
    options["prediction_target"] = prediction
    result = score_trajectory(episode, trajectory, official, **options)
    assert result["metrics"]["IM"] == im
    assert result["record"]["prediction"]["target"] == prediction
    assert result["metrics"]["GSR_hit"] == 1  # GSR is independent of IM.
    assert ("IM" in result["unavailable_metrics"]) == (prediction is None)
    assert result["aggregate_ready"] == (prediction is not None)


@pytest.mark.parametrize("missing", ["manifest", "objects", "objects_empty", "bbox", "geodesic", "target_position"])
def test_real_missing_inputs_are_unavailable_not_zero(official_case, missing):
    """Do not accept upstream's empty-assets or missing-geodesic fallback as a real score."""
    official, episode, trajectory, options = official_case
    if missing in ("objects", "objects_empty"):
        objects = options["scene_summary_root"] / episode.scene_id / "object_dict.json"
        if missing == "objects":
            objects.unlink()
        else:
            objects.write_text("{}", encoding="utf-8")
    elif missing in ("manifest", "bbox"):
        manifest = options["dataset_root"] / episode.scene_id / "manifest.json"
        if missing == "manifest":
            manifest.unlink()
        else:
            manifest.write_text("[]", encoding="utf-8")
    else:
        del episode._evaluation_meta["geodesic_to_target" if missing == "geodesic" else missing]
    result = score_trajectory(episode, trajectory, official, **options)
    expected = ({"G", "GSR_hit"} if missing in ("objects", "objects_empty", "manifest", "bbox")
                else {"SPL"} if missing == "geodesic" else {"SR_hit", "OSR_hit", "SPL", "GSR_hit"})
    assert set(result["unavailable_metrics"]) == expected
    assert all(result["metrics"][key] is None for key in expected)
    assert result["metrics"]["IM"] == 1
    assert not result["aggregate_ready"]
    if missing == "geodesic":
        assert result["metrics"]["SR_hit"] == result["metrics"]["GSR_hit"] == 1
        assert "geodesic_length" not in result["metrics"]


def test_record_roundtrips_through_real_official_aggregate(official_case, tmp_path):
    """The emitted record must reproduce the same per-item numbers in formal aggregation."""
    official, episode, trajectory, options = official_case
    result = score_trajectory(episode, trajectory, official, radius_m=2.0, **options)
    records = tmp_path / "isolated eval out"
    records.mkdir()
    (records / "record.json").write_text(json.dumps(result["record"]), encoding="utf-8")
    output = tmp_path / "aggregate output"
    env = dict(os.environ, EVAL_OUT_DIR=str(records),
               INTENTEQA_SCENE_SUMMARY=str(options["scene_summary_root"]))
    subprocess.run([sys.executable, "-B", str(official / "eval/aggregate/compute_metrics.py"),
                    "--out", str(output), "--radius-m", "2.0"],
                   capture_output=True, text=True, check=True, timeout=30, env=env)
    with (output / "per_item.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["style"] == "formal"
    assert rows[0]["prediction"] == "the sofa"
    assert rows[0]["target_category"] == "couch"
    for key in ("IM", "SR_hit", "OSR_hit", "SPL", "GSR_hit", "d_final", "path_length"):
        assert float(rows[0][key]) == result["metrics"][key]
    report = json.loads((output / "metrics.json").read_text())
    headline = report["headline_per_tier_model_style"]["goat/local-policy/formal"]
    assert report["radius_m"] == result["protocol"]["radius_m"]
    assert all(headline[key] == 1 for key in ("IM", "SR", "OSR", "SPL", "GSR"))
    assert "G" not in headline  # Actual upstream code, despite its docstring/header.
    assert report["style_sensitivity"]["goat/local-policy"]["CSR"] is None


@pytest.fixture
def navigation_batch(tmp_path):
    """Exercise the real batch CLI with tiny artifact writers, never models/simulators."""
    runner = Path(__file__).resolve().parents[1] / "scripts/run_navigation_batch.py"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "batch with spaces"
    script = (
        "import json, pathlib, sys; p = pathlib.Path(sys.argv[1]); p.mkdir(exist_ok=False); "
        "(p / 'launch.once').open('x').close(); "
        "(p / 'run_result.json').write_text(sys.argv[2]); "
        "(p / 'evaluation.json').write_text(sys.argv[3]); "
        "(p / 'literal_args.json').write_text(json.dumps(sys.argv[5:])); "
        "sys.exit(int(sys.argv[4]))"
    )

    def condition(scene, *, benchmark="intentionnav", status="completed", exit_code=0, score=None):
        result = {"status": status, "metrics": {"SR": 0, "SPL": 0.0},
                  "scene_load": "fixture only: no actual scene loaded", "future_field": [1, {"x": 2}]}
        score = score if score is not None else {
            "metrics": {"SR": 0, "SPL": 0.0}, "unavailable_metrics": {},
            "aggregate_ready": True, "future_schema": {"preserve": [0, None]},
        }
        return {
            "identity": {"benchmark": benchmark, "split": "val", "scene_id": scene},
            "protocol": {"style": "formal", "radius_m": 2.0},
            "versions": {"source": "fixture-source", "model": "none", "scorer": "fixture-scorer"},
            "command": [sys.executable, "-B", "-c", script, "{output}", json.dumps(result),
                        json.dumps(score), str(exit_code), "prefix{output}", "{identity}", "$(false)"],
            "assets": {}, "result_file": "run_result.json", "score_file": "evaluation.json",
        }

    def invoke(conditions=None, *, action="run", limit=None, code=0):
        if conditions is not None:
            manifest.write_text(json.dumps({"conditions": conditions}), encoding="utf-8")
        command = [sys.executable, "-B", str(runner), action, "--manifest", str(manifest), "--output", str(output)]
        if limit is not None:
            command += ["--limit", str(limit)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert process.returncode == code, process.stdout + process.stderr
        return json.loads(process.stdout) if code != 2 else process.stderr

    return condition, invoke, manifest, output


def _batch_key(condition):
    payload = {k: condition[k] for k in ("identity", "protocol", "versions")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def test_batch_zero_success_limit_resume_and_summary(navigation_batch):
    """Zero success completes; limits count new work, and summaries never pool scores."""
    make, invoke, manifest, output = navigation_batch
    conditions = [make("scene-1"), make("scene-2"), make("scene-3", benchmark="goat"), make("scene-4")]
    conditions[-1]["protocol"]["style"] = "casual"
    conditions[-1].pop("score_file")  # Optional scoring and custom runtime completion status.
    conditions[-1]["completed_status"] = "done"
    conditions[-1]["command"][5] = json.dumps({"status": "done", "metrics": {"SR": 0}})
    report = invoke(conditions, limit=1)
    assert (report["expected"], report["completed"], report["pending"], report["errors"], report["unresolved"]) == (4, 1, 3, 0, 0)
    first = output / _batch_key(conditions[0]) / "receipt.json"
    original = first.read_bytes()
    receipt = json.loads(original)
    assert receipt["condition_key"] == _batch_key(conditions[0])
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert receipt["command"] == conditions[0]["command"]
    assert receipt["exit_code"] == 0 and receipt["status"] == "completed"
    assert receipt["result"] == json.loads(conditions[0]["command"][5])
    assert receipt["score"] == json.loads(conditions[0]["command"][6])
    run = first.parent / "run"
    assert receipt["executed_command"][4] == str(run)
    assert json.loads((run / "literal_args.json").read_text()) == ["prefix" + str(run), "{identity}", "$(false)"]
    assert "not scene-load validation" in receipt["asset_preflight"]
    report = invoke(limit=1)
    assert (report["completed"], report["pending"]) == (2, 2)
    assert first.read_bytes() == original
    assert not (output / _batch_key(conditions[2])).exists()
    assert invoke()["coverage_complete"]
    frozen = {str(p): p.read_bytes() for p in output.glob("*/receipt.json")}
    report = invoke()
    assert {str(p): p.read_bytes() for p in output.glob("*/receipt.json")} == frozen
    assert len(list(output.glob("*/run/launch.once"))) == 4
    assert sorted(group["completed"] for group in report["groups"]) == [1, 1, 2]
    assert all("SR" not in group for group in report["groups"])
    assert {row["condition_key"] for row in report["conditions"]} == {_batch_key(c) for c in conditions}
    assert invoke(action="summary") == report == json.loads((output / "summary.json").read_text())


def test_batch_asset_errors_continue_and_are_not_retried(navigation_batch, tmp_path):
    """Missing and mismatched assets are not navigation failures or scene validation."""
    make, invoke, _, output = navigation_batch
    asset = tmp_path / "asset with spaces.bin"
    asset.write_bytes(b"fixture asset")
    missing = tmp_path / "missing.bin"
    conditions = [make("missing"), make("mismatch"), make("valid")]
    conditions[0]["assets"] = {str(missing): None}
    conditions[1]["assets"] = {str(asset): "0" * 64}
    conditions[2]["assets"] = {str(asset): hashlib.sha256(asset.read_bytes()).hexdigest(), str(tmp_path): None}
    report = invoke(conditions, code=1)
    assert (report["expected"], report["completed"], report["pending"], report["errors"]) == (3, 1, 0, 2)
    assert [row["status"] for row in report["conditions"]] == ["asset_error", "asset_error", "completed"]
    for condition in conditions[:2]:
        folder = output / _batch_key(condition)
        receipt = json.loads((folder / "receipt.json").read_text())
        assert receipt["exit_code"] is None and receipt["result"] is None
        assert not (folder / "run").exists()
    frozen = {str(p): p.read_bytes() for p in output.glob("*/receipt.json")}
    missing.touch()
    conditions[1]["assets"] = conditions[2]["assets"]
    assert invoke(conditions, code=1)["errors"] == 2
    assert {str(p): p.read_bytes() for p in output.glob("*/receipt.json")} == frozen


def test_batch_duplicate_keys_rejected_before_launch(navigation_batch):
    """Commands, resources and artifact paths cannot turn the same condition into two."""
    make, invoke, _, output = navigation_batch
    first = make("duplicate")
    duplicate = copy.deepcopy(first)
    duplicate["protocol"] = dict(reversed(list(first["protocol"].items())))
    duplicate.update(command=["different-command", "--gpu", "2", "--port", "1234"],
                     assets={}, result_file="different.json", resources={"gpu": 2, "port": 1234})
    assert _batch_key(first) == _batch_key(duplicate)
    assert "duplicate condition_key" in invoke([first, duplicate], code=2)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["exit", "runtime", "spawn", "unavailable", "not_ready", "missing_score"])
def test_batch_error_classification_and_evidence_protection(navigation_batch, failure):
    """Separate infrastructure/runtime errors from incomplete scoring and retain artifacts."""
    make, invoke, _, output = navigation_batch
    bad = make("bad", status="error" if failure == "runtime" else "completed",
               exit_code=7 if failure == "exit" else 0)
    if failure == "spawn":
        bad["command"] = [str(output.parent / "nonexistent-command")]
    elif failure == "unavailable":
        bad["command"][6] = json.dumps({"unavailable_metrics": {"G": "fixture missing"}})
    elif failure == "not_ready":
        bad["command"][6] = json.dumps({"aggregate_ready": False, "unavailable_metrics": {}})
    elif failure == "missing_score":
        bad["score_file"] = "missing.json"
    expected = "execution_error" if failure in ("exit", "runtime", "spawn") else "scoring_error"
    report = invoke([bad, make("good")], code=1)
    assert [row["status"] for row in report["conditions"]] == [expected, "completed"]
    assert (report["errors"], report["completed"], report["pending"]) == (1, 1, 0)
    receipt_path = output / _batch_key(bad) / "receipt.json"
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    assert receipt["exit_code"] == (None if failure == "spawn" else 7 if failure == "exit" else 0)
    if failure != "spawn":
        assert receipt["result"] == json.loads(bad["command"][5])
        if failure != "missing_score":
            assert receipt["score"] == json.loads(bad["command"][6])
    assert invoke(code=1)["errors"] == 1
    assert receipt_path.read_bytes() == original


def test_batch_running_orphaned_and_corrupt_evidence_is_blocked(navigation_batch):
    """Unresolved output is visible and cannot be silently overwritten on resume."""
    make, invoke, _, output = navigation_batch
    conditions = [make(name) for name in ("running", "orphan", "corrupt", "pending")]
    output.mkdir()
    evidence = {}
    for index, condition in enumerate(conditions[:3]):
        folder = output / _batch_key(condition)
        folder.mkdir()
        if index != 1:
            path = folder / "receipt.json"
            text = json.dumps({"condition_key": _batch_key(condition), "status": "running"}) if index == 0 else "{broken"
            path.write_text(text, encoding="utf-8")
            evidence[path] = path.read_bytes()
    report = invoke(conditions, limit=1, code=1)
    assert [row["status"] for row in report["conditions"]] == ["running", "orphaned_output", "receipt_error", "completed"]
    assert (report["expected"], report["completed"], report["pending"], report["errors"], report["unresolved"]) == (4, 1, 0, 1, 2)
    assert {path: path.read_bytes() for path in evidence} == evidence
    assert len(list(output.glob("*/run/launch.once"))) == 1
    assert invoke(action="summary", code=1) == report


def test_batch_lock_prevents_concurrent_launch(navigation_batch):
    """A second batch process fails explicitly while the output lock is held."""
    import fcntl

    make, invoke, _, output = navigation_batch
    output.mkdir()
    with (output / ".batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert "batch is locked" in invoke([make("locked")], code=2)
        assert not list(output.glob("*/receipt.json"))
    assert invoke()["completed"] == 1


@pytest.fixture
def launcher_case(dataset, fake_official, monkeypatch):
    """Run main with local protocol artifacts; no simulator, policy or scorer process."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts import launch_intentionnav_episode as launcher
    from scripts import serve_intentionnav as server_module

    root, meta, intent = dataset
    project = root / "launcher project"
    project.mkdir()
    (root / "results").mkdir()
    environment = {
        "STRIVE_ROOT": str(project), "INTENTIONNAV_DATA_ROOT": str(root),
        "INTENTIONNAV_ROOT": str(fake_official), "ISAACSIM_ROOT": str(root / "fake isaac"),
        "VLN_LVLM_API_KEY": "fixture-only-key", "HF_HOME_HOST": str(root / "fake cache"),
        "SAM_CHECKPOINT": str(root / "fake sam"),
        "GROUNDING_DINO_CHECKPOINT": str(root / "fake dino"),
        "PYTHONPATH": "fixture-pythonpath", "PYTHONHOME": "fixture-pythonhome",
        "LD_LIBRARY_PATH": "fixture-libraries", "LD_PRELOAD": "fixture-preload",
        "CUDA_VISIBLE_DEVICES": "fixture-gpu", "LLM_OFFLINE": "1",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assets = {}
    for scene_id in (meta["scene_id"], "kujiale_changed"):
        scene = root / "scenes" / scene_id / "start_result_navigation.usd"
        freemap = root / "metadata/metadata_train" / scene_id / "freemap.npy"
        for path in (scene, freemap):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture-only {scene_id}/{path.name}".encode())
        assets[scene_id] = (scene, freemap)

    case = SimpleNamespace(
        root=root, source=root / "public input.json", output=root / "launcher output",
        public={"selection_id": meta["selection_id"], "scene_id": meta["scene_id"],
                "style": "formal", "instruction": intent["formal_en"],
                "start_position": list(meta["start_position"]),
                "start_rotation_quat_wxyz": list(meta["start_rotation_quat_wxyz"])},
        assets=assets, consumers=[], events=[], snapshot_writes=[], public_supplied=False,
        mutate_source=False, server_exit=None,
        scorer_reply={"metrics": dict.fromkeys(("IM", "SR_hit", "OSR_hit", "SPL", "G", "GSR_hit"), 0),
                      "unavailable_metrics": {}, "visibility": {}},
    )

    def option(command, name):
        return command[command.index(name) + 1]

    def consume(command):
        path = Path(option(command, "--public-episode"))
        case.output = Path(option(command, "--output"))
        contents = path.read_bytes()
        case.consumers.append((path, contents))
        assert path == case.output / "public_episode.json"
        return json.loads(contents)

    def inspect_image(command, **kwargs):
        if case.public_supplied:
            assert [call.args[0] for call in case.validate.call_args_list] == [case.source]
        case.events.append("image_inspect")
        assert command == ["docker", "image", "inspect", "strive-hm3d:local", "--format", "{{.Id}}"]
        assert kwargs == {"text": True}
        return "sha256:fixture-only-image\n"

    def wait_server(timeout):
        assert timeout == 60
        case.server_exit = 0
        return 0

    server = SimpleNamespace(pid=424242, poll=lambda: case.server_exit,
                             wait=Mock(side_effect=wait_server))

    def start_server(command, **kwargs):
        case.events.append("server_start")
        public = consume(command)
        assert command[0] == str(root / "fake isaac/python.sh")
        assert Path(command[1]) == case.output / "source/scripts/serve_intentionnav.py"
        assert option(command, "--data-root") == str(root)
        assert kwargs["cwd"] == case.output / "source"
        assert kwargs["start_new_session"] is True
        assert not set(("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD",
                        "CUDA_VISIBLE_DEVICES")) & kwargs["env"].keys()
        case.ready = {"pid": server.pid, "port": int(option(command, "--port")),
                      "isaac_version": "fixture-only-not-loaded",
                      "start_pose": {"position": public["start_position"], "yaw": 0.0}}
        (case.output / "server_ready.json").write_text(json.dumps(case.ready), encoding="utf-8")
        if case.mutate_source:
            case.source.write_text('{"private_field": "changed after simulator launch"}', encoding="utf-8")
        return server

    def run_process(command, **kwargs):
        if command[0] == "docker":
            case.events.append("policy_run")
            assert command[:3] == ["docker", "run", "--rm"]
            assert "scripts/run_intentionnav.py" in command
            assert kwargs["timeout"] == 7200
            assert kwargs["env"]["VLN_LVLM_API_KEY"] == environment["VLN_LVLM_API_KEY"]
            public = consume(command)
            final = list(public["start_position"])
            final[0] += 0.25
            case.trajectory = [
                {"step": 0, "action": "observe", "position": public["start_position"], "yaw": 0.0},
                {"step": 1, "action": "move", "position": final, "yaw": 0.25},
                {"step": 2, "action": "STOP", "position": final, "yaw": 0.25},
            ]
            trajectory_bytes = ("\n".join(json.dumps(row) for row in case.trajectory) + "\n").encode()
            (case.output / "trajectory.jsonl").write_bytes(trajectory_bytes)
            case.policy = {
                "status": "completed", "reason": "max_moves", "task_complete": False,
                "predicted_target": "fixture-only prediction", "model_calls": 0, "moves": 1,
                "stop_receipt": {"trajectory_sha256": hashlib.sha256(trajectory_bytes).hexdigest()},
                "final_frame": {"pose": {"position": final, "yaw": 0.25}},
            }
            (case.output / "policy_result.json").write_text(json.dumps(case.policy), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)
        case.events.append("score")
        assert command[:3] == [sys.executable, "-B", "-c"]
        assert Path(command[-1]) == fake_official / "eval/aggregate/compute_metrics.py"
        assert kwargs["check"] is kwargs["text"] is kwargs["capture_output"] is True
        assert kwargs["timeout"] == 30
        case.score_payload = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(case.scorer_reply))

    def shutdown(port, action):
        case.events.append("shutdown")
        assert (port, action) == (case.ready["port"], "shutdown")
        return {"action": "shutdown", "fixture_only": True}

    case.processes = SimpleNamespace(check_output=Mock(side_effect=inspect_image),
        Popen=Mock(side_effect=start_server), run=Mock(side_effect=run_process), STDOUT=subprocess.STDOUT)
    monkeypatch.setattr(launcher, "subprocess", case.processes)
    monkeypatch.setattr(intentionnav, "subprocess", case.processes)
    monkeypatch.setattr(launcher, "shutil", SimpleNamespace(
        copytree=Mock(), copy2=Mock(), ignore_patterns=shutil.ignore_patterns))
    monkeypatch.setattr(launcher, "command", Mock(side_effect=shutdown))
    monkeypatch.setattr(launcher.os, "killpg", Mock(side_effect=AssertionError("unexpected process signal")))
    monkeypatch.setattr(launcher.time, "sleep", Mock(side_effect=AssertionError("unexpected readiness wait")))
    case.load = Mock(wraps=load_episode)
    case.score = Mock(wraps=score_trajectory)
    case.validate = Mock(wraps=server_module.load_public_episode)
    monkeypatch.setattr(intentionnav, "load_episode", case.load)
    monkeypatch.setattr(intentionnav, "score_trajectory", case.score)
    monkeypatch.setattr(server_module, "load_public_episode", case.validate)
    original_open = Path.open

    def track_open(path, mode="r", *args, **kwargs):
        if path.name == "public_episode.json" and any(flag in mode for flag in "wax+"):
            case.snapshot_writes.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)

    def invoke(*args, public=None, default_output=False, mutate_source=False):
        command = ["launch_intentionnav_episode.py", *args]
        case.public_supplied = public is not None
        if public is not None:
            case.source.write_text(json.dumps(public, ensure_ascii=False), encoding="utf-8")
            command += ["--public-episode", str(case.source)]
        if not default_output:
            command += ["--output", str(case.output)]
        case.mutate_source = mutate_source
        monkeypatch.setattr(sys, "argv", command)
        assert launcher.main() is None
        for name in ("execution", "evaluation", "run_result"):
            setattr(case, name, json.loads((case.output / (name + ".json")).read_text()))
        return case

    case.invoke = invoke
    return case


def _assert_launcher_completed(case, expected_public, *, scored):
    result, execution, evaluation = case.run_result, case.execution, case.evaluation
    assert result["status"] == "completed"
    assert result["policy_exit_code"] == result["server_exit_code"] == 0
    assert result["server"] == case.ready
    assert result["shutdown_receipt"] == {"action": "shutdown", "fixture_only": True}
    assert result["task_complete"] is evaluation["task_complete"] is False
    assert result["termination"] == evaluation["policy_termination"] == "max_moves"
    assert result["model_calls"] == 0 and result["moves"] == 1
    assert execution["run_id"] == result["run_id"]
    assert execution["policy_command"][execution["policy_command"].index("--name") + 1] == result["run_id"]
    assert "VLN_LVLM_API_KEY" not in execution["policy_env"]
    assert len(case.consumers) == 2 and case.consumers[0] == case.consumers[1]
    snapshot, contents = case.consumers[0]
    assert snapshot.read_bytes() == contents
    assert json.loads(contents) == expected_public
    assert set(expected_public) == {"selection_id", "scene_id", "style", "instruction",
                                    "start_position", "start_rotation_quat_wxyz"}
    assert case.snapshot_writes == [snapshot]
    if case.public_supplied:
        assert [call.args[0] for call in case.validate.call_args_list].count(case.source) == 1
    assert {artifact["public_episode_sha256"] for artifact in (execution, evaluation, result)} == {
        hashlib.sha256(contents).hexdigest()}
    trajectory_hash = hashlib.sha256((case.output / "trajectory.jsonl").read_bytes()).hexdigest()
    assert result["trajectory_sha256"] == evaluation["trajectory_sha256"] == trajectory_hash
    assert result["stop_receipt"] == case.policy["stop_receipt"] == {"trajectory_sha256": trajectory_hash}
    assert evaluation["trajectory_steps"] == len(case.trajectory)
    assert case.trajectory[-1]["action"] == "STOP"
    assert {key: case.trajectory[-1][key] for key in ("position", "yaw")} == case.policy["final_frame"]["pose"]
    scene, freemap = case.assets[expected_public["scene_id"]]
    assert evaluation["scene_sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert evaluation["freemap_sha256"] == hashlib.sha256(freemap.read_bytes()).hexdigest()
    binding = result["evaluation_binding"]
    assert binding == evaluation["evaluation_binding"] == execution["evaluation_binding"]
    assert result["metrics"] == evaluation["metrics"]
    assert result["aggregate_ready"] is evaluation["aggregate_ready"] is scored
    assert case.events == ["image_inspect", "server_start", "policy_run", *(["score"] if scored else []), "shutdown"]
    if scored:
        assert binding == {"status": "matched", "source": "official_dataset"}
        case.score.assert_called_once()
        episode = case.score.call_args.args[0]
        assert episode._evaluation_meta == evaluation["episode_meta"]
        assert evaluation["dataset_sha256"] == episode.source_sha256
        assert case.score_payload["record"] == evaluation["record"]
        assert evaluation["record"]["trajectory"] == case.trajectory
        assert evaluation["record"]["prediction"] == {"target": case.policy["predicted_target"]}
        assert case.score_payload["options"] == {
            "radius_m": 2.0, "dataset_root": str(case.root / "dataset"),
            "scene_summary_root": str(case.root / "metadata/kujiale_scene_summary"),
        }
        assert evaluation["protocol"]["radius_m"] == 2.0
        assert evaluation["metrics"] == case.scorer_reply["metrics"]
    else:
        case.score.assert_not_called()
        assert binding["status"] == "unavailable" and binding["reason"]
        assert evaluation["metrics"] == dict.fromkeys(("IM", "SR_hit", "OSR_hit", "SPL", "G", "GSR_hit"))
        assert set(evaluation["unavailable_metrics"]) == set(evaluation["metrics"])
        assert all(evaluation["unavailable_metrics"].values())
        assert not {"episode_meta", "dataset_sha256", "source_sha256", "record"} & evaluation.keys()


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit-selection-style"])
def test_launcher_golden_protocol_preserves_dataset_selection(launcher_case, dataset, explicit):
    root, meta, intent = dataset
    case = launcher_case
    args = []
    expected = copy.deepcopy(case.public)
    if explicit:
        other_meta = dict(meta, selection_id="SEL_OTHER")
        other_intent = dict(intent, selection_id="SEL_OTHER", natural_en="  A different instruction.\n")
        _write_dataset(root, [meta, other_meta], [intent, other_intent])
        args = ["--selection-id", "SEL_OTHER", "--style", "natural"]
        expected.update(selection_id="SEL_OTHER", style="natural", instruction=other_intent["natural_en"])
    case.invoke(*args)
    _assert_launcher_completed(case, expected, scored=True)
    case.load.assert_called_once_with(root, expected["selection_id"], expected["style"])
    assert case.score.call_args.args[0]._evaluation_meta == (other_meta if explicit else meta)
    assert case.run_result["run_id"].startswith(f"strive-{expected['selection_id']}-guided-isaac6-")


@pytest.mark.parametrize("style", ["formal", "natural"])
def test_launcher_matching_public_uses_original_gt_and_frozen_input(launcher_case, dataset, style):
    root, meta, intent = dataset
    intent[style + "_en"] = "  请找个地方休息。\nKeep this exact text.  "
    _write_dataset(root, [meta], [intent])
    case = launcher_case
    public = dict(case.public, style=style, instruction=intent[style + "_en"])
    case.invoke(public=public, mutate_source=True)
    _assert_launcher_completed(case, public, scored=True)
    case.load.assert_called_once_with(root, "SEL_001", style)
    episode = case.score.call_args.args[0]
    assert isinstance(episode.start_position, tuple)
    assert isinstance(public["start_position"], list)
    assert episode._evaluation_meta == meta
    assert case.source.read_bytes() != case.consumers[0][1]


@pytest.mark.parametrize("field,value", [
    ("instruction", "  A changed instruction.\n不要重写。  "),
    ("scene_id", "kujiale_changed"),
    ("start_position", [3.0, 4.0, 1.5]),
    ("start_rotation_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
    ("style", "casual"),
    ("selection_id", "SEL_UNKNOWN"),
])
def test_launcher_changed_public_completes_without_gt_scoring(launcher_case, field, value):
    case = launcher_case
    public = dict(case.public, **{field: value})
    case.invoke(public=public, mutate_source=True)
    _assert_launcher_completed(case, public, scored=False)
    case.load.assert_called_once_with(case.root, public["selection_id"], public["style"])
    assert case.source.read_bytes() != case.consumers[0][1]
    if field != "selection_id":
        binding = case.evaluation["evaluation_binding"]
        assert binding["reason"] == "public_episode_mismatch"
        assert binding["mismatched_fields"] == (["instruction"] if field == "style" else [field])


@pytest.mark.parametrize("original_dataset", ["missing", "invalid", "unreadable"])
def test_launcher_public_runs_without_usable_original_dataset(launcher_case, original_dataset):
    case = launcher_case
    if original_dataset == "missing":
        for path in (case.root / "dataset").iterdir():
            path.unlink()
        (case.root / "dataset").rmdir()
    elif original_dataset == "invalid":
        (case.root / "dataset/episodes.jsonl").write_text("{invalid json\n", encoding="utf-8")
    else:
        case.load.side_effect = PermissionError("fixture-only dataset permission failure")
    case.invoke(public=case.public)
    _assert_launcher_completed(case, case.public, scored=False)
    assert case.run_result["evaluation_binding"]["reason"].startswith("official_episode_unavailable:")


def test_launcher_public_snapshot_stays_within_consumer_size_limit(launcher_case):
    from backends.isaac.environment import MAX_BODY_BYTES

    case = launcher_case
    public = dict(case.public, instruction="")
    overhead = len(json.dumps(public, ensure_ascii=False).encode("utf-8"))
    public["instruction"] = "x" * (MAX_BODY_BYTES - overhead)
    case.invoke(public=public)
    _assert_launcher_completed(case, public, scored=False)
    assert len(case.source.read_bytes()) == MAX_BODY_BYTES
    assert len(case.consumers[0][1]) <= MAX_BODY_BYTES
    assert case.validate.call_args_list[-1].args == (case.consumers[0][0],)


@pytest.mark.parametrize("option", ["--selection-id", "--style"])
def test_launcher_empty_dataset_selector_is_not_replaced_by_a_default(launcher_case, option):
    case = launcher_case
    with pytest.raises(ValueError):
        case.invoke(option, "")
    for process in (case.processes.check_output, case.processes.Popen, case.processes.run):
        process.assert_not_called()
    assert not case.output.exists()


def test_launcher_public_run_id_and_asset_hash_use_validated_input(launcher_case):
    case = launcher_case
    public = dict(case.public, selection_id="../../unvalidated-selection", scene_id="kujiale_changed",
                  instruction="  原文保留。\nNo rewriting.  ")
    case.invoke(public=public, default_output=True, mutate_source=True)
    _assert_launcher_completed(case, public, scored=False)
    input_hash = hashlib.sha256(json.dumps(public, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    run_id = case.run_result["run_id"]
    assert run_id.startswith(f"strive-public-{input_hash[:12]}-guided-isaac6-")
    assert "/" not in run_id and ".." not in run_id and "unvalidated-selection" not in run_id
    assert case.output == case.root / "results" / run_id
    assert case.source.read_bytes() != case.consumers[0][1]
    original_scene, original_freemap = case.assets[case.public["scene_id"]]
    assert case.evaluation["scene_sha256"] != hashlib.sha256(original_scene.read_bytes()).hexdigest()
    assert case.evaluation["freemap_sha256"] != hashlib.sha256(original_freemap.read_bytes()).hexdigest()


@pytest.mark.parametrize("args", [("--selection-id", "SEL_001"), ("--style", "formal")])
def test_launcher_public_cli_conflicts_fail_before_subprocess(launcher_case, capsys, args):
    case = launcher_case
    with pytest.raises(SystemExit) as error:
        case.invoke(*args, public=case.public)
    assert error.value.code == 2
    assert args[0] in capsys.readouterr().err
    case.validate.assert_not_called()
    case.load.assert_not_called()
    case.score.assert_not_called()
    for process in (case.processes.check_output, case.processes.Popen, case.processes.run):
        process.assert_not_called()
    assert not case.output.exists()


@pytest.mark.parametrize("change,message", [
    ({"target_position": [8.0, 9.0, 1.0]}, "six public fields"),
    ({"scene_id": "../scene"}, "safe scene name"),
    ({"start_position": [1, 2]}, "start_position"),
    ({"start_rotation_quat_wxyz": [0, 0, 0, 0]}, "unit quaternion"),
    ({"instruction": " \n"}, "nonempty text"),
])
def test_launcher_invalid_public_fails_before_subprocess(launcher_case, change, message):
    from backends.isaac.environment import CommandError

    case = launcher_case
    with pytest.raises(CommandError, match=message):
        case.invoke(public=dict(case.public, **change))
    case.validate.assert_called_once_with(case.source)
    case.load.assert_not_called()
    case.score.assert_not_called()
    for process in (case.processes.check_output, case.processes.Popen, case.processes.run):
        process.assert_not_called()
    assert not case.output.exists()
