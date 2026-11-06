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
