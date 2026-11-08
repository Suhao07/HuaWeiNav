import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from scripts import prepare_demand_acceptance as freezer


STYLES = ("formal", "natural", "casual", "emotional")
RELATIONS = {"spatial", "temporal", "conditional", "exclusion", "reference"}
CONSTRAINTS = [
    {"kind": "attribute", "quote": "red", "requirement": "The chair must be red."},
    {"kind": "spatial", "quote": "beside the window", "requirement": "The chair must be beside the window."},
    {"kind": "state", "quote": "clean", "requirement": "The chair must be clean."},
    {"kind": "temporal", "quote": "before dusk", "requirement": "Arrive before dusk."},
    {"kind": "conditional", "quote": "if it rains", "requirement": "Go only if it rains."},
    {"kind": "functional", "quote": "for reading", "requirement": "The chair must support reading."},
    {"kind": "quantity", "quote": "two cushions", "requirement": "The chair must have two cushions."},
    {"kind": "exclusion", "quote": "not the sofa", "requirement": "Do not select the sofa."},
    {"kind": "reference", "quote": "that one", "requirement": "Select the previously indicated chair."},
    {"kind": "attribute", "quote": "large", "requirement": "The chair must be large."},
]


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def dataset(tmp_path):
    metadata, episodes, annotations = [], [], []
    patterns = [[], [0], [1], [0, 9], [0, 2], [1, 3], [0, 2, 5], [1, 7, 8],
                [0, 1, 3, 4], list(range(9)), [0, 9, 1], [0, 1, 2]]
    for number in range(500):
        selection_id = f"SEL_{number:03d}"
        scene_id = f"scene_{number % 17:02d}"
        item = {"selection_id": selection_id, "scene_id": scene_id,
                "difficulty": "historical-hard", "asset_available": False,
                "target_category": "private-metadata-category"}
        for style in STYLES:
            instruction = (f"  {style} request {number}: find the large red chair beside the window; "
                           "it must be clean, before dusk, if it rains, for reading, "
                           "with two cushions, not the sofa; use that one.\n保留原文。  ")
            item[style + "_en"] = instruction
            annotations.append({"condition_id": f"{selection_id}:{style}",
                                "instruction_sha256": _digest(instruction.encode("utf-8")),
                                "constraints": copy.deepcopy([CONSTRAINTS[i] for i in patterns[number % 12]]),
                                "annotation_call_id": f"local-qwen-call-{number}-{style}"})
        metadata.append(item)
        episodes.append({"selection_id": selection_id, "scene_id": scene_id,
                         "start_position": [number, 2.0, 1.5],
                         "start_rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                         "target_object_id": f"private-target-{number}",
                         "target_position": [number + 3.0, 9.0, 1.0],
                         "future_gt": {"z": "私有目标", "a": {"rank": number}}})
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("metadata", "episodes", "annotations")}
    _write_rows(paths["metadata"], metadata)
    _write_rows(paths["episodes"], list(reversed(episodes)))
    _write_rows(paths["annotations"], annotations)
    return paths, metadata, episodes, annotations


def _freeze(dataset, output, seed=20261108):
    paths = dataset[0]
    return freezer.freeze_demand(paths["metadata"], paths["episodes"], paths["annotations"], output, seed)


def _expected_rank(annotation, seed):
    kinds = [constraint["kind"] for constraint in annotation["constraints"]]
    n, r, t = len(kinds), sum(kind in RELATIONS for kind in kinds), len(set(kinds))
    return -n, -r, -t, _digest(f"{seed}:{annotation['condition_id']}".encode("utf-8"))


def test_full_freeze_preserves_text_binding_and_audit(dataset, tmp_path):
    paths, metadata, episodes, annotations = dataset
    originals = {name: path.read_bytes() for name, path in paths.items()}
    output = tmp_path / "frozen"
    receipt = _freeze(dataset, output)
    assert {path.name for path in output.iterdir()} == {
        "demand_2000.jsonl", "hard_500.jsonl", "regression_10.jsonl", "selection_receipt.json"}
    demand = _read_rows(output / "demand_2000.jsonl")
    hard = _read_rows(output / "hard_500.jsonl")
    regression = _read_rows(output / "regression_10.jsonl")
    assert len(demand) == len({row["condition_id"] for row in demand}) == 2000
    assert Counter(row["style"] for row in demand) == dict.fromkeys(STYLES, 500)
    intents = {row["selection_id"]: row for row in metadata}
    bound = {row["selection_id"]: row for row in episodes}
    by_id = {row["condition_id"]: row for row in demand}
    for row in demand:
        selection_id, style = row["selection_id"], row["style"]
        instruction = intents[selection_id][style + "_en"]
        canonical_episode = json.dumps(bound[selection_id], sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":"), allow_nan=False).encode("utf-8")
        assert row == {"condition_id": f"{selection_id}:{style}", "task_id": selection_id,
                       "selection_id": selection_id, "scene_id": bound[selection_id]["scene_id"],
                       "style": style, "instruction": instruction,
                       "instruction_sha256": _digest(instruction.encode("utf-8")),
                       "episode_sha256": _digest(canonical_episode)}
    eligible = [row for row in annotations if len(row["constraints"]) >= 2 or
                any(c["kind"] in RELATIONS for c in row["constraints"])]
    expected_hard = sorted(eligible, key=lambda row: _expected_rank(row, 20261108))[:500]
    assert len(hard) == len({row["condition_id"] for row in hard}) == 500
    assert [row["condition_id"] for row in hard] == [row["condition_id"] for row in expected_hard]
    for row, annotation in zip(hard, expected_hard):
        assert {key: row[key] for key in by_id[row["condition_id"]]} == by_id[row["condition_id"]]
        n, r, t, _ = _expected_rank(annotation, 20261108)
        assert row["difficulty"] == {"n": -n, "r": -r, "t": -t, "eligible": True}
        assert row["constraints"] == annotation["constraints"]
        assert row["annotation_call_id"] == annotation["annotation_call_id"]
    expected_regression, seen = [], set()
    for row in hard:
        if row["selection_id"] not in seen:
            expected_regression.append(row)
            seen.add(row["selection_id"])
    assert regression == expected_regression[:10]
    assert len({row["selection_id"] for row in regression}) == 10
    assert len({row["task_id"] for row in regression}) == 10
    assert receipt == json.loads((output / "selection_receipt.json").read_text(encoding="utf-8"))
    assert receipt["rubric"] == "complex_conditions_v1"
    assert receipt["seed"] == 20261108
    assert receipt["counts"] == {
        "demand_conditions": 2000, "demand_geometry": 500, "eligible_conditions": len(eligible),
        "hard_conditions": 500, "hard_geometry": len({row["selection_id"] for row in hard}),
        "hard_scenes": len({row["scene_id"] for row in hard}),
        "regression_conditions": 10, "regression_geometry": 10}
    assert receipt["subsets"] == {"hard_500_in_demand_2000": True, "regression_10_in_hard_500": True}
    assert receipt["selected10"] == [row["condition_id"] for row in regression]
    assert receipt["source_sha256"] == {"intention_metadata": _digest(originals["metadata"]),
                                         "episodes": _digest(originals["episodes"]),
                                         "annotations": _digest(originals["annotations"])}
    for name in ("demand_2000.jsonl", "hard_500.jsonl", "regression_10.jsonl"):
        assert receipt["manifest_sha256"][name] == _digest((output / name).read_bytes())
    for name, rows in (("demand_2000", annotations), ("hard_500", hard), ("regression_10", regression)):
        distribution = receipt["distributions"][name]
        style_counts = Counter(row["condition_id"].rsplit(":", 1)[1] for row in rows)
        assert distribution["style"] == {style: style_counts[style] for style in STYLES}
        for kind in {c["kind"] for c in CONSTRAINTS}:
            assert distribution["condition_kind"][kind] == sum(
                any(c["kind"] == kind for c in row["constraints"]) for row in rows)
            assert distribution["constraint_kind"][kind] == sum(
                c["kind"] == kind for row in rows for c in row["constraints"])
    assert receipt["ranking"]["eligible"] == "n >= 2 or r >= 1"
    assert set(receipt["ranking"]["relation_kinds"]) == RELATIONS
    assert "not navigation acceptance" in receipt["reporting"]
    assert "not runtime gates" in receipt["reporting"]
    assert all("private-" not in path.read_text(encoding="utf-8") for path in output.iterdir())
    assert {name: path.read_bytes() for name, path in paths.items()} == originals


def test_deterministic_order_and_episode_canonicalization(dataset, tmp_path):
    paths, metadata, episodes, annotations = dataset
    _freeze(dataset, tmp_path / "first")
    _freeze(dataset, tmp_path / "repeat")
    for path in (tmp_path / "first").iterdir():
        assert path.read_bytes() == (tmp_path / "repeat" / path.name).read_bytes()
    rng = random.Random(41)
    for name, rows in (("metadata", metadata), ("episodes", episodes), ("annotations", annotations)):
        rng.shuffle(rows)
        _write_rows(paths[name], [{key: row[key] for key in reversed(row)} for row in rows])
    _freeze(dataset, tmp_path / "shuffled")
    for name in ("demand_2000.jsonl", "hard_500.jsonl", "regression_10.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "shuffled" / name).read_bytes()
    receipt = _freeze(dataset, tmp_path / "other-seed", seed=123)
    eligible = [row for row in annotations if len(row["constraints"]) >= 2 or
                any(c["kind"] in RELATIONS for c in row["constraints"])]
    expected = sorted(eligible, key=lambda row: _expected_rank(row, 123))[:500]
    hard = _read_rows(tmp_path / "other-seed" / "hard_500.jsonl")
    assert [row["condition_id"] for row in hard] == [row["condition_id"] for row in expected]
    assert receipt["seed"] == 123
    assert hard != _read_rows(tmp_path / "first" / "hard_500.jsonl")


def test_rank_uses_n_then_r_then_t_not_historical_proxies(dataset, tmp_path):
    paths, metadata, _, annotations = dataset
    for row in annotations:
        row["constraints"] = [CONSTRAINTS[1]]
    for row, indices in zip(annotations, ([0, 9, 2], [0, 2, 5], [0, 9, 1], [0, 9, 2, 5])):
        row["constraints"] = [CONSTRAINTS[index] for index in indices]
    _write_rows(paths["annotations"], annotations)
    _freeze(dataset, tmp_path / "ranked")
    hard = _read_rows(tmp_path / "ranked" / "hard_500.jsonl")
    assert [row["condition_id"] for row in hard[:4]] == [annotations[i]["condition_id"] for i in (3, 2, 1, 0)]
    for item in metadata:
        item.update(difficulty="easy", asset_available=True, model_failed=True, historical_score=9999)
    _write_rows(paths["metadata"], metadata)
    _freeze(dataset, tmp_path / "changed-labels")
    assert (tmp_path / "ranked" / "hard_500.jsonl").read_bytes() == (
        tmp_path / "changed-labels" / "hard_500.jsonl").read_bytes()


def test_only_relation_kinds_are_eligible_as_single_constraints(dataset, tmp_path):
    paths, _, _, annotations = dataset
    for index, row in enumerate(annotations):
        row["constraints"] = [CONSTRAINTS[index % 9]]
    _write_rows(paths["annotations"], annotations)
    receipt = _freeze(dataset, tmp_path / "singletons")
    assert receipt["counts"]["eligible_conditions"] == sum(
        row["constraints"][0]["kind"] in RELATIONS for row in annotations)
    for row in _read_rows(tmp_path / "singletons" / "hard_500.jsonl"):
        assert row["difficulty"] == {"n": 1, "r": 1, "t": 1, "eligible": True}
        assert row["constraints"][0]["kind"] in RELATIONS


@pytest.mark.parametrize("existing", ["directory", "empty_directory", "file", "symlink"])
def test_existing_output_never_overwritten(dataset, tmp_path, existing):
    output = tmp_path / "historical"
    if existing == "directory":
        _freeze(dataset, output)
    elif existing == "empty_directory":
        output.mkdir()
    elif existing == "file":
        output.write_bytes(b"historical manifest")
    else:
        output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    before = ({path.name: path.read_bytes() for path in output.iterdir()} if output.is_dir()
              else output.read_bytes() if output.is_file() else output.readlink())
    with pytest.raises(FileExistsError, match="must be new"):
        _freeze(dataset, output)
    after = ({path.name: path.read_bytes() for path in output.iterdir()} if output.is_dir()
             else output.read_bytes() if output.is_file() else output.readlink())
    assert before == after


@pytest.mark.parametrize("case, message", [
    ("missing", "2000 annotation rows, found 1999"),
    ("extra", "2000 annotation rows, found 2001"),
    ("duplicate", "duplicate condition_id"),
    ("unknown", "missing 1, extra 1"),
    ("hash", "instruction_sha256 mismatch"),
    ("quote", "quote is not an exact"),
    ("empty_quote", "quote.*nonempty"),
    ("blank_quote", "quote.*nonempty"),
    ("bad_quote_type", "quote.*nonempty"),
    ("empty_requirement", "requirement.*nonempty"),
    ("bad_requirement_type", "requirement.*nonempty"),
    ("kind", "invalid constraint kind"),
    ("bad_kind_type", "invalid constraint kind"),
    ("call_id", "annotation_call_id.*nonempty"),
    ("missing_field", "invalid annotation fields"),
    ("extra_field", "invalid annotation fields"),
    ("constraints", "constraints must be a list"),
    ("constraint_fields", "invalid constraint fields"),
    ("constraint_type", "invalid constraint fields"),
    ("duplicate_requirement", "not an independent constraint"),
])
def test_invalid_annotations_fail_without_output(dataset, tmp_path, case, message):
    paths, _, _, annotations = dataset
    row = annotations[0]
    row["constraints"] = copy.deepcopy([CONSTRAINTS[0]])
    constraint = row["constraints"][0]
    if case == "missing":
        annotations.pop()
    elif case == "extra":
        annotations.append(copy.deepcopy(row))
    elif case == "duplicate":
        annotations[-1] = copy.deepcopy(row)
    elif case == "unknown":
        row["condition_id"] = "UNKNOWN:formal"
    elif case == "hash":
        row["instruction_sha256"] = "0" * 64
    elif case == "quote":
        constraint["quote"] = "this text is not in the instruction"
    elif case == "empty_quote":
        constraint["quote"] = ""
    elif case == "blank_quote":
        constraint["quote"] = "  "
    elif case == "bad_quote_type":
        constraint["quote"] = None
    elif case == "empty_requirement":
        constraint["requirement"] = " \n"
    elif case == "bad_requirement_type":
        constraint["requirement"] = []
    elif case == "kind":
        constraint["kind"] = "emotion"
    elif case == "bad_kind_type":
        constraint["kind"] = []
    elif case == "call_id":
        row["annotation_call_id"] = ""
    elif case == "missing_field":
        del row["annotation_call_id"]
    elif case == "extra_field":
        row["difficulty"] = 999
    elif case == "constraints":
        row["constraints"] = {}
    elif case == "constraint_fields":
        del constraint["requirement"]
    elif case == "constraint_type":
        row["constraints"] = ["red"]
    elif case == "duplicate_requirement":
        duplicate = dict(constraint, quote="red chair", kind="state", requirement="  THE CHAIR MUST BE RED. ")
        row["constraints"].append(duplicate)
    _write_rows(paths["annotations"], annotations)
    with pytest.raises(ValueError, match=message):
        _freeze(dataset, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("eligible_count", [0, 499])
def test_insufficient_eligible_reports_actual_count(dataset, tmp_path, eligible_count):
    paths, _, _, annotations = dataset
    for index, row in enumerate(annotations):
        row["constraints"] = [CONSTRAINTS[0], CONSTRAINTS[2]] if index < eligible_count else []
    _write_rows(paths["annotations"], annotations)
    with pytest.raises(ValueError, match=f"Only {eligible_count} eligible conditions; need at least 500"):
        _freeze(dataset, tmp_path / "too-few")
    assert not (tmp_path / "too-few").exists()


@pytest.mark.parametrize("case, message", [
    ("missing_task", "500 unique IntentionNav tasks, found 499"),
    ("extra_task", "500 unique IntentionNav tasks, found 501"),
    ("duplicate_task", "duplicate selection_id"),
    ("missing_episode", "Missing episode"),
    ("duplicate_episode", "duplicate selection_id"),
    ("scene_mismatch", "scene_id mismatch"),
    ("missing_style", "casual_en.*nonempty"),
    ("invalid_instruction", "emotional_en.*nonempty"),
])
def test_invalid_task_and_episode_inputs(dataset, tmp_path, case, message):
    paths, metadata, episodes, _ = dataset
    if case == "missing_task":
        metadata.pop()
    elif case == "extra_task":
        metadata.append(dict(metadata[0], selection_id="EXTRA"))
    elif case == "duplicate_task":
        metadata[-1] = metadata[0]
    elif case == "missing_episode":
        episodes.pop()
    elif case == "duplicate_episode":
        episodes.append(episodes[0])
    elif case == "scene_mismatch":
        episodes[0]["scene_id"] = "wrong-scene"
    elif case == "missing_style":
        del metadata[0]["casual_en"]
    elif case == "invalid_instruction":
        metadata[0]["emotional_en"] = 42
    _write_rows(paths["metadata"], metadata)
    _write_rows(paths["episodes"], episodes)
    with pytest.raises(ValueError, match=message):
        _freeze(dataset, tmp_path / "bad-input")
    assert not (tmp_path / "bad-input").exists()


@pytest.mark.parametrize("content", ["{bad json}\n", "[]\n", "\n", '{"x": NaN}\n', '{"x": 1, "x": 2}\n'])
def test_malformed_jsonl_identifies_source_line(dataset, tmp_path, content):
    paths = dataset[0]
    paths["annotations"].write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=r"annotations\.jsonl:1:"):
        _freeze(dataset, tmp_path / "bad-json")
    assert not (tmp_path / "bad-json").exists()


def test_explicit_regression_ids_are_fixed_members_of_hard_subset(dataset, tmp_path):
    _freeze(dataset, tmp_path / "baseline")
    rows = _read_rows(tmp_path / "baseline" / "hard_500.jsonl")
    chosen, seen = [], set()
    for row in reversed(rows):
        if row["selection_id"] not in seen:
            chosen.append(row["condition_id"])
            seen.add(row["selection_id"])
        if len(chosen) == 10:
            break
    paths = dataset[0]
    receipt = freezer.freeze_demand(paths["metadata"], paths["episodes"], paths["annotations"],
                                     tmp_path / "explicit", regression_ids=chosen)
    assert receipt["selected10"] == chosen
    assert "Explicit fixed" in receipt["ranking"]["regression_selection"]
    for invalid in (chosen[:9], chosen[:9] + chosen[:1], chosen[:9] + ["UNKNOWN:formal"]):
        with pytest.raises(ValueError):
            freezer.freeze_demand(paths["metadata"], paths["episodes"], paths["annotations"],
                                   tmp_path / "invalid", regression_ids=invalid)
    same_task = [r["condition_id"] for r in rows if r["selection_id"] == rows[0]["selection_id"]]
    with pytest.raises(ValueError, match="10 distinct geometric tasks"):
        freezer.freeze_demand(paths["metadata"], paths["episodes"], paths["annotations"],
                               tmp_path / "same-task", regression_ids=same_task[:2] + chosen[:8])


def test_cli_and_python39_syntax(dataset, tmp_path):
    paths = dataset[0]
    script = Path(freezer.__file__).resolve()
    ast.parse(script.read_text(encoding="utf-8"), feature_version=(3, 9))
    result = subprocess.run([sys.executable, str(script), "--intention-metadata", str(paths["metadata"]),
                             "--episodes", str(paths["episodes"]), "--annotations", str(paths["annotations"]),
                             "--output", str(tmp_path / "cli")], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["regression_conditions"] == 10
    receipt = json.loads((tmp_path / "cli" / "selection_receipt.json").read_text(encoding="utf-8"))
    assert receipt["seed"] == 20261108
