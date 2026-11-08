"""Freeze offline IntentionNav demand subsets; this is not navigation acceptance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


STYLES = ("formal", "natural", "casual", "emotional")
KINDS = ("attribute", "spatial", "state", "temporal", "conditional", "functional",
         "quantity", "exclusion", "reference")
RELATION_KINDS = frozenset(("spatial", "temporal", "conditional", "exclusion", "reference"))


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_jsonl(path):
    source = path.read_bytes()
    rows = []
    for number, line in enumerate(source.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line, parse_constant=_reject_constant,
                             object_pairs_hook=_unique_object)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    return rows, _sha256(source)


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a nonempty string")
    return value


def _index(rows, key, label):
    result = {}
    for row in rows:
        identity = _text(row.get(key), f"{label}.{key}")
        if identity in result:
            raise ValueError(f"{label}: duplicate {key} {identity}")
        result[identity] = row
    return result


def _conditions(metadata, episodes):
    intents = _index(metadata, "selection_id", "intention_metadata")
    if len(intents) != 500:
        raise ValueError(f"Expected 500 unique IntentionNav tasks, found {len(intents)}")
    bound = _index(episodes, "selection_id", "episodes")
    conditions = []
    for selection_id, item in sorted(intents.items()):
        if selection_id not in bound:
            raise ValueError(f"Missing episode for selection_id {selection_id}")
        scene_id = _text(item.get("scene_id"), f"{selection_id}.scene_id")
        episode = bound[selection_id]
        if episode.get("scene_id") != scene_id:
            raise ValueError(f"{selection_id}: scene_id mismatch between metadata and episode")
        episode_hash = _sha256(_canonical(episode))
        for style in STYLES:
            instruction = _text(item.get(style + "_en"), f"{selection_id}.{style}_en")
            conditions.append({
                "condition_id": f"{selection_id}:{style}",
                "task_id": selection_id,
                "selection_id": selection_id,
                "scene_id": scene_id,
                "style": style,
                "instruction": instruction,
                "instruction_sha256": _sha256(instruction.encode("utf-8")),
                "episode_sha256": episode_hash,
            })
    return conditions


def _annotated_conditions(conditions, annotations):
    if len(annotations) != 2000:
        raise ValueError(f"Expected 2000 annotation rows, found {len(annotations)}")
    indexed = _index(annotations, "condition_id", "annotations")
    expected = {row["condition_id"] for row in conditions}
    missing, extra = expected - indexed.keys(), indexed.keys() - expected
    if missing or extra:
        raise ValueError(f"Annotation coverage mismatch: missing {len(missing)}, extra {len(extra)}; "
                         f"missing IDs={sorted(missing)[:5]}, extra IDs={sorted(extra)[:5]}")
    annotated = []
    for condition in conditions:
        identity = condition["condition_id"]
        annotation = indexed[identity]
        if set(annotation) != {"condition_id", "instruction_sha256", "constraints", "annotation_call_id"}:
            raise ValueError(f"{identity}: invalid annotation fields")
        if annotation["instruction_sha256"] != condition["instruction_sha256"]:
            raise ValueError(f"{identity}: instruction_sha256 mismatch")
        _text(annotation["annotation_call_id"], f"{identity}.annotation_call_id")
        constraints = annotation["constraints"]
        if not isinstance(constraints, list):
            raise ValueError(f"{identity}: constraints must be a list")
        requirements = set()
        for constraint in constraints:
            if not isinstance(constraint, dict) or set(constraint) != {"kind", "quote", "requirement"}:
                raise ValueError(f"{identity}: invalid constraint fields")
            if constraint["kind"] not in KINDS:
                raise ValueError(f"{identity}: invalid constraint kind {constraint['kind']!r}")
            quote = _text(constraint["quote"], f"{identity}.quote")
            if quote not in condition["instruction"]:
                raise ValueError(f"{identity}: quote is not an exact instruction substring: {quote!r}")
            requirement = _text(constraint["requirement"], f"{identity}.requirement")
            independent_key = " ".join(requirement.split()).casefold()
            if independent_key in requirements:
                raise ValueError(f"{identity}: duplicate requirement is not an independent constraint")
            requirements.add(independent_key)
        kinds = [constraint["kind"] for constraint in constraints]
        n, r, t = len(constraints), sum(kind in RELATION_KINDS for kind in kinds), len(set(kinds))
        annotated.append(dict(condition, constraints=constraints,
                              annotation_call_id=annotation["annotation_call_id"],
                              difficulty={"n": n, "r": r, "t": t, "eligible": n >= 2 or r >= 1}))
    return annotated


def _rank(row, seed):
    difficulty = row["difficulty"]
    return (-difficulty["n"], -difficulty["r"], -difficulty["t"],
            _sha256(f"{seed}:{row['condition_id']}".encode("utf-8")))


def _distributions(rows):
    styles = Counter(row["style"] for row in rows)
    kinds = Counter(kind for row in rows for kind in {c["kind"] for c in row["constraints"]})
    constraints = Counter(c["kind"] for row in rows for c in row["constraints"])
    return {"style": {style: styles[style] for style in STYLES},
            "condition_kind": {kind: kinds[kind] for kind in KINDS},
            "constraint_kind": {kind: constraints[kind] for kind in KINDS}}


def freeze_demand(intention_metadata, episodes, annotations, output, seed=20261108, regression_ids=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory must be new: {output}")
    sources = {"intention_metadata": Path(intention_metadata), "episodes": Path(episodes),
               "annotations": Path(annotations)}
    inputs, source_hashes = {}, {}
    for name, path in sources.items():
        inputs[name], source_hashes[name] = _read_jsonl(path)
    demand = _conditions(inputs["intention_metadata"], inputs["episodes"])
    annotated = _annotated_conditions(demand, inputs["annotations"])
    eligible = [row for row in annotated if row["difficulty"]["eligible"]]
    if len(eligible) < 500:
        raise ValueError(f"Only {len(eligible)} eligible conditions; need at least 500 (no padding)")
    hard = sorted(eligible, key=lambda row: _rank(row, seed))[:500]
    regression, selected_tasks = [], set()
    for row in hard:
        if row["selection_id"] not in selected_tasks:
            regression.append(row)
            selected_tasks.add(row["selection_id"])
        if len(regression) == 10:
            break
    if regression_ids is not None:
        by_id = {row["condition_id"]: row for row in hard}
        if len(regression_ids) != 10 or len(set(regression_ids)) != 10:
            raise ValueError("Expected 10 distinct regression condition IDs")
        if not set(regression_ids) <= by_id.keys():
            raise ValueError("Regression conditions must belong to hard_500")
        regression = [by_id[identity] for identity in regression_ids]
        selected_tasks = {row["selection_id"] for row in regression}
        if len(selected_tasks) != 10:
            raise ValueError("Regression conditions must use 10 distinct geometric tasks")
    manifests = {"demand_2000.jsonl": demand, "hard_500.jsonl": hard,
                 "regression_10.jsonl": regression}
    payloads = {name: b"".join(_canonical(row) + b"\n" for row in rows)
                for name, rows in manifests.items()}
    receipt = {
        "rubric": "complex_conditions_v1",
        "seed": seed,
        "counts": {"demand_conditions": len(demand), "demand_geometry": len(inputs["intention_metadata"]),
                   "eligible_conditions": len(eligible), "hard_conditions": len(hard),
                   "hard_geometry": len({row["selection_id"] for row in hard}),
                   "hard_scenes": len({row["scene_id"] for row in hard}),
                   "regression_conditions": len(regression), "regression_geometry": len(selected_tasks)},
        "subsets": {
            "hard_500_in_demand_2000": {row["condition_id"] for row in hard} <=
                                      {row["condition_id"] for row in demand},
            "regression_10_in_hard_500": {row["condition_id"] for row in regression} <=
                                       {row["condition_id"] for row in hard},
        },
        "distributions": {"demand_2000": _distributions(annotated), "hard_500": _distributions(hard),
                          "regression_10": _distributions(regression)},
        "distribution_units": {"condition_kind": "conditions containing each kind (once per condition)",
                               "constraint_kind": "independent constraints of each kind"},
        "selected10": [row["condition_id"] for row in regression],
        "ranking": {
            "n": "Number of independently annotated constraints; duplicate normalized requirements rejected",
            "r": "Number of constraints whose kind is in relation_kinds",
            "t": "Number of distinct constraint kinds",
            "relation_kinds": sorted(RELATION_KINDS),
            "eligible": "n >= 2 or r >= 1",
            "sort_key": "(-n, -r, -t, sha256(f'{seed}:{condition_id}'.encode('utf-8')).hexdigest())",
            "hard_selection": "First 500 eligible conditions in rank order",
            "regression_selection": ("Explicit fixed original-text-reviewed IDs within hard_500"
                                     if regression_ids is not None else
                                     "First 10 distinct selection_id in hard rank order"),
            "ignored": ["historical difficulty labels", "emotion", "instruction length",
                        "asset availability", "model failures", "runtime outcomes"],
        },
        "source_sha256": source_hashes,
        "source_paths": {name: str(path.resolve()) for name, path in sources.items()},
        "manifest_sha256": {name: _sha256(data) for name, data in payloads.items()},
        "hashing": {"instruction_sha256": "SHA-256 of unchanged instruction UTF-8 bytes",
                    "episode_sha256": "SHA-256 of canonical JSON of the entire original episode row",
                    "canonical_json": {"sort_keys": True, "ensure_ascii": False,
                                       "separators": [",", ":"], "allow_nan": False, "encoding": "utf-8"}},
        "gt_binding": "Official episode joined only by selection_id; task_id equals selection_id; no GT rewritten",
        "reporting": "Offline selection only, not navigation acceptance; annotations and quotes are selection "
                     "evidence, not runtime gates or policy inputs",
    }
    payloads["selection_receipt.json"] = (json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                                                    indent=2, allow_nan=False) + "\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intention-metadata", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--regression-condition-ids", nargs=10)
    args = parser.parse_args(argv)
    receipt = freeze_demand(args.intention_metadata, args.episodes, args.annotations, args.output,
                           args.seed, args.regression_condition_ids)
    print(json.dumps(receipt["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
