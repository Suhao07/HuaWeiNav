"""Load policy-safe IntentionNav inputs and delegate offline scoring to official code."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


@dataclass
class IntentionNavEpisode:
    """Keep public episode inputs separate from unchanged private evaluator metadata."""

    selection_id: str
    scene_id: str
    style: str
    instruction: str
    start_position: tuple[float, float, float]
    start_rotation_quat_wxyz: tuple[float, float, float, float]
    _evaluation_meta: dict = field(repr=False)
    source_sha256: dict[str, str] = field(default_factory=dict, repr=False)

    def policy_input(self) -> dict:
        """Return only explicitly whitelisted task identifiers, instruction and start pose."""
        return {
            "selection_id": self.selection_id,
            "scene_id": self.scene_id,
            "style": self.style,
            "instruction": self.instruction,
            "start_position": self.start_position,
            "start_rotation_quat_wxyz": self.start_rotation_quat_wxyz,
        }


def _load_record(path: Path, selection_id: str) -> tuple[dict, str]:
    """Read exactly one matching JSONL record and hash its original source bytes."""
    source = path.read_bytes()
    matches = []
    for line_number, line in enumerate(source.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        if record.get("selection_id") == selection_id:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"{path}: expected exactly one {selection_id}, found {len(matches)}")
    return matches[0], hashlib.sha256(source).hexdigest()


def _finite_number(value, name: str) -> float:
    """Reject nonnumeric, boolean and nonfinite coordinates."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _finite_vector(value, length: int, name: str) -> tuple:
    """Validate a fixed-length finite vector without changing source metadata."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(_finite_number(component, name) for component in value)


def load_episode(
    data_root: Path, selection_id: str = "SEL_001", style: str = "formal"
) -> IntentionNavEpisode:
    """Join the two dataset JSONL files by selection ID and validate public inputs."""
    if not isinstance(selection_id, str) or not selection_id.strip():
        raise ValueError("selection_id must be a nonempty string")
    if style not in ("formal", "natural", "casual", "emotional"):
        raise ValueError("style must be formal, natural, casual or emotional")
    dataset = Path(data_root) / "dataset"
    meta, episode_hash = _load_record(dataset / "episodes.jsonl", selection_id)
    intent, intent_hash = _load_record(dataset / "selected_500_intents.jsonl", selection_id)
    scene_id = meta.get("scene_id")
    if not isinstance(scene_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", scene_id):
        raise ValueError("scene_id must be a safe scene name, not a path")
    if intent.get("scene_id") != scene_id:
        raise ValueError("scene_id mismatch between episode and instruction")
    instruction = intent.get(style + "_en")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{style}_en must be a nonempty instruction")
    position = _finite_vector(meta.get("start_position"), 3, "start_position")
    rotation = _finite_vector(meta.get("start_rotation_quat_wxyz"), 4, "start_rotation_quat_wxyz")
    if not math.isclose(math.hypot(*rotation), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("start_rotation_quat_wxyz must be a unit quaternion")
    return IntentionNavEpisode(
        selection_id=selection_id,
        scene_id=scene_id,
        style=style,
        instruction=instruction,
        start_position=position,
        start_rotation_quat_wxyz=rotation,
        _evaluation_meta=meta,
        source_sha256={
            "dataset/episodes.jsonl": episode_hash,
            "dataset/selected_500_intents.jsonl": intent_hash,
        },
    )


_SCORER_SCRIPT = """
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

scorer = Path(sys.argv[1])
sys.path[:0] = [str(scorer.parent), str(scorer.parent.parent / "agents")]
payload = json.load(sys.stdin)
record, options = payload["record"], payload["options"]
dependencies = {}
unavailable = {}

def dependency(path):
    path = Path(path).resolve()
    if not path.is_file():
        return False
    dependencies[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return True

def evaluate():
    def missing(names, reason):
        for name in names:
            metrics[name] = None
            unavailable[name] = reason

    spec = importlib.util.spec_from_file_location("intentionnav_official_metrics", scorer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.nav_metrics(record, radius_m=options["radius_m"])
    if not isinstance(metrics, dict):
        return {"metrics": metrics}
    for name in ("common", "visibility"):
        dependency(sys.modules[name].__file__)
    dependency(scorer)
    if "SR_hit" not in metrics:
        missing(("SR_hit", "OSR_hit", "SPL"), "official nav_metrics returned no navigation scores; target_position required")
    elif record["episode_meta"].get("geodesic_to_target") is None:
        metrics.pop("geodesic_length", None)
        missing(("SPL",), "episode_meta.geodesic_to_target is missing")

    prediction = record["prediction"]["target"]
    target = record.get("target_category")
    synonyms_path = scorer.parent.parent / "vocab/category_synonyms.yaml"
    if prediction is None:
        missing(("IM",), "public terminal prediction_target was not supplied")
    elif not target:
        missing(("IM",), "private target_category is missing")
    elif not dependency(synonyms_path):
        missing(("IM",), "missing synonym vocabulary: " + str(synonyms_path))
    elif importlib.util.find_spec("yaml") is None:
        missing(("IM",), "PyYAML is required for official synonym matching")
    else:
        metrics["IM"] = int(module.intent_match(prediction, target, module._load_synonyms()))

    visibility = sys.modules["visibility"]
    dataset_root = Path(options["dataset_root"])
    summary_root = options["scene_summary_root"] or os.environ.get(
        "INTENTEQA_SCENE_SUMMARY", str(getattr(visibility, "DEFAULT_SCENE_SUMMARY", "")))
    scene_id = record["scene_id"]
    assets = [dataset_root / "selected_500_intents.jsonl",
              dataset_root / scene_id / "manifest.json"]
    if summary_root:
        assets.append(Path(summary_root) / scene_id / "object_dict.json")
    absent = [str(path) for path in assets if not dependency(path)]
    if not summary_root:
        absent.append("scene_summary_root")
    vis = {}
    if absent:
        missing(("G", "GSR_hit"), "missing visibility assets: " + ", ".join(absent))
    else:
        # Relocate only data inputs in this isolated process; keep official geometry intact.
        visibility.DATASET_ROOT = dataset_root
        visibility.DATASET_JSONL = dataset_root / "selected_500_intents.jsonl"
        os.environ["INTENTEQA_SCENE_SUMMARY"] = str(summary_root)
        if record["selection_id"] not in visibility._items_by_selection_id():
            missing(("G", "GSR_hit"), "selection_id missing from visibility dataset")
        elif not visibility._scene_objects(scene_id):
            missing(("G", "GSR_hit"), "scene summary contains no usable object bounding boxes")
        else:
            vis = module.trajectory_visibility(record)
            if not vis:
                missing(("G", "GSR_hit"), "official visibility could not resolve the target bbox")
            else:
                metrics["G"] = int(vis["G_seen"])
                if metrics.get("SR_hit") is None:
                    missing(("GSR_hit",), "geometric SR is unavailable")
                else:
                    # Exactly aggregate()'s conjunction, independent of IM.
                    metrics["GSR_hit"] = int(metrics["SR_hit"] and metrics["G"])

    return {"metrics": metrics, "visibility": vis, "unavailable_metrics": unavailable,
            "scorer_dependencies_sha256": dependencies}

with contextlib.redirect_stdout(sys.stderr):
    scores = evaluate()
json.dump(scores, sys.stdout, allow_nan=False)
"""


def score_trajectory(
    episode: IntentionNavEpisode,
    trajectory,
    official_root: Path,
    *,
    prediction_target: str | None = None,
    tier: str = "",
    model: str = "",
    radius_m: float = 2.0,
    dataset_root: Path | None = None,
    scene_summary_root: Path | None = None,
) -> dict:
    """Score one active episode offline with the fixed upstream functions, never a judge.

    Args:
        episode: Loaded episode; private metadata is used only in this evaluator.
        trajectory: Nonempty position/yaw dictionaries, starting at the official
            start and ending at the actual stopped camera pose. Positions are
            world XYZ metres (Z up, camera height), yaw is radians from +X toward
            +Y; optional step is retained for visibility diagnostics. The official
            G uses a level, 90-degree, 1024-square projection, not RGB or pitch.
        official_root: Fixed upstream checkout containing eval/aggregate.
        prediction_target: Actual public terminal prediction, not a GT-derived
            guess. None means unavailable; an explicitly emitted empty string is
            scored by the official matcher as a mismatch.
        tier: Real experiment tier identifier for formal aggregate grouping.
        model: Real model identifier for formal aggregate grouping.
        radius_m: Explicit geometric radius; 2m matches upstream CLI primary.
            aggregate() itself defaults to 1m. G retains its official 3m radius.
        dataset_root: Actual directory containing selected_500_intents.jsonl and
            <scene_id>/manifest.json; defaults to official_root/results/dataset.
        scene_summary_root: Directory containing <scene_id>/object_dict.json;
            otherwise use INTENTEQA_SCENE_SUMMARY or the upstream default.

    Returns:
        Metrics (IM, SR_hit, OSR_hit, SPL, G, GSR_hit plus nav diagnostics), raw
        visibility, unavailable_metrics reasons, dependency SHA256s, protocol,
        and an evaluator-only record. Missing metrics are None, never fallback
        zeroes. SR/SPL come directly from nav_metrics; G is G_seen from
        trajectory_visibility; GSR is SR_hit AND G, as in aggregate().
        record contains selection_id/scene_id/tier/model/style/intent,
        prediction.target, private target_category, episode_meta and trajectory.
        style is formal/natural/casual/emotional (without _en). Never pass this
        record to the policy. Only export complete records with real tier/model
        into an isolated aggregate input directory; aggregate silently treats
        absent predictions/visibility as failures. Pass the same radius explicitly.
        Upstream aggregate emits GSR but does not persist G itself.

    Raises:
        ValueError: Invalid trajectory, prediction, radius, or scorer output.
        FileNotFoundError: The fixed scorer cannot be read.
        subprocess.CalledProcessError: The official scorer fails.
        subprocess.TimeoutExpired: The isolated scorer exceeds 30 seconds.
    """
    if not isinstance(trajectory, (list, tuple)) or not trajectory:
        raise ValueError("trajectory must be a nonempty sequence of position/yaw rows")
    official_start = _finite_vector(episode._evaluation_meta.get("start_position"), 3, "official start")
    for index, row in enumerate(trajectory):
        if not isinstance(row, dict):
            raise ValueError(f"trajectory[{index}] must be a position/yaw object")
        position = _finite_vector(row.get("position"), 3, f"trajectory[{index}].position")
        _finite_number(row.get("yaw"), f"trajectory[{index}].yaw")
        if index == 0 and math.dist(position, official_start) > 1e-6:
            raise ValueError("trajectory first position does not match the official start")
    if prediction_target is not None and not isinstance(prediction_target, str):
        raise ValueError("prediction_target must be a string or None")
    radius_m = _finite_number(radius_m, "radius_m")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    official_root = Path(official_root).resolve()
    scorer = official_root / "eval" / "aggregate" / "compute_metrics.py"
    scorer_hash = hashlib.sha256(scorer.read_bytes()).hexdigest()
    record = {
        "selection_id": episode.selection_id,
        "scene_id": episode.scene_id,
        "tier": tier,
        "model": model,
        "style": episode.style,
        "intent": episode.instruction,
        "prediction": {"target": prediction_target},
        "target_category": episode._evaluation_meta.get("target_category"),
        "trajectory": trajectory,
        "episode_meta": episode._evaluation_meta,
    }
    options = {
        "radius_m": radius_m,
        "dataset_root": str(Path(dataset_root).resolve() if dataset_root is not None
                            else official_root / "results" / "dataset"),
        "scene_summary_root": str(Path(scene_summary_root).resolve()) if scene_summary_root is not None else None,
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", _SCORER_SCRIPT, str(scorer)],
        input=json.dumps({"record": record, "options": options}, allow_nan=False),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    scores = json.loads(result.stdout)
    if not isinstance(scores.get("metrics"), dict):
        raise ValueError("official nav_metrics must return a JSON object")
    scores.update({
        "record": record,
        "scorer_sha256": scorer_hash,
        "source_sha256": dict(episode.source_sha256),
        "protocol": {"name": "active", "radius_m": radius_m,
                     "visibility": "official final-frame defaults"},
        "aggregate_ready": not scores["unavailable_metrics"] and bool(tier) and bool(model),
    })
    return scores
