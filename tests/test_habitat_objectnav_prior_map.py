import gzip
import json
import subprocess
import sys
from pathlib import Path

from prior_map.alignment import PriorMapAlignment
from prior_map.habitat_objectnav import (
    build_habitat_objectnav_prior_map,
    write_prior_map_with_alignment,
)
from prior_map.loaders import PriorMapLoader


def _write_dataset(path: Path) -> Path:
    payload = {
        "episodes": [
            {
                "episode_id": "0",
                "scene_id": "data/scene_datasets/hm3d/sceneA/sceneA.basis.glb",
                "object_category": "chair",
            },
            {
                "episode_id": "1",
                "scene_id": "data/scene_datasets/hm3d/sceneA/sceneA.basis.glb",
                "object_category": "tv_monitor",
            },
        ],
        "goals_by_category": {
            "sceneA_chair": [
                {"object_id": "chair_1", "object_category": "chair", "position": [0.0, 0.1, 1.0]},
            ],
            "sceneA_tv_monitor": [
                {
                    "object_id": "tv_1",
                    "object_category": "tv_monitor",
                    "position": [1.0, 0.2, 3.0],
                    "aliases": ["television"],
                },
                {
                    "object_id": "tv_2",
                    "object_category": "tv_monitor",
                    "view_points": [{"agent_state": {"position": [2.0, 0.2, 4.0]}}],
                },
            ],
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def _write_split_dataset(root_path: Path) -> Path:
    root_path.parent.mkdir(parents=True, exist_ok=True)
    content_dir = root_path.parent / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(root_path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": [], "category_to_task_category_id": {"tv": 0}}, f)
    _write_dataset(content_dir / "sceneA.json.gz")
    return root_path


def test_build_habitat_objectnav_prior_map_from_dataset_goals(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "val.json.gz")

    result = build_habitat_objectnav_prior_map(
        dataset_path,
        scene_id="sceneA",
        object_category="tv monitor",
    )

    prior_map = result.prior_map
    assert prior_map.scene_id == "sceneA"
    assert prior_map.source_format == "habitat_objectnav_json"
    assert prior_map.frame_id == "habitat_world"
    assert len(prior_map.rooms) == 1
    assert len(prior_map.objects) == 2
    assert len(prior_map.topology_edges) == 2
    assert prior_map.objects[0].label == "tv_monitor"
    assert prior_map.objects[0].position_xyz == (1.0, 0.2, 3.0)
    assert prior_map.objects[0].aliases == ("television",)
    assert prior_map.objects[1].position_xyz == (2.0, 0.2, 4.0)
    assert prior_map.objects[0].parent_room_uid == prior_map.rooms[0].uid
    assert prior_map.metadata["selected_episode_id"] == "1"
    assert result.goal_count == 2


def test_write_generated_prior_map_roundtrips_with_loader_and_alignment(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "val.json.gz")
    result = build_habitat_objectnav_prior_map(dataset_path, scene_id="sceneA", object_category="tv_monitor")
    prior_map_path = tmp_path / "sceneA_tv_prior_map.json"
    alignment_path = tmp_path / "sceneA_tv_alignment.json"

    paths = write_prior_map_with_alignment(
        result,
        prior_map_path,
        alignment_output_path=alignment_path,
        alignment_mode="unavailable",
    )

    loaded = PriorMapLoader().load(paths["prior_map"], source_format="canonical_json")
    alignment = PriorMapAlignment.load(paths["alignment"])
    assert loaded == result.prior_map
    assert alignment.alignment_type == "unavailable"
    assert alignment.enabled_for_ranking is False
    assert alignment.base_confidence == 0.0


def test_build_habitat_prior_map_from_split_root_content_dataset(tmp_path: Path) -> None:
    split_root = _write_split_dataset(tmp_path / "val" / "val.json.gz")

    result = build_habitat_objectnav_prior_map(split_root, scene_id="sceneA", object_category="tv")

    assert result.prior_map.scene_id == "sceneA"
    assert result.goal_count == 2
    assert result.prior_map.metadata["selected_object_category"] == "tv_monitor"


def test_build_habitat_prior_map_cli_writes_consumable_files(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "val.json.gz")
    prior_map_path = tmp_path / "prior_map.json"
    alignment_path = tmp_path / "alignment.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_habitat_prior_map.py",
            str(dataset_path),
            "--scene_id",
            "sceneA",
            "--object_category",
            "tv_monitor",
            "--output",
            str(prior_map_path),
            "--alignment_output",
            str(alignment_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["loader_source"] == "canonical_json"
    assert summary["goal_count"] == 2
    assert prior_map_path.exists()
    assert alignment_path.exists()
    assert PriorMapLoader().load(prior_map_path, source_format="canonical_json").objects
