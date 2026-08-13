import json
import subprocess
import sys
from pathlib import Path

from prior_map.alignment import PriorMapAlignment
from prior_map.hm3d_semantic import (
    build_hm3d_semantic_prior_map,
    write_hm3d_semantic_prior_map_with_alignment,
)
from prior_map.loaders import PriorMapLoader


def _write_semantic_txt(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "HM3D Semantic Annotations",
                '1,94210F,"ceiling",16',
                '2,DB9FFE,"wall",16',
                '3,414617,"book",16',
                '4,5F0846,"book",16',
                '5,ABCDEF,"tv",2',
                '6,ED7DB9,"unknown",2',
                '7,0CDC77,"dining table",2',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_build_hm3d_semantic_prior_map_filters_structural_and_unknown(tmp_path: Path) -> None:
    path = _write_semantic_txt(tmp_path / "sceneA.semantic.txt")

    result = build_hm3d_semantic_prior_map(path)

    prior_map = result.prior_map
    assert prior_map.scene_id == "sceneA"
    assert prior_map.source_format == "hm3d_semantic_txt"
    assert prior_map.frame_id == "habitat_world"
    assert {room.label for room in prior_map.rooms} == {"region_2", "region_16"}
    assert [obj.label for obj in prior_map.objects] == ["book", "book", "tv", "dining table"]
    assert all(obj.position_xyz is None for obj in prior_map.objects)
    assert all(obj.exact for obj in prior_map.objects)
    assert prior_map.metadata["raw_instance_count"] == 7
    assert prior_map.metadata["instance_count"] == 4
    assert prior_map.metadata["label_counts"]["book"] == 2
    assert len(prior_map.topology_edges) == 4


def test_write_hm3d_semantic_prior_map_roundtrips_with_loader_and_alignment(tmp_path: Path) -> None:
    path = _write_semantic_txt(tmp_path / "sceneA.semantic.txt")
    result = build_hm3d_semantic_prior_map(path)
    prior_map_path = tmp_path / "sceneA_semantic_prior_map.json"
    alignment_path = tmp_path / "sceneA_semantic_alignment.json"

    paths = write_hm3d_semantic_prior_map_with_alignment(
        result,
        prior_map_path,
        alignment_output_path=alignment_path,
    )

    loaded = PriorMapLoader().load(paths["prior_map"], source_format="canonical_json")
    alignment = PriorMapAlignment.load(paths["alignment"])
    assert loaded == result.prior_map
    assert alignment.alignment_type == "unavailable"
    assert alignment.enabled_for_ranking is False


def test_build_hm3d_semantic_prior_map_cli_writes_consumable_files(tmp_path: Path) -> None:
    path = _write_semantic_txt(tmp_path / "sceneA.semantic.txt")
    prior_map_path = tmp_path / "prior_map.json"
    alignment_path = tmp_path / "alignment.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_hm3d_semantic_prior_map.py",
            str(path),
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
    assert summary["instance_count"] == 4
    assert prior_map_path.exists()
    assert alignment_path.exists()
    assert PriorMapLoader().load(prior_map_path, source_format="canonical_json").objects
