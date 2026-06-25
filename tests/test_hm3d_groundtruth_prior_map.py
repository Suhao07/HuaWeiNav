import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from prior_map.alignment import PriorMapAlignment
from prior_map.hm3d_groundtruth import (
    HM3DGroundTruthBuildConfig,
    build_hm3d_groundtruth_prior_map_from_sim,
    write_hm3d_groundtruth_prior_map_with_alignment,
)
from prior_map.loaders import PriorMapLoader


@dataclass(frozen=True)
class _FakeCategory:
    label: str

    def name(self) -> str:
        return self.label


@dataclass(frozen=True)
class _FakeAABB:
    center: tuple[float, float, float]
    sizes: tuple[float, float, float]


@dataclass(frozen=True)
class _FakeRegion:
    id: str
    category: _FakeCategory
    aabb: _FakeAABB
    level_id: int = 0


@dataclass(frozen=True)
class _FakeObject:
    id: str
    category: _FakeCategory
    aabb: _FakeAABB
    region: _FakeRegion


@dataclass(frozen=True)
class _FakeSemanticScene:
    regions: tuple[_FakeRegion, ...]
    objects: tuple[_FakeObject, ...]


@dataclass(frozen=True)
class _FakePathfinder:
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    navigable_rectangles: tuple[tuple[float, float, float, float], ...]
    sample_height: float = 0.0

    def get_bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self.bounds

    def is_navigable(self, point: tuple[float, float, float]) -> bool:
        x, y, z = point
        if abs(y - self.sample_height) > 1e-6:
            return False
        return any(min_x <= x <= max_x and min_z <= z <= max_z for min_x, max_x, min_z, max_z in self.navigable_rectangles)

    def snap_point(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        for min_x, max_x, min_z, max_z in self.navigable_rectangles:
            if min_x <= x <= max_x and min_z <= z <= max_z:
                return point
        first = self.navigable_rectangles[0]
        return (min(max(x, first[0]), first[1]), y, min(max(z, first[2]), first[3]))


@dataclass(frozen=True)
class _FakeSim:
    semantic_scene: _FakeSemanticScene
    pathfinder: _FakePathfinder


def _base_fake_sim() -> _FakeSim:
    living = _FakeRegion(
        id="1",
        category=_FakeCategory("living room"),
        aabb=_FakeAABB(center=(1.0, 0.0, 1.0), sizes=(2.0, 2.0, 2.0)),
    )
    hall = _FakeRegion(
        id="2",
        category=_FakeCategory("hallway"),
        aabb=_FakeAABB(center=(3.0, 0.0, 1.0), sizes=(2.0, 2.0, 2.0)),
    )
    return _FakeSim(
        semantic_scene=_FakeSemanticScene(
            regions=(living, hall),
            objects=(
                _FakeObject(
                    id="349",
                    category=_FakeCategory("tv"),
                    aabb=_FakeAABB(center=(1.0, 0.0, 1.0), sizes=(0.4, 0.3, 0.1)),
                    region=living,
                ),
                _FakeObject(
                    id="350",
                    category=_FakeCategory("floor"),
                    aabb=_FakeAABB(center=(3.0, 0.0, 1.0), sizes=(2.0, 0.1, 2.0)),
                    region=hall,
                ),
            ),
        ),
        pathfinder=_FakePathfinder(
            bounds=((0.0, 0.0, 0.0), (4.0, 0.0, 2.0)),
            navigable_rectangles=((0.0, 4.0, 0.0, 2.0),),
        ),
    )


def test_builds_groundtruth_prior_from_fake_semantic_scene_and_navmesh() -> None:
    result = build_hm3d_groundtruth_prior_map_from_sim(
        _base_fake_sim(),
        scene_id="wcojb4TFT35",
        config=HM3DGroundTruthBuildConfig(topdown_resolution=1.0, mask_dilation_radius_m=1.0),
    )

    prior_map = result.prior_map
    assert prior_map.source_format == "hm3d_groundtruth_semantic_scene"
    assert prior_map.frame_id == "habitat_world"
    assert prior_map.metadata["no_episode_goal_positions"] is True
    assert len(prior_map.rooms) == 2
    assert all(room.boundary_xy for room in prior_map.rooms)
    assert all(room.centroid_xy is not None for room in prior_map.rooms)

    tv = prior_map.object_by_uid("prior_object:wcojb4TFT35:349")
    assert tv is not None
    assert tv.label == "tv"
    assert tv.position_xyz == pytest.approx((1.0, 0.0, 1.0))
    assert tv.parent_room_uid is not None
    assert prior_map.object_by_uid("prior_object:wcojb4TFT35:350") is None
    assert any(edge.edge_type == "room-object" and edge.target_uid == tv.uid for edge in prior_map.topology_edges)
    assert any(edge.edge_type == "room-room" for edge in prior_map.topology_edges)
    assert result.alignment.alignment_type == "identity"
    assert result.alignment.runtime_frame_id == "habitat_world"
    assert result.alignment.confidence() == pytest.approx(1.0)


def test_splits_disconnected_components_inside_one_semantic_region() -> None:
    room = _FakeRegion(
        id="7",
        category=_FakeCategory("office"),
        aabb=_FakeAABB(center=(3.0, 0.0, 1.0), sizes=(6.0, 2.0, 2.0)),
    )
    sim = _FakeSim(
        semantic_scene=_FakeSemanticScene(regions=(room,), objects=()),
        pathfinder=_FakePathfinder(
            bounds=((0.0, 0.0, 0.0), (6.0, 0.0, 2.0)),
            navigable_rectangles=((0.0, 1.0, 0.0, 1.0), (5.0, 6.0, 0.0, 1.0)),
        ),
    )

    result = build_hm3d_groundtruth_prior_map_from_sim(
        sim,
        scene_id="scene",
        config=HM3DGroundTruthBuildConfig(topdown_resolution=1.0, mask_dilation_radius_m=0.5),
    )

    assert len(result.prior_map.rooms) == 2
    assert {room.metadata["component_count"] for room in result.prior_map.rooms} == {2}
    assert all("component_" in room.uid for room in result.prior_map.rooms)


def test_build_from_sim_uses_mesh_bounds_when_semantic_aabb_is_degenerate() -> None:
    living = _FakeRegion(
        id="1",
        category=_FakeCategory("living room"),
        aabb=_FakeAABB(center=(1.0, 0.0, 1.0), sizes=(2.0, 2.0, 2.0)),
    )
    sim = _FakeSim(
        semantic_scene=_FakeSemanticScene(
            regions=(living,),
            objects=(
                _FakeObject(
                    id="tv_349",
                    category=_FakeCategory("tv"),
                    aabb=_FakeAABB(center=(0.0, 0.0, 0.0), sizes=(0.0, 0.0, 0.0)),
                    region=living,
                ),
            ),
        ),
        pathfinder=_FakePathfinder(
            bounds=((0.0, 0.0, 0.0), (2.0, 0.0, 2.0)),
            navigable_rectangles=((0.0, 2.0, 0.0, 2.0),),
        ),
    )

    result = build_hm3d_groundtruth_prior_map_from_sim(
        sim,
        scene_id="scene",
        config=HM3DGroundTruthBuildConfig(topdown_resolution=1.0),
        mesh_object_bounds={
            "349": {
                "center": (1.2, 0.4, 1.1),
                "sizes": (0.5, 0.6, 0.2),
                "vertex_count": 12,
            }
        },
    )

    tv = result.prior_map.object_by_uid("prior_object:scene:tv_349")
    assert tv is not None
    assert tv.position_xyz == pytest.approx((1.2, 0.4, 1.1))
    assert tv.metadata["geometry_source"] == "semantic_glb_texture_bounds"
    assert tv.metadata["semantic_glb_vertex_count"] == 12


def test_write_groundtruth_prior_roundtrips_with_loader_and_alignment(tmp_path: Path) -> None:
    result = build_hm3d_groundtruth_prior_map_from_sim(
        _base_fake_sim(),
        scene_id="wcojb4TFT35",
        config=HM3DGroundTruthBuildConfig(topdown_resolution=1.0),
    )
    prior_map_path = tmp_path / "groundtruth_prior_map.json"
    alignment_path = tmp_path / "alignment.json"

    paths = write_hm3d_groundtruth_prior_map_with_alignment(
        result,
        prior_map_path,
        alignment_output_path=alignment_path,
    )

    loaded = PriorMapLoader().load(paths["prior_map"], source_format="canonical_json")
    alignment = PriorMapAlignment.load(paths["alignment"])
    assert loaded == result.prior_map
    assert alignment.alignment_type == "identity"
    assert json.loads(prior_map_path.read_text(encoding="utf-8"))["frame_id"] == "habitat_world"


def test_groundtruth_prior_cli_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_hm3d_groundtruth_prior_map.py", "--help"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert "HM3D scene directory" in completed.stdout
