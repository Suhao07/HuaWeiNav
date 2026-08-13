"""Tests for the FloorPlan-compatible room-layout exchange contract."""

from __future__ import annotations

import json
from pathlib import Path

from prior_map.contracts import PriorMapData, PriorRoom, PriorTopologyEdge
from prior_map.floorplan_layout import FloorplanLayout
from prior_map.loaders import PriorMapLoader


class _Category:
    def __init__(self, label: str) -> None:
        self.label = label

    def name(self) -> str:
        return self.label


class _AABB:
    def __init__(self, center, sizes) -> None:
        self.center = center
        self.sizes = sizes


class _Region:
    def __init__(self, uid: str, label: str, center, sizes) -> None:
        self.id = uid
        self.category = _Category(label)
        self.aabb = _AABB(center, sizes)
        self.level_id = 0


class _SemanticScene:
    def __init__(self, regions) -> None:
        self.regions = tuple(regions)
        self.objects = ()


class _Pathfinder:
    def __init__(self) -> None:
        self.bounds = ((0.0, 0.0, 0.0), (4.0, 0.0, 2.0))

    def get_bounds(self):
        return self.bounds

    def is_navigable(self, point):
        x, y, z = point
        return abs(y) < 1e-6 and 0.0 <= x <= 4.0 and 0.0 <= z <= 2.0


class _Sim:
    def __init__(self) -> None:
        self.semantic_scene = _SemanticScene(
            (
                _Region("a", "living room", (1.0, 0.0, 1.0), (2.0, 2.0, 2.0)),
                _Region("b", "hallway", (3.0, 0.0, 1.0), (2.0, 2.0, 2.0)),
            )
        )
        self.pathfinder = _Pathfinder()


def test_layout_from_prior_map_keeps_room_geometry_and_flips_z() -> None:
    prior_map = PriorMapData(
        scene_id="scene",
        rooms=(
            PriorRoom(
                uid="room_a",
                label="living room",
                boundary_xy=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)),
                centroid_xy=(2.0, 3.0),
            ),
            PriorRoom(uid="room_b", label="hallway", centroid_xy=(4.0, 3.0)),
        ),
        topology_edges=(
            PriorTopologyEdge(
                uid="room_edge",
                source_uid="room_a",
                target_uid="room_b",
                edge_type="room-room",
                relation="connected",
            ),
        ),
        source_format="hm3d_floorplan_layout",
        frame_id="habitat_world",
    )

    payload = FloorplanLayout.from_prior_map(prior_map).to_dict()
    room = payload["levels"]["0"]["regions"]["room_a"]

    assert room["boundaries"] == [[1.0, -2.0], [3.0, -2.0], [3.0, -4.0], [1.0, -4.0]]
    assert room["center"] == [2.0, -3.0]
    assert room["connectivity"] == ["room_b"]
    assert payload["coordinate_convention"]["floorplan_axes"] == ["x", "-z"]
    assert payload["frame_id"] == "floorplan_metric"


def test_floorplan_layout_roundtrips_through_existing_loader(tmp_path) -> None:
    prior_map = PriorMapData(
        scene_id="scene",
        rooms=(
            PriorRoom(
                uid="room_a",
                label="office",
                boundary_xy=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
                centroid_xy=(1.0, 1.0),
            ),
        ),
        source_format="hm3d_floorplan_layout",
        frame_id="habitat_world",
    )
    path = tmp_path / "floorplan.json"
    path.write_text(json.dumps(FloorplanLayout.from_prior_map(prior_map).to_dict()), encoding="utf-8")

    loaded = PriorMapLoader().load(path, source_format="floorplan_vln_json")

    assert loaded.scene_id == "scene"
    assert loaded.frame_id == "habitat_world"
    assert len(loaded.rooms) == 1
    assert loaded.objects == ()
    assert loaded.room_by_uid("room_a").label == "office"
    assert loaded.room_by_uid("room_a").boundary_xy == ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    assert loaded.metadata["coordinate_normalization"]["operation"] == "reflect_second_plane_axis"


def test_hm3d_layout_builder_emits_room_only_layout() -> None:
    from prior_map.hm3d_layout import HM3DLayoutBuilder

    result = HM3DLayoutBuilder().build_from_sim(_Sim(), "scene")

    assert result.layout.metadata["object_instances_omitted"] is True
    assert result.prior_map.objects == ()
    assert len(result.layout.levels) == 1
    assert all(region.boundary_xy for region in result.layout.levels[0].regions)
    assert result.layout.frame_id == "floorplan_metric"
    assert result.layout.levels[0].height_range == (-1.0, 1.0)
    assert result.metadata["quality"]["room_boundary_coverage"] == 1.0
    assert result.metadata["semantic_bev_required"] is True


def test_hm3d_layout_writer_persists_semantic_bev_bundle(tmp_path) -> None:
    from prior_map.hm3d_layout import HM3DLayoutBuilder, write_hm3d_floorplan_layout

    result = HM3DLayoutBuilder().build_from_sim(_Sim(), "scene")
    paths = write_hm3d_floorplan_layout(
        result,
        tmp_path / "floorplan.json",
        quality_output_path=tmp_path / "quality.json",
    )

    assert Path(paths["prior_map_bev_png"]).is_file()
    assert Path(paths["prior_map_bev_svg"]).is_file()
    assert Path(paths["prior_map_bev_markers"]).is_file()
    assert Path(paths["manifest"]).is_file()
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["authority"] == "semantic_prior_only"
    assert not list(tmp_path.rglob("*.npy"))
