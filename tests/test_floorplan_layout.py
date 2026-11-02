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
    assert manifest["authority"] == "semantic_region_geometry"
    assert PriorMapLoader().load(paths["canonical_map"]).rooms == result.prior_map.rooms
    assert not list(tmp_path.rglob("*.npy"))


def test_geometry_regions_keep_corridor_and_unknown_semantics():
    """A narrow doorway splits seeds while all navigable cells remain assigned."""
    from types import SimpleNamespace
    from prior_map.hm3d_layout import build_navigation_region_layout
    from prior_map.hm3d_groundtruth import HM3DGroundTruthBuildConfig

    class Pathfinder:
        def get_bounds(self):
            return ((0., 0., 0.), (6., 0., 2.))

        def is_navigable(self, p):
            x, _, z = p
            return (x <= 2. or x >= 4.) or (.9 <= z <= 1.1)

    result = build_navigation_region_layout(SimpleNamespace(pathfinder=Pathfinder()), 'two_areas',
        floor_height=0., clearance_m=.35,
        config=HM3DGroundTruthBuildConfig(topdown_resolution=.1, min_room_area_m2=.25))
    rooms = result.prior_map.rooms
    assert len(rooms) == 2
    assert all(r.label == 'unknown' and len(r.neighbors) == 1 for r in rooms)
    assert abs(sum(r.metadata['area_m2'] for r in rooms) - result.metadata['free_cell_count'] * .01) < 1e-8
    assert result.metadata['semantic_gt_used'] is False
    assert all(-2.05 - 1e-10 <= y <= .05 + 1e-10 for room in rooms for _, y in room.boundary_xy)  # Cell boundary extends half a pixel.


def _write_box_capture(root, box_confidence=None):
    """Write three downward views at x=0, 0.5 and 3 m; boxes sit under x=0 and x=3 m."""
    import numpy as np
    from PIL import Image
    from prior_map.observations import COORDINATES, SCHEMA_VERSION, write_frame, write_json

    write_json(root / "manifest.json", {"coordinates": COORDINATES, "schema_version": SCHEMA_VERSION,
                                        "status": "complete", "scene_id": "box"})
    box = np.ones((64, 64), dtype=np.float32)
    box[24:40, 24:40] = .5  # 0.5m 高的箱体顶面，遮挡其下地面。
    k = np.array([[100., 0, 31.5], [0, 100., 31.5], [0, 0, 1]])
    # x=0.5m 的地面视野连通原点视野，x=3m 视野无轨迹落在可通行格。
    for index, (x, depth) in enumerate([(0., box), (.5, np.ones_like(box)), (3., box)]):
        pose = np.diag([1., -1., -1., 1.])
        pose[:3, 3] = x, 0., 1.
        metadata = {}
        if box_confidence is not None:
            (root / "confidence").mkdir(exist_ok=True)
            metadata["confidence_ref"] = f"confidence/{index:06d}.png"
            Image.fromarray(np.where(depth < 1, box_confidence, 2).astype(np.uint8)).save(
                root / metadata["confidence_ref"])
        write_frame(root, index, np.zeros((64, 64, 3), np.uint8), depth, k, pose, **metadata)


def test_low_confidence_depth_neither_blocks_nor_invents_floor(tmp_path):
    from prior_map.capture_builder import capture_free_cells

    _write_box_capture(tmp_path, box_confidence=0)
    settings = dict(resolution=.05, robot_height_m=.88, robot_radius_m=0., stride=1, max_depth_m=6.)
    _, _, kept = capture_free_cells(tmp_path, **settings)
    free, _, filtered = capture_free_cells(tmp_path, min_depth_confidence=1, **settings)
    assert kept["blocked_cell_count"] > 0 and filtered["blocked_cell_count"] == 0
    assert (0, 0) not in free and filtered["min_depth_confidence"] == 1


def test_capture_free_cells_block_obstacles_and_erode_footprint(tmp_path):
    """Downward views yield floor z, blocked box cells and a walked, robot-eroded free set."""
    from prior_map.capture_builder import capture_free_cells

    _write_box_capture(tmp_path)
    settings = dict(resolution=.05, robot_height_m=.88, stride=1, max_depth_m=6.)
    free, floor, report = capture_free_cells(tmp_path, robot_radius_m=0., **settings)
    robot, _, _ = capture_free_cells(tmp_path, robot_radius_m=.1, **settings)
    assert abs(floor) < .05 and report["blocked_cell_count"] > 0
    assert (0, 0) not in free and (5, 5) in free and (12, 0) in free
    assert (65, 5) not in free and report["reachable_cell_count"] < report["robot_cell_count"]
    assert robot < free and (3, 0) in free and (3, 0) not in robot and (4, 0) in robot


def test_capture_viewpoints_drop_handheld_poses_onto_reachable_floor(tmp_path):
    """Handheld poses become floor-level bases; blocked or unwalked floor is rejected."""
    import hashlib
    import numpy as np
    import pytest
    from prior_map.bundle import write_bundle
    from prior_map.capture_builder import resolve_capture_viewpoints
    from prior_map.contracts import PriorObject
    from prior_map.observations import write_json

    capture, bundle = tmp_path / "capture", tmp_path / "bundle"
    capture.mkdir()
    _write_box_capture(capture)
    level = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])  # 光轴朝 +x 的水平手持相机。
    views, observations = [], []
    for name, xy in [("walked", (.6, 0.)), ("box", (0., 0.)), ("unwalked", (3.25, .25))]:
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = level, (*xy, 1.3)
        observations.append(dict(observation_id=name, object_uids=["chair"],
                                 camera_to_map=pose.tolist(), intrinsics=np.eye(3).tolist()))
        views.append({"uid": f"view:{name}", "kind": "observation_pose",
                      "pose_in_map": pose.tolist(), "observation_ids": [name]})
    prior = PriorMapData(scene_id="box", frame_id="map_z_up", objects=(
        PriorObject(uid="chair", label="chair", position_xyz=(1., 0., .5)),))
    write_bundle(bundle, map_id="map", revision="1", prior_map=prior, observations=observations, viewpoints=views)
    digest = hashlib.sha256((capture / "manifest.json").read_bytes()).hexdigest()
    write_json(bundle / "build_receipt.json", {"capture_manifest_sha256": digest})
    settings = dict(resolution=.05, robot_height_m=.88, robot_radius_m=.1, stride=1, max_depth_m=6.)
    result = resolve_capture_viewpoints(bundle, capture, object_uid="chair", **settings)
    admitted, = result["viewpoints"]
    base = np.asarray(admitted["pose_in_map"])
    assert admitted["observation_ids"] == ["walked"] and abs(base[2, 3]) < .05
    np.testing.assert_allclose(base[:3, 0], [1., 0., 0.])
    assert {r["source_viewpoint"] for r in result["rejected"]} == {"view:box", "view:unwalked"}
    write_json(bundle / "build_receipt.json", {"capture_manifest_sha256": "other"})
    with pytest.raises(ValueError, match="different capture"):
        resolve_capture_viewpoints(bundle, capture, **settings)


def test_decimal_grid_contour_preserves_concavity():
    """Decimal cell sizes must preserve shared endpoints and the L-shaped area."""
    from prior_map.hm3d_groundtruth import _boundary_from_cells
    cells = [(x, z, 0) for x in range(3) for z in range(3) if x == 0 or z == 0]
    polygon = _boundary_from_cells(cells, .1)
    area = abs(sum(x * polygon[(i + 1) % len(polygon)][1] - z * polygon[(i + 1) % len(polygon)][0]
                   for i, (x, z) in enumerate(polygon))) / 2
    assert abs(area - .05) < 1e-10
    assert len(polygon) > 4
