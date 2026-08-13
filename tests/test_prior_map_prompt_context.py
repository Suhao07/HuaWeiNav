from pathlib import Path

from prior_map.contracts import (
    FrontierPrior,
    ObjectPrior,
    PriorMapData,
    PriorObject,
    PriorRoom,
    PriorTopologyEdge,
    RoomPrior,
    SearchPriorResult,
    SupportRegionPrior,
)
from prior_map.prompt_context import PriorMapPromptContextBuilder, summarize_prior_map, to_compact_xml
from prior_map.visualizer import (
    FloorPlanOverlay,
    FloorPlanOverlayPoint,
    PriorMapFloorPlanVisualizer,
    PriorMapSomVisualizer,
    render_floorplan_global_view,
    render_global_view,
    render_room_view,
    write_floorplan_artifacts,
)


def _map() -> PriorMapData:
    return PriorMapData(
        scene_id="lab_floor_1",
        rooms=(
            PriorRoom(
                uid="room_kitchen",
                label="kitchen",
                boundary_xy=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
                centroid_xy=(2.0, 1.5),
                neighbors=("room_living",),
                confidence=0.8,
            ),
            PriorRoom(
                uid="room_living",
                label="living room",
                boundary_xy=((4.0, 0.0), (8.0, 0.0), (8.0, 3.0), (4.0, 3.0)),
                centroid_xy=(6.0, 1.5),
                neighbors=("room_kitchen",),
                confidence=0.7,
            ),
        ),
        objects=(
            PriorObject(
                uid="obj_fridge",
                label="fridge",
                position_xyz=(1.0, 1.0, 0.0),
                parent_room_uid="room_kitchen",
                exact=True,
                confidence=0.9,
                aliases=("refrigerator",),
            ),
            PriorObject(
                uid="obj_book_hint",
                label="book",
                position_xyz=(6.5, 1.0, 0.0),
                parent_room_uid="room_living",
                exact=False,
                confidence=0.4,
            ),
        ),
        topology_edges=(
            PriorTopologyEdge(
                uid="edge_kitchen_living",
                source_uid="room_kitchen",
                target_uid="room_living",
                relation="adjacent",
            ),
        ),
        source_format="json",
        frame_id="prior_map",
        world_min=(0.0, 0.0),
        world_max=(8.0, 3.0),
    )


def _prior_result() -> SearchPriorResult:
    return SearchPriorResult(
        room_rankings=(RoomPrior(room_uid="room_kitchen", label="kitchen", score=1.2, reason="room hint"),),
        object_rankings=(
            ObjectPrior(object_uid="obj_fridge", label="fridge", score=1.6, reason="target concept match"),
        ),
        support_regions=(
            SupportRegionPrior(uid="support_counter", label="counter", score=0.4, room_uid="room_kitchen"),
        ),
        frontier_biases=(
            FrontierPrior(frontier_uid="frontier_1", score_delta=0.5, prior_room_uid="room_kitchen"),
        ),
    )


def test_prompt_context_summaries_are_bounded_and_traceable() -> None:
    builder = PriorMapPromptContextBuilder(max_chars=260)
    bundle = builder.build_bundle(_map(), _prior_result())

    assert len(bundle.natural_language) <= 260
    assert len(bundle.compact_xml) <= 260
    assert len(bundle.search_prior_summary) <= 260
    assert "lab_floor_1" in bundle.natural_language
    assert "kitchen" in bundle.natural_language
    assert "<prior_map" in bundle.compact_xml
    assert "room_kitchen" in bundle.compact_xml
    assert "ranking-only" in bundle.search_prior_summary
    assert bundle.metadata["room_count"] == 2


def test_prompt_context_convenience_functions() -> None:
    natural = summarize_prior_map(_map(), max_chars=500)
    xml = to_compact_xml(_map(), max_chars=1200)

    assert "fridge exact" in natural
    assert 'label="fridge"' in xml
    assert 'from="room_kitchen"' in xml


def test_som_visualizer_marker_ids_are_stable_and_labels_are_traceable() -> None:
    first = render_global_view(_map(), width=480, height=320)
    second = render_global_view(_map(), width=480, height=320)

    assert [marker.marker_id for marker in first.markers] == [marker.marker_id for marker in second.markers]
    assert "R_room_kitchen" in {marker.marker_id for marker in first.markers}
    assert "O_obj_fridge" in {marker.marker_id for marker in first.markers}
    assert any(marker.label == "fridge" and marker.uid == "obj_fridge" for marker in first.markers)
    assert "object marker" in first.svg
    assert first.legend["R_*"] == "prior room marker"


def test_som_visualizer_room_view_contains_only_room_objects() -> None:
    view = PriorMapSomVisualizer(width=480, height=320).render_room_view(_map(), "room_kitchen")

    marker_ids = {marker.marker_id for marker in view.markers}
    assert "R_room_kitchen" in marker_ids
    assert "O_obj_fridge" in marker_ids
    assert "O_obj_book_hint" not in marker_ids
    assert view.metadata["room_uid"] == "room_kitchen"


def test_floorplan_visualizer_renders_topology_and_runtime_overlay(tmp_path: Path) -> None:
    overlay = FloorPlanOverlay(
        target_prior_object_uids=("obj_fridge",),
        frontiers=(
            FloorPlanOverlayPoint(
                uid="frontier_1",
                label="frontier 1",
                xy=(3.0, 1.5),
                point_type="frontier",
                selected=True,
            ),
        ),
        live_detections=(
            FloorPlanOverlayPoint(
                uid="obj_fridge",
                label="fridge",
                xy=(1.0, 1.0),
                point_type="live_detection",
            ),
        ),
        trajectory_xy=((0.5, 0.5), (2.0, 1.0), (3.0, 1.5)),
        selected_frontier_uid="frontier_1",
    )

    view = render_floorplan_global_view(_map(), overlay=overlay, width=640, height=420)

    assert view.view_type == "floorplan_global"
    assert "room-room topology edge" in view.svg
    assert "selected frontier" in view.svg
    assert any(marker["marker_type"] == "room_edge" for marker in view.markers)
    assert any(marker["marker_type"] == "frontier" and marker["selected"] for marker in view.markers)

    paths = PriorMapFloorPlanVisualizer(width=640, height=420).write_global_artifacts(
        _map(),
        tmp_path,
        overlay=overlay,
    )
    assert Path(paths["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    markers = Path(paths["markers"]).read_text(encoding="utf-8")
    assert "frontier_1" in markers
    assert "obj_fridge" in markers


def test_write_floorplan_artifacts_uses_fixed_names(tmp_path: Path) -> None:
    artifacts = write_floorplan_artifacts(_map(), tmp_path, max_room_views=1, width=640, height=420)

    assert Path(artifacts["global"]["png"]).name == "floorplan_global.png"
    assert Path(artifacts["global"]["svg"]).name == "floorplan_global.svg"
    assert Path(artifacts["global"]["markers"]).name == "floorplan_global_markers.json"
    assert (tmp_path / "floorplan_global.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    room_paths = next(iter(artifacts["rooms"].values()))
    assert Path(room_paths["png"]).name.startswith("floorplan_room_")


def test_prompt_and_visualizer_stay_platform_neutral() -> None:
    for path in ("prior_map/prompt_context.py", "prior_map/visualizer.py"):
        source = Path(path).read_text(encoding="utf-8")
        for forbidden in ("rclpy", "habitat", "cv2", "open3d", "geometry_msgs"):
            assert forbidden not in source

    assert render_room_view(_map(), "room_living").view_type == "room"
