import json
from pathlib import Path

import pytest

from prior_map.contracts import (
    FrontierPrior,
    ObjectPrior,
    PriorMapData,
    PriorObject,
    PriorObservationRecord,
    PriorRoom,
    PriorTopologyEdge,
    RoomPrior,
    SearchPriorResult,
    SupportRegionPrior,
)


def _sample_prior_map() -> PriorMapData:
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
                source="floorplan",
                metadata={"source_page": 1},
            ),
            PriorRoom(
                uid="room_living",
                label="living room",
                centroid_xy=(5.0, 1.5),
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
                edge_type="room-room",
                relation="adjacent",
                confidence=0.75,
            ),
            PriorTopologyEdge(
                uid="edge_kitchen_fridge",
                source_uid="room_kitchen",
                target_uid="obj_fridge",
                edge_type="room-object",
                relation="contains",
                confidence=0.9,
            ),
        ),
        source_format="json",
        frame_id="prior_map",
        world_min=(0.0, 0.0),
        world_max=(8.0, 4.0),
        observations=(
            PriorObservationRecord(
                timestamp=12.5,
                pose_xyz=(0.5, 0.5, 0.0),
                observed_object_uids=("runtime_obj_1",),
                observed_object_labels=("chair",),
                room_hypothesis_uid="room_living",
                source="bag_replay",
            ),
        ),
        metadata={"building": "lab"},
    )


def test_prior_map_contracts_stay_platform_neutral() -> None:
    source = Path("prior_map/contracts.py").read_text(encoding="utf-8")

    forbidden = (
        "import rclpy",
        "import rospy",
        "import habitat",
        "import cv2",
        "import numpy",
        "from rclpy",
        "from rospy",
        "from habitat",
        "from cv2",
        "from numpy",
        "openai",
        "anthropic",
    )
    for text in forbidden:
        assert text not in source


def test_prior_map_data_json_roundtrip() -> None:
    prior_map = _sample_prior_map()

    encoded = json.dumps(prior_map.to_dict(), sort_keys=True)
    decoded = json.loads(encoded)
    restored = PriorMapData.from_dict(decoded)

    assert restored == prior_map
    assert restored.room_by_uid("room_kitchen").label == "kitchen"
    assert restored.object_by_uid("obj_fridge").exactness == "exact"
    assert restored.object_by_uid("obj_book_hint").exactness == "hypothesis"
    assert len(restored.topology_for_uid("room_kitchen")) == 2


def test_search_prior_result_json_roundtrip() -> None:
    result = SearchPriorResult(
        room_rankings=(
            RoomPrior(
                room_uid="room_kitchen",
                label="kitchen",
                score=0.82,
                reason="instruction mentions cooking objects",
                visit_state="unvisited",
                reachable_hint="connected",
            ),
        ),
        object_rankings=(
            ObjectPrior(
                object_uid="obj_fridge",
                label="fridge",
                score=0.91,
                reason="exact prior object",
                parent_room_uid="room_kitchen",
                exact=True,
                matched_runtime_uid="runtime_obj_7",
            ),
        ),
        frontier_biases=(
            FrontierPrior(
                frontier_uid="frontier_3",
                score_delta=0.25,
                reason="frontier leads toward kitchen prior",
                prior_room_uid="room_kitchen",
            ),
        ),
        support_regions=(
            SupportRegionPrior(
                uid="support_counter",
                label="countertop",
                score=0.4,
                reason="likely support for small kitchen objects",
                room_uid="room_kitchen",
                boundary_xy=((1.0, 1.0), (2.0, 1.0), (2.0, 1.5)),
            ),
        ),
        prompt_context={"top_rooms": ["kitchen"]},
        diagnostics={"weights": {"room": 0.5}},
    )

    restored = SearchPriorResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert restored == result
    assert restored.room_rankings[0].score == pytest.approx(0.82)
    assert restored.frontier_biases[0].score_delta == pytest.approx(0.25)


def test_prior_map_validation_rejects_invalid_geometry_and_duplicates() -> None:
    with pytest.raises(ValueError):
        PriorRoom(uid="room_bad", boundary_xy=((1.0, 2.0, 3.0),))

    with pytest.raises(ValueError):
        PriorObject(uid="obj_bad", label="chair", confidence=1.5)

    with pytest.raises(ValueError):
        PriorTopologyEdge(uid="edge_bad", source_uid="", target_uid="room_1")

    with pytest.raises(ValueError):
        PriorMapData(
            scene_id="scene",
            rooms=(PriorRoom(uid="room_1"), PriorRoom(uid="room_1")),
        )
