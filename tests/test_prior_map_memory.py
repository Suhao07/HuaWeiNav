from pathlib import Path
from types import SimpleNamespace

import pytest

from prior_map.alignment import PriorMapAlignment
from prior_map.contracts import PriorMapData, PriorObject, PriorRoom
from prior_map.memory import PriorMapMemory


def _base_map() -> PriorMapData:
    return PriorMapData(
        scene_id="memory_scene",
        rooms=(
            PriorRoom(uid="room_kitchen", label="kitchen", confidence=0.4),
            PriorRoom(uid="room_living", label="living room", confidence=0.6),
        ),
        objects=(
            PriorObject(
                uid="prior_fridge",
                label="fridge",
                parent_room_uid="room_kitchen",
                confidence=0.3,
                aliases=("refrigerator",),
            ),
            PriorObject(
                uid="prior_book",
                label="book",
                parent_room_uid="room_living",
                confidence=0.2,
            ),
        ),
    )


def test_memory_stays_platform_neutral() -> None:
    source = Path("prior_map/memory.py").read_text(encoding="utf-8")

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


def test_mark_room_visited_and_current_map_do_not_mutate_base() -> None:
    base = _base_map()
    memory = PriorMapMemory(base_map=base, alignment=PriorMapAlignment.identity())

    memory.mark_room_visited("room_kitchen", step=3)
    current = memory.current_map()

    base_room = base.room_by_uid("room_kitchen")
    current_room = current.room_by_uid("room_kitchen")
    assert base_room.metadata == {}
    assert current_room.metadata["runtime_state"]["visit_count"] == 1
    assert current_room.metadata["runtime_state"]["visited"] is True
    assert current_room.metadata["runtime_state"]["last_visited_step"] == 3
    assert current.metadata["runtime_memory"]["observation_count"] == 0


def test_object_verify_reject_and_confidence_moving_average() -> None:
    memory = PriorMapMemory(
        base_map=_base_map(),
        alignment=PriorMapAlignment.identity(),
        confidence_alpha=0.5,
    )

    memory._record_object_observation("prior_book", "runtime_book_1", confidence=0.8, step=4)
    memory._record_object_observation("prior_book", "runtime_book_1", confidence=0.4, step=5)
    memory.mark_object_verified("prior_book", "runtime_book_1", step=6)
    memory.mark_prior_rejected("prior_book", "wrong instance", step=7)

    state = memory.object_states["prior_book"]
    assert state.observation_count == 2
    assert state.confidence == pytest.approx(0.4)
    assert state.verified is False
    assert state.rejected is True
    assert state.rejection_reasons == ("wrong instance",)

    current_obj = memory.current_map().object_by_uid("prior_book")
    runtime_state = current_obj.metadata["runtime_state"]
    assert current_obj.confidence == pytest.approx(0.4)
    assert runtime_state["matched_runtime_uid"] == "runtime_book_1"
    assert runtime_state["verified"] is False
    assert runtime_state["rejected"] is True


def test_update_from_mapper_matches_room_and_object_by_label() -> None:
    memory = PriorMapMemory(base_map=_base_map(), alignment=PriorMapAlignment.identity())
    mapper = SimpleNamespace(
        current_room_uid="room_kitchen",
        robot_pose=(1.0, 2.0, 0.0),
        objects=[
            {
                "uid": "runtime_fridge_7",
                "label": "refrigerator",
                "confidence": 0.9,
                "room_uid": "room_kitchen",
            }
        ],
        rooms=[
            {
                "uid": "runtime_room_kitchen",
                "label": "kitchen",
                "confidence": 0.7,
            }
        ],
    )

    record = memory.update_from_mapper(mapper, step=8)

    assert record.pose_xyz == (1.0, 2.0, 0.0)
    assert record.source == "mapper"
    assert record.observed_object_uids == ("runtime_fridge_7",)
    assert record.observed_object_labels == ("refrigerator",)
    assert memory.room_states["room_kitchen"].visit_count == 1
    assert memory.room_states["room_kitchen"].observation_count == 2
    assert memory.object_states["prior_fridge"].observation_count == 1
    assert memory.object_states["prior_fridge"].matched_runtime_uid == "runtime_fridge_7"
    assert memory.object_states["prior_fridge"].confidence == pytest.approx(0.9)


def test_update_from_snapshot_uses_duck_typed_semantic_snapshot() -> None:
    memory = PriorMapMemory(base_map=_base_map(), alignment=PriorMapAlignment.identity())
    snapshot = SimpleNamespace(
        timestamp=12.0,
        source="sysnav_ros",
        robot_pose=SimpleNamespace(position=(0.0, 1.0, 0.0), frame_id="map"),
        objects=(
            SimpleNamespace(
                uid="runtime_book_2",
                label="book",
                confidence=0.75,
                room_id="room_living",
            ),
        ),
        rooms=(
            SimpleNamespace(
                uid="room_living",
                label="living room",
                confidence=0.8,
                explored=True,
            ),
        ),
    )

    record = memory.update_from_snapshot(snapshot)

    assert record.source == "sysnav_ros"
    assert record.pose_xyz == (0.0, 1.0, 0.0)
    assert record.room_hypothesis_uid == "room_living"
    assert memory.room_states["room_living"].visit_count == 1
    assert memory.object_states["prior_book"].observation_count == 1
    assert memory.object_states["prior_book"].matched_runtime_uid == "runtime_book_2"


def test_memory_rejects_unknown_prior_ids() -> None:
    memory = PriorMapMemory(base_map=_base_map(), alignment=PriorMapAlignment.identity())

    with pytest.raises(KeyError):
        memory.mark_room_visited("missing_room", step=1)

    with pytest.raises(KeyError):
        memory.mark_object_verified("missing_object", "runtime", step=1)

    with pytest.raises(KeyError):
        memory.mark_prior_rejected("missing_object", "reason", step=1)
