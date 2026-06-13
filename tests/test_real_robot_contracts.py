from pathlib import Path

import pytest

from real_robot.contracts import (
    CameraModel,
    DetectionFrame,
    EvidenceSource,
    MotionGoalMode,
    NavigationIntent,
    NavigationStatus,
    NavigationStatusCode,
    ObjectNodeSnapshot,
    Pose3D,
    RoomSnapshot,
    SemanticMapSnapshot,
    ViewEvidence,
    ViewpointGoal,
)


def test_contracts_stay_platform_neutral() -> None:
    source = Path("real_robot/contracts.py").read_text(encoding="utf-8")

    assert "import rclpy" not in source
    assert "import rospy" not in source
    assert "import habitat" not in source
    assert "import numpy" not in source
    assert "from rclpy" not in source
    assert "from rospy" not in source
    assert "from habitat" not in source
    assert "from numpy" not in source


def test_detection_frame_validates_parallel_fields() -> None:
    with pytest.raises(ValueError):
        DetectionFrame(
            timestamp=1.0,
            image_ref="frame.png",
            boxes_xyxy=((0.0, 0.0, 10.0, 10.0),),
            labels=("cup", "chair"),
            confidences=(0.9,),
        )

    with pytest.raises(ValueError):
        DetectionFrame(
            timestamp=1.0,
            image_ref="frame.png",
            boxes_xyxy=((10.0, 0.0, 5.0, 10.0),),
            labels=("cup",),
            confidences=(0.9,),
        )


def test_navigation_intent_converts_to_motion_goal() -> None:
    pose = Pose3D(position=(1.0, 2.0, 0.0), frame_id="map")
    intent = NavigationIntent(
        mode=MotionGoalMode.GO_TO_OBJECT,
        goal_pose=pose,
        target_object_uid="obj_tv_1",
        reason="semantic target confirmed",
        metadata={"instruction": "find the tv"},
    )

    goal = intent.to_motion_goal()

    assert goal.requires_motion()
    assert goal.mode == MotionGoalMode.GO_TO_OBJECT
    assert goal.goal_pose == pose
    assert goal.target_object_uid == "obj_tv_1"
    assert goal.metadata["instruction"] == "find the tv"


def test_viewpoint_goal_exports_motion_goal_with_evidence_context() -> None:
    viewpoint = ViewpointGoal(
        pose=Pose3D(position=(0.5, -1.0, 0.0)),
        purpose=MotionGoalMode.IMPROVE_VIEW,
        look_at=(1.0, -1.0, 1.2),
        target_object_uid="book_1",
        anchor_object_uid="shelf_1",
        relation_edge_id="book_1:on:shelf_1",
        evidence_requirements={"co_visible": True},
    )

    goal = viewpoint.as_motion_goal()

    assert goal.mode == MotionGoalMode.IMPROVE_VIEW
    assert goal.look_at == (1.0, -1.0, 1.2)
    assert goal.metadata["evidence_requirements"] == {"co_visible": True}
    assert goal.relation_edge_id == "book_1:on:shelf_1"


def test_semantic_map_snapshot_lookup() -> None:
    obj = ObjectNodeSnapshot(uid="obj_1", label="cabinet", position=(1.0, 0.0, 0.0))
    room = RoomSnapshot(uid="room_1", label="living room", objects=("obj_1",))
    snapshot = SemanticMapSnapshot(
        timestamp=3.0,
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0)),
        objects=(obj,),
        rooms=(room,),
    )

    assert snapshot.object_by_uid("obj_1") == obj
    assert snapshot.object_by_uid("missing") is None
    assert snapshot.room_by_uid("room_1") == room


def test_view_evidence_payload_is_verifier_friendly() -> None:
    evidence = ViewEvidence(
        source=EvidenceSource.VIEWPOINT_CAPTURE,
        timestamp=5.0,
        pose=Pose3D(position=(1.0, 2.0, 0.0), frame_id="map"),
        image_ref="evidence.png",
        camera_model=CameraModel.PANORAMA,
        bbox_xyxy=(10.0, 20.0, 100.0, 120.0),
        target_object_uid="obj_1",
        quality={"center_score": 0.8},
    )

    payload = evidence.for_verifier()

    assert payload["source"] == "viewpoint_capture"
    assert payload["pose"]["frame_id"] == "map"
    assert payload["camera_model"] == "panorama"
    assert payload["bbox_xyxy"] == (10.0, 20.0, 100.0, 120.0)
    assert payload["quality"] == {"center_score": 0.8}


def test_navigation_status_terminal_semantics() -> None:
    assert NavigationStatus(NavigationStatusCode.REACHED).is_terminal()
    assert NavigationStatus(NavigationStatusCode.REACHED).succeeded()
    assert NavigationStatus(NavigationStatusCode.RUNNING).is_terminal() is False
