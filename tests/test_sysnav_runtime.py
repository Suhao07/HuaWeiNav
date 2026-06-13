from types import SimpleNamespace

from real_robot.contracts import (
    CameraFrame,
    CameraModel,
    EvidenceSource,
    MotionGoal,
    MotionGoalMode,
    NavigationIntent,
    NavigationStatus,
    NavigationStatusCode,
    Pose3D,
    RealObservation,
    ViewEvidence,
    ViewpointGoal,
)
from real_robot.sysnav_runtime import (
    LatestObservationEvidenceProvider,
    SysNavInstructionRuntime,
    SysNavSemanticMapBridge,
    ViewpointEvidenceLoop,
)


def _point(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _header(sec=1, nanosec=0, frame_id="map"):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _object_list_msg():
    return SimpleNamespace(
        header=_header(sec=3),
        nodes=[
            SimpleNamespace(
                header=_header(sec=3),
                object_id=[11],
                label="book",
                position=_point(1.0, 2.0, 0.0),
                bbox3d=[],
                cloud=None,
                status=True,
                img_path="/tmp/book.npy",
                is_asked_vlm=False,
                viewpoint_id=5,
            )
        ],
    )


def _room_list_msg():
    return SimpleNamespace(
        nodes=[
            SimpleNamespace(
                id=2,
                show_id=2,
                centroid=_point(4.0, 0.0, 0.0),
                neighbors=[],
                is_connected=True,
                area=10.0,
                room_mask=None,
                polygon=None,
            )
        ]
    )


class FakePolicy:
    def __init__(self):
        self.calls = []

    def decide(self, snapshot, instruction=None):
        self.calls.append((snapshot, instruction))
        return NavigationIntent(
            mode=MotionGoalMode.GO_TO_OBJECT,
            goal_pose=Pose3D(position=(1.0, 2.0, -0.8)),
            target_object_uid=snapshot.objects[0].uid,
            reason="go to selected SysNav object",
        )


class FakeMotionController:
    def __init__(self, statuses=None):
        self.goals = []
        self.statuses = list(statuses or [NavigationStatus(NavigationStatusCode.REACHED)])

    def send_goal(self, goal: MotionGoal) -> str:
        self.goals.append(goal)
        return f"goal-{len(self.goals)}"

    def poll_status(self, goal_id: str) -> NavigationStatus:
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]


def test_semantic_map_bridge_builds_snapshot_from_cached_sysnav_topics() -> None:
    bridge = SysNavSemanticMapBridge(robot_pose_provider=lambda: Pose3D(position=(0.0, 0.0, 0.0)))

    assert bridge.build_snapshot() is None

    bridge.update_object_nodes(_object_list_msg())
    bridge.update_room_nodes(_room_list_msg())
    snapshot = bridge.build_snapshot(timestamp=10.0)

    assert snapshot.timestamp == 10.0
    assert snapshot.object_by_uid("sysnav_object:11").label == "book"
    assert snapshot.room_by_uid("sysnav_room:2").centroid == (4.0, 0.0, 0.0)


def test_instruction_runtime_waits_until_sysnav_object_nodes_arrive() -> None:
    bridge = SysNavSemanticMapBridge(robot_pose_provider=lambda: Pose3D(position=(0.0, 0.0, 0.0)))
    runtime = SysNavInstructionRuntime(
        semantic_map_bridge=bridge,
        high_level_policy=FakePolicy(),
        motion_controller=FakeMotionController(),
        now_fn=lambda: 1.0,
    )

    decision = runtime.step("find a book")

    assert decision.intent.mode == MotionGoalMode.WAIT
    assert decision.motion_goal is None
    assert decision.reason == "waiting for SysNav semantic map"


def test_instruction_runtime_dispatches_navigation_intent_to_motion_controller() -> None:
    bridge = SysNavSemanticMapBridge(robot_pose_provider=lambda: Pose3D(position=(0.0, 0.0, 0.0)))
    bridge.update_object_nodes(_object_list_msg())
    bridge.update_room_nodes(_room_list_msg())
    policy = FakePolicy()
    controller = FakeMotionController([NavigationStatus(NavigationStatusCode.RUNNING)])
    runtime = SysNavInstructionRuntime(
        semantic_map_bridge=bridge,
        high_level_policy=policy,
        motion_controller=controller,
        now_fn=lambda: 4.0,
    )

    decision = runtime.step("find a book")

    assert policy.calls[0][1] == "find a book"
    assert controller.goals[0].target_object_uid == "sysnav_object:11"
    assert decision.motion_goal.mode == MotionGoalMode.GO_TO_OBJECT
    assert decision.navigation_status.status == NavigationStatusCode.RUNNING
    assert decision.lower_planner_state["object_count"] == 1


class FakeEvidenceProvider:
    def __init__(self):
        self.calls = []

    def capture(self, goal, status):
        self.calls.append((goal, status))
        return ViewEvidence(
            source=EvidenceSource.VIEWPOINT_CAPTURE,
            timestamp=7.0,
            pose=status.current_pose,
            image_ref="/tmp/current_rgb.png",
            camera_model=CameraModel.PANORAMA,
            bbox_xyxy=(10.0, 20.0, 100.0, 160.0),
            target_object_uid=goal.target_object_uid,
            quality={"center_score": 0.8},
        )


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, evidence, context):
        self.calls.append((evidence, context))
        return {"decision": "accept", "reason": "current view satisfies instruction"}


def test_viewpoint_evidence_loop_waits_reached_then_runs_verifier() -> None:
    reached = NavigationStatus(
        NavigationStatusCode.REACHED,
        current_pose=Pose3D(position=(1.0, 2.0, -0.8)),
        message="reached waypoint",
    )
    controller = FakeMotionController([NavigationStatus(NavigationStatusCode.RUNNING), reached])
    evidence_provider = FakeEvidenceProvider()
    verifier = FakeVerifier()
    loop = ViewpointEvidenceLoop(
        motion_controller=controller,
        evidence_provider=evidence_provider,
        final_verifier=verifier,
        sleep_fn=lambda _: None,
        now_fn=lambda: 0.0,
    )
    goal = ViewpointGoal(
        pose=Pose3D(position=(1.0, 2.0, -0.8)),
        target_object_uid="sysnav_object:11",
        evidence_requirements={"target_visible": True},
    )

    result = loop.run(goal, context={"raw_instruction": "find a book"})

    assert result.status.status == NavigationStatusCode.REACHED
    assert result.evidence.target_object_uid == "sysnav_object:11"
    assert result.metadata["verifier_decision"]["decision"] == "accept"
    assert verifier.calls[0][1]["raw_instruction"] == "find a book"
    assert verifier.calls[0][1]["motion_status"] == "reached"


def test_viewpoint_evidence_loop_does_not_verify_when_motion_blocked() -> None:
    blocked = NavigationStatus(NavigationStatusCode.BLOCKED, message="local planner blocked")
    controller = FakeMotionController([blocked])
    evidence_provider = FakeEvidenceProvider()
    verifier = FakeVerifier()
    loop = ViewpointEvidenceLoop(
        motion_controller=controller,
        evidence_provider=evidence_provider,
        final_verifier=verifier,
    )

    result = loop.run(ViewpointGoal(pose=Pose3D(position=(1.0, 0.0, 0.0))))

    assert result.status.status == NavigationStatusCode.BLOCKED
    assert result.evidence is None
    assert evidence_provider.calls == []
    assert verifier.calls == []


def test_latest_observation_evidence_provider_uses_current_rgb_and_crop_payload() -> None:
    observation = RealObservation(
        timestamp=9.0,
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0)),
        camera_frames=(
            CameraFrame(
                image_ref="/tmp/rgb.png",
                camera_model=CameraModel.PINHOLE,
                frame_id="camera",
            ),
        ),
    )
    provider = LatestObservationEvidenceProvider(
        observation_provider=lambda: observation,
        crop_provider=lambda goal, obs: {
            "bbox_xyxy": (1.0, 2.0, 30.0, 40.0),
            "quality": {"bbox_area_ratio": 0.1},
            "metadata": {"crop_source": "object_node"},
        },
    )

    evidence = provider.capture(
        ViewpointGoal(
            pose=Pose3D(position=(0.0, 0.0, 0.0)),
            target_object_uid="sysnav_object:11",
        ),
        NavigationStatus(NavigationStatusCode.REACHED),
    )

    assert evidence.image_ref == "/tmp/rgb.png"
    assert evidence.camera_model == CameraModel.PINHOLE
    assert evidence.bbox_xyxy == (1.0, 2.0, 30.0, 40.0)
    assert evidence.metadata["crop_source"] == "object_node"
