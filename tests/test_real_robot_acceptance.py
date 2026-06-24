import sys
import types
from types import SimpleNamespace


sys.modules.setdefault("cv2", types.SimpleNamespace())

from instruction_adapter.contracts import ExecutionPolicy, InstructionPlan, TargetQuery
from planning.semantic_snapshot_context import SemanticMapSnapshotIntentAdapter, StaticInstructionPlanProvider
from real_robot.contracts import (
    MotionGoal,
    MotionGoalMode,
    NavigationIntent,
    NavigationStatus,
    NavigationStatusCode,
    Pose3D,
    RuntimeDecision,
    ViewpointGoal,
)
from real_robot.observation_cache import ObjectCropEvidenceProvider, RosObservationCache
from real_robot.sysnav_ros_adapters import RosNavigationStatusProvider, RosWaypointController
from real_robot.sysnav_runtime import (
    DryRunMotionController,
    RuntimeDecisionJsonlWriter,
    SysNavInstructionRuntime,
    SysNavSemanticMapBridge,
    ViewpointEvidenceLoop,
)


def _point(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _header(sec=1, nanosec=0, frame_id="map"):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _orientation(x=0.0, y=0.0, z=0.0, w=1.0):
    return SimpleNamespace(x=x, y=y, z=z, w=w)


def _pose_msg(x, y, z):
    return SimpleNamespace(position=_point(x, y, z), orientation=_orientation())


def _odom_msg(x, y, z, sec=1):
    return SimpleNamespace(header=_header(sec=sec), pose=SimpleNamespace(pose=_pose_msg(x, y, z)))


def _image_msg(sec=1, width=640, height=480):
    return SimpleNamespace(
        header=_header(sec=sec, frame_id="camera"),
        height=height,
        width=width,
        encoding="rgb8",
        step=width * 3,
        data=bytes(width * height * 3),
    )


def _detection_msg(sec=3):
    return SimpleNamespace(
        header=_header(sec=sec, frame_id="camera"),
        track_id=[11],
        x1=[100.0],
        y1=[120.0],
        x2=[220.0],
        y2=[260.0],
        label=["book"],
        confidence=[0.9],
        image=None,
    )


def _object_list_msg(sec=3):
    return SimpleNamespace(
        header=_header(sec=sec),
        nodes=[
            SimpleNamespace(
                header=_header(sec=sec),
                object_id=[11],
                label="book",
                position=_point(1.0, 2.0, 0.0),
                bbox3d=[],
                cloud=None,
                status=True,
                img_path="",
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


def _book_plan():
    return InstructionPlan(
        raw_instruction="find a book",
        targets=[
            TargetQuery(
                id="target_book",
                name="book",
                detector_terms=["book"],
                terminal=True,
            )
        ],
        execution=ExecutionPolicy(mode="any_target_success"),
        valid=True,
    )


def _semantic_bridge():
    bridge = SysNavSemanticMapBridge(robot_pose_provider=lambda: Pose3D(position=(0.0, 0.0, 0.0)))
    bridge.update_object_nodes(_object_list_msg())
    bridge.update_room_nodes(_room_list_msg())
    return bridge


class _FakePointStamped:
    def __init__(self):
        self.header = SimpleNamespace(frame_id="", stamp=None)
        self.point = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class _ScriptedMotionController:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.goals = []

    def send_goal(self, goal: MotionGoal) -> str:
        self.goals.append(goal)
        return "goal-1"

    def poll_status(self, goal_id: str) -> NavigationStatus:
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]


class _AcceptingVerifier:
    def __init__(self):
        self.calls = []

    def verify(self, evidence, context):
        self.calls.append((evidence, context))
        return {
            "satisfied": True,
            "decision": "accept",
            "reason": "acceptance smoke verified target evidence",
            "diagnostics": {
                "image_ref": evidence.image_ref,
                "verifier_payload": evidence.verifier_payload,
            },
        }


def test_acceptance_bag_replay_topics_build_semantic_snapshot() -> None:
    bridge = _semantic_bridge()

    snapshot = bridge.build_snapshot(timestamp=12.0)

    assert snapshot.source == "sysnav_ros"
    assert snapshot.object_by_uid("sysnav_object:11").label == "book"
    assert snapshot.room_by_uid("sysnav_room:2").centroid == (4.0, 0.0, 0.0)


def test_acceptance_dry_run_generates_navigation_intent_without_waypoint_publish() -> None:
    bridge = _semantic_bridge()
    policy = SemanticMapSnapshotIntentAdapter(StaticInstructionPlanProvider(_book_plan()))
    controller = DryRunMotionController()
    runtime = SysNavInstructionRuntime(
        semantic_map_bridge=bridge,
        high_level_policy=policy,
        motion_controller=controller,
        now_fn=lambda: 12.0,
    )

    decision = runtime.step("find a book")

    assert decision.intent.mode == MotionGoalMode.GO_TO_OBJECT
    assert decision.motion_goal.target_object_uid == "sysnav_object:11"
    assert decision.navigation_status.metadata["dry_run"] is True
    assert controller.goals == [decision.motion_goal]


def test_acceptance_waypoint_smoke_reports_running_then_reached() -> None:
    provider = RosNavigationStatusProvider(
        xy_tolerance_m=0.25,
        z_tolerance_m=1.0,
        timeout_s=30.0,
        no_progress_timeout_s=10.0,
        now_fn=lambda: 0.0,
    )
    provider.update_odometry(_odom_msg(0.0, 0.0, 0.0))
    publisher = _FakePublisher()
    controller = RosWaypointController(
        node=SimpleNamespace(),
        publisher=publisher,
        point_stamped_type=_FakePointStamped,
        status_provider=provider,
        waypoint_topic="/way_point",
    )
    goal = MotionGoal(
        mode=MotionGoalMode.GO_TO_OBJECT,
        goal_pose=Pose3D(position=(1.0, 0.0, 0.0)),
        target_object_uid="sysnav_object:11",
    )

    goal_id = controller.send_goal(goal)
    running = controller.poll_status(goal_id)
    provider.update_odometry(_odom_msg(0.9, 0.0, 0.0, sec=2))
    reached = controller.poll_status(goal_id)

    assert len(publisher.published) == 1
    assert publisher.published[0].point.x == 1.0
    assert running.status == NavigationStatusCode.RUNNING
    assert reached.status == NavigationStatusCode.REACHED


def test_acceptance_evidence_smoke_generates_view_evidence_and_payload(tmp_path) -> None:
    bridge = _semantic_bridge()
    cache = RosObservationCache()
    cache.update_pose(Pose3D(position=(1.0, 2.0, 0.0), stamp=3.0))
    cache.update_rgb_image(_image_msg(sec=3))
    cache.update_detection_result(_detection_msg(sec=3))
    evidence_provider = ObjectCropEvidenceProvider(
        observation_provider=cache.latest_observation,
        object_provider=lambda: bridge.build_snapshot(timestamp=12.0),
        detection_provider=lambda: cache.latest_detection_frame,
        mode="bbox_crop",
        now_fn=lambda: 12.0,
    )
    verifier = _AcceptingVerifier()
    loop = ViewpointEvidenceLoop(
        motion_controller=_ScriptedMotionController([NavigationStatus(NavigationStatusCode.REACHED)]),
        evidence_provider=evidence_provider,
        final_verifier=verifier,
        now_fn=lambda: 12.0,
        sleep_fn=lambda _: None,
    )

    result = loop.verify_reached(
        ViewpointGoal(
            pose=Pose3D(position=(1.0, 2.0, 0.0)),
            target_object_uid="sysnav_object:11",
        ),
        NavigationStatus(
            NavigationStatusCode.REACHED,
            goal_id="goal-1",
            current_pose=Pose3D(position=(1.0, 2.0, 0.0)),
        ),
        context={"raw_instruction": "find a book"},
    )
    writer = RuntimeDecisionJsonlWriter(tmp_path / "runtime_decisions.jsonl")
    payload = writer.write(
        RuntimeDecision(
            timestamp=12.0,
            intent=NavigationIntent(
                mode=MotionGoalMode.VERIFY_TARGET,
                target_object_uid=result.goal.target_object_uid,
                reason="acceptance evidence smoke",
            ),
            navigation_status=result.status,
            verifier_decision=result.metadata["verifier_decision"],
            reason=result.reason,
            metadata={"viewpoint_result": result},
        )
    )

    assert result.evidence.image_ref.startswith("ros:///camera/image/image/")
    assert result.evidence.bbox_xyxy == (100.0, 120.0, 220.0, 260.0)
    assert result.evidence.verifier_payload["target_object_uid"] == "sysnav_object:11"
    assert result.metadata["verifier_decision"]["decision"] == "accept"
    assert payload["metadata"]["viewpoint_result"]["evidence"]["verifier_payload"]["bbox_source"] == "detection_track"
    assert (tmp_path / "runtime_decisions.jsonl").read_text(encoding="utf-8").strip()


def test_acceptance_end_to_end_snapshot_to_waypoint_to_final_verifier() -> None:
    bridge = _semantic_bridge()
    cache = RosObservationCache()
    cache.update_pose(Pose3D(position=(1.0, 2.0, 0.0), stamp=3.0))
    cache.update_rgb_image(_image_msg(sec=3))
    cache.update_detection_result(_detection_msg(sec=3))
    policy = SemanticMapSnapshotIntentAdapter(StaticInstructionPlanProvider(_book_plan()))
    controller = _ScriptedMotionController(
        [
            NavigationStatus(NavigationStatusCode.RUNNING, goal_id="goal-1", message="accepted"),
            NavigationStatus(
                NavigationStatusCode.REACHED,
                goal_id="goal-1",
                current_pose=Pose3D(position=(1.0, 2.0, 0.0)),
                message="reached",
            ),
        ]
    )
    verifier = _AcceptingVerifier()
    runtime = SysNavInstructionRuntime(
        semantic_map_bridge=bridge,
        high_level_policy=policy,
        motion_controller=controller,
        viewpoint_evidence_loop=ViewpointEvidenceLoop(
            motion_controller=controller,
            evidence_provider=ObjectCropEvidenceProvider(
                observation_provider=cache.latest_observation,
                object_provider=lambda: bridge.build_snapshot(timestamp=12.0),
                detection_provider=lambda: cache.latest_detection_frame,
                mode="bbox_crop",
                now_fn=lambda: 12.0,
            ),
            final_verifier=verifier,
            now_fn=lambda: 12.0,
            sleep_fn=lambda _: None,
        ),
        now_fn=lambda: 12.0,
    )

    first = runtime.step("find a book")
    second = runtime.step("find a book")

    assert first.intent.mode == MotionGoalMode.GO_TO_OBJECT
    assert first.motion_goal.target_object_uid == "sysnav_object:11"
    assert first.navigation_status.status == NavigationStatusCode.RUNNING
    assert second.intent.mode == MotionGoalMode.STOP
    assert second.verifier_decision["decision"] == "accept"
    assert second.accepted_candidate_uid
    assert second.metadata["viewpoint_result"]["evidence"]["verifier_payload"]["target_object_uid"] == "sysnav_object:11"
    assert len(controller.goals) == 1
