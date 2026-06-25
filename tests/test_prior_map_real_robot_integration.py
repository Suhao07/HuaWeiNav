import json
import sys
import types
from pathlib import Path


sys.modules.setdefault("cv2", types.SimpleNamespace())

from instruction_adapter.contracts import ExecutionPolicy, InstructionPlan, TargetQuery
from planning.semantic_snapshot_context import SemanticMapSnapshotIntentAdapter, StaticInstructionPlanProvider
from prior_map.alignment import PriorMapAlignment
from prior_map.contracts import ObjectPrior, PriorMapData, PriorObject, PriorRoom, SearchPriorResult
from prior_map.policy_adapter import PriorMapPolicyAdapter
from prior_map.real_robot import PriorMapRealRobotRuntime
from real_robot.contracts import MotionGoalMode, ObjectNodeSnapshot, Pose3D, SemanticMapSnapshot


def _base_map() -> PriorMapData:
    return PriorMapData(
        scene_id="real_lab",
        rooms=(
            PriorRoom(uid="room_kitchen", label="kitchen", centroid_xy=(1.0, 1.0), confidence=0.8),
            PriorRoom(uid="room_live", label="live lab", centroid_xy=(3.0, 1.0), confidence=0.6),
        ),
        objects=(
            PriorObject(
                uid="prior_mug",
                label="mug",
                position_xyz=(1.2, 1.0, 0.0),
                parent_room_uid="room_kitchen",
                aliases=("cup",),
                confidence=0.7,
            ),
        ),
        source_format="json",
        frame_id="prior_map",
        world_min=(0.0, 0.0),
        world_max=(4.0, 3.0),
    )


def _plan() -> InstructionPlan:
    return InstructionPlan(
        raw_instruction="find the book",
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


def _snapshot(*objects: ObjectNodeSnapshot) -> SemanticMapSnapshot:
    return SemanticMapSnapshot(
        timestamp=10.0,
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0), frame_id="map"),
        objects=tuple(objects),
        source="fake_replay",
    )


def test_real_robot_prior_runtime_updates_from_snapshot_and_writes_artifacts(tmp_path: Path) -> None:
    runtime = PriorMapRealRobotRuntime(
        base_map=_base_map(),
        alignment=PriorMapAlignment.identity(),
        run_directory=tmp_path,
    )
    snapshot = _snapshot(
        ObjectNodeSnapshot(
            uid="runtime_mug",
            label="mug",
            position=(2.0, 1.0, 0.0),
            room_id="room_live",
            confidence=0.9,
        )
    )
    plan = types.SimpleNamespace(
        raw_instruction="find the cup",
        dataset_target="mug",
        target_detector_prompts=("mug",),
        target_match_terms=("mug", "cup"),
        search_priors=types.SimpleNamespace(room_hints=("kitchen",), support_objects=(), affordances=()),
    )

    context = runtime.update_and_query(snapshot=snapshot, plan=plan, step=5)
    result = context["prior_result"]

    assert result.object_rankings[0].object_uid == "prior_mug"
    assert context["prior_map_diagnostics"]["authority"] == "ranking_only"
    assert context["prior_map_diagnostics"]["live_evidence_priority"] is True
    assert result.diagnostics["live_conflicts"][0]["resolution"] == "live_evidence_priority"
    assert not hasattr(result, "motion_goal")
    assert not hasattr(result, "navigation_intent")

    prior_dir = tmp_path / "prior_map"
    assert (prior_dir / "base_map.json").exists()
    assert (prior_dir / "alignment.json").exists()
    assert (prior_dir / "som_global.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (prior_dir / "som_global_markers.json").exists()
    assert (prior_dir / "query_000005.json").exists()
    assert (prior_dir / "search_prior_result_000005.json").exists()
    assert (prior_dir / "runtime_state_000005.json").exists()
    assert (prior_dir / "debug_000005.json").exists()
    assert (prior_dir / "failure_modes_000005.json").exists()
    query_payload = json.loads((prior_dir / "query_000005.json").read_text(encoding="utf-8"))
    assert query_payload["authority"] == "ranking_only"
    assert query_payload["live_evidence_priority"] is True
    debug_payload = json.loads((prior_dir / "debug_000005.json").read_text(encoding="utf-8"))
    assert debug_payload["failure_modes"]["live_conflict"] == 1


def test_semantic_snapshot_adapter_consumes_prior_ranking_context_without_stop_authority() -> None:
    prior_result = SearchPriorResult(
        object_rankings=(
            ObjectPrior(
                object_uid="prior_book",
                label="book",
                score=2.0,
                reason="prior-selected instance",
                matched_runtime_uid="book-1",
            ),
        ),
        diagnostics={"authority": "ranking_only", "live_conflicts": []},
    )

    def provider(snapshot, plan, step):
        return {
            "prior_result": prior_result,
            "prior_map_policy_adapter": PriorMapPolicyAdapter(),
            "prior_map_diagnostics": {"enabled": True, "authority": "ranking_only"},
        }

    adapter = SemanticMapSnapshotIntentAdapter(
        StaticInstructionPlanProvider(_plan()),
        prior_context_provider=provider,
    )
    intent = adapter.decide(
        _snapshot(
            ObjectNodeSnapshot(uid="book-1", label="book", position=(1.0, 0.0, 0.0), confidence=0.6),
            ObjectNodeSnapshot(uid="book-2", label="book", position=(2.0, 0.0, 0.0), confidence=0.95),
        ),
        "find the book",
    )

    assert intent.mode == MotionGoalMode.GO_TO_OBJECT
    assert intent.target_object_uid == "book-1"
    assert intent.stop_allowed is False
    assert intent.metadata["prior_map"]["enabled"] is True
    assert intent.metadata["prior_map"]["authority"] == "ranking_only"
    assert intent.metadata["concept_debug"][-1]["source"] == "prior_map_target_annotation"


def test_real_robot_node_and_launch_expose_prior_map_parameters() -> None:
    node_source = Path(
        "real_robot/ros2_ws/src/strive_sysnav_bringup/strive_sysnav_bringup/instruction_runtime_node.py"
    ).read_text(encoding="utf-8")
    launch_source = Path(
        "real_robot/ros2_ws/src/strive_sysnav_bringup/launch/strive_instruction_runtime.launch.py"
    ).read_text(encoding="utf-8")

    for text in ("prior_map_path", "prior_map_source", "prior_map_alignment"):
        assert text in node_source
        assert text in launch_source
    assert "build_prior_map_real_robot_runtime" in node_source
    assert "prior_context_provider" in node_source
