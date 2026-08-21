import sys
import types


sys.modules.setdefault("cv2", types.SimpleNamespace())

from instruction_adapter.contracts import (
    ConceptQuery,
    Constraint,
    ExecutionPolicy,
    InstructionPlan,
    TargetQuery,
)
from instruction_adapter.verifier import VerificationResult, candidate_from_object
from real_robot.contracts import (
    MotionGoalMode,
    NavigationStatus,
    NavigationStatusCode,
    ObjectNodeSnapshot,
    Pose3D,
    SemanticMapSnapshot,
    ViewpointGoal,
    ViewpointResult,
)

from planning.semantic_snapshot_context import (
    SemanticMapSnapshotPolicyContext,
    SemanticMapSnapshotIntentAdapter,
    StaticInstructionPlanProvider,
    select_target_candidate_from_snapshot,
)


def _object(uid, label, position, confidence=0.9):
    return ObjectNodeSnapshot(
        uid=uid,
        label=label,
        position=position,
        confidence=confidence,
    )


def _snapshot(*objects):
    return SemanticMapSnapshot(
        timestamp=3.0,
        robot_pose=Pose3D(position=(0.0, 0.0, 0.0), frame_id="map"),
        objects=tuple(objects),
    )


def _book_plan():
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


def _book_on_table_plan():
    table_concept = ConceptQuery(id="anchor_table", name="table", role="anchor", terminal=False)
    return InstructionPlan(
        raw_instruction="find the book on the table",
        targets=[
            TargetQuery(
                id="target_book",
                name="book",
                detector_terms=["book"],
                terminal=True,
            ),
            TargetQuery(
                id="anchor_table",
                name="table",
                detector_terms=["table"],
                role="anchor",
                terminal=False,
                concept=table_concept,
            ),
        ],
        constraints=[
            Constraint(
                type="spatial",
                subject="book",
                relation="on",
                object="table",
                hardness="hard",
                object_concept=table_concept,
            )
        ],
        execution=ExecutionPolicy(mode="anchor_first_relation_search"),
        valid=True,
    )


def test_snapshot_context_reuses_existing_target_selection_policy() -> None:
    plan = _book_plan()
    selection = select_target_candidate_from_snapshot(
        _snapshot(_object("book-1", "book", (1.0, 2.0, 0.0))),
        plan=plan,
    )

    assert selection.result.found is True
    assert selection.selected_snapshot_uid == "book-1"
    assert selection.selected_pose.position == (1.0, 2.0, 0.0)
    assert selection.result.obj.tag == "book"


def test_snapshot_context_selects_anchor_through_existing_anchor_policy() -> None:
    plan = _book_on_table_plan()
    selection = select_target_candidate_from_snapshot(
        _snapshot(_object("table-1", "table", (2.0, 0.0, 0.0), confidence=0.95)),
        plan=plan,
    )

    assert selection.result.found is True
    assert selection.is_anchor_reference is True
    assert selection.selected_snapshot_uid == "table-1"
    assert selection.result.obj._instruction_reference_role == "anchor"


def test_snapshot_context_hard_reject_is_instance_scoped() -> None:
    plan = _book_plan()
    snapshot = _snapshot(
        _object("book-1", "book", (1.0, 0.0, 0.0), confidence=0.95),
        _object("book-2", "book", (2.0, 0.0, 0.0), confidence=0.75),
    )
    context = SemanticMapSnapshotPolicyContext(snapshot=snapshot, plan=plan)
    rejected_candidate = candidate_from_object(
        context.objects[0],
        canonical_label=context.target,
    )
    context.verification_ledger.put(
        plan.raw_instruction,
        rejected_candidate.uid,
        VerificationResult(satisfied=False, decision="reject_candidate", reason="wrong instance"),
    )

    selection = select_target_candidate_from_snapshot(snapshot, plan=plan, context=context)

    assert selection.result.found is True
    assert selection.selected_snapshot_uid == "book-2"
    assert selection.result.skipped_objs


def test_snapshot_intent_adapter_returns_target_navigation_intent() -> None:
    plan = _book_plan()
    adapter = SemanticMapSnapshotIntentAdapter(StaticInstructionPlanProvider(plan))

    intent = adapter.decide(_snapshot(_object("book-1", "book", (1.0, 2.0, 0.0))), "find the book")

    assert intent.mode == MotionGoalMode.GO_TO_OBJECT
    assert intent.target_object_uid == "book-1"
    assert intent.goal_pose is None
    assert intent.metadata["candidate_instance"].detector_label == "book"


def test_snapshot_intent_adapter_returns_anchor_navigation_intent() -> None:
    plan = _book_on_table_plan()
    adapter = SemanticMapSnapshotIntentAdapter(StaticInstructionPlanProvider(plan))

    intent = adapter.decide(_snapshot(_object("table-1", "table", (2.0, 0.0, 0.0))), "find the book on the table")

    assert intent.mode == MotionGoalMode.GO_TO_ANCHOR
    assert intent.anchor_object_uid == "table-1"
    assert intent.target_object_uid is None


def test_snapshot_intent_adapter_records_reject_candidate_feedback() -> None:
    plan = _book_plan()
    snapshot = _snapshot(
        _object("book-1", "book", (1.0, 0.0, 0.0), confidence=0.95),
        _object("book-2", "book", (2.0, 0.0, 0.0), confidence=0.75),
    )
    adapter = SemanticMapSnapshotIntentAdapter(StaticInstructionPlanProvider(plan))
    first = adapter.decide(snapshot, "find the book")
    result = ViewpointResult(
        goal=ViewpointGoal(pose=first.goal_pose, target_object_uid=first.target_object_uid),
        status=NavigationStatus(NavigationStatusCode.REACHED),
        metadata={
            "verifier_decision": {
                "satisfied": False,
                "decision": "reject_candidate",
                "reason": "wrong instance",
            }
        },
    )

    changed = adapter.handle_viewpoint_result(result, first)
    second = adapter.decide(snapshot, "find the book")

    assert changed is True
    assert first.target_object_uid == "book-1"
    assert second.target_object_uid == "book-2"


def test_snapshot_intent_adapter_records_accept_and_then_stops() -> None:
    plan = _book_plan()
    adapter = SemanticMapSnapshotIntentAdapter(StaticInstructionPlanProvider(plan))
    first = adapter.decide(_snapshot(_object("book-1", "book", (1.0, 0.0, 0.0))), "find the book")
    result = ViewpointResult(
        goal=ViewpointGoal(pose=first.goal_pose, target_object_uid=first.target_object_uid),
        status=NavigationStatus(NavigationStatusCode.REACHED),
        metadata={
            "verifier_decision": {
                "satisfied": True,
                "decision": "accept",
                "reason": "target verified",
            }
        },
    )

    changed = adapter.handle_viewpoint_result(result, first)
    final = adapter.decide(_snapshot(_object("book-1", "book", (1.0, 0.0, 0.0))), "find the book")

    assert changed is True
    assert final.mode == MotionGoalMode.STOP
