import sys
import types


sys.modules.setdefault("cv2", types.SimpleNamespace())

from instruction_adapter.view_control import ViewControlState


def test_view_control_keeps_attempts_for_same_candidate_objective_rewording():
    """Reworded VLM objectives must not reset an active better-view subgoal."""

    state = ViewControlState()
    state.start(
        "tv:abc",
        {"required_stop_distance": 1.5, "reason": "move closer"},
        {"view_quality_facts": {"bbox_area_ratio": 0.02, "center_offset_norm": 0.2}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.9, "distance_to_target": 1.3},
            {"pose": [1, 0, 0], "score": 0.8, "distance_to_target": 1.4},
        ]
    )
    first = state.next_proposal()
    assert first is not None

    state.start(
        "tv:abc",
        {"required_stop_distance": 1.5, "reason": "approach the television"},
        {"view_quality_facts": {"bbox_area_ratio": 0.03, "center_offset_norm": 0.1}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.9, "distance_to_target": 1.3},
            {"pose": [1, 0, 0], "score": 0.8, "distance_to_target": 1.4},
        ]
    )
    second = state.next_proposal()

    assert second is not None
    assert second.pose == [1.0, 0.0, 0.0]


def test_view_control_prioritizes_view_score_without_pending_physical_contract():
    """Pure soft-view optimization still ranks by predicted evidence quality."""

    state = ViewControlState()
    state.start(
        "tv:abc",
        {"required_stop_distance": 1.5},
        {"view_quality_facts": {"bbox_area_ratio": 0.02}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.95, "distance_to_target": 2.4},
            {"pose": [1, 0, 0], "score": 0.65, "distance_to_target": 1.4},
        ]
    )

    proposal = state.next_proposal()

    assert proposal is not None
    assert proposal.pose == [0.0, 0.0, 0.0]


def test_view_control_prioritizes_physical_stop_contract_when_pending():
    """Unmet stop distance is planner-owned and takes priority over soft view score."""

    state = ViewControlState()
    state.start(
        "book:abc",
        {
            "required_stop_distance": 1.5,
            "current_distance_to_object": 2.7,
            "hard_stop_constraints": {
                "satisfied": False,
                "failed": [
                    {
                        "name": "within_final_stop_distance",
                        "current_distance_to_object": 2.7,
                        "required_stop_distance": 1.5,
                    }
                ],
            },
        },
        {"view_quality_facts": {"bbox_area_ratio": 0.02}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.95, "distance_to_target": 2.4},
            {"pose": [1, 0, 0], "score": 0.65, "distance_to_target": 1.4},
            {"pose": [2, 0, 0], "score": 0.85, "distance_to_target": 1.7},
        ]
    )

    proposal = state.next_proposal()

    assert proposal is not None
    assert proposal.pose == [1.0, 0.0, 0.0]


def test_view_control_pins_first_semantic_visual_reference():
    """Later geometry-heavy views must not overwrite the stable target reference."""

    state = ViewControlState()
    state.start(
        "tv:abc",
        {"required_stop_distance": 1.5},
        {
            "current_rgb_with_bbox_path": "first_tv.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.02, "center_offset_norm": 0.2},
        },
    )
    state.pin_visual_evidence(
        {
            "current_rgb_with_bbox_path": "first_tv.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.02, "center_offset_norm": 0.2},
        },
        step=10,
        decision="need_better_view",
        reason="Target is semantically correct but the view can improve.",
    )
    state.pin_visual_evidence(
        {
            "current_rgb_with_bbox_path": "later_fireplace_drift.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.5, "center_offset_norm": 0.01},
        },
        step=20,
        decision="need_better_view",
        reason="Later projection is geometrically larger but may be drifted.",
    )

    context = state.as_context()

    assert context["pinned_visual_evidence"]["image_paths"]["current_rgb_with_bbox_path"] == "first_tv.jpg"
    assert context["latest_visual_evidence"]["image_paths"]["current_rgb_with_bbox_path"] == "later_fireplace_drift.jpg"


def test_view_control_budget_exhausts_after_max_attempts(monkeypatch):
    """The controller must not ask for unlimited better-view retries."""

    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_VERIFIER_CALLS", "4")
    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_NO_IMPROVEMENT_ROUNDS", "4")

    state = ViewControlState()
    state.start(
        "tv:abc",
        {"required_stop_distance": 1.5},
        {"view_quality_facts": {"bbox_area_ratio": 0.01, "center_offset_norm": 0.3}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.9, "distance_to_target": 1.3},
            {"pose": [1, 0, 0], "score": 0.8, "distance_to_target": 1.2},
            {"pose": [2, 0, 0], "score": 0.7, "distance_to_target": 1.1},
        ]
    )

    assert state.next_proposal() is not None
    state.record_attempt(
        10,
        {
            "current_rgb_with_bbox_path": "attempt1.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.02, "center_offset_norm": 0.2},
        },
        "need_better_view",
        semantic_satisfied=True,
        reason="visible but can improve",
    )
    assert not state.budget_exhausted()

    assert state.next_proposal() is not None
    state.record_attempt(
        11,
        {
            "current_rgb_with_bbox_path": "attempt2.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.05, "center_offset_norm": 0.1},
        },
        "need_better_view",
        semantic_satisfied=True,
        reason="still clipped",
    )

    context = state.as_context()

    assert state.budget_exhausted()
    assert context["budget_exhausted"] is True
    assert context["attempt_count"] == 2
    assert state.next_proposal() is None


def test_no_improvement_stalls_but_does_not_exhaust_remaining_proposals(monkeypatch):
    """Poor progress should switch proposals, not claim no better view exists."""

    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_VERIFIER_CALLS", "5")
    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_NO_IMPROVEMENT_ROUNDS", "2")

    state = ViewControlState()
    state.start(
        "book:abc",
        {"required_stop_distance": 1.5},
        {"view_quality_facts": {"bbox_area_ratio": 0.02, "center_offset_norm": 0.1}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.9, "distance_to_target": 2.4},
            {"pose": [1, 0, 0], "score": 0.8, "distance_to_target": 1.6},
            {"pose": [2, 0, 0], "score": 0.7, "distance_to_target": 1.3},
        ]
    )

    assert state.next_proposal() is not None
    for step in (10, 11):
        state.record_attempt(
            step,
            {
                "current_rgb_with_bbox_path": f"attempt{step}.jpg",
                "view_quality_facts": {"bbox_area_ratio": 0.01, "center_offset_norm": 0.5},
            },
            "need_better_view",
            semantic_satisfied=True,
            reason="no clear improvement yet",
        )
        if step == 10:
            assert state.next_proposal() is not None

    context = state.as_context()

    assert context["progress_stalled"] is True
    assert context["budget_exhausted"] is False
    assert context["remaining_feasible_proposals"] == 1
    assert context["closest_remaining_proposal_distance"] == 1.3
    assert state.next_proposal() is not None


def test_attempt_budget_does_not_exhaust_unmet_physical_contract(monkeypatch):
    """Soft attempt limits cannot close the loop while closer physical proposals remain."""

    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_VERIFIER_CALLS", "1")

    state = ViewControlState()
    state.start(
        "book:abc",
        {
            "required_stop_distance": 1.5,
            "current_distance_to_object": 2.5,
            "hard_stop_constraints": {
                "satisfied": False,
                "failed": [
                    {
                        "name": "within_final_stop_distance",
                        "current_distance_to_object": 2.5,
                        "required_stop_distance": 1.5,
                    }
                ],
            },
        },
        {"view_quality_facts": {"bbox_area_ratio": 0.02}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.9, "distance_to_target": 1.9},
            {"pose": [1, 0, 0], "score": 0.7, "distance_to_target": 1.3},
        ]
    )
    assert state.next_proposal() is not None
    state.record_attempt(
        10,
        {
            "current_rgb_with_bbox_path": "far.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.01, "center_offset_norm": 0.5},
        },
        "need_better_view",
        semantic_satisfied=True,
        reason="still outside stop distance",
    )

    context = state.as_context()

    assert context["physical_contract_pending"] is True
    assert context["remaining_physical_contract_proposals"] == 1
    assert context["attempt_budget_exhausted"] is True
    assert context["budget_exhausted"] is False
    assert state.next_proposal() is not None


def test_view_control_tracks_best_semantic_evidence(monkeypatch):
    """Best-available stop evidence should be the highest-quality semantic view."""

    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("STRIVE_VIEW_CONTROL_MAX_NO_IMPROVEMENT_ROUNDS", "5")

    state = ViewControlState()
    state.start(
        "tv:abc",
        {"required_stop_distance": 1.5},
        {"view_quality_facts": {"bbox_area_ratio": 0.01, "center_offset_norm": 0.3}},
    )
    state.set_proposals(
        [
            {"pose": [0, 0, 0], "score": 0.9, "distance_to_target": 1.3},
            {"pose": [1, 0, 0], "score": 0.8, "distance_to_target": 1.2},
        ]
    )

    assert state.next_proposal() is not None
    state.record_attempt(
        20,
        {
            "current_rgb_with_bbox_path": "better.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.08, "center_offset_norm": 0.05},
        },
        "need_better_view",
        semantic_satisfied=True,
        reason="best semantic view so far",
    )
    assert state.next_proposal() is not None
    state.record_attempt(
        21,
        {
            "current_rgb_with_bbox_path": "worse.jpg",
            "view_quality_facts": {"bbox_area_ratio": 0.01, "center_offset_norm": 0.5},
        },
        "need_better_view",
        semantic_satisfied=True,
        reason="worse view",
    )

    context = state.as_context()

    assert context["best_visual_evidence"]["image_paths"]["current_rgb_with_bbox_path"] == "better.jpg"
    assert context["best_attempt"]["step"] == 20
