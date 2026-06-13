import sys
import types


sys.modules.setdefault("cv2", types.SimpleNamespace())

from instruction_adapter.verifier import (
    FinalInstructionVerifier,
    _planner_hard_constraint_infeasible,
    hard_stop_constraints_from_evidence,
)


def test_view_guidance_reports_distance_as_prompt_fact():
    """Geometry distance should guide the prompt without hard-blocking accept."""

    guidance, preferred = FinalInstructionVerifier._view_guidance(
        {
            "geometry": {
                "distance_to_object": 2.42,
                "required_stop_distance": 1.5,
            },
            "view_quality_facts": {
                "bbox_center_norm": [0.5, 0.5],
                "bbox_area_ratio": 0.05,
                "center_offset_norm": 0.0,
                "border_margin_norm": 0.5,
                "projection_failed": False,
            },
        }
    )

    assert any("current_distance_to_object=2.420m" in item for item in guidance)
    assert any("benchmark_success_distance=1.500m" in item for item in guidance)
    assert "closest executable stop view" in preferred


def test_view_guidance_reports_projection_failure_as_prompt_fact():
    """Projection failures are VLM guidance, not Python-side rejection rules."""

    guidance, preferred = FinalInstructionVerifier._view_guidance(
        {
            "geometry": {"projection_failed_in_final_view": True},
            "view_quality_facts": {
                "bbox_area_ratio": 0.001,
                "center_offset_norm": 0.6,
                "border_margin_norm": 0.0,
                "projection_failed": True,
            },
            "view_control": {
                "budget_exhausted": True,
                "best_visual_evidence": {
                    "image_paths": {"current_rgb_with_bbox_path": "best_target.jpg"},
                },
            },
        }
    )

    assert "current target projection failed" in guidance[0]
    assert any("view_control_budget_exhausted=True" in item for item in guidance)
    assert "candidate" in preferred


def test_hard_stop_constraints_report_unsatisfied_distance():
    """Distance is a structured stop contract, not only natural-language advice."""

    constraints = hard_stop_constraints_from_evidence(
        {
            "geometry": {
                "distance_to_object": 2.512,
                "required_stop_distance": 1.5,
            },
            "view_control": {
                "best_visual_evidence": {
                    "image_paths": {"current_rgb_with_bbox_path": "best.jpg"},
                },
            },
        }
    )

    assert constraints["satisfied"] is False
    assert constraints["failed"][0]["name"] == "within_final_stop_distance"
    assert constraints["failed"][0]["margin"] < 0


def test_vlm_report_cannot_override_planner_hard_stop_contract():
    """Only planner/geometry evidence may mark a physical stop contract infeasible."""

    constraints = hard_stop_constraints_from_evidence(
        {
            "geometry": {
                "distance_to_object": 2.278,
                "required_stop_distance": 1.5,
            }
        }
    )

    allowed, reason = _planner_hard_constraint_infeasible({
        **constraints,
        "vlm_report": {
            "infeasible_or_not_applicable": True,
            "reason": "The image suggests closer views might crop the target.",
        },
    })

    assert allowed is False
    assert reason == ""


def test_planner_proof_can_mark_hard_stop_contract_infeasible():
    """Best-available stop requires an explicit geometry-side infeasibility proof."""

    constraints = hard_stop_constraints_from_evidence(
        {
            "geometry": {
                "distance_to_object": 2.278,
                "required_stop_distance": 1.5,
            },
            "planner_infeasibility_proof": {
                "infeasible_by_geometry": True,
                "reason": "all pathfinder-validated closer stop poses collide",
            },
        }
    )

    allowed, reason = _planner_hard_constraint_infeasible(constraints)

    assert allowed is True
    assert "collide" in reason
