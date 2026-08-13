import json
from pathlib import Path

from prior_map.alignment import PriorMapAlignment
from prior_map.contracts import ObjectPrior, PriorMapData, PriorObject, PriorRoom, SearchPriorResult
from prior_map.evaluation import (
    prior_map_failure_modes,
    prior_map_metrics_fields,
    prior_map_metrics_summary,
    write_prior_map_static_artifacts,
    write_prior_map_step_artifacts,
)
from prior_map.memory import PriorMapMemory


def _base_map() -> PriorMapData:
    return PriorMapData(
        scene_id="eval_lab",
        rooms=(
            PriorRoom(uid="room_kitchen", label="kitchen", centroid_xy=(1.0, 1.0)),
            PriorRoom(uid="room_living", label="living room", centroid_xy=(3.0, 1.0)),
        ),
        objects=(
            PriorObject(
                uid="prior_book",
                label="book",
                position_xyz=(3.2, 1.1, 0.0),
                parent_room_uid="room_living",
            ),
        ),
        world_min=(0.0, 0.0),
        world_max=(4.0, 2.0),
    )


def _result() -> SearchPriorResult:
    return SearchPriorResult(
        object_rankings=(
            ObjectPrior(
                object_uid="prior_book",
                label="book",
                score=1.5,
                reason="target concept match",
            ),
        ),
        diagnostics={
            "authority": "ranking_only",
            "live_conflicts": [{"type": "object_room_mismatch"}],
        },
    )


def test_evaluation_artifacts_write_map_alignment_som_debug_and_failure_files(tmp_path: Path) -> None:
    memory = PriorMapMemory(base_map=_base_map(), alignment=PriorMapAlignment.identity())
    memory.mark_prior_rejected("prior_book", "wrong visible instance", step=4)
    memory.room_states["room_kitchen"].metadata["exhausted"] = True

    static_paths = write_prior_map_static_artifacts(output_dir=tmp_path, memory=memory)
    step_paths = write_prior_map_step_artifacts(
        output_dir=tmp_path,
        step=4,
        memory=memory,
        prior_result=_result(),
        query_payload={"step": 4, "authority": "ranking_only"},
        prompt_context_payload={"natural_language": "summary"},
    )

    assert Path(static_paths["base_map"]).exists()
    assert Path(static_paths["alignment"]).exists()
    assert Path(static_paths["som"]["global"]["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert Path(static_paths["som"]["global"]["markers"]).exists()
    assert Path(step_paths["runtime_state"]).exists()
    assert Path(step_paths["query"]).exists()
    assert Path(step_paths["debug"]).exists()
    failure_modes = json.loads(Path(step_paths["failure_modes"]).read_text(encoding="utf-8"))
    assert failure_modes["wrong_prior"] == 1
    assert failure_modes["live_conflict"] == 1
    assert failure_modes["prior_exhausted"] == 1


def test_prior_map_metrics_summary_and_csv_fields() -> None:
    memory = PriorMapMemory(
        base_map=_base_map(),
        alignment=PriorMapAlignment.unavailable(reason="not enough calibration points"),
    )
    summary = prior_map_metrics_summary(
        enabled=True,
        alignment=memory.alignment,
        memory=memory,
        prior_result=_result(),
    )
    fields = prior_map_metrics_fields(summary)

    assert summary["alignment_confidence"] == 0.0
    assert summary["failure_modes"]["alignment_mismatch"] == 1
    assert fields["prior_map_enabled"] is True
    assert fields["prior_map_top_object_uid"] == "prior_book"
    assert fields["prior_map_failure_live_conflict"] == 1


def test_failure_modes_disabled_prior_map_are_zero() -> None:
    failure_modes = prior_map_failure_modes(
        alignment=None,
        memory=None,
        prior_result=None,
    )

    assert failure_modes["wrong_prior"] == 0
    assert failure_modes["alignment_mismatch"] == 0
    assert failure_modes["live_conflict"] == 0
    assert failure_modes["prior_exhausted"] == 0
