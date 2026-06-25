import json
from pathlib import Path
from types import SimpleNamespace

from prior_map.alignment import PriorMapAlignment
from prior_map.contracts import PriorMapData, PriorObject, PriorRoom
from prior_map.simulation import (
    PriorMapSimulationRuntime,
    configure_mapper_prior_map,
    refresh_mapper_prior_map_query,
)


def _base_map() -> PriorMapData:
    return PriorMapData(
        scene_id="sim_lab",
        rooms=(
            PriorRoom(uid="room_kitchen", label="kitchen", centroid_xy=(1.0, 1.0), confidence=0.8),
            PriorRoom(uid="room_living", label="living room", centroid_xy=(4.0, 1.0), confidence=0.6),
        ),
        objects=(
            PriorObject(
                uid="prior_mug",
                label="mug",
                position_xyz=(1.2, 1.2, 0.0),
                parent_room_uid="room_kitchen",
                aliases=("cup",),
                confidence=0.7,
            ),
        ),
        world_min=(0.0, 0.0),
        world_max=(5.0, 3.0),
        source_format="json",
    )


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        raw_instruction="find the cup in the kitchen",
        dataset_target="mug",
        target_detector_prompts=("mug",),
        target_match_terms=("mug", "cup"),
        search_priors=SimpleNamespace(room_hints=("kitchen",), support_objects=(), affordances=()),
        targets=(SimpleNamespace(name="mug", terminal=True, detector_terms=("mug",), aliases=("cup",)),),
    )


def _mapper(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        save_dir=str(tmp_path),
        target="mug",
        objects=(
            SimpleNamespace(uid="runtime_mug", tag="cup", confidence=0.9, parent_room_uid="room_kitchen"),
        ),
        room_nodes=(
            SimpleNamespace(uid="room_kitchen", label="kitchen"),
            SimpleNamespace(uid="room_living", label="living room"),
        ),
        rooms=(
            SimpleNamespace(uid="room_kitchen", label="kitchen"),
            SimpleNamespace(uid="room_living", label="living room"),
        ),
        frontiers=(SimpleNamespace(uid="frontier_1", room_uid="room_kitchen"),),
        current_room_uid="room_kitchen",
        robot_pose=(0.5, 0.5, 0.0),
    )


def test_simulation_runtime_updates_queries_and_writes_artifacts(tmp_path: Path) -> None:
    runtime = PriorMapSimulationRuntime(
        base_map=_base_map(),
        alignment=PriorMapAlignment.identity(),
        artifact_root=tmp_path / "prior_map",
    )
    mapper = _mapper(tmp_path)
    configure_mapper_prior_map(mapper, runtime)
    runtime.begin_episode(tmp_path / "episode-0", 0)

    result = refresh_mapper_prior_map_query(mapper, plan=_plan(), step=7, episode_idx=0)

    assert result is mapper.search_prior_result
    assert result.object_rankings[0].object_uid == "prior_mug"
    assert result.room_rankings[0].room_uid == "room_kitchen"
    assert mapper.prior_map_prompt_context is not None
    assert not hasattr(result, "motion_goal")
    assert not hasattr(result, "navigation_intent")

    prior_dir = tmp_path / "episode-0" / "prior_map"
    assert (prior_dir / "base_map.json").exists()
    assert (prior_dir / "alignment.json").exists()
    assert (prior_dir / "query_000007.json").exists()
    assert (prior_dir / "search_prior_result_000007.json").exists()
    query_payload = json.loads((prior_dir / "query_000007.json").read_text(encoding="utf-8"))
    assert query_payload["authority"] == "ranking_only"
    assert query_payload["runtime_counts"]["objects"] == 1


def test_simulation_runtime_reuses_same_step_query_without_duplicate_observation(tmp_path: Path) -> None:
    runtime = PriorMapSimulationRuntime(
        base_map=_base_map(),
        alignment=PriorMapAlignment.identity(),
        artifact_root=tmp_path / "prior_map",
    )
    mapper = _mapper(tmp_path)
    configure_mapper_prior_map(mapper, runtime)

    first = refresh_mapper_prior_map_query(mapper, plan=_plan(), step=3, episode_idx=0)
    second = refresh_mapper_prior_map_query(mapper, plan=_plan(), step=3, episode_idx=0)

    assert first is second
    assert len(runtime.memory.observations) == 1


def test_simulation_entrypoints_expose_prior_map_cli_and_hooks() -> None:
    entrypoint = Path("objnav_benchmark_with_process_obs.py").read_text(encoding="utf-8")
    mapper_source = Path("mapper_with_process_obs.py").read_text(encoding="utf-8")

    for cli_arg in ("--enable_prior_map", "--prior_map_path", "--prior_map_source", "--prior_map_alignment"):
        assert cli_arg in entrypoint
    assert "build_prior_map_simulation_runtime" in entrypoint
    assert "configure_mapper_prior_map" in entrypoint
    assert "refresh_mapper_prior_map_query" in mapper_source
    assert "search_prior_result" in mapper_source
