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
    assert (prior_dir / "som_global.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (prior_dir / "som_global_markers.json").exists()
    assert (prior_dir / "floorplan_global.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (prior_dir / "floorplan_global_markers.json").exists()
    assert (prior_dir / "query_000007.json").exists()
    assert (prior_dir / "search_prior_result_000007.json").exists()
    assert (prior_dir / "runtime_state_000007.json").exists()
    assert (prior_dir / "debug_000007.json").exists()
    assert (prior_dir / "failure_modes_000007.json").exists()
    assert (prior_dir / "floorplan_step_000007.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (prior_dir / "floorplan_step_000007_markers.json").exists()
    assert (prior_dir / "dynamic_prior_map_bev_000007.png").exists()
    assert (prior_dir / "dynamic_prior_map_bev_000007_markers.json").exists()
    query_payload = json.loads((prior_dir / "query_000007.json").read_text(encoding="utf-8"))
    assert query_payload["authority"] == "ranking_only"
    assert query_payload["runtime_counts"]["objects"] == 1
    debug_payload = json.loads((prior_dir / "debug_000007.json").read_text(encoding="utf-8"))
    assert debug_payload["enabled"] is True
    assert debug_payload["top_object"]["uid"] == "prior_mug"


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


def test_shared_region_snapshot_and_event_decision(tmp_path, monkeypatch):
    """Real room_id/idx contracts, pose projection and persistent region choice."""
    import numpy as np
    from unittest.mock import Mock
    from prior_map.high_level_selector import HighLevelSelectionResult
    import planning.exploration_policy as policy

    monkeypatch.setattr(policy, '_geodesic_distances_to_nodes',
                        lambda mapper, nodes: np.array([float(n.idx + 1) for n in nodes]))
    selector = Mock()
    selector.select.return_value = HighLevelSelectionResult(
        selected_uid='runtime_room:1', selected_type='room', decision='select', source='test')
    runtime = PriorMapSimulationRuntime(base_map=_base_map(), alignment=PriorMapAlignment.identity(),
                                       artifact_root=tmp_path, high_level_selector=selector)
    mapper = _mapper(tmp_path)
    nodes = [SimpleNamespace(idx=i, state=0, has_frontier=True, position=np.array([i, 0., 0.])) for i in range(2)]
    mapper.room_nodes = [SimpleNamespace(room_id=i, nodes=[n]) for i, n in enumerate(nodes)]
    mapper.initial_position = np.zeros(3)
    mapper.current_rotation = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    mapper.trajectory_position = [np.array([0., 0., 0.]), np.array([0.5, 0., 0.])]
    runtime.begin_episode(tmp_path, 0)
    runtime.update_and_query(mapper=mapper, plan=_plan(), step=1)
    snapshot = json.loads((tmp_path/'prior_map/shared_map_000001.json').read_text())
    assert snapshot['pose']['heading_rad'] == -np.pi / 2
    assert snapshot['trajectory_xy'][-1] == [0.5, 0.]
    assert [r['uid'] for r in snapshot['regions']] == ['runtime_room:0', 'runtime_room:1']
    assert all(r.candidate_type == 'room' for r in selector.select.call_args.kwargs['candidates'])
    assert len(runtime.base_map.rooms) == 2
    runtime.update_and_query(mapper=mapper, plan=_plan(), step=20)
    assert selector.select.call_count == 1  # 位姿/时间变化不重复审批局部探索。
    nodes[1].state = 1
    runtime.update_and_query(mapper=mapper, plan=_plan(), step=21)
    assert runtime.last_high_level_selection.selected_uid == 'runtime_room:0'
    assert selector.select.call_count == 1  # 唯一可执行区域由确定性代码选择。

    mapper.trajectory_position.append(np.array([1., 0., 0.]))
    runtime.finish_episode(mapper=mapper, plan=_plan(), step=30)
    terminal = json.loads((tmp_path/'prior_map/shared_map_000030.json').read_text())
    assert terminal['terminal'] and terminal['trajectory_xy'][-1] == [1., 0.]
    assert selector.select.call_count == 1


def test_prior_polygons_split_one_online_room_without_guessing_overlap(tmp_path, monkeypatch):
    """Only unique polygon membership grants a semantic region identity."""
    import numpy as np
    import planning.exploration_policy as policy
    monkeypatch.setattr(policy, '_geodesic_distances_to_nodes', lambda mapper, nodes: np.ones(len(nodes)))
    base = PriorMapData(scene_id='partition', rooms=(
        PriorRoom(uid='left', label='kitchen', boundary_xy=((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))),
        PriorRoom(uid='right', label='study', boundary_xy=((2., -1.), (4., -1.), (4., 1.), (2., 1.))),
        PriorRoom(uid='overlap', boundary_xy=((2.5, -.5), (3.5, -.5), (3.5, .5), (2.5, .5))),
    ))
    runtime = PriorMapSimulationRuntime(base_map=base, alignment=PriorMapAlignment.identity(), artifact_root=tmp_path)
    nodes = [SimpleNamespace(idx=i, state=0, has_frontier=True, position=np.array([x, 0., 0.]))
             for i, x in enumerate([0., 2.25, 3.])]
    mapper = SimpleNamespace(room_nodes=[SimpleNamespace(room_id=0, nodes=nodes)],
                             initial_position=np.zeros(3), robot_pose=(0., 0., 0.))
    runtime.begin_episode(tmp_path, 0)
    runtime.update_and_query(mapper=mapper, plan=_plan(), step=1)
    groups = {r.uid: r.metadata['node_ids'] for r in runtime.region_candidates}
    assert groups == {'left': (0,), 'right': (1,), 'runtime_room:0': (2,)}
    assert len(base.rooms) == 3
    assert runtime.shared_map.room_by_uid('left').boundary_xy == base.rooms[0].boundary_xy


def test_region_layout_composition_preserves_objects_and_checks_frame(tmp_path):
    from dataclasses import replace
    import pytest
    from prior_map.simulation import PriorMapSimulationConfig
    base = _base_map()
    objects_path, layout_path = tmp_path/'objects.json', tmp_path/'layout.json'
    objects_path.write_text(json.dumps(base.to_dict()))
    layout = replace(base, objects=())
    layout_path.write_text(json.dumps(layout.to_dict()))
    config = PriorMapSimulationConfig(enabled=True, prior_map_path=str(objects_path),
        prior_map_layout=str(layout_path), artifact_root=str(tmp_path/'run'))
    runtime = PriorMapSimulationRuntime.from_config(config)
    assert runtime.base_map.objects == base.objects
    assert runtime.base_map.rooms == layout.rooms
    assert runtime.base_map.metadata['region_layout']['sha256']
    layout_path.write_text(json.dumps(replace(layout, frame_id='different').to_dict()))
    with pytest.raises(ValueError, match='scene_id and frame_id'):
        PriorMapSimulationRuntime.from_config(config)


def test_static_prior_region_survives_frontier_replacement(tmp_path):
    from prior_map.high_level_selector import HighLevelSelectionResult
    runtime = PriorMapSimulationRuntime(base_map=_base_map(), alignment=PriorMapAlignment.identity(), artifact_root=tmp_path)
    runtime.last_high_level_selection = HighLevelSelectionResult(selected_uid='room_kitchen', decision='select', selected_type='room')
    runtime.last_region_signature = (('room_kitchen', (1,)),)
    runtime.region_signature = (('room_kitchen', (2,)),)
    assert runtime._region_decision_due() is False
    runtime.region_signature = ()
    assert runtime._region_decision_due() is True
