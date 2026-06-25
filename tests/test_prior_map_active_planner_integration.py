import json
from types import SimpleNamespace

import numpy as np

from planning.exploration_policy import find_closest_nodes
from planning.room_policy import select_nearest_frontier_room
from prior_map.alignment import PriorMapAlignment
from prior_map.contracts import FrontierPrior, PriorMapData, RoomPrior, SearchPriorResult
from prior_map.simulation import PriorMapSimulationRuntime


class _FakeSim:
    def geodesic_distance(self, start, goal):
        return float(np.linalg.norm(np.asarray(start, dtype=float) - np.asarray(goal, dtype=float)))


def _node(idx: int, position, room_idx: int = 0):
    return SimpleNamespace(idx=idx, position=np.asarray(position, dtype=float), room_idx=room_idx, has_frontier=1)


def _mapper(tmp_path, *, prior_result=None, runtime=None):
    return SimpleNamespace(
        initial_position=np.asarray((0.0, 0.0, 0.0), dtype=float),
        current_position=np.asarray((0.0, 0.0, 0.0), dtype=float),
        env=SimpleNamespace(sim=_FakeSim()),
        search_prior_result=prior_result,
        prior_map_policy_adapter=SimpleNamespace(enabled=True),
        prior_map_current_step=5,
        prior_map_runtime=runtime,
        prior_map_frontier_distance_bias_m=1.0,
        prior_map_room_distance_bias_m=10.0,
        save_dir=str(tmp_path),
        room_nodes=(
            SimpleNamespace(uid="room_near", label="near room"),
            SimpleNamespace(uid="room_target", label="target room"),
        ),
        nodes=(),
    )


def test_active_frontier_prior_off_preserves_nearest_node(tmp_path) -> None:
    mapper = _mapper(tmp_path)
    nodes = [_node(1, (1.0, 0.0, 0.0)), _node(2, (5.0, 0.0, 0.0))]

    selected = find_closest_nodes(mapper, nodes)

    assert selected.idx == 1
    assert mapper.prior_map_last_chosen_frontier["prior_enabled"] is False
    assert mapper.prior_map_last_chosen_frontier["prior_changed_selection"] is False


def test_active_frontier_bias_changes_selection_and_writes_artifact(tmp_path) -> None:
    prior_result = SearchPriorResult(
        frontier_biases=(FrontierPrior(frontier_uid="2", score_delta=10.0, reason="near target prior"),)
    )
    runtime = PriorMapSimulationRuntime(
        base_map=PriorMapData(scene_id="scene"),
        alignment=PriorMapAlignment.identity(),
        artifact_root=tmp_path / "prior_map",
    )
    runtime.begin_episode(tmp_path / "episode-0", 0)
    mapper = _mapper(tmp_path, prior_result=prior_result, runtime=runtime)
    nodes = [_node(1, (1.0, 0.0, 0.0)), _node(2, (5.0, 0.0, 0.0))]

    selected = find_closest_nodes(mapper, nodes)

    assert selected.idx == 2
    payload = mapper.prior_map_last_chosen_frontier
    assert payload["baseline_selected_uid"] == "1"
    assert payload["selected_uid"] == "2"
    assert payload["prior_changed_selection"] is True
    assert payload["candidates"][1]["prior_score"] == 10.0
    artifact = tmp_path / "episode-0" / "prior_map" / "chosen_frontier_000005.json"
    assert artifact.exists()
    assert json.loads(artifact.read_text(encoding="utf-8"))["selected_uid"] == "2"


def test_room_policy_records_baseline_and_prior_adjusted_distance(tmp_path) -> None:
    prior_result = SearchPriorResult(
        room_rankings=(RoomPrior(room_uid="room_target", label="target room", score=1.0),)
    )
    mapper = _mapper(tmp_path, prior_result=prior_result)
    mapper.nodes = (
        _node(1, (1.0, 0.0, 0.0), room_idx=0),
        _node(2, (5.0, 0.0, 0.0), room_idx=1),
    )

    selection = select_nearest_frontier_room(mapper)

    assert selection.baseline_closest_node_idx == 1
    assert selection.closest_node_idx == 2
    assert selection.prior_changed_selection is True
    assert selection.selected_prior_score > 0.0
    assert selection.adjusted_distances[1] < selection.distances[1]
