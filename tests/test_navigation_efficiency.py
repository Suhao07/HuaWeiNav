"""Behavioral checks for constrained views, coverage ranking and evidence reuse."""
from types import SimpleNamespace as NS
import pytest
import numpy as np
from navigation_core.view_geometry import interpolate_polyline
from navigation.semantic_label_cache import LabelEvidence, reusable_label, matching_instance
from planning.exploration_policy import (
    select_baseline_exploration_node, select_coverage_frontier, select_room_first,
)
from instruction_adapter.view_control import ViewControlState

@pytest.mark.parametrize("semantic", [True, False])
def test_shared_room_first_order_and_local_side_effects(semantic):
    events, room, node = [], NS(state=0), object()
    def local(current):
        assert current is room
        events.append("local")
        current.state = 1
        return None
    def remaining():
        events.append("remaining")
        return node
    assert select_room_first(current_room=room, rooms=[room], local_choice=local,
        remaining_choice=remaining, room_choice=lambda: events.append("room"),
        closest_in_room=lambda selected: events.append("closest"), semantic=semantic) == (node, "remaining_nodes")
    assert events == ["local", "remaining"]
    room.state = 0
    events.clear()
    def choose():
        events.append("room")
        return room
    def closest(selected):
        assert selected is room
        events.append("closest")
        return node
    result = select_room_first(current_room=room, rooms=[room],
        local_choice=lambda current: events.append("local"), remaining_choice=remaining,
        room_choice=choose, closest_in_room=closest, semantic=semantic)
    assert result == (node, "room_semantic_policy" if semantic else "room_distance_policy")
    assert events == ["local", "room", "closest"]


def test_habitat_relocation_callback_mutation_precedes_room_exhaustion():
    events, room, node = [], NS(state=0), NS(room_idx=0)
    def relocate(current):
        events.append("reset_and_relocate")
        current.state = 1
    mapper = NS(nodes=[node], room_nodes=[room], explore_in_room_relocate=relocate,
                explore_after_fully_explored=lambda: events.append("remaining") or node)
    agent = NS(mapper=mapper, current_node_idx=0, relocate=True, gpt_relocate=True)
    assert select_baseline_exploration_node(agent, episode_idx=0, step=0) == (node, "remaining_nodes")
    assert events == ["reset_and_relocate", "remaining"]


def test_path_sampling_preserves_short_segment_endpoint_and_max_spacing():
    points = interpolate_polyline(np.array([[0, 0, 0], [.1, 0, 0], [.66, 0, 0]]))
    assert np.allclose(points[-1], [.66, 0, 0])
    assert max(np.linalg.norm(b-a) for a, b in zip(points, points[1:])) <= .25


def test_coverage_only_breaks_near_cost_ties_and_deduplicates():
    nodes = [NS(idx=i, frontier_idxs=cells) for i, cells in enumerate([
        [(0, 0)], [(0, 0)], [(1, 0), (1, 1)], [(2, k) for k in range(100)]])]
    assert select_coverage_frontier(nodes, np.array([1.1, 1., 1.2, 2.]), .25) == 2
    assert select_coverage_frontier(nodes, np.array([1.1, 1., 1.3, 2.]), .25) == 1


def test_feasible_view_uses_shortest_travel_before_visual_score():
    state = ViewControlState()
    state.set_proposals([
        {'pose':[0,0,0], 'score':.99, 'distance_to_target':1.2},
        {'pose':[1,0,0], 'score':.9, 'stop_feasible':True,'travel_distance':3},
        {'pose':[2,0,0], 'score':.8, 'stop_feasible':True,'travel_distance':1}])
    assert state.next_proposal().pose == [2,0,0]


def test_instance_label_reuse_rechecks_better_or_conflicting_evidence():
    record = LabelEvidence('chair', 'chair', 100, 10, 1, 20)
    args = dict(detector_label='chair', mapped_label='chair', area=100, sharpness=10, task_revision=1)
    assert reusable_label(record, **args)
    for key, value in [('area',150),('sharpness',15),('detector_label','table'),('mapped_label','table'),('task_revision',2)]:
        assert not reusable_label(record, **{**args,key:value})


def test_instance_overlap_requires_unique_bidirectional_match():
    pts = np.array([[x,0,0] for x in np.arange(0,1,.05)])
    def obj(points, uid):
        return NS(uid=uid, alive=True, pcd=NS(point=NS(positions=NS(cpu=lambda:NS(numpy=lambda:points)))))
    a = obj(pts,'a')
    assert matching_instance(pts, [a]).uid == 'a'
    assert matching_instance(pts, [a,obj(pts,'b')]) is None
    assert matching_instance(pts, [obj(pts+2,'b')]) is None


@pytest.mark.parametrize("empty_first_frame", [False, True])
def test_partial_scan_resumes_missing_headings_and_keeps_sensor_alignment(monkeypatch, tmp_path, empty_first_frame):
    import ast
    from pathlib import Path
    cloud = NS(voxel_down_sample=lambda resolution: object())
    def empty_cloud(points, colors, device):
        assert points.shape == colors.shape == (0, 3)
        return cloud
    path = Path(__file__).parents[1] / 'navigation/observation_pipeline.py'
    functions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
                 and node.name in {'reset_panoramic_buffers', 'collect_adaptive_observations'}]
    namespace = {'np': np, 'Any': object, 'gpu_merge_pointcloud': lambda *a: cloud,
                 'gpu_pointcloud_from_array': empty_cloud}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    module = NS(**namespace)
    mapper = NS(current_position=np.zeros(3), pcd_resolution=.1, current_navigable_pcd=cloud, current_pcd=cloud,
                objects=[NS(tag='chair')], _is_target_tag=lambda tag:tag=='chair')
    agent=NS(mapper=mapper, task=NS(spec=NS(revision=1)), semantic_scan_headings={}, episode_steps=1,
             episode_samples=1, save_dir=str(tmp_path), env=NS(episode_over=False))
    (tmp_path/'episode-0').mkdir()
    mapper.pcd_device="test"
    heading=[0];batches=[]
    def update():
        mapper.current_navigable_pcd = mapper.current_pcd = None if empty_first_frame and heading[0]==0 else cloud
        angle=heading[0]*np.pi/6
        mapper.current_rotation=np.array([[0,0,-np.cos(angle)],[0,0,-np.sin(angle)],[0,1,0]])
        agent.rotation=NS(w=np.cos(angle/2))
        agent.rgb_trajectory=[heading[0]];agent.temporary_depths=[heading[0]+100]
    def step(action):
        assert action==3
        heading[0]=(heading[0]+1)%12;agent.episode_steps+=1
    agent.execution=NS(step=step);agent.update_trajectory=update
    def detect(images,depths,positions,rotations,partial):
        assert partial and depths==[x+100 for x in images]
        batches.append(list(images));mapper.current_obj_indices=[0]
    agent.rotate_segmentation=detect;update()
    module.collect_adaptive_observations(agent)
    assert batches==[[0,1,2,3]] and not agent.scan_complete
    module.collect_adaptive_observations(agent)
    assert batches[-1]==[4,5,6,7]
    module.collect_adaptive_observations(agent)
    assert batches[-1]==[8,9,10,11] and agent.scan_complete
    assert agent.episode_steps==12  # 初始一步 + 11 次转向；没有恢复朝向的第 12 次。
    module.collect_adaptive_observations(agent)
    assert len(batches)==3 and not agent.temporary_pcd
