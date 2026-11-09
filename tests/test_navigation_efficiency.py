"""Behavioral checks for constrained views, coverage ranking and evidence reuse."""
from types import SimpleNamespace as NS
import pytest
import numpy as np
from navigation_core.view_geometry import interpolate_polyline
from navigation.semantic_label_cache import LabelEvidence, reusable_label, matching_instance
from planning.exploration_policy import select_coverage_frontier
from instruction_adapter.view_control import ViewControlState


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
def test_adaptive_scan_resumes_missing_headings_and_keeps_sensor_alignment(tmp_path, empty_first_frame):
    """Four-view batches retain aligned evidence without repeating covered headings."""
    import json

    ns = adaptive_observation_functions()
    cloud = ns['cloud']
    collect = ns['collect_adaptive_observations']
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
        assert len(images) == len(positions) == len(rotations) == 4
        for image, position, rotation in zip(images, positions, rotations):
            np.testing.assert_array_equal(position, mapper.current_position)
            assert position is not mapper.current_position and rotation is not mapper.current_rotation
            angle = image * np.pi / 6
            np.testing.assert_allclose(rotation[:, 2], [-np.cos(angle), -np.sin(angle), 0])
        batches.append(list(images));mapper.current_obj_indices=[0]
    agent.rotate_segmentation=detect;update()
    collect(agent)
    assert batches==[[0,1,2,3]] and not agent.scan_complete
    collect(agent)
    assert batches[-1]==[4,5,6,7]
    collect(agent)
    assert batches[-1]==[8,9,10,11] and agent.scan_complete
    assert agent.episode_steps==12  # 初始一步 + 11 次转向；没有恢复朝向的第 12 次。
    collect(agent)
    assert len(batches)==3 and not agent.temporary_pcd
    assert ns['gpu_pointcloud_from_array'].call_count == int(empty_first_frame)
    if empty_first_frame:
        points, colors, device = ns['gpu_pointcloud_from_array'].call_args.args
        assert points.shape == colors.shape == (0, 3) and device == 'test'
    records = [json.loads(line) for line in
               (tmp_path / 'episode-0/observation_cycles.jsonl').read_text().splitlines()]
    assert records == [dict(mode='multi', cycle_complete=True, start_step=start, end_step=end,
                           views=4, covered_headings=list(range(covered)), complete=covered == 12)
                       for start, end, covered in [(1, 4, 4), (4, 8, 8), (8, 12, 12)]]
    mapper.current_position = np.array([1., 0., 0.])
    collect(agent)
    assert batches[-1] == [11, 0, 1, 2] and not agent.scan_complete
    assert agent.semantic_scan_headings[(1, 0., 0., 0.)] == set(range(12))
    assert agent.semantic_scan_headings[(1, 1., 0., 0.)] == {11, 0, 1, 2}


def test_observation_pipeline_exposes_only_multiview_collectors():
    """Removed single-view collection cannot remain an internal planner dependency."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'navigation/observation_pipeline.py').read_text())
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {'collect_adaptive_observations', 'collect_panoramic_observations'} <= names
    assert 'collect_single_observation' not in names
    planner = ast.parse((root / 'navigation/planner_loop.py').read_text())
    assert not any(isinstance(node, ast.alias) and node.name == 'collect_single_observation'
                   for node in ast.walk(planner))
    assert not any(isinstance(node, ast.Attribute) and node.attr == 'observation_mode'
                   or isinstance(node, ast.Constant) and node.value == 'observation_mode'
                   for node in ast.walk(planner))


@pytest.mark.parametrize('adaptive', [None, False, True])
@pytest.mark.parametrize('patrol', [False, True])
def test_observation_dispatch_preserves_multiview_and_patrol(monkeypatch, adaptive, patrol):
    """Adaptive configuration takes precedence; internal/patrol defaults stay panoramic."""
    import sys
    from unittest.mock import Mock
    from navigation.planner_loop import run_observation_mapping_cycle
    calls = []
    monkeypatch.setitem(sys.modules, 'navigation.observation_pipeline', NS(
        collect_adaptive_observations=lambda agent: calls.append('adaptive')))
    mapper = NS(update_obj=Mock(), current_obj_indices=[2], current_node_idx=8,
                traversable_pcd_all=NS(is_empty=lambda: False), process_nav_pcd=NS(is_empty=lambda: False))
    agent = NS(patrol=patrol, mapper=mapper, rotate_panoramic=lambda: calls.append('panorama'),
               temporary_pcd=[object()], env=NS(episode_over=False), episode_steps=7, current_node_idx=0,
               _merge_temporary_pointclouds=Mock(return_value='merged'), _save_obs_pointcloud=Mock(),
               _log_mapper_state_before_after_get_nodes=Mock())
    if adaptive is not None:
        agent.adaptive_observation = adaptive
    result = run_observation_mapping_cycle(agent, node=None, episode_idx=0)
    assert calls == ['adaptive' if adaptive else 'panorama']
    assert result.step == 7 and result.geometry_available and not result.episode_over
    agent._merge_temporary_pointclouds.assert_called_once_with()
    agent._save_obs_pointcloud.assert_called_once_with('merged', idx=0, step=7)
    assert mapper.current_node_idx == agent.current_node_idx
    mapper.update_obj.assert_called_once_with(0, [2])
    agent._log_mapper_state_before_after_get_nodes.assert_called_once_with(7, None, 0)


def adaptive_observation_functions():
    """Isolate the real adaptive collector with inert point-cloud dependencies."""
    import __future__
    import ast
    from pathlib import Path
    from unittest.mock import Mock

    cloud = NS(voxel_down_sample=Mock())
    cloud.voxel_down_sample.return_value = cloud
    namespace = dict(np=np, cloud=cloud, gpu_merge_pointcloud=Mock(return_value=cloud),
                     gpu_pointcloud_from_array=Mock(return_value=cloud))
    path = Path(__file__).resolve().parents[1] / 'navigation/observation_pipeline.py'
    tree = ast.parse(path.read_text())
    names = {'reset_panoramic_buffers', 'collect_adaptive_observations'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@pytest.mark.parametrize('episode_over', [False, True])
def test_completed_adaptive_coverage_skips_duplicate_map_updates(monkeypatch, episode_over):
    """A scan with no missing headings must not merge stale observation buffers."""
    import sys
    from unittest.mock import Mock
    from navigation.planner_loop import run_observation_mapping_cycle

    def collect(agent):
        agent.temporary_pcd = []
        agent.scan_complete = True

    monkeypatch.setitem(sys.modules, 'navigation.observation_pipeline', NS(collect_adaptive_observations=collect))
    mapper = NS(update_obj=Mock())
    agent = NS(adaptive_observation=True, mapper=mapper, current_node_idx=2,
               temporary_pcd=[object()], env=NS(episode_over=episode_over), episode_steps=12,
               _merge_temporary_pointclouds=Mock(), _save_obs_pointcloud=Mock(),
               _log_mapper_state_before_after_get_nodes=Mock())
    result = run_observation_mapping_cycle(agent, node=None, episode_idx=0)
    assert result.step == 12 and result.episode_over is episode_over
    assert agent.scan_complete and not agent.temporary_pcd
    agent._merge_temporary_pointclouds.assert_not_called()
    agent._save_obs_pointcloud.assert_not_called()
    agent._log_mapper_state_before_after_get_nodes.assert_not_called()
    mapper.update_obj.assert_not_called()


@pytest.mark.parametrize('adaptive', [False, True])
def test_multiview_cycle_stops_mapping_at_episode_boundary(monkeypatch, adaptive):
    """An exhausted action budget cannot trigger graph mutation after collection."""
    import sys
    from unittest.mock import Mock
    from navigation.planner_loop import run_observation_mapping_cycle

    def collect(agent):
        agent.env.episode_over = True
        agent.episode_steps += 1

    monkeypatch.setitem(sys.modules, 'navigation.observation_pipeline', NS(collect_adaptive_observations=collect))
    mapper = NS(update_obj=Mock())
    agent = NS(adaptive_observation=adaptive, mapper=mapper, current_node_idx=2,
               temporary_pcd=[object()], env=NS(episode_over=False), episode_steps=12,
               _merge_temporary_pointclouds=Mock(), _save_obs_pointcloud=Mock(),
               _log_mapper_state_before_after_get_nodes=Mock())
    agent.rotate_panoramic = lambda: collect(agent)
    result = run_observation_mapping_cycle(agent, node=None, episode_idx=0)
    assert result.step == 13 and result.episode_over
    agent._merge_temporary_pointclouds.assert_not_called()
    agent._save_obs_pointcloud.assert_not_called()
    agent._log_mapper_state_before_after_get_nodes.assert_not_called()
    mapper.update_obj.assert_not_called()
