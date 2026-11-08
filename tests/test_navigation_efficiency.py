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
def test_partial_scan_resumes_missing_headings_and_keeps_sensor_alignment(monkeypatch, tmp_path, empty_first_frame):
    import sys, importlib.util
    from pathlib import Path
    monkeypatch.setitem(sys.modules, 'open3d', NS())
    monkeypatch.setitem(sys.modules, 'quaternion', NS())
    monkeypatch.setitem(sys.modules, 'loguru', NS(logger=NS()))
    cloud = NS(voxel_down_sample=lambda resolution: object())
    def empty_cloud(points, colors, device):
        assert points.shape == colors.shape == (0, 3)
        return cloud
    monkeypatch.setitem(sys.modules, 'mapping_utils.geometry', NS(gpu_merge_pointcloud=lambda *a:cloud, gpu_pointcloud_from_array=empty_cloud))
    spec = importlib.util.spec_from_file_location('scan_under_test', Path(__file__).parents[1]/'navigation/observation_pipeline.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
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


@pytest.mark.parametrize("empty_cloud", [False, True])
@pytest.mark.parametrize("do_seg", [False, True])
def test_single_observation_keeps_alignment_and_history_without_motion(monkeypatch, tmp_path, empty_cloud, do_seg):
    import sys, importlib.util, json
    from pathlib import Path
    from unittest.mock import Mock
    monkeypatch.setitem(sys.modules, 'open3d', NS())
    monkeypatch.setitem(sys.modules, 'quaternion', NS())
    monkeypatch.setitem(sys.modules, 'loguru', NS(logger=NS()))
    cloud = NS(voxel_down_sample=Mock(return_value='current_cloud'))
    empty = Mock(return_value='empty_cloud')
    merge = Mock(return_value=cloud)
    monkeypatch.setitem(sys.modules, 'mapping_utils.geometry', NS(
        gpu_merge_pointcloud=merge, gpu_pointcloud_from_array=empty))
    spec = importlib.util.spec_from_file_location('single_scan_under_test', Path(__file__).parents[1]/'navigation/observation_pipeline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    position, rotation = np.array([1., 2., 3.]), np.eye(3)
    rgb, depth = np.zeros((8, 10, 3)), np.full((8, 10, 1), 2.)
    objects, covered = [object()], {(1, 1, 2, 3): {0}}
    mapper = NS(current_position=position, current_rotation=rotation, pcd_resolution=.1, pcd_device='CPU:0',
                current_navigable_pcd=None if empty_cloud else cloud,
                current_pcd=None if empty_cloud else cloud, objects=objects,
                process_nav_pcd=NS(is_empty=lambda: False), process_obs_pcd=NS(is_empty=lambda: False))
    agent = NS(mapper=mapper, rgb_trajectory=[rgb], temporary_depths=[depth], rotation=NS(w=1.),
               episode_steps=7, episode_samples=1, save_dir=str(tmp_path), do_seg=do_seg,
               execution=NS(step=Mock()), update_trajectory=Mock(), rotate_segmentation=Mock(),
               semantic_scan_headings=covered, scan_complete=True)
    (tmp_path/'episode-0').mkdir()
    module.collect_single_observation(agent)
    agent.execution.step.assert_not_called()
    agent.update_trajectory.assert_not_called()
    assert agent.episode_steps == 7 and not agent.scan_complete
    assert agent.mapper.objects is objects and agent.semantic_scan_headings is covered
    assert covered == {(1, 1, 2, 3): {0}}
    assert agent.temporary_pcd == ['empty_cloud' if empty_cloud else 'current_cloud']
    assert empty.call_count == int(empty_cloud) and merge.call_count == int(not empty_cloud)
    assert agent.rotate_segmentation.call_count == int(do_seg)
    if do_seg:
        images, depths, positions, rotations = agent.rotate_segmentation.call_args.args
        assert images[0] is rgb and depths[0] is depth
        assert np.array_equal(positions[0], position) and np.array_equal(rotations[0], rotation)
        assert positions[0] is not position and rotations[0] is not rotation
        assert agent.rotate_segmentation.call_args.kwargs == {'partial': True, 'single_view': True}
    record = json.loads((tmp_path/'episode-0/observation_cycles.jsonl').read_text())
    assert record == dict(mode='single', start_step=7, end_step=7, views=1, cycle_complete=True, complete=False)


@pytest.mark.parametrize('mode,adaptive,patrol,expected', [
    ('single', True, False, 'single'), ('single', False, False, 'single'),
    ('multi', True, False, 'adaptive'), ('multi', False, False, 'panorama'),
    ('single', False, True, 'panorama')])
def test_observation_dispatch_preserves_multiview_and_patrol(monkeypatch, mode, adaptive, patrol, expected):
    import sys
    from unittest.mock import Mock
    from navigation.planner_loop import run_observation_mapping_cycle
    calls = []
    monkeypatch.setitem(sys.modules, 'navigation.observation_pipeline', NS(
        collect_single_observation=lambda agent, **kwargs: calls.append('single'),
        collect_adaptive_observations=lambda agent: calls.append('adaptive')))
    mapper = NS(update_obj=Mock(), current_obj_indices=[2], current_node_idx=8,
                traversable_pcd_all=NS(is_empty=lambda: False), process_nav_pcd=NS(is_empty=lambda: False))
    agent = NS(observation_mode=mode, adaptive_observation=adaptive, patrol=patrol,
               mapper=mapper, rotate_panoramic=lambda: calls.append('panorama'),
               temporary_pcd=[object()], env=NS(episode_over=False), episode_steps=7, current_node_idx=0,
               _merge_temporary_pointclouds=Mock(return_value='merged'), _save_obs_pointcloud=Mock(),
               _log_mapper_state_before_after_get_nodes=Mock())
    result = run_observation_mapping_cycle(agent, node=None, episode_idx=0)
    assert calls == [expected] and result.step == 7 and result.geometry_available
    assert mapper.current_node_idx == agent.current_node_idx
    mapper.update_obj.assert_called_once_with(0, [2])
    agent._log_mapper_state_before_after_get_nodes.assert_called_once_with(7, None, 0)


def pointcloud_observation_functions():
    import __future__
    import ast
    import math
    from pathlib import Path
    from unittest.mock import Mock
    from mapping_utils.preprocess import ori_preprocess_depth, preprocess_depth, preprocess_image
    o3d = pytest.importorskip('open3d')
    namespace = dict(np=np, o3d=o3d, math=math, logger=Mock(),
                     ori_preprocess_depth=ori_preprocess_depth,
                     preprocess_depth=preprocess_depth, preprocess_image=preprocess_image)
    for filename, names in (
        ('mapping_utils/geometry.py', {'gpu_merge_pointcloud', 'gpu_pointcloud_from_array',
                                       'get_pointcloud_from_depth', 'translate_to_world'}),
        ('navigation/observation_pipeline.py', {'reset_panoramic_buffers', 'collect_single_observation',
                                               'merge_temporary_pointclouds'}),
        ('mapper_with_process_obs.py', {'update', 'get_nodes', 'keep_the_max_connect_component',
                                        'get_closest_disances_and_points'}),
    ):
        path = Path(__file__).parents[1] / filename
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        nodes += [n for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@pytest.mark.parametrize('geometry_source', ['accumulated', 'current', 'node'])
def test_single_cycle_retains_next_frontier_after_arrival(monkeypatch, tmp_path, geometry_source):
    import sys
    from unittest.mock import Mock
    from navigation.planner_loop import run_observation_mapping_cycle
    ns = pointcloud_observation_functions()
    monkeypatch.setitem(sys.modules, 'loguru', NS(logger=Mock()))
    from mapping import topology_graph as topology
    from mapping.node_candidate_builder import add_nodes_from_candidates
    from mapping_utils.representation import our_Node, NodeState
    device = ns['o3d'].core.Device('CPU:0')
    cloud = ns['gpu_pointcloud_from_array'](
        np.array([[2.8, 0, -.8], [2.8, .1, -.8]]), np.ones((2, 3)), device)
    nodes = [our_Node(None, None, cloud, np.array([x, 0., 0.]), None, i,
                      bool(i), np.array([[10+i, 10]])) for i, x in enumerate([0., 2.])]
    nodes[0].state = NodeState.EXPLORED
    mapper = NS(current_node_idx=0, current_position=np.array([2., 0., 0.]),
                pcd_device=device, pcd_resolution=.05, current_navigable_pcd=cloud, current_pcd=cloud,
                traversable_pcd_all=cloud, process_nav_pcd=cloud, process_obs_pcd=cloud, nodes=nodes, node_cnt=2,
                initial_position=np.zeros(3), nodes_pos_to_idx={tuple(n.position): n.idx for n in nodes},
                neighbors=[[1], [0]], frontiers_considered=np.zeros((50, 50)),
                env=NS(sim=NS(geodesic_distance=lambda a, b: float(np.linalg.norm(np.asarray(a)-b)))),
                normalize_angle=lambda a: a % (2*np.pi))
    if geometry_source != 'accumulated':
        mapper.process_nav_pcd = ns['o3d'].t.geometry.PointCloud(device)
        mapper.process_obs_pcd = ns['o3d'].t.geometry.PointCloud(device)
        mapper.floor_height = -.8
    if geometry_source == 'node':
        mapper.current_navigable_pcd = mapper.current_pcd = None
    mapper.add_node = lambda *args: topology.add_node(mapper, *args)
    mapper.add_edge = lambda *args: topology.add_edge(mapper, *args)
    mapper.update_obj = lambda *args: topology.update_obj(mapper, *args)
    agent = NS(mapper=mapper, current_node_idx=1, observation_mode='single', patrol=False,
               rotation=NS(w=1.), do_seg=False, save_dir=str(tmp_path), episode_samples=1,
               episode_steps=10, env=NS(episode_over=False), execution=NS(step=Mock()),
               update_trajectory=Mock(), _save_obs_pointcloud=Mock())
    (tmp_path / 'episode-0').mkdir()
    monkeypatch.setitem(sys.modules, 'navigation.observation_pipeline', NS(
        collect_single_observation=ns['collect_single_observation']))
    agent._merge_temporary_pointclouds = lambda: ns['merge_temporary_pointclouds'](agent)
    agent._log_mapper_state_before_after_get_nodes = lambda *args: add_nodes_from_candidates(
        mapper, valid_centers=[(np.array([2.8, 0., 0.]), True, np.array([[12, 10]]))],
        current_node_pcd=agent.current_pcd,
        angles_current_node=np.zeros(agent.current_pcd.point.positions.shape[0]))
    result = run_observation_mapping_cycle(agent, node=nodes[1], episode_idx=0)
    topology.visit_node(mapper, 1)
    assert result.geometry_available and result.step == 10
    assert mapper.current_node_idx == 1
    assert [n.idx for n in mapper.nodes if n.has_frontier] == [2]
    assert (1, 2) in topology.get_edges(mapper) and (0, 2) not in topology.get_edges(mapper)
    assert topology.find_closest_unexplored_node(mapper).idx == 2
    agent.execution.step.assert_not_called()
    agent.update_trajectory.assert_not_called()


@pytest.mark.parametrize('depth_value', [0., .5])
@pytest.mark.parametrize('global_history', [False, True])
def test_single_cycle_rejects_unavailable_geometry_before_node_extraction(
        monkeypatch, tmp_path, depth_value, global_history):
    import sys
    from types import MethodType
    from unittest.mock import Mock
    from navigation.planner_loop import run_observation_mapping_cycle
    ns = pointcloud_observation_functions()
    device = ns['o3d'].core.Device('CPU:0')
    traversable = ns['o3d'].t.geometry.PointCloud(device)
    if global_history:
        traversable = ns['gpu_pointcloud_from_array'](np.ones((12, 3)), np.ones((12, 3)), device)
    mapper = NS(pcd_device=device, pcd_resolution=.05, initial_position=np.zeros(3),
                translation_func=lambda p: p.copy(), rotation_func=lambda r: r.copy(),
                lower_img=.51, upper_img=4.99, trajectory_position=[], traj_on_trav=[],
                traversable_pcd_all=traversable, process_nav_pcd=ns['o3d'].t.geometry.PointCloud(device),
                process_obs_pcd=ns['o3d'].t.geometry.PointCloud(device), update_obj=Mock())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.full((8, 8, 1), depth_value)
    ns['update'](mapper, image, depth, np.zeros(3), np.eye(3), 0, 1, False, 0)
    assert mapper.current_pcd is mapper.current_navigable_pcd is None
    np.testing.assert_array_equal(depth, np.full((8, 8, 1), depth_value))
    mapper.keep_the_max_connect_component = MethodType(ns['keep_the_max_connect_component'], mapper)
    agent = NS(mapper=mapper, current_node_idx=0, observation_mode='single', patrol=False,
               rotation=NS(w=1.), do_seg=False, save_dir=str(tmp_path), episode_samples=1,
               episode_steps=1, env=NS(episode_over=False), execution=NS(step=Mock()),
               update_trajectory=Mock(), _save_obs_pointcloud=Mock())
    (tmp_path / 'episode-0').mkdir()
    monkeypatch.setitem(sys.modules, 'navigation.observation_pipeline', NS(
        collect_single_observation=ns['collect_single_observation']))
    agent._merge_temporary_pointclouds = lambda: ns['merge_temporary_pointclouds'](agent)
    agent._log_mapper_state_before_after_get_nodes = lambda step, node, idx: ns['get_nodes'](
        mapper, agent.temporary_pcd, agent.angles, node, idx, step)
    result = run_observation_mapping_cycle(agent, node=None, episode_idx=0)
    assert not result.geometry_available and not result.episode_over and result.step == 1
    assert mapper.traversable_pcd_all is traversable
    agent.execution.step.assert_not_called()
    agent.update_trajectory.assert_not_called()
    agent._save_obs_pointcloud.assert_not_called()
    mapper.update_obj.assert_not_called()
