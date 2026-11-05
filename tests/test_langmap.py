"""Contracts that differ from category-only ObjectNav evaluation."""
import copy
from benchmark.langmap import convert_scene, terminal_metrics


def test_same_category_tasks_keep_distinct_instances():
    goals = [dict(object_id=f'chair_{i}', object_category='chair', position=[i, 0, 0],
                  view_points=[], annot_unique_concise_description=f'chair number {i}') for i in (1, 2)]
    data = dict(goals=goals, region_annotation={}, episodes_by_object_level=[],
                episodes_by_room_level=[], episodes_by_region_level=[],
                episodes_by_instance_level=[dict(episode_id=i, instance_id=f'chair_{i}',
                    target_object_ids=[f'chair_{i}'], start_position=[0, 0, 0],
                    start_rotation=[0, 0, 0, 1]) for i in (1, 2)])
    original = copy.deepcopy(data)
    episodes, skipped = convert_scene(data, '00802-wcojb4TFT35')
    assert not skipped
    assert data == original
    assert [ep['goals'][0]['object_id'] for ep in episodes] == ['chair_1', 'chair_2']
    assert episodes[0]['episode_id'] == '00802-wcojb4TFT35_instance_1'
    assert episodes[1]['info']['instruction'] == 'Find the chair number 2'
    assert 'target_object_ids' not in episodes[0]['info']


def test_official_terminal_metrics_strict_radius_and_unreachable():
    assert terminal_metrics(4, 0.249, 8)['spl'] == 0.5
    assert terminal_metrics(4, 0.25, 8)['success'] == 0
    assert terminal_metrics(float('inf'), float('inf'), 8)['spl'] == 0
    assert terminal_metrics(0, 0, 0)['spl'] == 1


def test_habitat_loader_preserves_task_identity_and_roundtrip():
    """Exercise the actual Habitat decoder when running in the maintained image."""
    import json
    import importlib.util
    if importlib.util.find_spec('habitat') is None:
        import pytest
        pytest.skip('Run the Habitat roundtrip contract inside the maintained image')
    from benchmark.instance_goal_dataset import InstanceGoalDataset
    rows = []
    for i in (1, 2):
        rows.append(dict(episode_id=f'scene_instance_{i}', scene_id='scene.glb',
            start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1],
            object_category='chair', goals=[dict(object_id=str(i), object_category='chair',
                position=[i, 0, 0], view_points=[dict(iou=0.5,
                    agent_state=dict(position=[i, 0, 1], rotation=[0, 0, 0, 1]))])]))
    payload = dict(episodes=rows, category_to_task_category_id={'chair': 0},
                   category_to_scene_annotation_category_id={'chair': 0})
    dataset = InstanceGoalDataset()
    dataset.from_json(json.dumps(payload))
    restored = InstanceGoalDataset()
    restored.from_json(dataset.to_json())
    assert [ep.episode_id for ep in restored.episodes] == ['scene_instance_1', 'scene_instance_2']
    assert [ep.goals[0].object_id for ep in restored.episodes] == ['1', '2']
    assert [ep.goals[0].view_points[0].agent_state.position for ep in restored.episodes] == [[1, 0, 1], [2, 0, 1]]
