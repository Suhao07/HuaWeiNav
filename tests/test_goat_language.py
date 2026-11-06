"""Preserve GOAT provenance and the independent-task denominator."""
import pytest
from benchmark.goat_language import language_tasks, freeze_subset


def test_language_subset_preserves_targets_and_removes_repeated_subtasks():
    goal=dict(object_id='chair_1',object_category='chair',position=[1,0,0],
              view_points=[],lang_desc='the red chair next to the table')
    parent=dict(episode_id=7,scene_id='hm3d/val//00802-wcojb4TFT35/wcojb4TFT35.basis.glb',
                start_position=[0,0,0],start_rotation=[0,0,0,1],
                tasks=[['chair','object',None],['chair','description','chair_1'],
                       ['chair','description','chair_1']])
    tasks=language_tasks({'goals':{'chair':[goal]},'episodes':[parent]},split='val_unseen',source_file='source.gz')
    assert len(tasks)==2
    assert tasks[0]['info']['source_subtask_index']==1
    assert tasks[0]['info']['instruction']==goal['lang_desc']
    assert tasks[0]['goals'][0]['object_id']=='chair_1'
    assert tasks[0]['start_position']==parent['start_position']
    selected,unique=freeze_subset(tasks,1,20261105)
    assert unique==1
    assert selected[0]['episode_id'].endswith(':7:1')
    with pytest.raises(ValueError,match='available 1'):
        freeze_subset(tasks,2,20261105)


@pytest.mark.parametrize('invalid', [None, 'identity', 'instruction', 'nonfinite', 'unfinished'])
def test_language_launcher_binds_frozen_task_and_score(tmp_path, invalid):
    import csv
    import gzip
    import json
    from scripts.launch_goat_episode import language_inputs, language_score

    row = dict(episode_id='val:scene:3:2', scene_id='scene/a.glb',
               start_position=[0, 0, 0], start_rotation=[0, 0, 0, 1],
               object_category='chair', info=dict(instruction='the red chair',
               protocol='goat_language_single_parent_start_v1'))
    source = tmp_path / 'tasks.json.gz'
    with gzip.open(source, 'wt') as stream:
        json.dump({'episodes': [row]}, stream)
    argv, identity, instruction = language_inputs(source, 0, tmp_path, tmp_path, 500, 60)
    assert argv[argv.index('--benchmark') + 1] == 'goat_language'
    assert argv[argv.index('--eval_episodes') + 1] == '1'
    assert instruction == row['info']['instruction']
    assert json.loads((tmp_path / 'runtime_episodes.json').read_text()) == {'runtime_episodes': [identity]}
    folder = tmp_path / 'navigation/task'
    episode = folder / 'episode-0'
    (episode / 'harness').mkdir(parents=True)
    actual = dict(identity)
    if invalid == 'identity':
        actual['episode_id'] = 'wrong'
    (episode / 'episode_identity.json').write_text(json.dumps(actual))
    (episode / 'harness/termination.json').write_text(json.dumps({'reason': 'episode_limit'}))
    metrics = dict(Episode='0', object_goal=instruction, benchmark='goat_language',
                   benchmark_success_distance='0.25', success='0', spl='0', distance_to_goal='2', End='1')
    if invalid == 'instruction':
        metrics['object_goal'] = 'the blue chair'
    if invalid == 'nonfinite':
        metrics['spl'] = 'nan'
    if invalid == 'unfinished':
        metrics['End'] = '0'
    with (folder / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    if invalid:
        with pytest.raises(ValueError):
            language_score(tmp_path, 0, identity, instruction)
    else:
        score = language_score(tmp_path, 0, identity, instruction)
        assert score['aggregate_ready'] and score['metrics'] == {'SR': 0, 'SPL': 0, 'distance_to_goal': 2}
        assert score['protocol'] == 'goat_language_single_parent_start_v1'
