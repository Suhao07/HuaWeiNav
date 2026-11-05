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
