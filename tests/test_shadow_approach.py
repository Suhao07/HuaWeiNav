"""Shadow geometry must not manufacture a goal for an invalid/frontless target."""
import pytest
from scripts.probe_real_robot_shadow import approach_candidate


def test_front_standoff_is_not_object_center():
    assert approach_candidate((0,0,0), 0, (2,0,1), .8) == pytest.approx([1.2,0,0])


@pytest.mark.parametrize('center', [(-2,0,0), (.5,0,0), (float('nan'),0,0)])
def test_invalid_or_already_close_target_is_rejected(center):
    with pytest.raises(ValueError):
        approach_candidate((0,0,0),0,center,.8)


def test_base_extrinsic_composes_translation_and_heading():
    import math
    from scripts.probe_real_robot_shadow import base_pose
    p,q=base_pose([1,2,3],[0,0,math.sqrt(.5),math.sqrt(.5)],[-.2,0,-math.pi/2])
    assert p == pytest.approx([1,1.8,3])
    assert q == pytest.approx([0,0,0,1])
