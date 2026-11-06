"""Contracts for current-instance distance evidence, independent of model accuracy."""
import numpy as np
from navigation_core.surface_evidence import measure_instance_surface


def sample():
    projected = np.array([[2, 2, 2.], [3, 2, 2.], [2, 3, 2.], [7, 7, .1]])
    depth = np.full((10, 10), 2.)
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 1:5, 1:5] = True
    k = np.array([[10., 0, 4.5], [0, 10., 4.5], [0, 0, 1.]])
    # Mapper plane is camera x/-z; elevation is camera y.
    rotation = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    return projected, depth, mask, k, rotation, np.zeros(3), np.zeros(3)


def test_distance_uses_current_region_and_raw_surface():
    args = sample()
    result, _, indices = measure_instance_surface(*args)
    assert result['surface_match']
    assert result['distance_to_object'] >= 2.
    assert indices.tolist() == [0, 1, 2]
    # A predicted point in front of the surface cannot supply a near distance.
    args[0][:3, 2] = .8
    result, _, indices = measure_instance_surface(*args)
    assert not result['surface_match']
    assert result['distance_to_object'] is None


def test_occluded_single_and_ambiguous_support_require_observation():
    args = sample()
    args[0][:2, 2] = 3.
    assert not measure_instance_surface(*args)[0]['surface_match']
    args = list(sample())
    args[2] = np.repeat(args[2], 2, axis=0)
    assert measure_instance_surface(*args)[0]['surface_match']
    # 同一实例的重复mask不会制造歧义；两个等量独立区域没有唯一多数。
    args[0] = np.array([[2,2,2.], [3,2,2.], [2,3,2.], [6,6,2.], [7,6,2.], [6,7,2.]])
    args[2][1] = False
    args[2][1, 5:9, 5:9] = True
    assert measure_instance_surface(*args)[0]['surface_status'] == 'ambiguous_instance'


def test_mapper_preprocessing_preserves_sensor_depth_and_mapping_values():
    """Run the production depth assignments with the actual preprocessing functions."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from mapping_utils.preprocess import ori_preprocess_depth, preprocess_depth

    source = ast.parse((Path(__file__).resolve().parents[1]/'mapper_with_process_obs.py').read_text())
    mapper = next(n for n in source.body if isinstance(n, ast.ClassDef))
    update = next(n for n in mapper.body if isinstance(n, ast.FunctionDef) and n.name == 'update')
    names = {'self.original_depth', 'mapping_depth', 'self.ori_current_depth', 'self.current_depth'}
    assignments = [n for n in update.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in names]
    original = np.array([[0., .4], [1., 2.], [3., 6.], [2., 3.]], dtype=np.float32)[..., None]
    raw = original.copy()
    state = SimpleNamespace(lower_img=np.full((4, 2), .51), upper_img=np.full((4, 2), 4.9))
    old = raw.copy()
    expected_ori = ori_preprocess_depth(old)
    expected_current = preprocess_depth(old, state.lower_img, state.upper_img)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), 'mapper_depth_contract', 'exec'),
         dict(self=state, depth=raw, ori_preprocess_depth=ori_preprocess_depth, preprocess_depth=preprocess_depth))
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(state.ori_current_depth, expected_ori)
    np.testing.assert_array_equal(state.current_depth, expected_current)
