import numpy as np

from navigation.panoramic_detection import filter_center_panel, triplet_indices


def test_triplet_indices_wrap_circular_panorama():
    """Panorama triplets should wrap at both ends of the sweep."""

    assert triplet_indices(0, total_views=12) == (11, 0, 1)
    assert triplet_indices(11, total_views=12) == (10, 11, 0)
    assert triplet_indices(5, total_views=12) == (4, 5, 6)


def test_filter_center_panel_keeps_only_middle_view_boxes():
    """Only detections centered in the middle stitched panel should survive."""

    classes = np.array(["left", "middle", "right", "middle_edge"])
    boxes = np.array([
        [10, 10, 30, 30],
        [650, 20, 750, 120],
        [1300, 20, 1400, 120],
        [1200, 50, 1270, 110],
    ])
    masks = np.arange(4)
    confidences = np.array([0.1, 0.9, 0.2, 0.8])

    result = filter_center_panel(
        classes,
        boxes,
        masks,
        confidences,
        image_width=640,
    )

    assert result.has_boxes
    assert result.classes.tolist() == ["middle", "middle_edge"]
    assert result.masks.tolist() == [1, 3]
    assert result.confidences.tolist() == [0.9, 0.8]


def test_single_view_object_projection_uses_only_supplied_camera(monkeypatch):
    import sys, importlib.util
    from pathlib import Path
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    import torch
    monkeypatch.setitem(sys.modules, 'cv_utils.stitch', NS(
        image_stitch_and_crop=Mock(side_effect=AssertionError('must not stitch'))))
    project = Mock(return_value=(np.array([2, 8]), np.array([3, 9]), np.array([1., 1.])))
    monkeypatch.setitem(sys.modules, 'mapping_utils.geometry', NS(project_to_camera=project))
    monkeypatch.setitem(sys.modules, 'mapping_utils.transform', NS(habitat_rotation=Mock()))
    spec = importlib.util.spec_from_file_location('projector_single_test', Path(__file__).parents[1]/'navigation/object_view_projector.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    image, rotation, position, intrinsic = np.zeros((20, 30, 3)), np.eye(3), np.array([1., 2., 3.]), np.eye(3)
    cloud = NS(point=NS(positions=NS(cpu=lambda: NS(numpy=lambda: np.array([[1., 2., 1.]])))))
    obj = NS(pcd=cloud, pcd_all=cloud)
    agent = NS(mapper=NS(current_position=position, camera_intrinsic=intrinsic))
    evidence = module.project_object_view(agent, obj, [image], [rotation], torch, partial=True)
    assert evidence.image is image and evidence.bbox_xyxy.tolist() == [0, 0, 11, 12]
    assert evidence.bbox_real_xyxy.tolist() == [2, 3, 8, 9]
    assert project.call_count == 2
    for call in project.call_args_list:
        assert call.args[0] is cloud and call.args[1] is intrinsic
        assert call.args[2] is position and call.args[3] is rotation
    project.return_value = ([], [], [])
    assert module.project_object_view(agent, obj, [image], [rotation], torch, partial=True) is None
