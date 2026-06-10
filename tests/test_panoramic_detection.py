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
