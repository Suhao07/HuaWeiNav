"""Run in the maintained dualmap environment with upstream on PYTHONPATH."""
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
import numpy as np
import torch
DUALMAP_AVAILABLE = find_spec("dualmap") is not None
if DUALMAP_AVAILABLE:
    from utils.object_detector import Detector


@unittest.skipUnless(DUALMAP_AVAILABLE, "Requires the official DualMap checkout on PYTHONPATH")
class ColorBoundaryTest(unittest.TestCase):
    def test_rgb_stays_rgb_until_ultralytics_boundary(self):
        calls = []
        boxes = SimpleNamespace(conf=torch.tensor([.9]), cls=torch.tensor([0.]),
                                xyxy=torch.tensor([[0., 0., 2., 2.]]))
        result = SimpleNamespace(boxes=boxes, masks=SimpleNamespace(data=torch.ones((1, 2, 2))))
        def predict(image, **kwargs):
            calls.append(image.copy())
            return [result]
        detector = Detector.__new__(Detector)
        detector.yolo = SimpleNamespace(predict=predict)
        detector.sam = SimpleNamespace(predict=predict)
        detector.fastsam = predict
        detector.obj_classes = SimpleNamespace(get_classes_arr=lambda: ['chair'])
        detector.cfg = SimpleNamespace(fastsam_confidence=.2)
        detector.unknown_class_id = 1
        detector.timing_results = {}
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 251
        detector.process_yolo_and_sam(rgb)
        detector.process_fastsam_results(rgb)
        self.assertEqual(len(calls), 3)
        for image in calls:
            np.testing.assert_array_equal(image[0, 0], [0, 0, 251])
        np.testing.assert_array_equal(rgb[0, 0], [251, 0, 0])


if __name__ == '__main__':
    unittest.main()
