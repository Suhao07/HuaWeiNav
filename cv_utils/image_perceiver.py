import numpy as np
import torch

from constants import *

from .sam import MMDINO_Grounded_SAM


class ImagePerceiver:

    def __init__(self, classes, box_threshold=0.3, text_threshold=0.3, device="cuda"):
        self.classes = classes
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def segment(self, img):
        raise NotImplementedError("Subclasses must implement segment().")

    def perceive(self,
                 img,
                 area_threshold=100,
                 target=None,
                 target_list=[],
                 save_dir="",
                 episode_idx=-1,
                 episode_step=-1):
        """Filter detector masks by area while preserving empty detections.

        ``target=None`` keeps vocabulary detections and an empty target group.
        Detector contract violations propagate to the collection driver.
        """
        B_boxes, B_confidences, B_classes, B_masks, C_boxes, C_confidences, C_masks = \
            self.segment(img, target, target_list, save_dir, episode_idx, episode_step)

        # 空检测由零长度张量表示；维度或索引错误直接向上传播，
        # 保留失败帧的异常栈，避免将感知失败记成“当前区域没有对象”。
        mask_area = torch.tensor([mask.sum() for mask in B_masks])
        areas = torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in B_boxes
                             ])
        too_large_flag = (areas < 0.8 * img.shape[0] * img.shape[1])
        B_flag = (mask_area > area_threshold) & too_large_flag
        B_classes = B_classes[B_flag.cpu().numpy()]

        C_classes = np.array([target] * C_boxes.shape[0])
        mask_area = torch.tensor([mask.sum() for mask in C_masks])
        areas = torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in C_boxes
                             ])
        too_large_flag = (areas < 0.8 * img.shape[0] * img.shape[1])
        C_flag = (mask_area > area_threshold) & too_large_flag
        C_classes = C_classes[C_flag.cpu().numpy()]

        return B_classes, B_boxes[B_flag], B_masks[B_flag], B_confidences[B_flag], \
                C_classes, C_boxes[C_flag], C_masks[C_flag], C_confidences[C_flag]


class MMDINOSAM_Perceiver(ImagePerceiver):

    def __init__(self,
                 classes,
                 box_threshold=0.3,
                 text_threshold=0.3,
                 no_gpt_seg=False,
                 device="cuda"):
        super().__init__(classes, box_threshold, text_threshold, device)

        self.sam = MMDINO_Grounded_SAM(classes, box_threshold, text_threshold, no_gpt_seg,
                                       device)

    def segment(self,
                img,
                target=None,
                target_list=[],
                save_dir="",
                episode_idx=-1,
                episode_step=-1):
        return self.sam.segment(img, target, target_list, save_dir, episode_idx,
                                episode_step)
