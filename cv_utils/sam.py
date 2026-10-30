import numpy as np
import torch
from loguru import logger
from mmdet.apis import DetInferencer
from segment_anything import SamPredictor, build_sam
from torchvision.ops import nms

from config_utils import (GROUNDING_DINO_CHECKPOINT, GROUNDING_DINO_CONFIG,
                          SAM_CHECKPOINT)


class MMDINO_Grounded_SAM:

    TARGET_PRESENCE_THRESHOLD = 0.5
    TARGET_BOX_THRESHOLD = 0.25
    NMS_IOU_THRESHOLD = 0.8

    def __init__(self,
                 classes,
                 box_threshold=0.2,
                 text_threshold=0.25,
                 no_gpt_seg=False,
                 device="cuda"):
        self.classes = classes
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.DEVICE = device

        self.sam_model = build_sam(checkpoint=SAM_CHECKPOINT).to(self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam_model)

        inferencer = DetInferencer(model=GROUNDING_DINO_CONFIG,
                                   weights=GROUNDING_DINO_CHECKPOINT,
                                   device=self.DEVICE)
        self.grounding_model = inferencer

    def _run_grounding(self, img, text_prompt):
        # segment 的图像协议为 RGB；MMDetection ndarray loader 接收 BGR，
        # 再由模型的 bgr_to_rgb 预处理还原 RGB。SAM 始终读取原始 RGB。
        result = self.grounding_model(inputs=img[..., ::-1].copy(),
                                      texts=text_prompt,
                                      pred_score_thr=self.TEXT_THRESHOLD,
                                      custom_entities=True)
        return result["predictions"][0]

    def _predict_masks(self, img, boxes):
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes, img.shape[:2]).to(self.DEVICE)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks.cpu()

    def initialize(self, target):
        TEXT_PROMPTs = []
        class_labels = []
        classes = self.classes.copy()
        # classes.remove(target)

        class_num = len(classes)
        if class_num == 0:
            raise ValueError("Detector vocabulary must contain at least one concept")
        split = min(4, class_num)
        step = class_num // split
        start_and_end = [(i * step, (i + 1) * step if i < split - 1 else class_num)
                         for i in range(split)]
        for start, end in start_and_end:
            TEXT_PROMPT = ""
            class_labels.append(np.asarray(classes[start:end]))
            for obj in classes[start:end]:
                TEXT_PROMPT += f"{obj.lower()} . "
            TEXT_PROMPT = TEXT_PROMPT.strip()
            TEXT_PROMPTs.append(TEXT_PROMPT)
        self.TEXT_PROMPTs = TEXT_PROMPTs
        self.class_labels = class_labels
        self.split = split

    def segment(self,
                img,
                target=None,
                target_list=[],
                save_dir="",
                episode_idx=-1,
                episode_step=-1):
        """Segment a uint8 RGB image using vocabulary and optional task target."""
        self.sam_predictor.set_image(img)
        boxes = []
        confidences = []
        labels = []

        # 离线先验建图没有任务目标；只执行共享词表检测与 SAM 分割。
        if target is not None:
            target_predictions = self._run_grounding(img, target)
            target_confidences = torch.tensor(target_predictions["scores"])
            target_boxes = torch.tensor(target_predictions["bboxes"]).reshape(-1, 4)
            target_mask = target_confidences >= self.TARGET_PRESENCE_THRESHOLD

        # do detection for all categories
        for i in range(self.split):
            predictions = self._run_grounding(img, self.TEXT_PROMPTs[i])
            boxes = boxes + predictions["bboxes"]
            confidences = confidences + predictions["scores"]
            labels = labels + self.class_labels[i][predictions["labels"]].tolist()

        boxes = torch.tensor(boxes).reshape(-1, 4)
        confidences = torch.tensor(confidences)

        mask = confidences > self.BOX_THRESHOLD
        B_boxes = boxes[mask]
        B_confidences = confidences[mask]
        B_labels = np.array([labels[i] for i in range(len(labels)) if mask[i]])

        C_boxes = torch.zeros((0, 4))
        C_confidences = torch.zeros(0)

        if target is not None and (target in labels or target_mask.sum() > 0):
            target_mask = target_confidences >= self.TARGET_BOX_THRESHOLD
            if target_mask.sum() > 0:
                C_boxes = target_boxes[target_mask]
                C_confidences = target_confidences[target_mask]

        h, w, _ = img.shape

        if B_boxes.shape[0] == 0:
            return B_boxes, B_confidences, B_labels, torch.zeros(
                (0, h, w)), C_boxes, C_confidences, torch.zeros((0, h, w))

        boxes_xyxy = B_boxes

        # NMS post process
        logger.info(f"Before NMS: {boxes_xyxy.shape[0]} boxes")
        nms_idx = nms(boxes_xyxy, B_confidences, self.NMS_IOU_THRESHOLD)

        boxes_xyxy = boxes_xyxy[nms_idx, :]
        B_boxes = B_boxes[nms_idx, :]
        B_confidences = B_confidences[nms_idx]
        B_labels = B_labels[nms_idx.cpu().numpy()]

        logger.info(f"After NMS: {boxes_xyxy.shape[0]} boxes")

        B_masks = self._predict_masks(img, boxes_xyxy)
        B_confidences = B_confidences.cpu()

        logger.info("{} {} {}", B_labels.shape, B_confidences.shape, B_masks.shape)

        if C_boxes.shape[0] == 0:
            return B_boxes, B_confidences, B_labels, B_masks, C_boxes, C_confidences, torch.zeros(
                (0, h, w))

        nms_idx = nms(C_boxes, C_confidences, self.NMS_IOU_THRESHOLD)

        C_boxes = C_boxes[nms_idx, :]
        C_confidences = C_confidences[nms_idx]

        C_masks = self._predict_masks(img, C_boxes)
        C_confidences = C_confidences.cpu()

        # remove target from background detections
        keep_mask_np = (B_labels != target)
        keep_mask_torch = torch.as_tensor(keep_mask_np, device=B_boxes.device)
        B_boxes = B_boxes[keep_mask_torch]
        B_confidences = B_confidences[keep_mask_torch]
        B_labels = B_labels[keep_mask_np]
        B_masks = B_masks[keep_mask_torch]

        return B_boxes, B_confidences, B_labels, B_masks, C_boxes, C_confidences, C_masks
