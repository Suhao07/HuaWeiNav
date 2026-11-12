import numpy as np
import torch
from loguru import logger
from mmdet.apis import DetInferencer
from segment_anything import SamPredictor, build_sam
from torchvision.ops import nms

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
        from config_utils import (GROUNDING_DINO_CHECKPOINT, GROUNDING_DINO_CONFIG,
                                  SAM_CHECKPOINT)

        self.classes = list(classes)
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.DEVICE = device

        self.sam_model = build_sam(checkpoint=SAM_CHECKPOINT).to(self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam_model)

        inferencer = DetInferencer(model=GROUNDING_DINO_CONFIG,
                                   weights=GROUNDING_DINO_CHECKPOINT,
                                   device=self.DEVICE)
        self.grounding_model = inferencer
        language = inferencer.model.language_model
        self.tokenizer, self.max_text_tokens = language.tokenizer, language.max_tokens

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

    def segment_current_target(self, img, target):
        """Detect and segment current RGB instances for open-vocabulary verification.

        Reuses the deployed detector, thresholds, NMS and SAM. A historical
        projection never supplies the segmentation box or certifies identity.
        """
        prediction = self._run_grounding(img, target)
        scores = torch.tensor(prediction["scores"])
        boxes = torch.tensor(prediction["bboxes"]).reshape(-1, 4)
        keep = scores >= self.TARGET_BOX_THRESHOLD
        boxes, scores = boxes[keep], scores[keep]
        if len(boxes) == 0:
            return np.empty((0, *img.shape[:2]), dtype=bool)
        boxes = boxes[nms(boxes, scores, self.NMS_IOU_THRESHOLD)]
        self.sam_predictor.set_image(img)
        return self._predict_masks(img, boxes).numpy().astype(bool)

    def initialize(self, target):
        """Reset cached prompts to the fixed base vocabulary without reloading models."""
        self._prompt_classes = None
        self._update_vocabulary()

    def _update_vocabulary(self, target_list=None):
        """Pack the vocabulary into the fewest captions the text encoder accepts.

        Grounding DINO scores each query against every phrase of one caption, so
        the canonical use is a single caption holding the whole vocabulary; it is
        only divided when the BERT token budget would be exceeded. 每个 caption
        形如 ``"chair . table ."``，长度以 tokenizer 实际 token 数计量，而不是固定组数。

        Args:
            target_list: Dynamic retrieval terms appended after the base classes.
        """
        classes = list(dict.fromkeys([*self.classes, *(target_list or ())]))
        if classes == self._prompt_classes:
            return
        if not classes:
            raise ValueError("Detector vocabulary must contain at least one concept")
        prompts, groups = [], []
        for concept in classes:
            phrase = f"{concept.lower()} ."
            if prompts and self._token_count(f"{prompts[-1]} {phrase}") <= self.max_text_tokens:
                prompts[-1] = f"{prompts[-1]} {phrase}"
                groups[-1].append(concept)
            else:
                prompts.append(phrase)
                groups.append([concept])
        self.TEXT_PROMPTs = prompts
        self.class_labels = [np.asarray(group) for group in groups]
        self._prompt_classes = classes

    def _token_count(self, caption):
        return self.tokenizer([caption], return_tensors="pt").input_ids.shape[1]

    def segment(self,
                img,
                target=None,
                target_list=None,
                save_dir="",
                episode_idx=-1,
                episode_step=-1):
        """Segment RGB using base classes, current retrieval terms and an optional target."""
        self._update_vocabulary(target_list)
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
        for prompt, prompt_labels in zip(self.TEXT_PROMPTs, self.class_labels):
            predictions = self._run_grounding(img, prompt)
            boxes = boxes + predictions["bboxes"]
            confidences = confidences + predictions["scores"]
            labels = labels + prompt_labels[predictions["labels"]].tolist()

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

        # SAM 的图像编码（ViT-H）只在存在待分割框时执行；无检测的视图直接返回。
        if B_boxes.shape[0] > 0 or C_boxes.shape[0] > 0:
            self.sam_predictor.set_image(img)

        # 词表检测为空时目标检测仍独立成立，目标框必须继续生成对应掩码。
        B_masks = torch.zeros((0, h, w))
        if B_boxes.shape[0] > 0:
            logger.info(f"Before NMS: {B_boxes.shape[0]} boxes")
            nms_idx = nms(B_boxes, B_confidences, self.NMS_IOU_THRESHOLD)
            B_boxes = B_boxes[nms_idx, :]
            B_confidences = B_confidences[nms_idx]
            B_labels = B_labels[nms_idx.cpu().numpy()]
            logger.info(f"After NMS: {B_boxes.shape[0]} boxes")

            B_masks = self._predict_masks(img, B_boxes)
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
        keep_mask_np = np.array([label != target for label in B_labels], dtype=bool)
        keep_mask_torch = torch.as_tensor(keep_mask_np, device=B_boxes.device)
        B_boxes = B_boxes[keep_mask_torch]
        B_confidences = B_confidences[keep_mask_torch]
        B_labels = B_labels[keep_mask_np]
        B_masks = B_masks[keep_mask_torch]

        return B_boxes, B_confidences, B_labels, B_masks, C_boxes, C_confidences, C_masks
