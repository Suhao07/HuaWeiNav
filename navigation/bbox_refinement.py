"""VLM-assisted bbox tag refinement for object entities.

This module wraps the legacy bbox verifier used during object-map construction.
It refines detector labels for mapper objects only. Final instruction
satisfaction remains in ``instruction_adapter.verifier``.
"""

from __future__ import annotations

from typing import Any

import torch

from cv_utils.gpt_utils import ask_gpt_object_in_box, refine_tag_with_target_obj_list


def refine_bbox_tag(agent: Any, image: Any, bbox_tensor: Any, object_index: int) -> str:
    """Ask the configured VLM to refine the detector tag inside one bbox."""

    tag = ask_gpt_object_in_box(
        image,
        bbox_tensor,
        agent.save_dir,
        agent.episode_samples - 1,
        agent.episode_steps,
        object_index,
        agent.vlm,
    )
    if tag not in agent.mapper.object_perceiver.classes:
        tag = refine_tag_with_target_obj_list(
            tag,
            agent.mapper.target,
            agent.save_dir,
            agent.episode_samples - 1,
            agent.episode_steps,
            object_index,
            agent.vlm,
        )
    return tag


def apply_refined_tag(agent: Any, obj: Any, refined_tag: str, image: Any) -> None:
    """Apply a refined tag and confidence update to an object entity."""

    target_aliases = {agent.mapper.target}
    target_aliases.update(getattr(agent.mapper, "target_aliases", []) or [])
    if refined_tag in target_aliases:
        refined_tag = agent.mapper.target

    # 这里更新的是 mapper 内对象实例的类别置信度；不代表任务已经成功。
    obj.num_list[refined_tag] = obj.num_list.pop(obj.tag)
    obj.conf_list[refined_tag] = obj.conf_list.pop(obj.tag)
    obj.tag = refined_tag

    if refined_tag == agent.mapper.target:
        obj.confidence = (0.9 * 2 + obj.confidence) / 3
    else:
        obj.confidence = 0.9 / (0.9 + obj.confidence)
    if refined_tag == "unknown":
        obj.confidence = 0.0

    if not isinstance(obj.confidence, torch.Tensor):
        obj.confidence = torch.tensor(obj.confidence)
    obj.rgb = image
    obj.conf_list[obj.tag] = obj.confidence


def refine_instance_tag(agent, obj, evidence, object_index):
    """Resolve current instance identity, then reuse or refresh its label evidence."""
    import cv2
    import json
    from pathlib import Path
    from llm_utils.lvlm_call_tracker import record_cache_hit
    from navigation.semantic_label_cache import LabelEvidence, matching_instance, reusable_label

    x1, y1, x2, y2 = map(int, evidence.bbox_xyxy)
    crop = evidence.image[y1:y2 + 1, x1:x2 + 1]
    area = float(crop.shape[0] * crop.shape[1])
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
    instance = matching_instance(obj.pcd.point.positions.cpu().numpy(), agent.mapper.objects)
    record = agent.semantic_label_evidence.get(instance.uid) if instance is not None else None
    reuse = record is not None and reusable_label(
        record, detector_label=obj.tag, mapped_label=instance.tag, area=area,
        sharpness=sharpness, task_revision=agent.task.spec.revision)
    if reuse:
        tag = record.label
        record_cache_hit("bbox_instance_label")
    else:
        tag = refine_bbox_tag(agent, evidence.image, evidence.bbox_tensor, object_index)
        record = LabelEvidence(obj.tag, tag, area, sharpness, agent.task.spec.revision, agent.episode_steps)
    path = Path(agent.save_dir) / f"episode-{agent.episode_samples - 1}" / "semantic_reuse.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"step": agent.episode_steps, "instance_uid": instance.uid if instance else None,
                                 "reused": reuse, "evidence_step": record.step, "label": tag}) + "\n")
    return tag, record
