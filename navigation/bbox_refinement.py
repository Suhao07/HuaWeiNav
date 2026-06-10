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
