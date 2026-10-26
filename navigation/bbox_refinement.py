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


def apply_refined_tag(agent: Any, obj: Any, refined_tag: str, image: Any, *, new_judgment: bool = True) -> None:
    """Apply a refined tag and confidence update to an object entity."""

    target_aliases = {agent.mapper.target}
    target_aliases.update(getattr(agent.mapper, "target_aliases", []) or [])
    if refined_tag in target_aliases:
        refined_tag = agent.mapper.target

    # 这里更新的是 mapper 内对象实例的类别置信度；不代表任务已经成功。
    obj.num_list[refined_tag] = obj.num_list.pop(obj.tag)
    obj.conf_list[refined_tag] = obj.conf_list.pop(obj.tag)
    obj.tag = refined_tag

    # 缓存命中复用同一语义判断，只有新模型判断更新语义置信度。
    if new_judgment:
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



def refine_candidates(agent, candidates):
    """Resolve one frozen observation with parallel inference and ordered commits.

    Args:
        agent: Main-thread owner of episode state, map and artifact paths.
        candidates: Ordered (index, object, projected evidence) tuples.

    Returns:
        Label evidence indexed by observation object UID for post-association binding.

    网络等待并发，地图写入顺序保持 candidates 顺序；任一请求失败时传播异常，
    已保存的原始模型响应保留。线程池退出形成观察屏障，随后才更新对象地图。
    """
    import os
    import json
    import time
    import hashlib
    import cv2
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import asdict
    from cv_utils.gpt_utils import (prepare_bbox_request, classify_bbox_request,
                                   load_bbox_cache, save_bbox_cache)
    from llm_utils.lvlm_call_tracker import record_cache_hit
    from navigation.semantic_label_cache import (LabelEvidence, matching_instance,
                                                 reusable_label, instance_signature)
    from llm_utils.structured_output import request_fingerprint
    from prompting.schemas import BBoxObjectLabelResponse, TagRefineWithObjectListResponse
    from prompting.templates import BBOX_OBJECT_LABEL_PROMPT, TAG_REFINE_WITH_OBJECT_LIST_PROMPT
    from constants import DETECT_OBJECTS

    started = time.perf_counter()
    enabled = os.getenv("STRIVE_EVIDENCE_REUSE", "1") == "1"
    workers = int(os.getenv("STRIVE_VLM_WORKERS", "2"))
    directory = Path(agent.save_dir) / f"episode-{agent.episode_samples - 1}"
    cache = load_bbox_cache(agent.save_dir, agent.episode_samples - 1) if enabled else {}
    classes = tuple(agent.mapper.object_perceiver.classes)
    target = agent.mapper.target
    revision = agent.task.spec.revision
    jobs = []
    for index, obj, evidence in candidates:
        request = prepare_bbox_request(evidence.image, evidence.bbox_tensor, agent.save_dir,
                                       agent.episode_samples - 1, agent.episode_steps, index, agent.vlm)
        configuration = request_fingerprint(request["model"],
            [BBOX_OBJECT_LABEL_PROMPT, TAG_REFINE_WITH_OBJECT_LIST_PROMPT,
             (TagRefineWithObjectListResponse.model_json_schema() if hasattr(TagRefineWithObjectListResponse, "model_json_schema")
              else TagRefineWithObjectListResponse.schema()), DETECT_OBJECTS, classes, target], BBoxObjectLabelResponse,
            provider=agent.vlm)
        request["key"] = hashlib.sha256((request["key"] + configuration).encode()).hexdigest()
        x1, y1, x2, y2 = map(int, evidence.bbox_xyxy)
        crop = evidence.image[y1:y2 + 1, x1:x2 + 1]
        area = float(crop.shape[0] * crop.shape[1])
        sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        instance = matching_instance(obj.pcd.point.positions.cpu().numpy(), agent.mapper.objects) if enabled else None
        previous = agent.semantic_label_evidence.get(instance.uid) if instance is not None else None
        valid = (previous is not None and previous.configuration == configuration
                 and previous.instance_dependency == instance_signature(instance)
                 and reusable_label(previous, detector_label=obj.tag, mapped_label=instance.tag,
                                    area=area, sharpness=sharpness, task_revision=revision))
        jobs.append(dict(index=index, obj=obj, evidence=evidence, request=request,
                         configuration=configuration, area=area, sharpness=sharpness,
                         instance_uid=instance.uid if instance else None,
                         previous=previous if valid else None,
                         source="instance" if valid else "exact" if request["key"] in cache else "model"))

    # Workers receive only prepared requests and immutable normalization context.
    def classify(request):
        tag = classify_bbox_request(request)
        if tag not in classes:
            tag = refine_tag_with_target_obj_list(tag, target, str(directory.parent),
                agent_episode, observation_step, request["index"], provider)
        return tag

    agent_episode, observation_step, provider = agent.episode_samples - 1, agent.episode_steps, agent.vlm
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {j["index"]: pool.submit(classify, j["request"])
                   for j in jobs if j["source"] == "model"}
        for job in jobs:
            if job["source"] == "model":
                job["label"] = futures[job["index"]].result()
            elif job["source"] == "instance":
                job["label"] = job["previous"].label
                record_cache_hit("bbox_instance_label")
            else:
                job["label"] = cache[job["request"]["key"]]["res"]
                record_cache_hit("bbox_object_in_box")
    records = {}
    with (directory / "semantic_reuse.jsonl").open("a") as stream:
        for job in jobs:
            obj, evidence, label = job["obj"], job["evidence"], job["label"]
            record = job["previous"] or LabelEvidence(obj.tag, label, job["area"], job["sharpness"],
                revision, observation_step, source_id=job["request"]["key"], configuration=job["configuration"])
            records[obj.uid] = record
            apply_refined_tag(agent, obj, label, evidence.image, new_judgment=job["source"] == "model")
            if evidence.bbox_real_xyxy is not None:
                obj.bbox = evidence.bbox_real_xyxy
            if enabled and job["source"] != "instance":
                cache[job["request"]["key"]] = {"res": label}
            stream.write(json.dumps({"step": observation_step, "instance_uid": job["instance_uid"],
                "source": job["source"], "reused": job["source"] != "model",
                "evidence": asdict(record)}) + "\n")
    if enabled:
        save_bbox_cache(agent.save_dir, agent_episode, cache)
    with (directory / "candidate_cycles.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": observation_step, "workers": workers, "candidates": len(jobs),
            "model_requests": len(futures), "elapsed_s": time.perf_counter() - started}) + "\n")
    return records
