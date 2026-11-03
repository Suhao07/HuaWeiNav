"""Image-oriented VLM helper functions for detector refinement.

Prompt text and response schemas live under ``prompting``. This module handles
image crops, local artifacts, cache keys, and legacy-compatible return values.
"""

import base64
import hashlib
import json
import os

import cv2

from constants import DETECT_OBJECTS
from cv_utils.visualizer import visualize_mask
from llm_utils.cognav_llm_adapter import get_client_and_model
from llm_utils.lvlm_call_tracker import record_cache_hit
from prompting.registry import (
    BBOX_OBJECT_LABEL,
    CHECK_AGAIN_BBOX,
    SIMILAR_OBJECTS,
    TAG_REFINE,
    TAG_REFINE_OBJECT_LIST,
)
from prompting.schemas import (
    BBoxObjectLabelResponse,
    CheckAgainBBoxResponse,
    SimilarObjectsResponse,
    TagRefineResponse,
    TagRefineWithObjectListResponse,
)
from prompting.templates import (
    BBOX_OBJECT_LABEL_PROMPT,
    CHECK_AGAIN_BBOX_PROMPT,
    SIMILAR_OBJECTS_PROMPT,
    TAG_REFINE_PROMPT,
    TAG_REFINE_WITH_OBJECT_LIST_PROMPT,
)


def _get_client_and_model(vlm: str):
    # 上层保留 OpenAI parse 调用形式；cognav 分支由适配器转到 CogNav LLMClient。
    return get_client_and_model(vlm)


def _encode_image_base64(image) -> str:
    image_jpg = cv2.imencode(".jpg", image)[1]
    return base64.b64encode(image_jpg).decode("utf-8")


def _image_jpg_bytes(image) -> bytes:
    return cv2.imencode(".jpg", image)[1].tobytes()


def _step_dir(save_dir, episode_idx, episode_step) -> str:
    path = f"{save_dir}/episode-{episode_idx}/detection/step_{episode_step}"
    os.makedirs(path, exist_ok=True)
    return path


def prepare_bbox_request(img, boxes, save_dir, episode_idx, episode_step, ind, vlm):
    """Freeze the exact visual request; artifact writes belong to the caller thread."""
    step_dir = _step_dir(save_dir, episode_idx, episode_step)
    marked = visualize_mask(img, boxes, None, None, None)
    x1, y1, x2, y2 = map(int, boxes[0])
    x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
    x2, y2 = min(img.shape[1], x2 + 5), min(img.shape[0], y2 + 5)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"{step_dir}/real_C_image_bbox_for_gpt_{ind}.jpg", marked)
    cv2.imwrite(f"{step_dir}/real_C_image_cropped_for_gpt_{ind}.jpg", crop)
    prompt = BBOX_OBJECT_LABEL_PROMPT.format(detect_objects=DETECT_OBJECTS)
    client, model = _get_client_and_model(vlm)
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": [
        {"type": "text", "text": "This is the whole image with the bounding box."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _encode_image_base64(marked)}},
        {"type": "text", "text": "This is the cropped image inside the bounding box."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _encode_image_base64(crop)}},
    ]}]
    from llm_utils.structured_output import request_fingerprint
    key = request_fingerprint(model, messages, BBoxObjectLabelResponse, provider=vlm)
    return {"client": client, "model": model, "messages": messages, "key": key,
            "step_dir": step_dir, "index": ind}


def classify_bbox_request(request):
    """Execute one frozen request; cache and map ownership remain with the caller."""
    completion = request["client"].beta.chat.completions.parse(
        model=request["model"], messages=request["messages"],
        response_format=BBoxObjectLabelResponse, trace_label=BBOX_OBJECT_LABEL.trace_label)
    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("VLM returned empty parsed response for object detection")
    return answer.res


def load_bbox_cache(save_dir, episode_idx):
    """Read the episode cache; malformed files are explicit errors."""
    path = f"{save_dir}/episode-{episode_idx}/detection/object_box_cache.json"
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return {}


def save_bbox_cache(save_dir, episode_idx, cache):
    """Publish cache entries after the observation barrier on the main thread."""
    path = f"{save_dir}/episode-{episode_idx}/detection/object_box_cache.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(cache, stream, ensure_ascii=False, indent=2, sort_keys=True)


def ask_gpt_object_in_box(img, boxes, save_dir, episode_idx, episode_step, ind, vlm):
    """Classify an exact full-image/crop request through the shared cache contract."""
    request = prepare_bbox_request(img, boxes, save_dir, episode_idx, episode_step, ind, vlm)
    cache = load_bbox_cache(save_dir, episode_idx)
    if request["key"] in cache:
        record_cache_hit("bbox_object_in_box")
        return cache[request["key"]]["res"]
    label = classify_bbox_request(request)
    cache[request["key"]] = {"res": label}
    save_bbox_cache(save_dir, episode_idx, cache)
    return label


def refine_tag_with_target(res, target, save_dir, episode_idx, episode_step, ind, vlm):
    tags = [item.tag for item in res]

    PROMPT = TAG_REFINE_PROMPT
    INPUT = f"List: {tags}\nTarget: {target}\n"

    step_dir = _step_dir(save_dir, episode_idx, episode_step)
    with open(
            f"{step_dir}/refine_prompt_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Prompt: {PROMPT}\n')
        f.write(f'Input: {INPUT}\n')

    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": INPUT
            },
        ],
        response_format=TagRefineResponse,
        trace_label=TAG_REFINE.trace_label,
    )

    ans = completion.choices[0].message.parsed
    if ans is None:
        raise ValueError("VLM returned empty parsed response for tag refinement.")

    with open(
            f"{step_dir}/refine_output_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Answer: {ans}\n')
        f.write(f'\n')
        f.write(f'\n')

    assert len(ans.res) == len(tags)
    for i, item in enumerate(res):
        item.tag = ans.res[i]
    return res


def refine_tag_with_target_obj_list(res, target, save_dir, episode_idx, episode_step, ind,
                                    vlm):

    object_list = list(DETECT_OBJECTS)
    if target not in object_list:
        object_list.append(target)

    PROMPT = TAG_REFINE_WITH_OBJECT_LIST_PROMPT.format(object_list=object_list)

    INPUT = f"""
            tag1: {res}
            """

    step_dir = _step_dir(save_dir, episode_idx, episode_step)
    with open(
            f"{step_dir}/refine_prompt_{ind}.txt",
            "w",
    ) as f:
        f.write(PROMPT)
        f.write(f'\n')
        f.write(INPUT)
        f.write(f'\n')

    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": INPUT
            },
        ],
        response_format=TagRefineWithObjectListResponse,
        trace_label=TAG_REFINE_OBJECT_LIST.trace_label,
    )
    ans = completion.choices[0].message.parsed
    if ans is None:
        raise ValueError("VLM returned empty parsed response for object-list refinement.")
    res = ans.output

    with open(
            f"{step_dir}/refine_output_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Answer: {ans}\n')
        f.write(f'\n')
        f.write(f'\n')

    return res


def ask_gpt_similar_objects(obj_list, target, vlm="cognav"):
    obj_str = ", ".join(obj_list)

    prompt = SIMILAR_OBJECTS_PROMPT.format(object_list=obj_str)

    user_input = f"""
    Target object: {target}
    """

    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        response_format=SimilarObjectsResponse,
        trace_label=SIMILAR_OBJECTS.trace_label,
    )

    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("VLM returned empty parsed response for similar-objects query.")

    result = answer.object_list
    if target not in result:
        result.append(target)
    return result


def check_again_object_in_bbox(img_vis, target, save_dir, episode_idx, episode_step, vlm):
    check_dir = f"{save_dir}/episode-{episode_idx}/check_again"
    os.makedirs(check_dir, exist_ok=True)

    prompt = CHECK_AGAIN_BBOX_PROMPT

    prompt_info = (
        f"Whether the object within the bbox in the above image is {target}? "
        "If yes, please output True in the flag field. If no, please output False in the flag field."
    )
    img_base64 = _encode_image_base64(img_vis)

    with open(f"{check_dir}/prompt_{episode_step}.txt", "w") as f:
        f.write(f"Input: {prompt}\n")
        f.write(f"Target: {prompt_info}\n\n")

    client, model_name = _get_client_and_model(vlm)
    messages = [{
        "role": "system",
        "content": prompt
    }, {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            },
            {
                "type": "text",
                "text": prompt_info
            },
        ],
    }]

    try:
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=CheckAgainBBoxResponse,
            trace_label=CHECK_AGAIN_BBOX.trace_label,
        )
    except Exception:
        if vlm != "gemini":
            raise
        completion = client.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=messages,
            response_format=CheckAgainBBoxResponse,
            trace_label=f"{CHECK_AGAIN_BBOX.trace_label}_retry",
        )

    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("VLM returned empty parsed response for check-again.")

    with open(f"{check_dir}/answer_{episode_step}.txt", "w") as f:
        f.write(f"Answer: {answer}\n\n")

    return answer.flag
