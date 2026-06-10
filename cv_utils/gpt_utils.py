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


def ask_gpt_object_in_box(img, boxes, save_dir, episode_idx, episode_step, ind, vlm):
    step_dir = _step_dir(save_dir, episode_idx, episode_step)
    img_vis = visualize_mask(img, boxes, None, None, None)

    cv2.imwrite(
        f"{step_dir}/real_C_image_bbox_for_gpt_{ind}.jpg",
        img_vis,
    )

    img_vis_base64 = _encode_image_base64(img_vis)

    # crop the image to the bounding box
    box = boxes[0]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    # bigger the bbox by 5 pixels
    x1 = max(0, x1 - 5)
    y1 = max(0, y1 - 5)
    x2 = min(img.shape[1], x2 + 5)
    y2 = min(img.shape[0], y2 + 5)

    img_cropped = img[y1:y2, x1:x2]

    cv2.imwrite(
        f"{step_dir}/real_C_image_cropped_for_gpt_{ind}.jpg",
        img_cropped,
    )

    crop_bytes = _image_jpg_bytes(img_cropped)
    img_cropped_base64 = base64.b64encode(crop_bytes).decode("utf-8")
    cache_key_raw = json.dumps({
        "bbox": [x1, y1, x2, y2],
        "crop_sha1": hashlib.sha1(crop_bytes).hexdigest(),
        "classes_sha1": hashlib.sha1(json.dumps(DETECT_OBJECTS, sort_keys=True).encode("utf-8")).hexdigest(),
    }, sort_keys=True)
    cache_key = hashlib.sha1(cache_key_raw.encode("utf-8")).hexdigest()
    cache_path = f"{save_dir}/episode-{episode_idx}/detection/object_box_cache.json"
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}
    if cache_key in cache:
        cached = cache[cache_key]
        label = str(cached.get("res", "unknown") or "unknown")
        record_cache_hit("bbox_object_in_box")
        nwlabels = [label]
        img_vis = visualize_mask(img, boxes, None, nwlabels, None)
        cv2.imwrite(f"{step_dir}/real_C_image_gpt_output_{ind}.jpg", img_vis)
        with open(f"{step_dir}/real_C_image_gpt_output_{ind}.txt", "w", encoding="utf-8") as f:
            f.write("Cache-Hit: true\n")
            f.write(f"Cache-Key: {cache_key}\n")
            f.write(f"Answer: {label}\n")
        return label

    PROMPT = BBOX_OBJECT_LABEL_PROMPT.format(detect_objects=DETECT_OBJECTS)

    prompt_info = f'box: {boxes[0].tolist()}'

    with open(
            f"{step_dir}/real_C_image_gpt_input_{ind}.txt",
            "a",
    ) as f:
        f.write(f'Input: {PROMPT}\n')
        f.write(prompt_info)
        f.write(f'\n')
        f.write(f'\n')

    # 这里的 client 可能是 OpenAI/Gemini，也可能是 CogNav 适配后的兼容对象。
    client, model_name = _get_client_and_model(vlm)
    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[{
            "role": "system",
            "content": PROMPT
        }, {
            "role":
                "user",
            "content": [
                {
                    "type": "text",
                    "text": 'This is the whole image with the bounding box.'
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_vis_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": 'This is the cropped image inside the bounding box.'
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_cropped_base64}"
                    }
                },
            ]
        }],
        response_format=BBoxObjectLabelResponse,
        trace_label=BBOX_OBJECT_LABEL.trace_label,
    )

    answer = completion.choices[0].message.parsed
    if answer is None:
        raise ValueError("VLM returned empty parsed response for object detection.")
    nwlabels = [answer.res]
    img_vis = visualize_mask(img, boxes, None, nwlabels, None)
    cv2.imwrite(
        f"{step_dir}/real_C_image_gpt_output_{ind}.jpg",
        img_vis,
    )

    with open(
            f"{step_dir}/real_C_image_gpt_output_{ind}.txt",
            "w",
    ) as f:
        f.write(f'Answer: {answer}\n')
        f.write(f'\n')
        f.write(f'\n')

    cache[cache_key] = {
        "res": answer.res,
        "cache_key": cache_key,
        "bbox": [x1, y1, x2, y2],
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)

    return answer.res


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
