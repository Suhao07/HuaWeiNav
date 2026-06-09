import os
# from cv_utils.object_list.matterport_categories_1_10 import categories
from cv_utils.object_list.nyu_categories import categories

DETECT_OBJECTS = [cat['name'].lower() for cat in categories]
INTEREST_OBJECTS = ['bed', 'chair', 'toilet', 'potted_plant', 'tv_monitor', 'sofa']

# 默认使用 CogNav_ObjNav 的 LLMClient 风格配置；Gemini 仅作为显式后端保留。
DEFAULT_VLM = os.getenv("STRIVE_LLM_CLIENT", "cognav").lower()
COGNAV_MODEL_NAME = (
    os.getenv("VLM_MODEL")
    or os.getenv("LLM_MODEL")
    or os.getenv("ARK_MODEL")
    or "doubao-seed-2-0-pro-260215"
)

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MODEL_NAME = GEMINI_MODEL_NAME
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
USE_COGNAV_LLM = (
    DEFAULT_VLM == "cognav"
    or bool(os.getenv("COGNAV_OBJNAV_PATH"))
    or bool(os.getenv("ARK_API_KEY"))
    or bool(os.getenv("LLM_API_BASE_URL"))
)


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required only when --vlm gemini is used")
    return GEMINI_API_KEY
