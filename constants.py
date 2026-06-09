import os
# from cv_utils.object_list.matterport_categories_1_10 import categories
from cv_utils.object_list.nyu_categories import categories

DETECT_OBJECTS = [cat['name'].lower() for cat in categories]
INTEREST_OBJECTS = ['bed', 'chair', 'toilet', 'potted_plant', 'tv_monitor', 'sofa']

MODEL_NAME = 'gemini-2.5-flash'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_USE_COGNAV_LLM = (
    os.getenv("STRIVE_LLM_CLIENT", "").lower() == "cognav"
    or bool(os.getenv("COGNAV_OBJNAV_PATH"))
    or bool(os.getenv("ARK_API_KEY"))
)
if not GEMINI_API_KEY and not _USE_COGNAV_LLM:
    raise ValueError("GEMINI_API_KEY is not set")
