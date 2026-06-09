import os
import sys


def check_path(name):
    path = os.environ.get(name, "")
    ok = bool(path) and os.path.exists(path)
    print(f"{name}: {path or '(unset)'} [{'OK' if ok else 'MISS'}]")
    if not ok:
        raise SystemExit(2)


def main():
    sys.path.insert(0, "/workspace/STRIVE")
    cognav = os.environ.get("COGNAV_OBJNAV_PATH")
    if cognav:
        sys.path.insert(0, cognav)

    for key in [
        "HABITAT_LAB_PATH",
        "HM3D_DATA_PATH",
        "SAM_CHECKPOINT",
        "GROUNDING_DINO_PATH",
        "GROUNDING_DINO_CONFIG",
        "GROUNDING_DINO_CHECKPOINT",
    ]:
        check_path(key)

    import habitat
    import habitat_sim
    import torch
    from mmdet.apis import DetInferencer
    from segment_anything import build_sam
    from utils.llm_client import LLMClient

    from config_utils import hm3d_config
    from mapping_utils.transform import habitat_camera_intrinsic

    cfg = hm3d_config(stage="val", episodes=1)
    intrinsic = habitat_camera_intrinsic(cfg)
    llm = LLMClient(apikey_file=os.environ.get("COGNAV_APIKEY_FILE", "./apikey.txt"))

    print("habitat:", getattr(habitat, "__version__", "unknown"))
    print("habitat_sim:", getattr(habitat_sim, "__version__", "unknown"))
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("mmdet DetInferencer:", DetInferencer.__name__)
    print("sam builder:", build_sam.__name__)
    print("intrinsic:", intrinsic.tolist())
    print("llm model:", llm.default_model)
    print("preflight OK")


if __name__ == "__main__":
    main()
