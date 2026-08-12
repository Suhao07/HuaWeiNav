import os
import sys


def check_path(name, *, required=True):
    """Check whether an environment variable points to an existing path.

    Args:
        name: Environment variable name to inspect.
        required: Whether a missing path should fail the preflight.

    Returns:
        True when the path exists, otherwise False.

    Raises:
        SystemExit: If the path is required and missing.
    """
    path = os.environ.get(name, "")
    ok = bool(path) and os.path.exists(path)
    print(f"{name}: {path or '(unset)'} [{'OK' if ok else 'MISS'}]")
    if required and not ok:
        raise SystemExit(2)
    return ok


def main():
    """Run deterministic STRIVE HM3D simulation preflight checks.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If required paths or imports are unavailable.
    """
    sys.path.insert(0, "/workspace/STRIVE")

    check_path("HABITAT_LAB_PATH", required=False)
    for key in ["HM3D_DATA_PATH", "SAM_CHECKPOINT", "GROUNDING_DINO_PATH", "GROUNDING_DINO_CONFIG", "GROUNDING_DINO_CHECKPOINT"]:
        check_path(key)

    import habitat
    import habitat_sim
    import torch
    from mmdet.apis import DetInferencer
    from segment_anything import build_sam

    from config_utils import hm3d_config
    from mapping_utils.transform import habitat_camera_intrinsic

    cfg = hm3d_config(stage="val", episodes=1)
    intrinsic = habitat_camera_intrinsic(cfg)

    print("habitat:", getattr(habitat, "__version__", "unknown"))
    print("habitat_sim:", getattr(habitat_sim, "__version__", "unknown"))
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("mmdet DetInferencer:", DetInferencer.__name__)
    print("sam builder:", build_sam.__name__)
    print("intrinsic:", intrinsic.tolist())
    print("preflight OK")


if __name__ == "__main__":
    main()
