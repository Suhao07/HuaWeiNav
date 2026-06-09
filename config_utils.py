import os
from pathlib import Path

import habitat

try:
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
    from habitat.config.read_write import read_write
except Exception:
    CollisionsMeasurementConfig = None
    FogOfWarConfig = None
    TopDownMapMeasurementConfig = None
    read_write = None


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def _first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


HABITAT_LAB_PATH = _get_env("HABITAT_LAB_PATH")
_HABITAT_ROOT = Path(HABITAT_LAB_PATH)
HM3D_CONFIG_PATH = _first_existing([
    str(_HABITAT_ROOT / "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"),
    str(_HABITAT_ROOT / "habitat-lab/configs/tasks/objectnav_hm3d.yaml"),
    str(_HABITAT_ROOT / "configs/tasks/objectnav_hm3d.yaml"),
])
MP3D_CONFIG_PATH = _first_existing([
    str(_HABITAT_ROOT / "habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"),
    str(_HABITAT_ROOT / "habitat-lab/configs/tasks/objectnav_mp3d.yaml"),
    str(_HABITAT_ROOT / "configs/tasks/objectnav_mp3d.yaml"),
])

SAM_CHECKPOINT = _get_env("SAM_CHECKPOINT")
GROUNDING_DINO_PATH = _get_env("GROUNDING_DINO_PATH")
GROUNDING_DINO_CONFIG = os.getenv(
    "GROUNDING_DINO_CONFIG",
    os.path.join(
        GROUNDING_DINO_PATH,
        "configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py",
    ),
)
GROUNDING_DINO_CHECKPOINT = _get_env("GROUNDING_DINO_CHECKPOINT")


class _ConfigWrite:
    def __init__(self, cfg):
        self.cfg = cfg

    def __enter__(self):
        if hasattr(self.cfg, "defrost"):
            self.cfg.defrost()
        return self.cfg

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self.cfg, "freeze"):
            self.cfg.freeze()


def _write_context(cfg):
    # Habitat 0.2 使用 yacs defrost/freeze，新版 Habitat 使用 read_write。
    # 这里做统一上下文，避免 benchmark 绑定某个 Habitat 版本。
    if read_write is not None and hasattr(cfg, "habitat"):
        return read_write(cfg)
    return _ConfigWrite(cfg)


def _legacy_hm3d_data_path(data_path: str, stage: str) -> str:
    # CogNav 和不同 Habitat 发行版的 ObjectNav episode 路径命名不完全一致。
    # 按本机实际存在的文件优先选择，保证同一脚本可复用多个数据布局。
    candidates = [
        f"objectnav_hm3d_v2/{stage}/{stage}.json.gz",
        f"objectgoal_hm3d/{stage}/{stage}.json.gz",
        f"datasets/objectnav/hm3d/v1/{stage}/{stage}.json.gz",
    ]
    for rel in candidates:
        if os.path.exists(os.path.join(data_path, rel)):
            return os.path.join(data_path, rel)
    return os.path.join(data_path, candidates[0])


def _set_modern_common(cfg, data_path: str, stage: str, episodes: int, dataset_rel: str):
    # 新版 Habitat config 字段位于 cfg.habitat.*。
    cfg.habitat.dataset.split = stage
    cfg.habitat.dataset.scenes_dir = os.path.join(data_path, "scene_datasets")
    cfg.habitat.dataset.data_path = os.path.join(data_path, dataset_rel)
    scene_dataset = os.path.join(
        data_path,
        "scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json",
    )
    if hasattr(cfg.habitat.simulator, "scene_dataset"):
        cfg.habitat.simulator.scene_dataset = scene_dataset
    cfg.habitat.environment.iterator_options.num_episode_sample = episodes
    cfg.habitat.environment.iterator_options.shuffle = False
    cfg.habitat.environment.iterator_options.group_by_scene = False
    if TopDownMapMeasurementConfig is not None:
        cfg.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=79),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
    depth = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    depth.max_depth = 5.0
    depth.normalize_depth = False
    cfg.habitat.task.measurements.success.success_distance = 1.0
    cfg.habitat.environment.max_episode_steps = 500


def _set_legacy_common(cfg, data_path: str, stage: str, episodes: int, dataset_path: str):
    # CogNav 基础镜像中的 Habitat 0.2 使用大写字段。
    # 同时关闭 top-down map 中依赖 semantic scene 的目标绘制，避免 HM3D 旧数据触发崩溃。
    cfg.DATASET.SPLIT = stage
    cfg.DATASET.SCENES_DIR = os.path.join(data_path, "scene_datasets")
    cfg.DATASET.DATA_PATH = dataset_path
    if hasattr(cfg.SIMULATOR, "SCENE_DATASET"):
        cfg.SIMULATOR.SCENE_DATASET = os.path.join(
            data_path,
            "scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json",
        )
    if hasattr(cfg, "ENVIRONMENT"):
        cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 500
        if hasattr(cfg.ENVIRONMENT, "ITERATOR_OPTIONS"):
            cfg.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = episodes
            cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
            cfg.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = False
    if hasattr(cfg.SIMULATOR, "DEPTH_SENSOR"):
        cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
        if hasattr(cfg.SIMULATOR.DEPTH_SENSOR, "NORMALIZE_DEPTH"):
            cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    if hasattr(cfg.TASK, "SUCCESS"):
        cfg.TASK.SUCCESS.SUCCESS_DISTANCE = 1.0
    if hasattr(cfg.TASK, "MEASUREMENTS"):
        measurements = list(cfg.TASK.MEASUREMENTS)
        for name in ("TOP_DOWN_MAP", "COLLISIONS"):
            if name not in measurements:
                measurements.append(name)
        cfg.TASK.MEASUREMENTS = measurements
    if hasattr(cfg.TASK, "TOP_DOWN_MAP"):
        top_down_map = cfg.TASK.TOP_DOWN_MAP
        for key, value in {
            "MAP_PADDING": 3,
            "MAP_RESOLUTION": 1024,
            "DRAW_SOURCE": True,
            "DRAW_BORDER": True,
            "DRAW_SHORTEST_PATH": False,
            "DRAW_VIEW_POINTS": False,
            "DRAW_GOAL_POSITIONS": False,
            "DRAW_GOAL_AABBS": False,
        }.items():
            if hasattr(top_down_map, key):
                setattr(top_down_map, key, value)


def hm3d_config(path: str = HM3D_CONFIG_PATH, stage: str = "val", episodes=200):
    cfg = habitat.get_config(path)
    data_path = _get_env("HM3D_DATA_PATH")
    with _write_context(cfg):
        if hasattr(cfg, "habitat"):
            dataset_rel = os.path.relpath(_legacy_hm3d_data_path(data_path, stage), data_path)
            _set_modern_common(cfg, data_path, stage, episodes, dataset_rel)
        else:
            _set_legacy_common(cfg, data_path, stage, episodes, _legacy_hm3d_data_path(data_path, stage))
    return cfg


def mp3d_config(path: str = MP3D_CONFIG_PATH, stage: str = "val", episodes=200):
    cfg = habitat.get_config(path)
    data_path = _get_env("MP3D_DATA_PATH")
    with _write_context(cfg):
        if hasattr(cfg, "habitat"):
            _set_modern_common(cfg, data_path, stage, episodes, "habitat_task/objectnav/{split}/{split}.json.gz")
        else:
            _set_legacy_common(
                cfg,
                data_path,
                stage,
                episodes,
                os.path.join(data_path, "habitat_task/objectnav/{split}/{split}.json.gz"),
            )
    return cfg
