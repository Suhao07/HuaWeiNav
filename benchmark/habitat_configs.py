"""Habitat config factory used by benchmark providers."""

from __future__ import annotations

from .contracts import BenchmarkSpec


def make_habitat_config(spec: BenchmarkSpec, episodes: int):
    # 延迟导入 config_utils，避免 benchmark provider 的纯数据测试提前依赖
    # Habitat 安装环境；只有真正构造 Habitat config 时才需要这些依赖。
    from config_utils import gibson_config, hm3d_config

    if spec.benchmark in {"hm3d_objectnav", "hm3d_ovon", "hm3d_instruction"}:
        return hm3d_config(
            stage=spec.split,
            episodes=episodes,
            dataset_path=spec.dataset_path,
            success_distance=spec.success_distance,
        )
    if spec.benchmark in {"gibson_objectnav", "gibson_custom"}:
        return gibson_config(
            stage=spec.split,
            episodes=episodes,
            dataset_path=spec.dataset_path,
            success_distance=spec.success_distance,
        )
    raise ValueError(f"Unsupported benchmark for Habitat config: {spec.benchmark}")
