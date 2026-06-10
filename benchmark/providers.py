"""Benchmark provider registry."""

from __future__ import annotations

from typing import Dict, Type

from .contracts import BenchmarkProvider
from .gibson_objectnav import GibsonCustomProvider, GibsonObjectNavProvider
from .hm3d_objectnav import HM3DInstructionProvider, HM3DObjectNavProvider
from .hm3d_ovon import HM3DOVONProvider


_PROVIDERS: Dict[str, Type[BenchmarkProvider]] = {
    "hm3d_objectnav": HM3DObjectNavProvider,
    "hm3d_instruction": HM3DInstructionProvider,
    "hm3d_ovon": HM3DOVONProvider,
    "gibson_objectnav": GibsonObjectNavProvider,
    "gibson_custom": GibsonCustomProvider,
}


def infer_benchmark(args) -> str:
    requested = str(getattr(args, "benchmark", "auto") or "auto").strip().lower()
    if requested and requested != "auto":
        return requested

    dataset_path = str(getattr(args, "dataset_path", "") or "")
    if "gibson" in dataset_path.lower():
        return "gibson_objectnav"
    if "hm3d_ovon" in dataset_path.lower() or "objectgoal_hm3d_ovon" in dataset_path.lower():
        return "hm3d_ovon"
    # Backwards-compatible scene/object debugging used the OVON/instruction
    # search path first.  Keep that behavior in auto mode, but explicit
    # benchmark runs should set --benchmark and --benchmark_split.
    if getattr(args, "scene_id", None) and getattr(args, "object_category", None):
        return "hm3d_ovon"
    return "hm3d_objectnav"


def get_provider(args) -> BenchmarkProvider:
    name = infer_benchmark(args)
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown benchmark {name!r}. Available: {sorted(_PROVIDERS)}"
        )
    return provider_cls()


def available_benchmarks():
    return sorted(_PROVIDERS)
