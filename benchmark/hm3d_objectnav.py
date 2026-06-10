"""HM3D ObjectNav benchmark provider."""

from __future__ import annotations

import os
from pathlib import Path

from .contracts import BenchmarkProvider, BenchmarkSpec
from .habitat_configs import make_habitat_config


class HM3DObjectNavProvider(BenchmarkProvider):
    """Provider for standard HM3D ObjectNav-compatible splits."""

    name = "hm3d_objectnav"

    def prepare(self, args) -> BenchmarkSpec:
        data_root = str(getattr(args, "dataset_root", "") or os.getenv("HM3D_DATA_PATH", ""))
        split = str(
            getattr(args, "benchmark_split", "")
            or getattr(args, "split", "")
            or "val"
        )
        dataset_path = str(getattr(args, "dataset_path", "") or "")
        if not dataset_path and data_root:
            candidates = [
                Path(data_root) / f"objectnav_hm3d_v2/{split}/{split}.json.gz",
                Path(data_root) / f"objectgoal_hm3d/{split}/{split}.json.gz",
                Path(data_root) / f"datasets/objectnav/hm3d/v1/{split}/{split}.json.gz",
            ]
            for path in candidates:
                if path.exists():
                    dataset_path = str(path)
                    break
        success_distance = (
            float(args.success_distance)
            if getattr(args, "success_distance", None) is not None
            else 1.0
        )
        return BenchmarkSpec(
            benchmark="hm3d_objectnav",
            split=split,
            dataset_root=data_root,
            dataset_path=dataset_path,
            scene_id=str(getattr(args, "scene_id", "") or ""),
            scene_id_contains=str(getattr(args, "scene_id_contains", "") or ""),
            object_category=str(getattr(args, "object_category", "") or ""),
            success_distance=success_distance,
            provenance={
                "provider": self.name,
                "dataset_path_source": "args_or_env",
            },
        )

    def make_config(self, spec: BenchmarkSpec, episodes: int):
        return make_habitat_config(spec, episodes)


class HM3DInstructionProvider(HM3DObjectNavProvider):
    """Provider name for HM3D instruction-style ObjectNav splits."""

    name = "hm3d_instruction"

    def prepare(self, args) -> BenchmarkSpec:
        spec = super().prepare(args)
        data = spec.as_dict()
        data["benchmark"] = self.name
        data["provenance"] = {
            **dict(spec.provenance or {}),
            "provider": self.name,
        }
        return BenchmarkSpec(**data)
