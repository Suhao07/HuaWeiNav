"""Gibson ObjectNav benchmark providers.

Gibson is deliberately separated from HM3D/OVON because CogNav's custom Gibson
path has different semantic maps and metric plumbing.  The standard provider
below only resolves Habitat-compatible ObjectNav JSON paths; custom Gibson is
kept as an explicit provider boundary until its wrapper is migrated.
"""

from __future__ import annotations

import os
from pathlib import Path

from .contracts import BenchmarkProvider, BenchmarkSpec
from .habitat_configs import make_habitat_config


class GibsonObjectNavProvider(BenchmarkProvider):
    name = "gibson_objectnav"

    def _data_root(self, args) -> str:
        return str(
            getattr(args, "dataset_root", "")
            or os.getenv("GIBSON_DATA_PATH", "")
            or os.getenv("HM3D_DATA_PATH", "")
        )

    def _success_distance(self, args) -> float:
        if getattr(args, "success_distance", None) is not None:
            return float(args.success_distance)
        return 1.5

    def prepare(self, args) -> BenchmarkSpec:
        data_root = self._data_root(args)
        if not data_root:
            raise ValueError("GIBSON_DATA_PATH or HM3D_DATA_PATH must be set for Gibson.")
        split = str(
            getattr(args, "benchmark_split", "")
            or getattr(args, "split", "")
            or "val"
        )
        dataset_path = str(getattr(args, "dataset_path", "") or "")
        if not dataset_path:
            dataset_path = str(
                Path(data_root) / f"datasets/objectnav/gibson/v1.1/{split}/{split}.json.gz"
            )
        return BenchmarkSpec(
            benchmark=self.name,
            split=split,
            dataset_root=data_root,
            dataset_path=dataset_path,
            scene_id=str(getattr(args, "scene_id", "") or ""),
            scene_id_contains=str(getattr(args, "scene_id_contains", "") or ""),
            object_category=str(getattr(args, "object_category", "") or ""),
            episode_rank=int(getattr(args, "episode_rank", 0) or 0),
            success_distance=self._success_distance(args),
            provenance={
                "provider": self.name,
                "mode": "standard_habitat_objectnav",
            },
        )

    def make_config(self, spec: BenchmarkSpec, episodes: int):
        return make_habitat_config(spec, episodes)


class GibsonCustomProvider(GibsonObjectNavProvider):
    """Boundary for CogNav custom Gibson migration.

    The custom path needs CogNav's semantic map / pbz2 episode metadata and an
    ObjectGoal_Env wrapper.  Keeping it explicit prevents HM3D/OVON benchmark
    code from silently inheriting custom Gibson metric behavior.
    """

    name = "gibson_custom"

    def prepare(self, args) -> BenchmarkSpec:
        spec = super().prepare(args)
        return BenchmarkSpec(
            **{
                **spec.as_dict(),
                "benchmark": self.name,
                "provenance": {
                    **dict(spec.provenance or {}),
                    "mode": "custom_cognav_gibson_pending_wrapper_migration",
                },
            }
        )
