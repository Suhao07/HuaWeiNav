"""Benchmark provider contracts for STRIVE ObjectNav entrypoints.

The benchmark layer is intentionally thin: it selects a Habitat-compatible
dataset/config and records provenance.  It must not change navigation policy,
instruction parsing, target verification, or planner behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BenchmarkSpec:
    """Resolved benchmark input for one STRIVE run.

    ``dataset_path`` is the concrete Habitat dataset JSON used by the run.  For
    single-scene debugging this may be a filtered temporary dataset; in that
    case ``source_file`` and ``source_episode_id`` preserve the original
    benchmark provenance.
    """

    benchmark: str
    split: str
    dataset_path: str
    dataset_root: str = ""
    scene_id: str = ""
    scene_id_contains: str = ""
    object_category: str = ""
    episode_rank: int = 0
    source_file: str = ""
    source_episode_id: str = ""
    filtered_dataset_path: str = ""
    success_distance: float = 1.0
    scene_dataset: str = ""
    is_filtered: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            "dataset_path": self.dataset_path,
            "dataset_root": self.dataset_root,
            "scene_id": self.scene_id,
            "scene_id_contains": self.scene_id_contains,
            "object_category": self.object_category,
            "episode_rank": self.episode_rank,
            "source_file": self.source_file,
            "source_episode_id": self.source_episode_id,
            "filtered_dataset_path": self.filtered_dataset_path,
            "success_distance": self.success_distance,
            "scene_dataset": self.scene_dataset,
            "is_filtered": self.is_filtered,
            "provenance": dict(self.provenance or {}),
        }


class BenchmarkProvider:
    """Base class for Habitat ObjectNav benchmark adapters."""

    name = "base"

    def prepare(self, args) -> BenchmarkSpec:
        raise NotImplementedError

    def make_config(self, spec: BenchmarkSpec, episodes: int):
        raise NotImplementedError

    def filter_env(self, env, args, spec: BenchmarkSpec) -> None:
        """Optional post-construction episode filtering.

        Providers that materialize a filtered dataset usually do not need this.
        The method exists for backwards-compatible generic HM3D filtering.
        """

    def resolve_episode_context(self, env, spec: BenchmarkSpec) -> Optional[Dict[str, Any]]:
        episode = getattr(env, "current_episode", None)
        if episode is None:
            return None
        return {
            "episode_id": getattr(episode, "episode_id", ""),
            "scene_id": getattr(episode, "scene_id", ""),
            "object_category": getattr(episode, "object_category", ""),
        }
