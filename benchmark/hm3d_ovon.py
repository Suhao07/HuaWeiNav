"""HM3D-OVON / instruction ObjectNav benchmark provider."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .contracts import BenchmarkProvider, BenchmarkSpec
from .episode_index import find_matching_episode_file, safe_name, write_json_gz
from .habitat_configs import make_habitat_config


class HM3DOVONProvider(BenchmarkProvider):
    """Provider for converted HM3D-OVON ObjectNav-compatible datasets.

    Explicit split selection is the benchmark-safe path:

    ``--benchmark hm3d_ovon --benchmark_split val_seen_instruction_balanced_3k``.

    ``split=auto`` is retained only for backwards-compatible single-scene
    debugging.  It searches historical local layouts in priority order and
    records that behavior in BenchmarkSpec.provenance.
    """

    name = "hm3d_ovon"

    def _default_root(self, args) -> Path:
        if getattr(args, "dataset_root", None):
            return Path(args.dataset_root)
        data_root = os.getenv("HM3D_DATA_PATH")
        if not data_root:
            raise ValueError("HM3D_DATA_PATH is not set; cannot locate HM3D-OVON data.")
        return Path(data_root) / "datasets/objectnav/hm3d_ovon/v1"

    def _spec_root(self, args, dataset_path: str = "") -> str:
        if getattr(args, "dataset_root", None):
            return str(Path(args.dataset_root))
        if dataset_path:
            return str(Path(dataset_path).resolve().parent)
        return str(self._default_root(args))

    def _explicit_split_candidates(self, root: Path, split: str, scene_id: str) -> Iterable[Path]:
        scene_file = f"{scene_id}.json.gz"
        yield root / split / "content" / scene_file
        yield root / split / f"{split}.json.gz"

    def _legacy_auto_candidates(self, scene_id: str) -> Iterable[Path]:
        data_root = os.getenv("HM3D_DATA_PATH")
        if not data_root:
            raise ValueError("HM3D_DATA_PATH is not set; cannot locate HM3D-OVON data.")
        scene_file = f"{scene_id}.json.gz"
        explicit = os.getenv("STRIVE_SCENE_OBJECT_DATASET")
        if explicit:
            yield Path(explicit)
        for rel in [
            f"datasets/objectnav/hm3d_ovon/v1/val_seen_complex_balanced_2k/content/{scene_file}",
            f"datasets/objectnav/hm3d_ovon/v1/val_seen_complex_instruction/content/{scene_file}",
            f"datasets/objectnav/hm3d_ovon/v1/val_seen_instruction_balanced_3k/content/{scene_file}",
            f"datasets/objectnav/hm3d_ovon/v1/val_seen_instruction_2k/content/{scene_file}",
            f"objectgoal_hm3d_ovon/val_seen/content/{scene_file}",
        ]:
            yield Path(data_root) / rel

    def _candidate_files(self, args, split: str, scene_id: str) -> Iterable[Path]:
        dataset_path = str(getattr(args, "dataset_path", "") or "")
        if dataset_path:
            yield Path(dataset_path)
            return
        if split == "auto":
            yield from self._legacy_auto_candidates(scene_id)
            return
        yield from self._explicit_split_candidates(self._default_root(args), split, scene_id)

    def _materialize_filtered_dataset(self, args, split: str) -> BenchmarkSpec:
        scene_id = str(getattr(args, "scene_id", "") or "").strip()
        object_category = str(getattr(args, "object_category", "") or "").strip()
        if not scene_id or not object_category:
            dataset_path = str(getattr(args, "dataset_path", "") or os.getenv("HM3D_DATASET_PATH", ""))
            if not dataset_path:
                root = self._default_root(args)
                dataset_path = str(root / split / f"{split}.json.gz")
            return BenchmarkSpec(
                benchmark=self.name,
                split=split,
                dataset_root=self._spec_root(args, dataset_path),
                dataset_path=dataset_path,
                success_distance=self._success_distance(args),
                provenance={
                    "provider": self.name,
                    "selection": "full_split",
                },
            )

        source_file, source_dataset, matched = find_matching_episode_file(
            self._candidate_files(args, split, scene_id),
            object_category,
        )
        episode_rank = int(getattr(args, "episode_rank", 0) or 0)
        if episode_rank < 0 or episode_rank >= len(matched):
            raise IndexError(
                f"episode_rank={episode_rank} is out of range; "
                f"{len(matched)} matched episodes are available."
            )

        filtered = dict(source_dataset)
        filtered["episodes"] = [matched[episode_rank]]
        output_dir = Path(getattr(args, "filtered_dataset_dir", "logs/datasets"))
        output_path = output_dir / (
            f"{safe_name(scene_id)}_{safe_name(object_category)}_rank{episode_rank}.json.gz"
        )
        write_json_gz(filtered, output_path)
        selected_episode = filtered["episodes"][0]

        # Keep legacy scripts working, but BenchmarkSpec is now the authoritative
        # source used by config construction.
        os.environ["HM3D_DATASET_PATH"] = str(output_path.resolve())
        args.dataset_episodes = 1
        args.eval_episodes = 1
        args.start_episode = 0

        return BenchmarkSpec(
            benchmark=self.name,
            split=split,
            dataset_root=self._spec_root(args, str(source_file)),
            dataset_path=str(output_path.resolve()),
            scene_id=scene_id,
            object_category=object_category,
            episode_rank=episode_rank,
            source_file=str(source_file),
            source_episode_id=str(selected_episode.get("episode_id", "")),
            filtered_dataset_path=str(output_path.resolve()),
            success_distance=self._success_distance(args),
            is_filtered=True,
            provenance={
                "provider": self.name,
                "selection": "single_scene_object",
                "matched_count": len(matched),
                "split_auto": split == "auto",
            },
        )

    def _success_distance(self, args) -> float:
        if getattr(args, "success_distance", None) is not None:
            return float(args.success_distance)
        # CogNav converted OVON configs use 1.5m; keep benchmark parity unless
        # the caller explicitly asks for a different threshold.
        return 1.5

    def prepare(self, args) -> BenchmarkSpec:
        split = str(
            getattr(args, "benchmark_split", "")
            or getattr(args, "split", "")
            or "val_seen"
        )
        # Auto mode is explicit: backwards compatibility is useful for smoke
        # tests, but benchmark runs should name their split.
        if str(getattr(args, "benchmark", "auto") or "auto") == "auto":
            split = "auto"
        return self._materialize_filtered_dataset(args, split)

    def make_config(self, spec: BenchmarkSpec, episodes: int):
        return make_habitat_config(spec, episodes)
