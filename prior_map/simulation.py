"""Simulation integration helpers for prior-map mode.

The helpers here keep the benchmark entrypoint thin: they load prior-map
artifacts, update runtime memory from a mapper, query soft rankings, and write
debug JSON. They do not publish motion goals or final stop decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .alignment import PriorMapAlignment
from .contracts import PriorMapData, SearchPriorResult
from .evaluation import (
    prior_map_metrics_summary,
    write_prior_map_static_artifacts,
    write_prior_map_step_artifacts,
)
from .loaders import PriorMapLoader
from .memory import PriorMapMemory
from .policy_adapter import PriorMapPolicyAdapter
from .prompt_context import PriorMapPromptContextBuilder, PromptContextBundle
from .query import PriorMapQueryService


@dataclass(frozen=True)
class PriorMapSimulationConfig:
    """Configuration for prior-map simulation integration.

    Args:
        enabled: Whether prior-map integration is active.
        prior_map_path: Prior map file path.
        prior_map_source: Loader source format, or ``auto``.
        prior_map_alignment: Alignment mode or JSON path.
        artifact_root: Root directory for prior-map runtime artifacts.
    """

    enabled: bool = False
    prior_map_path: str = ""
    prior_map_source: str = "auto"
    prior_map_alignment: str = "identity"
    artifact_root: str = ""

    @classmethod
    def from_args(cls, args: Any, save_dir: str) -> "PriorMapSimulationConfig":
        """Build config from an argparse namespace-like object.

        Args:
            args: Object exposing prior-map CLI attributes.
            save_dir: Run output directory.

        Returns:
            Simulation prior-map config.
        """

        return cls(
            enabled=bool(getattr(args, "enable_prior_map", False)),
            prior_map_path=str(getattr(args, "prior_map_path", "") or ""),
            prior_map_source=str(getattr(args, "prior_map_source", "auto") or "auto"),
            prior_map_alignment=str(getattr(args, "prior_map_alignment", "identity") or "identity"),
            artifact_root=str(Path(save_dir) / "prior_map"),
        )


class PriorMapSimulationRuntime:
    """Runtime owner for simulation prior-map memory/query artifacts.

    Args:
        base_map: Loaded prior map.
        alignment: Prior-to-runtime alignment.
        artifact_root: Run-level artifact directory.
        query_service: Optional query service.
        prompt_builder: Optional prompt context builder.
        policy_adapter: Optional policy adapter.
    """

    def __init__(
        self,
        *,
        base_map: PriorMapData,
        alignment: PriorMapAlignment,
        artifact_root: str | Path,
        query_service: Optional[PriorMapQueryService] = None,
        prompt_builder: Optional[PriorMapPromptContextBuilder] = None,
        policy_adapter: Optional[PriorMapPolicyAdapter] = None,
    ) -> None:
        """Create simulation runtime state."""

        self.base_map = base_map
        self.alignment = alignment
        self.memory = PriorMapMemory(base_map=base_map, alignment=alignment)
        self.artifact_root = Path(artifact_root)
        self.query_service = query_service or PriorMapQueryService()
        self.prompt_builder = prompt_builder or PriorMapPromptContextBuilder(max_chars=4000)
        self.policy_adapter = policy_adapter or PriorMapPolicyAdapter(enabled=True)
        self.episode_dir: Optional[Path] = None
        self.last_observation: Any = None
        self.last_query_result: Optional[SearchPriorResult] = None
        self.last_prompt_context: Optional[PromptContextBundle] = None
        self.last_chosen_frontier: Optional[dict[str, Any]] = None
        self.last_query_step: Optional[int] = None

    @classmethod
    def from_config(cls, config: PriorMapSimulationConfig) -> Optional["PriorMapSimulationRuntime"]:
        """Load a simulation runtime from config.

        Args:
            config: Prior-map simulation config.

        Returns:
            Runtime instance when enabled, otherwise ``None``.

        Raises:
            ValueError: If prior-map mode is enabled without a map path.
        """

        if not config.enabled:
            return None
        if not config.prior_map_path:
            raise ValueError("--enable_prior_map requires --prior_map_path")
        base_map = PriorMapLoader().load(config.prior_map_path, source_format=config.prior_map_source)
        alignment = _load_alignment(config.prior_map_alignment, base_map)
        return cls(base_map=base_map, alignment=alignment, artifact_root=config.artifact_root)

    def begin_episode(self, episode_dir: str | Path, episode_idx: int) -> None:
        """Initialize per-episode artifact paths and reset memory.

        Args:
            episode_dir: Episode output directory.
            episode_idx: Episode index for diagnostics.
        """

        self.memory = PriorMapMemory(base_map=self.base_map, alignment=self.alignment)
        self.episode_dir = Path(episode_dir) / "prior_map"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.last_observation = None
        self.last_query_result = None
        self.last_prompt_context = None
        self.last_chosen_frontier = None
        self.last_query_step = None
        write_prior_map_static_artifacts(output_dir=self.episode_dir, memory=self.memory)
        _write_json(
            self.episode_dir / "manifest.json",
            {
                "episode_idx": int(episode_idx),
                "scene_id": self.base_map.scene_id,
                "source_format": self.base_map.source_format,
                "frame_id": self.base_map.frame_id,
                "authority": "ranking_only",
            },
        )

    def update_and_query(
        self,
        *,
        mapper: Any,
        plan: Any,
        step: int,
        episode_idx: Optional[int] = None,
    ) -> SearchPriorResult:
        """Update memory from mapper and query current prior rankings.

        Args:
            mapper: Simulation mapper-like object.
            plan: Instruction plan-like object.
            step: Planning step index.
            episode_idx: Optional episode index used when lazy-initializing
                artifacts.

        Returns:
            Search prior result.
        """

        self._ensure_episode_dir(mapper, episode_idx)
        if self.last_query_step == int(step) and self.last_query_result is not None:
            return self.last_query_result
        self.last_observation = self.memory.update_from_mapper(mapper, step=step)
        result = self.query_service.query(plan, mapper, self.memory)
        prompt_context = self.prompt_builder.build_bundle(self.memory.current_map(), result)
        self.last_query_result = result
        self.last_prompt_context = prompt_context
        self.last_query_step = int(step)
        self._write_step_artifacts(step=step, plan=plan, mapper=mapper, result=result, prompt_context=prompt_context)
        return result

    def metrics_summary(self) -> dict[str, Any]:
        """Return compact prior-map metrics for benchmark rows.

        Returns:
            JSON-friendly prior-map metrics/debug summary.
        """

        return prior_map_metrics_summary(
            enabled=True,
            alignment=self.alignment,
            memory=self.memory,
            prior_result=self.last_query_result,
        )

    def record_chosen_frontier(self, payload: dict[str, Any], step: Optional[int] = None) -> Optional[Path]:
        """Write the active planner's chosen-frontier debug artifact.

        Args:
            payload: JSON-friendly selection payload produced by the active
                frontier policy.
            step: Optional planning step. If omitted, the last query step is
                used.

        Returns:
            Written path when an episode directory is available, otherwise
            ``None``.
        """

        self.last_chosen_frontier = dict(payload)
        if self.episode_dir is None:
            return None
        step_value = int(step if step is not None and int(step) >= 0 else self.last_query_step or 0)
        output = self.episode_dir / f"chosen_frontier_{step_value:06d}.json"
        enriched = {
            "step": step_value,
            "authority": "ranking_only",
            **dict(payload),
        }
        _write_json(output, enriched)
        return output

    def _ensure_episode_dir(self, mapper: Any, episode_idx: Optional[int]) -> None:
        if self.episode_dir is not None:
            return
        save_dir = Path(str(getattr(mapper, "save_dir", self.artifact_root.parent or ".")))
        idx = int(episode_idx if episode_idx is not None else getattr(mapper, "episode_idx", 0))
        self.begin_episode(save_dir / f"episode-{idx}", idx)

    def _write_step_artifacts(
        self,
        *,
        step: int,
        plan: Any,
        mapper: Any,
        result: SearchPriorResult,
        prompt_context: PromptContextBundle,
    ) -> None:
        assert self.episode_dir is not None
        suffix = f"{int(step):06d}"
        query_payload = {
            "step": int(step),
            "raw_instruction": str(getattr(plan, "raw_instruction", "") or getattr(mapper, "target", "")),
            "dataset_target": str(getattr(plan, "dataset_target", "") or getattr(mapper, "target", "")),
            "runtime_counts": {
                "objects": len(list(getattr(mapper, "objects", []) or [])),
                "rooms": len(list(getattr(mapper, "room_nodes", []) or [])),
                "frontiers": len(list(getattr(mapper, "nodes", []) or [])),
            },
            "observation": self.last_observation.to_dict() if hasattr(self.last_observation, "to_dict") else {},
            "authority": "ranking_only",
        }
        paths = write_prior_map_step_artifacts(
            output_dir=self.episode_dir,
            step=int(step),
            memory=self.memory,
            prior_result=result,
            query_payload=query_payload,
            prompt_context_payload=prompt_context.to_dict(),
        )
        _write_json(
            self.episode_dir / f"artifact_manifest_{suffix}.json",
            {
                "step": int(step),
                "artifacts": paths,
                "authority": "ranking_only",
            },
        )


def build_prior_map_simulation_runtime(args: Any, save_dir: str) -> Optional[PriorMapSimulationRuntime]:
    """Build a simulation prior-map runtime from CLI args.

    Args:
        args: argparse namespace-like object.
        save_dir: Run output directory.

    Returns:
        Runtime instance when enabled, otherwise ``None``.
    """

    return PriorMapSimulationRuntime.from_config(PriorMapSimulationConfig.from_args(args, save_dir))


def configure_mapper_prior_map(mapper: Any, runtime: Optional[PriorMapSimulationRuntime]) -> None:
    """Attach prior-map runtime hooks to a mapper-like object.

    Args:
        mapper: Mapper-like object to annotate.
        runtime: Optional simulation prior-map runtime.
    """

    setattr(mapper, "prior_map_runtime", runtime)
    setattr(mapper, "prior_map_policy_adapter", getattr(runtime, "policy_adapter", None))
    setattr(mapper, "search_prior_result", None)
    setattr(mapper, "prior_map_prompt_context", None)
    setattr(mapper, "prior_map_last_observation", None)
    setattr(mapper, "prior_map_last_chosen_frontier", None)
    setattr(mapper, "prior_map_current_step", None)


def refresh_mapper_prior_map_query(
    mapper: Any,
    *,
    plan: Any,
    step: int,
    episode_idx: Optional[int] = None,
) -> Optional[SearchPriorResult]:
    """Refresh mapper-attached prior-map query state.

    Args:
        mapper: Mapper-like object configured by ``configure_mapper_prior_map``.
        plan: Instruction plan-like object.
        step: Planning step index.
        episode_idx: Optional episode index for artifact paths.

    Returns:
        Search prior result when prior map is enabled, otherwise ``None``.
    """

    runtime = getattr(mapper, "prior_map_runtime", None)
    if runtime is None or plan is None:
        return None
    result = runtime.update_and_query(mapper=mapper, plan=plan, step=int(step), episode_idx=episode_idx)
    setattr(mapper, "search_prior_result", result)
    setattr(mapper, "prior_map_policy_adapter", runtime.policy_adapter)
    setattr(mapper, "prior_map_prompt_context", runtime.last_prompt_context)
    setattr(mapper, "prior_map_last_observation", runtime.last_observation)
    setattr(mapper, "prior_map_current_step", int(step))
    return result


def _load_alignment(value: str, base_map: PriorMapData) -> PriorMapAlignment:
    normalized = str(value or "identity").strip()
    if normalized in {"", "identity", "auto"}:
        return PriorMapAlignment.identity(prior_frame_id=base_map.frame_id, runtime_frame_id="map")
    if normalized in {"unavailable", "disabled", "prompt_context_only"}:
        return PriorMapAlignment.unavailable(prior_frame_id=base_map.frame_id, runtime_frame_id="map")
    return PriorMapAlignment.load(normalized)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "PriorMapSimulationConfig",
    "PriorMapSimulationRuntime",
    "build_prior_map_simulation_runtime",
    "configure_mapper_prior_map",
    "refresh_mapper_prior_map_query",
]
