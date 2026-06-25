"""Real-robot prior-map runtime helpers.

This module is platform-neutral despite the name: it consumes
``SemanticMapSnapshot``-like objects by duck typing and writes diagnostics. ROS
subscription, waypoint publishing, and safety gates remain in the real-robot
adapter layer.
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
class PriorMapRealRobotConfig:
    """Configuration for real-robot prior-map integration.

    Args:
        prior_map_path: Prior map file path. Empty disables prior map.
        prior_map_source: Loader source format, or ``auto``.
        prior_map_alignment: Alignment mode or JSON path.
        run_directory: Runtime output directory.
    """

    prior_map_path: str = ""
    prior_map_source: str = "auto"
    prior_map_alignment: str = "identity"
    run_directory: str = "/tmp/strive_real_robot_runtime"

    @property
    def enabled(self) -> bool:
        """Return whether prior map should be enabled."""

        return bool(str(self.prior_map_path or "").strip())


class PriorMapRealRobotRuntime:
    """Runtime owner for real-robot prior-map memory/query artifacts.

    Args:
        base_map: Loaded prior map.
        alignment: Prior-to-runtime alignment.
        run_directory: Runtime output directory.
        query_service: Optional query service.
        prompt_builder: Optional prompt context builder.
        policy_adapter: Optional policy adapter.
    """

    def __init__(
        self,
        *,
        base_map: PriorMapData,
        alignment: PriorMapAlignment,
        run_directory: str | Path,
        query_service: Optional[PriorMapQueryService] = None,
        prompt_builder: Optional[PriorMapPromptContextBuilder] = None,
        policy_adapter: Optional[PriorMapPolicyAdapter] = None,
    ) -> None:
        """Create a real-robot prior-map runtime."""

        self.base_map = base_map
        self.alignment = alignment
        self.memory = PriorMapMemory(base_map=base_map, alignment=alignment)
        self.run_directory = Path(run_directory)
        self.artifact_dir = self.run_directory / "prior_map"
        self.query_service = query_service or PriorMapQueryService()
        self.prompt_builder = prompt_builder or PriorMapPromptContextBuilder(max_chars=4000)
        self.policy_adapter = policy_adapter or PriorMapPolicyAdapter(enabled=True)
        self.last_observation: Any = None
        self.last_query_result: Optional[SearchPriorResult] = None
        self.last_prompt_context: Optional[PromptContextBundle] = None
        self.last_step: Optional[int] = None
        self._write_static_artifacts()

    @classmethod
    def from_config(cls, config: PriorMapRealRobotConfig) -> Optional["PriorMapRealRobotRuntime"]:
        """Build runtime from config.

        Args:
            config: Real-robot prior-map config.

        Returns:
            Runtime when config is enabled, otherwise ``None``.
        """

        if not config.enabled:
            return None
        base_map = PriorMapLoader().load(config.prior_map_path, source_format=config.prior_map_source)
        alignment = _load_alignment(config.prior_map_alignment, base_map)
        return cls(base_map=base_map, alignment=alignment, run_directory=config.run_directory)

    def update_and_query(self, *, snapshot: Any, plan: Any, step: int) -> dict[str, Any]:
        """Update memory from semantic snapshot and query prior rankings.

        Args:
            snapshot: ``SemanticMapSnapshot``-like object.
            plan: Instruction plan-like object.
            step: Runtime decision step.

        Returns:
            Dictionary that can be passed as `SemanticMapSnapshotPolicyContext`
            constructor kwargs.
        """

        if self.last_step == int(step) and self.last_query_result is not None:
            return self._context_payload()

        self.last_observation = self.memory.update_from_snapshot(snapshot)
        result = self.query_service.query(plan, snapshot, self.memory)
        conflicts = detect_live_prior_conflicts(snapshot=snapshot, base_map=self.base_map)
        result.diagnostics.setdefault("authority", "ranking_only")
        result.diagnostics["live_conflicts"] = conflicts
        result.diagnostics["live_evidence_priority"] = True
        prompt_context = self.prompt_builder.build_bundle(self.memory.current_map(), result)
        self.last_query_result = result
        self.last_prompt_context = prompt_context
        self.last_step = int(step)
        self._write_step_artifacts(step=int(step), plan=plan, result=result, prompt_context=prompt_context)
        return self._context_payload()

    def _context_payload(self) -> dict[str, Any]:
        summary = self.metrics_summary()
        return {
            "prior_result": self.last_query_result,
            "prior_map_policy_adapter": self.policy_adapter,
            "prior_map_prompt_context": self.last_prompt_context,
            "prior_map_diagnostics": {
                "enabled": True,
                "scene_id": self.base_map.scene_id,
                "alignment": self.alignment.diagnostics_payload(),
                "metrics_summary": summary,
                "live_conflicts": list(
                    (self.last_query_result.diagnostics or {}).get("live_conflicts", [])
                    if self.last_query_result is not None
                    else []
                ),
                "live_evidence_priority": True,
                "authority": "ranking_only",
            },
        }

    def metrics_summary(self) -> dict[str, Any]:
        """Return compact prior-map metrics/debug summary.

        Returns:
            JSON-friendly prior-map metrics/debug payload.
        """

        return prior_map_metrics_summary(
            enabled=True,
            alignment=self.alignment,
            memory=self.memory,
            prior_result=self.last_query_result,
        )

    def _write_static_artifacts(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_prior_map_static_artifacts(output_dir=self.artifact_dir, memory=self.memory)
        _write_json(
            self.artifact_dir / "manifest.json",
            {
                "scene_id": self.base_map.scene_id,
                "source_format": self.base_map.source_format,
                "frame_id": self.base_map.frame_id,
                "authority": "ranking_only",
            },
        )

    def _write_step_artifacts(
        self,
        *,
        step: int,
        plan: Any,
        result: SearchPriorResult,
        prompt_context: PromptContextBundle,
    ) -> None:
        suffix = f"{int(step):06d}"
        query_payload = {
            "step": int(step),
            "raw_instruction": str(getattr(plan, "raw_instruction", "")),
            "dataset_target": str(getattr(plan, "dataset_target", "")),
            "observation": self.last_observation.to_dict() if hasattr(self.last_observation, "to_dict") else {},
            "authority": "ranking_only",
            "live_evidence_priority": True,
        }
        paths = write_prior_map_step_artifacts(
            output_dir=self.artifact_dir,
            step=int(step),
            memory=self.memory,
            prior_result=result,
            query_payload=query_payload,
            prompt_context_payload=prompt_context.to_dict(),
        )
        _write_json(
            self.artifact_dir / f"artifact_manifest_{suffix}.json",
            {
                "step": int(step),
                "artifacts": paths,
                "authority": "ranking_only",
            },
        )


def detect_live_prior_conflicts(*, snapshot: Any, base_map: PriorMapData) -> list[dict[str, Any]]:
    """Detect simple live/prior conflicts while keeping live evidence authoritative.

    Args:
        snapshot: Semantic snapshot-like object.
        base_map: Loaded prior map.

    Returns:
        Conflict diagnostics. These records are evidence for ranking/debug only
        and never block live observations.
    """

    conflicts: list[dict[str, Any]] = []
    for obj in getattr(snapshot, "objects", ()) or ():
        label = _norm(getattr(obj, "label", ""))
        room_id = str(getattr(obj, "room_id", "") or "")
        if not label or not room_id:
            continue
        for prior in base_map.objects:
            prior_terms = {_norm(prior.label), *(_norm(alias) for alias in prior.aliases)}
            if label not in prior_terms:
                continue
            if prior.parent_room_uid and prior.parent_room_uid != room_id:
                conflicts.append(
                    {
                        "type": "object_room_mismatch",
                        "runtime_object_uid": str(getattr(obj, "uid", "")),
                        "label": getattr(obj, "label", ""),
                        "live_room_uid": room_id,
                        "prior_object_uid": prior.uid,
                        "prior_room_uid": prior.parent_room_uid,
                        "resolution": "live_evidence_priority",
                    }
                )
    return conflicts


def build_prior_map_real_robot_runtime(config: PriorMapRealRobotConfig) -> Optional[PriorMapRealRobotRuntime]:
    """Build a real-robot prior-map runtime.

    Args:
        config: Prior-map config.

    Returns:
        Runtime when enabled, otherwise ``None``.
    """

    return PriorMapRealRobotRuntime.from_config(config)


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


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


__all__ = [
    "PriorMapRealRobotConfig",
    "PriorMapRealRobotRuntime",
    "build_prior_map_real_robot_runtime",
    "detect_live_prior_conflicts",
]
