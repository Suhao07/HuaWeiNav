"""Simulation integration helpers for prior-map mode.

The helpers here keep the benchmark entrypoint thin: they load prior-map
artifacts, update runtime memory from a mapper, query soft rankings, and write
debug JSON. They do not publish motion goals or final stop decisions.
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from .alignment import PriorMapAlignment
from .contracts import PriorMapData, PriorRoom, SearchPriorResult
from .evaluation import (
    prior_map_metrics_summary,
    write_prior_map_static_artifacts,
    write_prior_map_step_artifacts,
)
from .loaders import PriorMapLoader
from .memory import PriorMapMemory
from .multimodal import PriorMapMultimodalContext
from .room_semantics import RoomSemanticCache, RoomSemanticClassifier, room_evidence_from_record
from .policy_adapter import PriorMapPolicyAdapter
from .prompt_context import PriorMapPromptContextBuilder, PromptContextBundle
from .query import PriorMapQueryService
from .high_level_selector import (
    HighLevelCandidate, HighLevelSelectionResult,
    PriorMapHighLevelSelector,
    build_runtime_candidates,
    runtime_candidate_payloads,
)
from .visualizer import PriorMapFloorPlanVisualizer, build_floorplan_overlay


@dataclass(frozen=True)
class PriorMapSimulationConfig:
    """Configuration for prior-map simulation integration.

    Args:
        enabled: Whether prior-map integration is active.
        prior_map_path: Prior map file path.
        prior_map_layout: Optional canonical region layout in the same scene/frame.
        prior_map_source: Loader source format, or ``auto``.
        prior_map_alignment: Alignment mode or JSON path.
        artifact_root: Root directory for prior-map runtime artifacts.
        enable_high_level_vlm: Whether to invoke the optional BEV selector.
        vlm: LVLM backend name.
        high_level_interval: Minimum planning-step interval between selector calls.
        room_semantic_interval: Minimum planning-step interval between room LVLM calls.
        enable_room_semantics: Whether to invoke online room annotation.
    """

    enabled: bool = False
    enable_object_evidence: bool = False
    prior_map_path: str = ""
    prior_map_layout: str = ""
    prior_map_source: str = "auto"
    prior_map_alignment: str = "identity"
    artifact_root: str = ""
    enable_high_level_vlm: bool = False
    vlm: str = "cognav"
    high_level_interval: int = 10
    room_semantic_interval: int = 10
    enable_room_semantics: bool = False

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
            prior_map_layout=str(getattr(args, "prior_map_layout", "") or ""),
            enable_object_evidence=bool(getattr(args, "enable_prior_object_evidence", False)),
            prior_map_path=str(getattr(args, "prior_map_path", "") or ""),
            prior_map_source=str(getattr(args, "prior_map_source", "auto") or "auto"),
            prior_map_alignment=str(getattr(args, "prior_map_alignment", "identity") or "identity"),
            artifact_root=str(Path(save_dir) / "prior_map"),
            enable_high_level_vlm=bool(getattr(args, "enable_prior_map_vlm", False)),
            vlm=str(getattr(args, "vlm", "cognav") or "cognav"),
            high_level_interval=max(1, int(getattr(args, "prior_map_vlm_interval", 10) or 10)),
            room_semantic_interval=max(1, int(getattr(args, "room_semantic_interval", 10) or 10)),
            enable_room_semantics=bool(getattr(args, "enable_room_semantics", False)),
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
        high_level_selector: Optional[PriorMapHighLevelSelector] = None,
        high_level_interval: int = 10,
        room_semantic_interval: int = 10,
        room_semantic_classifier: Optional[RoomSemanticClassifier] = None,
        evidence_bundle: str | Path | None = None,
    ) -> None:
        """Create simulation runtime state."""

        self.base_map = base_map
        self.evidence_bundle = evidence_bundle
        self.alignment = alignment
        self.memory = PriorMapMemory(base_map=base_map, alignment=alignment)
        self.artifact_root = Path(artifact_root)
        self.query_service = query_service or PriorMapQueryService()
        self.prompt_builder = prompt_builder or PriorMapPromptContextBuilder(max_chars=4000)
        self.policy_adapter = policy_adapter or PriorMapPolicyAdapter(enabled=True)
        self.high_level_selector = high_level_selector
        self.high_level_interval = max(1, int(high_level_interval))
        self.room_semantic_interval = max(1, int(room_semantic_interval))
        self.room_semantic_classifier = room_semantic_classifier
        self.episode_dir: Optional[Path] = None
        self.last_observation: Any = None
        self.last_query_result: Optional[SearchPriorResult] = None
        self.last_prompt_context: Optional[PromptContextBundle] = None
        self.last_chosen_frontier: Optional[dict[str, Any]] = None
        self.last_query_step: Optional[int] = None
        self.last_multimodal_context: Optional[PriorMapMultimodalContext] = None
        self.last_high_level_selection: Any = None
        self.last_high_level_step: Optional[int] = None
        self.last_room_semantic_step: Optional[int] = None
        self.last_room_semantics: dict[str, Any] = {}
        self.shared_snapshot: dict[str, Any] = {}
        self.shared_map = base_map
        self.region_candidates: tuple[HighLevelCandidate, ...] = ()
        self.region_signature = None
        self.last_region_signature = None

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
        if config.enable_object_evidence and not config.enable_high_level_vlm:
            raise ValueError("Historical object evidence requires the high-level selector")
        if config.enable_object_evidence and not Path(config.prior_map_path).is_dir():
            raise ValueError("Historical object evidence requires a complete bundle directory")
        base_map = PriorMapLoader().load(config.prior_map_path, source_format=config.prior_map_source)
        if config.prior_map_layout:
            layout_path = Path(config.prior_map_layout)
            layout = PriorMapLoader().load(layout_path)
            if (layout.scene_id, layout.frame_id) != (base_map.scene_id, base_map.frame_id):
                raise ValueError("Region layout and object map must share scene_id and frame_id")
            base_map = replace(base_map, rooms=layout.rooms, topology_edges=layout.topology_edges,
                metadata={**base_map.metadata, "region_layout": {
                    "sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
                    "source": layout.metadata, "path": str(layout_path)}})
        alignment = _load_alignment(config.prior_map_alignment, base_map)
        selector = (
            PriorMapHighLevelSelector(vlm=config.vlm, scene_id=base_map.scene_id)
            if config.enable_high_level_vlm
            else None
        )
        room_classifier = (
            RoomSemanticClassifier(vlm=config.vlm, scene_id=base_map.scene_id)
            if config.enable_room_semantics
            else None
        )
        return cls(
            base_map=base_map,
            evidence_bundle=config.prior_map_path if config.enable_object_evidence else None,
            alignment=alignment,
            artifact_root=config.artifact_root,
            high_level_selector=selector,
            high_level_interval=config.high_level_interval,
            room_semantic_interval=config.room_semantic_interval,
            room_semantic_classifier=room_classifier,
        )

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
        self.last_multimodal_context = None
        self.last_high_level_selection = None
        self.last_high_level_step = None
        self.last_room_semantic_step = None
        self.last_room_semantics = {}
        self.shared_snapshot = {}
        self.shared_map = self.base_map
        self.region_candidates = ()
        self.region_signature = None
        self.last_region_signature = None
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
        self.last_room_semantics = self._classify_runtime_rooms(mapper, step=int(step))
        self._publish_shared_snapshot(mapper, step=int(step))
        result = self.query_service.query(plan, mapper, self.memory)
        result.diagnostics["room_semantics"] = dict(self.last_room_semantics)
        runtime_rooms = tuple(getattr(mapper, "room_nodes", ()) or getattr(mapper, "rooms", ()) or ())
        runtime_frontiers = tuple(getattr(mapper, "frontiers", ()) or ())
        if not runtime_frontiers:
            runtime_frontiers = tuple(
                node for node in (getattr(mapper, "nodes", ()) or ())
                if bool(getattr(node, "has_frontier", False) or getattr(node, "is_frontier", False))
            )
        multimodal_context = self._write_dynamic_bev(
            step=step,
            plan=plan,
            result=result,
            room_candidates=runtime_rooms,
            frontier_candidates=runtime_frontiers,
        )
        self.last_query_result = result
        self.last_query_step = int(step)
        if self.high_level_selector is not None and self._region_decision_due():
            self.select_high_level(
                plan=plan,
                result=result,
                room_candidates=runtime_rooms,
                frontier_candidates=runtime_frontiers,
            )
            # 高层选择结果只作为后续 room/frontier policy 的候选上下文，不拥有 STOP 权限。
            multimodal_context = self._write_dynamic_bev(
                step=step,
                plan=plan,
                result=result,
                room_candidates=runtime_rooms,
                frontier_candidates=runtime_frontiers,
            )
        if self.last_high_level_selection is not None:
            result.diagnostics["high_level_selection"] = self.last_high_level_selection.to_dict()
        prompt_context = self.prompt_builder.build_bundle(
            self.shared_map, result, multimodal_context=multimodal_context
        )
        self.last_prompt_context = prompt_context
        self.shared_snapshot["selection"] = (
            self.last_high_level_selection.to_dict() if self.last_high_level_selection else None)
        _write_json(self.episode_dir / f"shared_map_{step:06d}.json", self.shared_snapshot)
        self._write_step_artifacts(step=step, plan=plan, mapper=mapper, result=result, prompt_context=prompt_context)
        return result

    def finish_episode(self, *, mapper: Any, plan: Any, step: int) -> None:
        """Publish the terminal pose and complete observed trajectory without model calls.

        Goal approach can end between planning cycles. This final publication
        keeps the shared map aligned with actual execution rather than the last
        search decision. STOP authority remains with the existing verifier.
        """
        self.last_observation = self.memory.update_from_mapper(mapper, step=step)
        self._publish_shared_snapshot(mapper, step=step)
        result = self.query_service.query(plan, mapper, self.memory)
        self.last_high_level_selection = None
        self.last_chosen_frontier = None
        self.shared_snapshot["terminal"] = True
        self._write_dynamic_bev(step=step, plan=plan, result=result)
        self.shared_snapshot["selection"] = None
        _write_json(self.episode_dir / f"shared_map_{step:06d}.json", self.shared_snapshot)

    def _region_decision_due(self) -> bool:
        """Keep the selected region while its observed node membership overlaps.

        Room IDs are reassigned during segmentation. Shared nodes bind an active
        decision to its region; exhaustion or repartition triggers a new decision.
        """
        if self.last_region_signature is None:
            return True
        selected_uid = getattr(self.last_high_level_selection, "selected_uid", "")
        old_members = dict(self.last_region_signature).get(selected_uid, ())
        current_members = dict(self.region_signature).get(selected_uid, ())
        # 固定先验区域有持久ID；新frontier出现不改变区域身份。
        if current_members and self.base_map.room_by_uid(selected_uid) is not None:
            return False
        if old_members and set(old_members).intersection(current_members):
            return False
        return self.region_signature != self.last_region_signature

    def _publish_shared_snapshot(self, mapper: Any, *, step: int) -> None:
        """Publish one version for map rendering, localization and region decisions.

        Runtime rooms reuse the mapper partition. Reachability and path costs
        reuse baseline NavMesh queries. Missing room semantics remain unknown;
        centroids describe sampled regions, never fabricated wall boundaries.
        """
        from planning.exploration_policy import _geodesic_distances_to_nodes

        from matplotlib.path import Path as Polygon

        regions = []
        room_views = []
        # 先验多边形只负责区域归属；候选来自当前可达frontier，不产生虚构运动目标。
        polygons = [(room, Polygon(room.boundary_xy)) for room in self.base_map.rooms
                    if len(room.boundary_xy) >= 3]
        groups = {}
        for room in getattr(mapper, "room_nodes", ()):
            nodes = [n for n in getattr(room, "nodes", ()) if n.has_frontier and n.state == 0]
            if not nodes:
                continue
            distances = _geodesic_distances_to_nodes(mapper, nodes)
            for node, distance in zip(nodes, distances):
                if not math.isfinite(distance):
                    continue
                xyz = (float(node.position[0] + mapper.initial_position[0]),
                       -float(node.position[1] + mapper.initial_position[1]),
                       float(node.position[2] + mapper.initial_position[2]))
                xy = self.alignment.runtime_to_prior(xyz)
                matches = []
                for prior, polygon in polygons:
                    height = prior.metadata.get("region_height_range")
                    if height is not None and not height[0] <= xyz[2] - self.alignment.translation_xyz[2] <= height[1]:
                        continue
                    if polygon.contains_point(xy):
                        matches.append(prior)
                # 多边形重叠或缺失时保留在线区域，绝不以最近中心代替包含关系。
                prior = matches[0] if len(matches) == 1 else None
                uid = prior.uid if prior is not None else f"runtime_room:{room.room_id}"
                group = groups.setdefault(uid, {"prior": prior, "room": room, "members": []})
                group["members"].append((node, float(distance), xy))
        for uid, group in groups.items():
            prior, room, members = group["prior"], group["room"], group["members"]
            center = tuple(sum(item[2][k] for item in members) / len(members) for k in (0, 1))
            semantic = self.last_room_semantics.get(str(room.room_id), {})
            label = prior.label if prior is not None else str(semantic.get("label", getattr(room, "label", "unknown")))
            node_ids = tuple(sorted(int(item[0].idx) for item in members))
            source = "prior_polygon_binding" if prior is not None else "runtime_room_partition"
            evidence = self.base_map.metadata.get("region_observation_summaries", {}).get(uid, {})
            regions.append(HighLevelCandidate(
                uid=uid, room_uid=uid, label=label,
                distance_m=min(item[1] for item in members),
                objects=tuple(sorted(set(evidence.get("label_frame_counts", {})) |
                    {obj.label for obj in self.base_map.objects if obj.parent_room_uid == uid})),
                metadata={"node_ids": node_ids, "position": center, "source": source,
                          "observed_from_region": evidence},
            ))
            if prior is None:
                room_views.append(PriorRoom(uid=uid, label=label, centroid_xy=center,
                    source=source, metadata={"node_ids": node_ids,
                    "semantic_status": "unknown" if label == "unknown" else "inferred"}))
        self.region_candidates = tuple(regions)
        # 节点成员绑定区域身份；重分区或区域耗尽使旧选择失效。普通位姿变化不重新请求模型。
        self.region_signature = tuple((r.uid, r.metadata["node_ids"]) for r in regions)
        current = self.memory.current_map()
        self.shared_map = replace(current, rooms=current.rooms + tuple(room_views))
        observation = self.last_observation
        pose = self.alignment.runtime_to_prior(observation.pose_xyz) if observation.pose_xyz else None
        heading = None
        if hasattr(mapper, "current_rotation"):
            rotation = mapper.current_rotation
            # 相机前向为-R[:,2]；STRIVE(x,z,y)转为地图(x,-z,y)，再应用平面对齐。
            heading = math.atan2(float(rotation[1, 2]), -float(rotation[0, 2])) - self.alignment.rotation_rad
        if hasattr(mapper, "trajectory_position"):
            # 复用每个实际感知步保存的轨迹，避免只显示高层决策点之间的直线。
            trajectory = [list(self.alignment.runtime_to_prior((
                float(p[0] + mapper.initial_position[0]),
                -float(p[1] + mapper.initial_position[1]),
                float(p[2] + mapper.initial_position[2]),
            ))) for p in mapper.trajectory_position]
        else:
            trajectory = [list(self.alignment.runtime_to_prior(o.pose_xyz))
                          for o in self.memory.observations if o.pose_xyz is not None]
        self.shared_snapshot = {
            "map_id": self.base_map.metadata.get("map_id", self.base_map.scene_id),
            "revision": len(self.memory.observations), "step": step,
            "alignment_revision": hashlib.sha256(json.dumps(self.alignment.to_dict(), sort_keys=True).encode()).hexdigest()[:16],
            "alignment": self.alignment.to_dict(),
            "frame_id": self.base_map.frame_id,
            "pose": {"position_xy": pose, "heading_rad": heading,
                     "source": "habitat_ground_truth_pose", "covariance": None},
            "trajectory_xy": trajectory,
            "regions": [r.to_dict() for r in regions],
            "map": self.shared_map.to_dict(),
        }
        _write_json(self.episode_dir / f"shared_map_{step:06d}.json", self.shared_snapshot)

    def _classify_runtime_rooms(self, mapper: Any, *, step: int) -> dict[str, Any]:
        """Annotate rooms when a new local RGB evidence version is available."""

        if self.room_semantic_classifier is None:
            return {}
        if (
            self.last_room_semantic_step is not None
            and int(step) - int(self.last_room_semantic_step) < self.room_semantic_interval
        ):
            return dict(self.last_room_semantics)
        output: dict[str, Any] = {}
        rooms = list(getattr(mapper, "room_nodes", ()) or getattr(mapper, "rooms", ()) or ())
        for room in rooms:
            evidence = room_evidence_from_record(
                room,
                fallback_rgb_path=str(getattr(mapper, "current_rgb_path", "") or ""),
                pose=tuple(float(item) for item in getattr(mapper, "current_position", ()) or ()),
                source="simulation_room_observation",
            )
            output[evidence.room_uid] = self.room_semantic_classifier.classify(evidence).to_dict()
        self.last_room_semantic_step = int(step)
        return output

    def select_high_level(
        self,
        *,
        plan: Any,
        result: Optional[SearchPriorResult] = None,
        room_candidates: tuple[Any, ...] = (),
        frontier_candidates: tuple[Any, ...] = (),
    ) -> Any:
        """Select an existing room candidate with the optional dynamic-BEV LVLM service.

        Args:
            plan: Current instruction plan.
            result: Search result; defaults to the last query result.

        Returns:
            High-level selection result, or ``None`` when the service is disabled.
        """

        if self.high_level_selector is None or self.last_multimodal_context is None:
            return None
        result = result or self.last_query_result
        if result is None:
            return None
        # 只提交有当前可达frontier的运行时区域；历史房间记录保留为上下文。
        candidates = self.region_candidates
        self.last_region_signature = self.region_signature
        if len(candidates) <= 1:
            self.last_high_level_selection = HighLevelSelectionResult(
                selected_uid=candidates[0].uid if candidates else "",
                selected_type="room" if candidates else "",
                decision="select" if candidates else "uncertain",
                reason="single executable region" if candidates else "no executable region",
                source="deterministic",
            )
            return self.last_high_level_selection
        evidence = ()
        if self.evidence_bundle is not None:
            from .query_binding import historical_object_evidence
            evidence = historical_object_evidence(self.evidence_bundle, plan, result,
                self.episode_dir / f"historical_evidence_{self.last_query_step:06d}")
        selection = self.high_level_selector.select(
            evidence_contexts=evidence,
            instruction=str(getattr(plan, "raw_instruction", "") or ""),
            instruction_plan=plan,
            context=self.last_multimodal_context,
            candidates=candidates,
            runtime_state={
                "step": self.last_query_step,
                "authority": "ranking_only",
                "candidate_types": sorted({candidate.candidate_type for candidate in candidates}),
            },
        )
        self.last_high_level_selection = selection
        self.last_high_level_step = self.last_query_step
        if self.last_query_result is not None:
            self.last_query_result.diagnostics["high_level_selection"] = selection.to_dict()
        return selection

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
            "room_semantics": self.last_room_semantics,
            **dict(payload),
        }
        _write_json(output, enriched)
        overlay = build_floorplan_overlay(
            self.memory.base_map,
            prior_result=self.last_query_result,
            chosen_frontier=enriched,
            observations=self.memory.observations,
            object_states=self.memory.object_states,
            robot_position_xyz=getattr(self.last_observation, "pose_xyz", None),
            current_room_uid=getattr(self.last_observation, "room_hypothesis_uid", None),
            candidate_room_uids=tuple(
                item.room_uid for item in getattr(self.last_query_result, "room_rankings", ()) or ()
            ),
            selected_room_uid=getattr(self.last_high_level_selection, "selected_uid", None),
            alignment=self.alignment,
        )
        floorplan_paths = PriorMapFloorPlanVisualizer().write_global_artifacts(
            self.memory.base_map,
            self.episode_dir,
            stem=f"floorplan_chosen_frontier_{step_value:06d}",
            overlay=overlay,
        )
        _write_json(
            self.episode_dir / f"floorplan_chosen_frontier_{step_value:06d}_manifest.json",
            {
                "step": step_value,
                "authority": "diagnostic_only",
                "chosen_frontier": str(output),
                "floorplan": floorplan_paths,
            },
        )
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
            "room_semantic_artifact": f"room_semantics_{suffix}.json",
            "high_level_selection_artifact": (
                f"high_level_selection_{suffix}.json"
                if self.last_high_level_selection is not None else None
            ),
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
        _write_json(
            self.episode_dir / f"room_semantics_{suffix}.json",
            {
                "step": int(step),
                "rooms": dict(self.last_room_semantics),
                "authority": "semantic_annotation_only",
            },
        )
        if self.last_high_level_selection is not None:
            _write_json(
                self.episode_dir / f"high_level_selection_{suffix}.json",
                self.last_high_level_selection.to_dict(),
            )

    def _write_dynamic_bev(
        self,
        *,
        step: int,
        plan: Any,
        result: SearchPriorResult,
        room_candidates: tuple[Any, ...] = (),
        frontier_candidates: tuple[Any, ...] = (),
    ) -> PriorMapMultimodalContext:
        """Render the current map state and return its LVLM context."""

        assert self.episode_dir is not None
        chosen_frontier = dict(self.last_chosen_frontier or {})
        if (
            self.last_high_level_selection is not None
            and getattr(self.last_high_level_selection, "selected_type", "") == "frontier"
        ):
            chosen_frontier["selected_uid"] = self.last_high_level_selection.selected_uid
        runtime_candidates = build_runtime_candidates(
            rooms=room_candidates,
            frontiers=frontier_candidates,
            room_rankings=result.room_rankings,
            frontier_biases=result.frontier_biases,
        )
        if not chosen_frontier.get("candidates"):
            chosen_frontier["candidates"] = [
                {
                    "uid": candidate.uid,
                    "frontier_uid": candidate.uid,
                    "room_uid": candidate.room_uid,
                    "position": candidate.metadata.get("position"),
                }
                for candidate in runtime_candidates
                if candidate.candidate_type == "frontier"
            ]
        overlay = build_floorplan_overlay(
            self.shared_map,
            prior_result=result,
            chosen_frontier=chosen_frontier or None,
            observations=self.memory.observations,
            object_states=self.memory.object_states,
            robot_position_xyz=getattr(self.last_observation, "pose_xyz", None),
            current_room_uid=getattr(self.last_observation, "room_hypothesis_uid", None),
            candidate_room_uids=tuple(
                str(getattr(room, "uid", "") or getattr(room, "room_uid", ""))
                for room in room_candidates
                if str(getattr(room, "uid", "") or getattr(room, "room_uid", ""))
            ) or tuple(item.room_uid for item in result.room_rankings),
            selected_room_uid=getattr(self.last_high_level_selection, "selected_uid", None),
            alignment=self.alignment,
        )
        overlay = replace(overlay,
            robot_heading_rad=self.shared_snapshot.get("pose", {}).get("heading_rad"),
            trajectory_xy=tuple(tuple(p[:2]) for p in self.shared_snapshot.get("trajectory_xy", ())),
            candidate_room_uids=tuple(r.uid for r in self.region_candidates),
            metadata={**overlay.metadata, "revision": self.shared_snapshot.get("revision"),
                      "map_id": self.shared_snapshot.get("map_id"),
                      "alignment_revision": self.shared_snapshot.get("alignment_revision")})
        paths = PriorMapFloorPlanVisualizer().write_global_artifacts(
            self.shared_map,
            self.episode_dir,
            stem=f"dynamic_prior_map_bev_{int(step):06d}",
            overlay=overlay,
        )
        context = PriorMapMultimodalContext(
            image_path=paths["png"],
            image_role="dynamic_bev",
            map_frame_id=self.base_map.frame_id,
            alignment_status=str(self.alignment.diagnostics_payload().get("fallback", "aligned")),
            text_context=json.dumps(
                {
                    "instruction": str(getattr(plan, "raw_instruction", "") or ""),
                    "room_rankings": [
                        {"uid": item.room_uid, "label": item.label, "score": item.score}
                        for item in result.room_rankings[:8]
                    ],
                    "room_semantics": self.last_room_semantics,
                    "candidate_types": ["room"] if self.region_candidates else [],
                    "candidates": [r.to_dict() for r in self.region_candidates],
                    "snapshot_revision": self.shared_snapshot.get("revision"),
                    "pose": self.shared_snapshot.get("pose"),
                    "authority": "high_level_room_selection_only",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            metadata={"step": int(step), "artifact_paths": paths, "authority": "ranking_only"},
        )
        self.last_multimodal_context = context
        return context


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
    setattr(mapper, "prior_map_high_level_selection", None)
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
    setattr(
        mapper,
        "prior_map_high_level_selection",
        runtime.last_high_level_selection.to_dict()
        if hasattr(runtime.last_high_level_selection, "to_dict")
        else runtime.last_high_level_selection,
    )
    setattr(mapper, "prior_map_current_step", int(step))
    return result


def _load_alignment(value: str, base_map: PriorMapData) -> PriorMapAlignment:
    normalized = str(value or "identity").strip()
    if normalized in {"", "identity", "auto"}:
        # Canonical capture maps use global (x,-z,y); frontier extraction converts
        # Habitat/world or mapper-local coordinates into this plane before alignment.
        runtime_frame = "map_z_up" if base_map.frame_id == "map_z_up" else "map"
        return PriorMapAlignment.identity(prior_frame_id=base_map.frame_id, runtime_frame_id=runtime_frame)
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
