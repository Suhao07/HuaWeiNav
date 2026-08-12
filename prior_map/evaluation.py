"""Evaluation and artifact summaries for prior-map mode.

The functions here summarize prior-map context for metrics/debug artifacts.
They only report ranking and failure diagnostics; they do not create navigation
goals or stop authority.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .alignment import PriorMapAlignment
from .contracts import SearchPriorResult
from .memory import PriorMapMemory
from .visualizer import (
    PriorMapFloorPlanVisualizer,
    build_floorplan_overlay,
    write_floorplan_artifacts,
    write_som_artifacts,
)


def prior_map_metrics_summary(
    *,
    enabled: bool,
    alignment: Optional[PriorMapAlignment] = None,
    memory: Optional[PriorMapMemory] = None,
    prior_result: Optional[SearchPriorResult] = None,
) -> dict[str, Any]:
    """Return a compact prior-map metrics/debug summary.

    Args:
        enabled: Whether prior map was enabled.
        alignment: Optional alignment used by the run.
        memory: Optional runtime memory.
        prior_result: Optional latest query result.

    Returns:
        JSON-friendly metrics payload with top rankings and failure counts.
    """

    top_room = _top_room(prior_result)
    top_object = _top_object(prior_result)
    alignment_payload = alignment.diagnostics_payload() if alignment is not None else {}
    return {
        "enabled": bool(enabled),
        "top_room": top_room,
        "top_object": top_object,
        "alignment": alignment_payload,
        "alignment_confidence": float(alignment_payload.get("confidence", 0.0) or 0.0),
        "failure_modes": prior_map_failure_modes(
            alignment=alignment,
            memory=memory,
            prior_result=prior_result,
        ),
        "authority": "ranking_only",
    }


def prior_map_metrics_fields(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a prior-map summary for CSV metrics rows.

    Args:
        summary: Payload from ``prior_map_metrics_summary``.

    Returns:
        Flat dictionary suitable for appending to benchmark metrics.
    """

    top_room = dict(summary.get("top_room") or {})
    top_object = dict(summary.get("top_object") or {})
    failure_modes = dict(summary.get("failure_modes") or {})
    return {
        "prior_map_enabled": bool(summary.get("enabled", False)),
        "prior_map_top_room_uid": top_room.get("uid", ""),
        "prior_map_top_room_label": top_room.get("label", ""),
        "prior_map_top_room_score": top_room.get("score", ""),
        "prior_map_top_object_uid": top_object.get("uid", ""),
        "prior_map_top_object_label": top_object.get("label", ""),
        "prior_map_top_object_score": top_object.get("score", ""),
        "prior_map_alignment_confidence": summary.get("alignment_confidence", 0.0),
        "prior_map_failure_wrong_prior": failure_modes.get("wrong_prior", 0),
        "prior_map_failure_alignment_mismatch": failure_modes.get("alignment_mismatch", 0),
        "prior_map_failure_live_conflict": failure_modes.get("live_conflict", 0),
        "prior_map_failure_prior_exhausted": failure_modes.get("prior_exhausted", 0),
    }


def prior_map_failure_modes(
    *,
    alignment: Optional[PriorMapAlignment],
    memory: Optional[PriorMapMemory],
    prior_result: Optional[SearchPriorResult],
) -> dict[str, Any]:
    """Compute prior-map failure mode counters.

    Args:
        alignment: Optional alignment used by runtime.
        memory: Optional prior-map memory.
        prior_result: Optional latest query result.

    Returns:
        Counters for wrong prior, alignment mismatch, live conflict, and prior
        exhausted cases.
    """

    wrong_prior = 0
    prior_exhausted = 0
    rejected_priors: list[str] = []
    exhausted_rooms: list[str] = []
    if memory is not None:
        for uid, state in memory.object_states.items():
            if getattr(state, "rejection_count", 0) > 0 or getattr(state, "rejected", False):
                wrong_prior += 1
                rejected_priors.append(uid)
        for uid, state in memory.room_states.items():
            if bool(getattr(state, "metadata", {}).get("exhausted", False)):
                prior_exhausted += 1
                exhausted_rooms.append(uid)

    alignment_mismatch = 0
    if alignment is not None and not alignment.can_rank_geometry():
        alignment_mismatch = 1

    live_conflicts = list((getattr(prior_result, "diagnostics", {}) or {}).get("live_conflicts", []))
    return {
        "wrong_prior": wrong_prior,
        "alignment_mismatch": alignment_mismatch,
        "live_conflict": len(live_conflicts),
        "prior_exhausted": prior_exhausted,
        "rejected_prior_uids": rejected_priors,
        "exhausted_room_uids": exhausted_rooms,
        "live_conflicts": live_conflicts,
        "authority": "ranking_only",
    }


def write_prior_map_static_artifacts(
    *,
    output_dir: str | Path,
    memory: PriorMapMemory,
    max_room_views: int = 4,
) -> dict[str, Any]:
    """Write static map/evaluation artifacts.

    Args:
        output_dir: Prior-map artifact directory.
        memory: Runtime memory containing base map and alignment.
        max_room_views: Maximum room SoM views to write.

    Returns:
        Artifact path summary.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_map_path = output / "base_map.json"
    alignment_path = output / "alignment.json"
    base_map_path.write_text(
        json.dumps(memory.base_map.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    alignment_path.write_text(
        json.dumps(memory.alignment.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    som = write_som_artifacts(memory.base_map, output, max_room_views=max_room_views)
    floorplan = write_floorplan_artifacts(memory.base_map, output, max_room_views=max_room_views)
    marker_manifest = {
        "global": som["global"].get("markers"),
        "rooms": {room_uid: paths.get("markers") for room_uid, paths in som.get("rooms", {}).items()},
        "floorplan_global": floorplan["global"].get("markers"),
        "floorplan_rooms": {room_uid: paths.get("markers") for room_uid, paths in floorplan.get("rooms", {}).items()},
    }
    (output / "som_markers_manifest.json").write_text(
        json.dumps(marker_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "base_map": str(base_map_path),
        "alignment": str(alignment_path),
        "som": som,
        "floorplan": floorplan,
        "marker_manifest": str(output / "som_markers_manifest.json"),
    }


def write_prior_map_step_artifacts(
    *,
    output_dir: str | Path,
    step: int,
    memory: PriorMapMemory,
    prior_result: SearchPriorResult,
    query_payload: dict[str, Any],
    prompt_context_payload: dict[str, Any],
) -> dict[str, str]:
    """Write per-step prior-map artifacts.

    Args:
        output_dir: Prior-map artifact directory.
        step: Runtime step index.
        memory: Runtime memory.
        prior_result: Latest query result.
        query_payload: Query input/debug payload.
        prompt_context_payload: Prompt context payload.

    Returns:
        Dictionary of written artifact paths.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"{int(step):06d}"
    paths = {
        "query": output / f"query_{suffix}.json",
        "search_prior_result": output / f"search_prior_result_{suffix}.json",
        "prompt_context": output / f"prompt_context_{suffix}.json",
        "runtime_state": output / f"runtime_state_{suffix}.json",
        "debug": output / f"debug_{suffix}.json",
        "failure_modes": output / f"failure_modes_{suffix}.json",
    }
    summary = prior_map_metrics_summary(
        enabled=True,
        alignment=memory.alignment,
        memory=memory,
        prior_result=prior_result,
    )
    failure_modes = dict(summary.get("failure_modes") or {})
    payloads = {
        paths["query"]: query_payload,
        paths["search_prior_result"]: prior_result.to_dict(),
        paths["prompt_context"]: prompt_context_payload,
        paths["runtime_state"]: memory.state_dict(),
        paths["debug"]: summary,
        paths["failure_modes"]: failure_modes,
    }
    for path, payload in payloads.items():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    overlay = build_floorplan_overlay(
        memory.base_map,
        prior_result=prior_result,
        observations=memory.observations,
        object_states=memory.object_states,
    )
    floorplan_paths = PriorMapFloorPlanVisualizer().write_global_artifacts(
        memory.base_map,
        output,
        stem=f"floorplan_step_{suffix}",
        overlay=overlay,
    )
    renamed = {f"floorplan_step_{key}": Path(value) for key, value in floorplan_paths.items()}
    return {name: str(path) for name, path in {**paths, **renamed}.items()}


def _top_room(prior_result: Optional[SearchPriorResult]) -> dict[str, Any]:
    if prior_result is None or not prior_result.room_rankings:
        return {}
    item = prior_result.room_rankings[0]
    return {"uid": item.room_uid, "label": item.label, "score": item.score, "reason": item.reason}


def _top_object(prior_result: Optional[SearchPriorResult]) -> dict[str, Any]:
    if prior_result is None or not prior_result.object_rankings:
        return {}
    item = prior_result.object_rankings[0]
    return {"uid": item.object_uid, "label": item.label, "score": item.score, "reason": item.reason}


__all__ = [
    "prior_map_failure_modes",
    "prior_map_metrics_fields",
    "prior_map_metrics_summary",
    "write_prior_map_static_artifacts",
    "write_prior_map_step_artifacts",
]
