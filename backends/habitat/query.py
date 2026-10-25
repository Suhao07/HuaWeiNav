"""Habitat binding of the shared scene query to existing geometry execution."""
from __future__ import annotations

import json
from pathlib import Path

from backends.habitat.scene import snapshot_from_mapper
from prior_map.simulation import refresh_mapper_prior_map_query
from harness.tools import query_scene


def select_observed_candidate(agent, *, episode_idx=None, step=0):
    """Query native snapshots and return the existing object for local geometry."""
    session, mapper = agent.task, agent.mapper
    snapshot, objects = snapshot_from_mapper(mapper, step=step)
    session.snapshot = snapshot
    active_id = session.progress.active_target().id
    active = next(t for t in session.plan.terminal_targets if t.id == active_id)
    session.target = active.name
    if active is not None and session.plan.execution.ordered:
        mapper.target = active.name
        mapper.target_list = list(active.detector_terms or [active.name])
        mapper.target_aliases = list(active.match_terms)
    refresh_mapper_prior_map_query(mapper, plan=session.plan, step=step, episode_idx=episode_idx)
    session.constraint_evaluator.register_mapper_objects(mapper=session, step=step,
                                                        current_node_idx=mapper.current_node_idx)
    session.prior_map_policy_adapter = mapper.prior_map_policy_adapter
    session.prior_result = mapper.search_prior_result
    result = query_scene(session, step=step)
    selected = result.selected
    agent.selected_candidate = selected
    if selected is not None and selected.role == "anchor":
        session.anchor_search_ledger.mark(
            raw_instruction=session.plan.raw_instruction, concept_id=selected.target_id,
            anchor_uid=selected.obj.uid, status="navigating_to_anchor", step=step,
            reason=result.reason,
        )
    output = Path(agent.save_dir) / f"episode-{episode_idx}" / "scene_query"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"query_{step}.json").write_text(json.dumps({
        "reason": result.reason, "selected_uid": selected.obj.uid if selected else None,
        "target_id": selected.target_id if selected else None,
        "role": selected.role if selected else None,
        "skipped": result.skipped_objs, "concept_matches": result.concept_debug,
    }, ensure_ascii=False, indent=2))
    return selected is not None, objects[selected.obj.uid] if selected else None
