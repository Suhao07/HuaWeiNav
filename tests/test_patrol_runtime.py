"""Patrol lifecycle checks using the actual shared motion supervisor."""
from types import SimpleNamespace
import json

import pytest

from backends.habitat.motion import HabitatExecution
from backends.habitat import patrol_runtime


def _agent(tmp_path, monkeypatch, actions):
    env = SimpleNamespace(episode_over=False)
    executed, visited, scanned = [], [], []
    env.step = lambda action: executed.append(action) or {}
    execution = HabitatExecution(env)
    node = SimpleNamespace(idx=1, position=(1., 0., 0.))
    mapper = SimpleNamespace(traj=[0], change_state=lambda item: visited.append(item.idx))
    agent = SimpleNamespace(patrol=True, task=None, save_dir=str(tmp_path), execution=execution,
        env=env, mapper=mapper, episode_steps=1, current_node_idx=0, travel_distance=0.)
    def update(on_node_flag):
        agent.episode_steps += 1
    agent.update_trajectory = update
    def observe(agent, *, node, episode_idx):
        scanned.append(None if node is None else node.idx)
        agent.episode_steps += 12
        return SimpleNamespace(episode_over=False)
    selections = iter([(node, "current_room"), (None, "remaining_nodes")])
    monkeypatch.setattr(patrol_runtime, "run_observation_mapping_cycle", observe)
    monkeypatch.setattr(patrol_runtime, "select_exploration_node", lambda *a, **k: next(selections))
    commands = iter(actions)
    monkeypatch.setattr("navigation.action_controller.next_action_to_waypoint",
                        lambda *a, **k: next(commands))
    return agent, executed, visited, scanned


def test_patrol_marks_coverage_after_arrival_and_observation(tmp_path, monkeypatch):
    agent, executed, visited, scanned = _agent(tmp_path, monkeypatch, [1, 0])
    receipt = patrol_runtime.run_patrol(agent, episode_idx=0, max_steps=40)
    assert receipt["termination"] == "frontiers_exhausted"
    assert receipt["complete_panoramas"] == 2
    assert scanned == [None, 1] and visited == [1]
    assert executed == [1]  # waypoint completion produces no task STOP.
    assert receipt["task_complete"] is None
    assert agent.execution.motion.active_request_id is None


def test_patrol_budget_cancels_motion_and_preserves_unobserved_node(tmp_path, monkeypatch):
    agent, executed, visited, scanned = _agent(tmp_path, monkeypatch, [1, 1])
    receipt = patrol_runtime.run_patrol(agent, episode_idx=0, max_steps=14)
    assert receipt["termination"] == "step_budget"
    assert executed == [1] and visited == [] and scanned == [None]
    assert agent.execution.motion.active_request_id is None
    assert agent.execution.controller.waypoint_goal is None
    events = [json.loads(line) for line in
              (tmp_path / "episode-0/patrol/decisions.jsonl").read_text().splitlines()]
    assert events[-1]["status"] == "preempted"
    assert events[-1]["reason"] == "patrol_exit"


def test_patrol_motion_failure_is_explicit(tmp_path, monkeypatch):
    agent, executed, visited, _ = _agent(tmp_path, monkeypatch, [None])
    receipt = patrol_runtime.run_patrol(agent, episode_idx=0, max_steps=40)
    assert receipt["termination"] == "motion_blocked"
    assert executed == [] and visited == []


def test_patrol_waits_for_full_scan_budget(tmp_path, monkeypatch):
    agent, executed, visited, scanned = _agent(tmp_path, monkeypatch, [])
    receipt = patrol_runtime.run_patrol(agent, episode_idx=0, max_steps=12)
    assert receipt["complete_panoramas"] == 0
    assert executed == [] and visited == [] and scanned == []


def test_patrol_rejects_task_bearing_agent(tmp_path, monkeypatch):
    agent, _, _, _ = _agent(tmp_path, monkeypatch, [])
    agent.task = object()
    with pytest.raises(ValueError, match="task-free reset"):
        patrol_runtime.run_patrol(agent, episode_idx=0, max_steps=40)
