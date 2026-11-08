"""GOAT language base-motion evidence and deterministic offline viewpoint scoring."""
import argparse
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

from .episode_selection import episode_identity


_SCHEMA = "goat_language_viewpoints_v1"
_POSITION_TOLERANCE = 1e-5


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return float(value)


def _vector(value, size=3):
    result = [_number(item) for item in value]
    if len(result) != size:
        raise ValueError("Invalid position or rotation dimension")
    return result


def _distance(value):
    if value is None:
        return None
    result = _number(value)
    if result < 0:
        raise ValueError("Negative geodesic distance")
    return result


def _minimum(paths):
    distances = [path["distance"] for path in paths if path["distance"] is not None]
    return min(distances) if distances else None


def _continuous(before, previous):
    # 仅容忍仿真浮点噪声，不修补或投影轨迹。
    if any(abs(a - b) > _POSITION_TOLERANCE for a, b in zip(before, previous)):
        raise ValueError("Base position continuity differs from initial/previous position")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def strict_json_loads(text):
    """Decode evidence JSON, rejecting nonfinite constants and overflowing exponents."""
    return json.loads(text, parse_constant=_reject_constant,
                      parse_float=lambda value: _number(float(value)))


def _pose(state):
    rotation = state.rotation
    if hasattr(rotation, "imag") and isinstance(rotation.real, Real):
        rotation = [*rotation.imag, rotation.real]
    return {"position": _vector(state.position), "rotation": _vector(rotation, 4)}


def _settings(pathfinder):
    settings = pathfinder.nav_mesh_settings
    values = {}
    for name in dir(settings):
        if name.startswith("_"):
            continue
        value = getattr(settings, name)
        if isinstance(value, bool):
            values[name] = value
        elif isinstance(value, Real):
            values[name] = _number(value)
    if not values:
        raise ValueError("Missing runtime NavMesh settings")
    return values


def _viewpoints(goals):
    for goal_index, goal in enumerate(goals):
        for viewpoint_index, position in enumerate(goal["view_points"]):
            yield goal_index, viewpoint_index, _vector(position)


def _load_pathfinder(mesh):
    """Load the saved runtime mesh into a new, simulator-independent pathfinder."""
    import habitat_sim

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(mesh)) or not pathfinder.is_loaded:
        raise ValueError("Cannot load frozen runtime NavMesh")
    return pathfinder


def _paths(pathfinder, goals, start):
    """Query every frozen viewpoint from a base position on the supplied mesh."""
    import habitat_sim

    paths = []
    for goal_index, viewpoint_index, end in _viewpoints(goals):
        query = habitat_sim.ShortestPath()
        query.requested_start, query.requested_end = start, end
        found = pathfinder.find_path(query)
        distance = _distance(query.geodesic_distance) if found else None
        points = [_vector(point) for point in query.points] if found else []
        if found and (distance is None or not points):
            raise ValueError("Reachable viewpoint lacks distance or path points")
        paths.append({"goal_index": goal_index, "viewpoint_index": viewpoint_index,
                      "distance": distance, "path_points": points})
    return paths


class HabitatEvaluationTrace:
    """Freeze GOAT viewpoint GT and record each owned simulator step exactly once."""

    def __init__(self, env, episode_dir, *, provenance=None):
        import habitat_sim

        self.env = env
        self._runtime_pathfinder = env.sim.pathfinder
        if not self._runtime_pathfinder.is_loaded:
            raise ValueError("Runtime NavMesh is not loaded")
        settings = _settings(self._runtime_pathfinder)
        episode = env.current_episode
        initial = _pose(env.sim.get_agent_state())
        self._goals = [{"object_id": str(goal.object_id),
                        "view_points": [_vector(view.agent_state.position)
                                        for view in goal.view_points]}
                       for goal in episode.goals]
        self._previous = initial["position"]
        self._step_index = 0
        self._stopped = False
        self.directory = Path(episode_dir) / "evaluation"
        self.directory.mkdir(exist_ok=False)
        mesh = self.directory / "runtime.navmesh"
        if not self._runtime_pathfinder.save_nav_mesh(str(mesh)):
            raise RuntimeError("Cannot save runtime NavMesh")
        self.pathfinder = _load_pathfinder(mesh)
        paths = _paths(self.pathfinder, self._goals, initial["position"])
        reference = {
            "schema": _SCHEMA, "benchmark": "goat_language",
            "episode_identity": episode_identity(episode),
            "episode_info": dict(episode.info or {}),
            "input_provenance": dict(provenance or {}),
            "coordinates": "habitat_world_xyz_metres", "rotation_order": "xyzw",
            "pose_source": "env.sim.get_agent_state().position",
            "initial_base_pose": initial, "goals": self._goals,
            "navmesh": {"file": mesh.name, "sha256": _sha256(mesh),
                        "settings": settings,
                        "settings_source": "runtime_pathfinder.nav_mesh_settings",
                        "build_profile_verified": False},
            "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
            "viewpoint_paths": paths, "reference_distance": _minimum(paths),
            "success_distance": 0.25, "success_rule": "terminal_STOP_and_distance_lt",
        }
        (self.directory / "reference.json").write_text(_json(reference) + "\n")
        with (self.directory / "trajectory.jsonl").open("x"):
            pass

    def step(self, action):
        """Record base XYZ around the same env.step and return its original observation."""
        if isinstance(action, bool) or not isinstance(action, Integral) or action < 0:
            raise ValueError("Expected an integer Habitat action")
        if self._stopped:
            raise ValueError("Cannot record actions after terminal STOP")
        if self.env.sim.pathfinder is not self._runtime_pathfinder:
            raise ValueError("Runtime NavMesh pathfinder changed")
        before = _vector(self.env.sim.get_agent_state().position)
        _continuous(before, self._previous)
        observation = self.env.step(action)
        after = _vector(self.env.sim.get_agent_state().position)
        record = {"step": self._step_index, "action": int(action),
                  "before": before, "after": after}
        if action == 0:
            paths = _paths(self.pathfinder, self._goals, after)
            record.update(final_viewpoint_paths=paths, final_distance=_minimum(paths))
        with (self.directory / "trajectory.jsonl").open("a") as stream:
            stream.write(_json(record) + "\n")
        self._previous = after
        self._step_index += 1
        self._stopped = action == 0
        return observation


def _validate_paths(paths, recomputed, minimum):
    """Check receipts against fresh mesh queries, allowing float32-scale distance noise."""
    if len(paths) != len(recomputed):
        raise ValueError("Incomplete viewpoint paths")
    for path, fresh in zip(paths, recomputed):
        if ((path["goal_index"], path["viewpoint_index"])
                != (fresh["goal_index"], fresh["viewpoint_index"])):
            raise ValueError("Viewpoint path identity differs")
        distance = _distance(path["distance"])
        points = [_vector(point) for point in path["path_points"]]
        if (distance is None and points) or (distance is not None and not points):
            raise ValueError("Viewpoint path reachability differs")
        if (distance is None) != (fresh["distance"] is None):
            raise ValueError("Viewpoint reachability differs from frozen NavMesh")
        if distance is not None and not math.isclose(
                distance, fresh["distance"], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Viewpoint distance differs from frozen NavMesh")
    if _distance(minimum) != _minimum(paths):
        raise ValueError("Viewpoint minimum distance differs")


def score_habitat_evaluation(episode_dir, *, expected_identity=None):
    """Reload the frozen NavMesh and compute independent SR and SPL=SR*L/max(L,P)."""
    directory = Path(episode_dir) / "evaluation"
    provenance = {"schema": _SCHEMA, "evaluation_directory": str(directory)}

    def unavailable(reason):
        return {"status": "unavailable", "reason": reason, "provenance": provenance}

    required = ("reference.json", "trajectory.jsonl", "runtime.navmesh")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return unavailable("missing_evaluation_files: " + ", ".join(missing))
    reference = strict_json_loads((directory / "reference.json").read_text())
    if (reference["schema"] != _SCHEMA or reference["benchmark"] != "goat_language"
            or reference["success_distance"] != 0.25
            or reference["success_rule"] != "terminal_STOP_and_distance_lt"
            or reference["coordinates"] != "habitat_world_xyz_metres"
            or reference["rotation_order"] != "xyzw"
            or reference["pose_source"] != "env.sim.get_agent_state().position"):
        raise ValueError("Independent evaluation protocol differs")
    identity = reference["episode_identity"]
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Independent evaluation episode identity differs")
    _vector(identity["start_position"])
    _vector(identity["start_rotation"], 4)
    mesh = reference["navmesh"]
    if mesh["file"] != "runtime.navmesh" or _sha256(directory / "runtime.navmesh") != mesh["sha256"]:
        raise ValueError("Runtime NavMesh hash differs")
    provenance.update(episode_identity=identity, episode_info=reference["episode_info"],
                      input_provenance=reference["input_provenance"], navmesh=mesh,
                      reference_sha256=_sha256(directory / "reference.json"),
                      trajectory_sha256=_sha256(directory / "trajectory.jsonl"))
    goals = reference["goals"]
    previous = _vector(reference["initial_base_pose"]["position"])
    _vector(reference["initial_base_pose"]["rotation"], 4)
    pathfinder = _load_pathfinder(directory / "runtime.navmesh")
    initial_paths = _paths(pathfinder, goals, previous)
    _validate_paths(reference["viewpoint_paths"], initial_paths, reference["reference_distance"])
    length = _minimum(initial_paths)
    path_length = 0.0
    terminal = None
    with (directory / "trajectory.jsonl").open() as stream:
        for index, line in enumerate(stream):
            record = strict_json_loads(line)
            if type(record["step"]) is not int or record["step"] != index:
                raise ValueError("Trajectory step sequence differs")
            action = record["action"]
            if type(action) is not int or action < 0:
                raise ValueError("Invalid trajectory action")
            if terminal is not None:
                raise ValueError("Trajectory contains actions after terminal STOP")
            before, after = _vector(record["before"]), _vector(record["after"])
            _continuous(before, previous)
            path_length += math.dist(before, after)
            previous = after
            if action == 0:
                terminal = record
    path_length = _number(path_length)
    if terminal is None:
        return unavailable("missing_terminal_STOP")
    if "final_distance" not in terminal or "final_viewpoint_paths" not in terminal:
        return unavailable("missing_STOP_viewpoint_distance")
    final_paths = _paths(pathfinder, goals, previous)
    _validate_paths(terminal["final_viewpoint_paths"], final_paths, terminal["final_distance"])
    final_distance = _minimum(final_paths)
    if length is None:
        return unavailable("no_reachable_reference_viewpoint")
    success = int(final_distance is not None and final_distance < 0.25)
    # Project convention: stationary zero-length success has SR=SPL=1, avoiding 0/0.
    spl = success * (length / max(length, path_length) if max(length, path_length) else 1.0)
    return {"status": "available", "SR": success, "SPL": spl,
            "distance_to_goal": final_distance, "reference_distance": length,
            "path_length": path_length, "provenance": provenance}


def main():
    """Score one saved episode offline and emit strict JSON on stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args()
    print(_json(score_habitat_evaluation(args.episode_dir)))


if __name__ == "__main__":
    main()
