"""Synchronous, geometry-only execution for the isolated Isaac process."""
from bisect import bisect_left, bisect_right
import heapq
import hashlib
import json
import math
from pathlib import Path
import threading


MAX_BODY_BYTES = 16 * 1024
MAX_MOVE_DISTANCE = 1.7


class CommandError(ValueError):
    """Identify invalid client input without masking simulator failures."""


def require_main_thread():
    """Reject simulator work outside Python's main thread."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Isaac commands must run on the main thread")


def finite_number(value, name):
    """Validate a finite JSON number, excluding booleans and numeric strings."""
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError
        result = float(value)
        if not math.isfinite(result):
            raise ValueError
        return result
    except (ValueError, OverflowError):
        raise CommandError(f"{name} must be a finite number") from None


def finite_vector(value, name, size=3):
    """Validate an exact-length JSON array of finite coordinates."""
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise CommandError(f"{name} must contain {size} numbers")
    return [finite_number(item, name) for item in value]


def decode_json(body):
    """Decode bounded UTF-8 JSON, rejecting duplicates and nonfinite numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CommandError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise CommandError("nonfinite JSON number")

    def number(value):
        return finite_number(float(value), "JSON number")

    if not body or len(body) > MAX_BODY_BYTES:
        raise CommandError("JSON body is empty or oversized")
    try:
        return json.loads(body.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant, parse_float=number)
    except (UnicodeError, ValueError, RecursionError, OverflowError) as exc:
        raise CommandError("invalid JSON body") from exc


def validate_command(request):
    """Allow only the fixed geometry-command schema."""
    if not isinstance(request, dict) or not isinstance(request.get("action"), str):
        raise CommandError("command must be an object with an action")
    action = request["action"]
    fields = {"observe": set(), "look": {"yaw"}, "viewpoints": {"position"},
              "move": {"position", "yaw", "max_distance"}, "STOP": set(),
              "finish": set(), "shutdown": set()}
    if action not in fields:
        raise CommandError("unknown action")
    if set(request) - {"action"} - fields[action]:
        raise CommandError("unexpected command fields")
    result = {"action": action}
    if action in {"move", "viewpoints"}:
        result["position"] = finite_vector(request.get("position"), "position")
    if action in {"move", "look"}:
        result["yaw"] = finite_number(request.get("yaw"), "yaw")
    if action == "move":
        distance = finite_number(request.get("max_distance", MAX_MOVE_DISTANCE), "max_distance")
        if distance <= 0:
            raise CommandError("max_distance must be positive")
        result["max_distance"] = min(distance, MAX_MOVE_DISTANCE)
    return result


class FreemapPlanner:
    """Plan exact-endpoint XY paths using strict, corner-safe 8-neighbor A*."""

    def __init__(self, walkable_map):
        self.wm = walkable_map
        self.axes = [tuple(float(v) for v in walkable_map.x_coords),
                     tuple(float(v) for v in walkable_map.y_coords)]
        self.edges = []
        spacings = []
        for axis in self.axes:
            if len(axis) < 2 or not all(math.isfinite(v) for v in axis):
                raise ValueError("freemap axes must have at least two finite coordinates")
            diffs = [b - a for a, b in zip(axis, axis[1:])]
            if not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
                raise ValueError("freemap axes must be strictly monotonic")
            values = sorted(axis)
            self.edges.append([values[0] - (values[1] - values[0]) / 2,
                               *[(a + b) / 2 for a, b in zip(values, values[1:])],
                               values[-1] + (values[-1] - values[-2]) / 2])
            spacings.extend(abs(d) for d in diffs)
        if walkable_map.grid.shape != (len(self.axes[1]), len(self.axes[0])):
            raise ValueError("freemap grid shape disagrees with coordinates")
        if not all(value in (0, 1) for row in walkable_map.grid for value in row):
            raise ValueError("freemap grid must be binary")
        self.sample_step = min(spacings) / 2
        self._edge_cache = {}

    def _indices(self, value, dimension):
        edges = self.edges[dimension]
        if not math.isfinite(value) or value < edges[0] or value > edges[-1]:
            return []
        # Closed-cell supercover: touching an obstacle edge/corner is blocked.
        low = max(0, bisect_left(edges, value - 1e-10) - 1)
        high = min(len(edges) - 2, bisect_right(edges, value + 1e-10) - 1)
        indices = range(low, high + 1)
        if self.axes[dimension][0] > self.axes[dimension][-1]:
            return [len(edges) - 2 - i for i in indices]
        return list(indices)

    def is_walkable(self, point):
        """Require in-bounds official walkability and all touched binary cells."""
        x, y = point
        cols, rows = self._indices(x, 0), self._indices(y, 1)
        return bool(cols and rows and self.wm.is_walkable(x, y)
                    and all(self.wm.grid[r, c] == 1 for r in rows for c in cols))

    def segment_is_clear(self, start, end):
        """Check half-cell samples plus every crossed cell and boundary exactly."""
        if not self.is_walkable(start) or not self.is_walkable(end):
            return False
        length = math.dist(start, end)
        count = max(1, math.ceil(length / self.sample_step))
        times = {i / count for i in range(count + 1)}
        for dimension, edges in enumerate(self.edges):
            delta = end[dimension] - start[dimension]
            if delta == 0:
                continue
            low, high = sorted((start[dimension], end[dimension]))
            for edge in edges[bisect_right(edges, low):bisect_left(edges, high)]:
                times.add((edge - start[dimension]) / delta)
        ordered = sorted(times)
        # Midpoints catch arbitrarily short passages through a blocked cell.
        samples = ordered + [(a + b) / 2 for a, b in zip(ordered, ordered[1:])]
        return all(self.is_walkable(tuple(a + (b - a) * t for a, b in zip(start, end)))
                   for t in samples)

    @staticmethod
    def path_length(path):
        """Return the sum of XY subsegment lengths, not the endpoint chord."""
        return sum(math.dist(a, b) for a, b in zip(path, path[1:]))

    def plan(self, start, goal):
        """Return a fully verified path with exact endpoints, or None without snapping."""
        start, goal = tuple(start), tuple(goal)
        if not self.is_walkable(start) or not self.is_walkable(goal):
            return None
        first, last = self.wm.world_to_cell(*start), self.wm.world_to_cell(*goal)
        if first == last:
            return [start, goal] if self.segment_is_clear(start, goal) else None
        world = self.wm.cell_to_world
        if not (self.segment_is_clear(start, world(*first))
                and self.segment_is_clear(world(*last), goal)):
            return None
        rows, cols = self.wm.grid.shape

        def neighbors(cell):
            row, col = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or not self.wm.grid[nr, nc]:
                    continue
                if dr and dc and not (self.wm.grid[row, nc] and self.wm.grid[nr, col]):
                    continue
                yield nr, nc

        # Connect the exact start to adjacent safe centers too: tiny successive
        # movement budgets must not repeatedly backtrack to the same cell center.
        costs, parents, pending = {}, {}, []
        for cell in [first, *neighbors(first)]:
            if self.segment_is_clear(start, world(*cell)):
                cost = math.dist(start, world(*cell))
                costs[cell], parents[cell] = cost, None
                heapq.heappush(pending, (cost + math.dist(world(*cell), world(*last)), cost, cell))
        while pending:
            _, cost, cell = heapq.heappop(pending)
            if cost != costs[cell]:
                continue
            if cell == last:
                cells = [cell]
                while parents[cell] is not None:
                    cell = parents[cell]
                    cells.append(cell)
                points = [start] + [tuple(world(*c)) for c in reversed(cells)] + [goal]
                path = [points[0]]
                for point in points[1:]:
                    if point != path[-1]:
                        path.append(point)
                if all(self.segment_is_clear(a, b) for a, b in zip(path, path[1:])):
                    return path
                return None
            for nxt in neighbors(cell):
                candidate = cost + math.dist(world(*cell), world(*nxt))
                if candidate >= costs.get(nxt, math.inf):
                    continue
                key = tuple(sorted((cell, nxt)))
                if key not in self._edge_cache:
                    self._edge_cache[key] = self.segment_is_clear(world(*cell), world(*nxt))
                if not self._edge_cache[key]:
                    continue
                costs[nxt], parents[nxt] = candidate, cell
                heapq.heappush(pending, (candidate + math.dist(world(*nxt), world(*last)),
                                         candidate, nxt))
        return None


class IsaacEnvironment:
    """Execute geometry commands and persist actual poses without any target metadata."""

    def __init__(self, env, walkable_map, output, capture):
        require_main_thread()
        self.env = env
        self.planner = FreemapPlanner(walkable_map)
        self.output = Path(output).resolve(strict=True)
        if not self.output.is_dir():
            raise ValueError("output must be an existing run directory")
        self.capture_root = self.output / "capture"
        self.capture_root.mkdir(exist_ok=False)
        self._trajectory = self.output / "trajectory.jsonl"
        self._journal = self.output / "actions.jsonl"
        for path in (self._trajectory, self._journal):
            with path.open("x", encoding="utf-8"):
                pass
        self._capture = capture
        self._frame_id = 0
        self._step = 0
        self._command_id = 0
        self._final_receipt = None
        self._trace("START")

    @staticmethod
    def _append(path, record):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _trace(self, action):
        pose = self.env.get_pose()
        if action in {"START", "MOVE"}:
            self.planner.wm.mark_explored(*pose["position"][:2])
        self._append(self._trajectory, dict(step=self._step, action=action,
                                            position=pose["position"], yaw=pose["yaw"]))
        self._step += 1
        return pose

    def _update(self, action, operation, *args, **kwargs):
        try:
            operation(*args, **kwargs)
        finally:
            # A simulator exception can follow a successful position update.
            pose = self._trace(action)
        return pose

    def record_rejection(self, error):
        """Journal a boundary rejection without copying untrusted fields or paths."""
        self._append(self._journal, {"event": "rejected", "error": str(error)})

    def command(self, request):
        """Execute one validated command on the main thread, journaling start and outcome."""
        require_main_thread()
        command = validate_command(request)
        action = command["action"]
        command_id = self._command_id
        self._command_id += 1
        self._append(self._journal, {"command_id": command_id, "event": "begin", "command": command})
        try:
            if self._final_receipt is not None and action in {"move", "look"}:
                raise CommandError("episode trajectory is finalized")
            if action == "observe":
                result = self._observe()
            elif action == "look":
                result = {"pose": self._update("LOOK", self.env.look_at_yaw, command["yaw"])}
            elif action == "viewpoints":
                result = self._viewpoints(command["position"])
            elif action == "move":
                result = self._move(command["position"], command["yaw"], command["max_distance"])
            else:
                result = self._final_receipt or {
                    "status": "reached", "pose": self._trace("STOP"), "zero_velocity": True}
                if action == "finish" and self._final_receipt is None:
                    self._final_receipt = result = {**result, "trajectory_steps": self._step,
                        "trajectory_sha256": hashlib.sha256(self._trajectory.read_bytes()).hexdigest()}
            # Fail in the execution journal too if an environment returns non-JSON data.
            json.dumps(result, allow_nan=False)
            self._append(self._journal, {"command_id": command_id, "event": "complete", "response": result})
            return result
        except Exception as exc:
            self._append(self._journal, {"command_id": command_id, "event": "error", "error": str(exc)})
            raise

    def _observe(self):
        frame_id = self._frame_id
        self._frame_id += 1  # Never reuse an ID after a partially failed capture.
        frame = self._capture(self.env, self.capture_root, frame_id)
        pose = self.env.get_pose()
        start = pose["position"][:2]
        frontiers = []
        for waypoint in self.env.sample_frontier_waypoints(self.planner.wm, K=8, seed=0):
            x, y = float(waypoint[0]), float(waypoint[1])
            if self.planner.segment_is_clear(start, (x, y)):
                frontiers.append({"uid": f"frontier:{frame['frame_id']}:{len(frontiers)}",
                                  "position": [x, y, 1.5], "score": math.dist(start, (x, y)),
                                  "metadata": {"frame_id": "map",
                                               "explored_density": self.planner.wm.explored_density_at(x, y, radius_m=0.5),
                                               "source": "official_freemap_action_proposer"}})
        return {"frame": frame, "capture_root": str(self.capture_root),
                "pose": pose, "frontiers": frontiers}

    def _viewpoints(self, position):
        start = self.env.get_pose()["position"][:2]
        candidates = []
        # Only the caller's observed estimate is used; the centroid is never a goal.
        for radius in (1.0, 1.5):
            for index in range(8):
                angle = index * math.pi / 4
                goal = (position[0] + radius * math.cos(angle),
                        position[1] + radius * math.sin(angle))
                path = self.planner.plan(start, goal)
                if path is not None:
                    candidates.append({"position": [*goal, 1.5],
                                       "yaw": math.atan2(position[1] - goal[1], position[0] - goal[0]),
                                       "path_length": self.planner.path_length(path),
                                       "source": "estimated_object_freemap_standoff"})
        return sorted(candidates, key=lambda item: item["path_length"])[:4]

    def _move(self, position, yaw, max_distance):
        pose = self.env.get_pose()
        start, goal = pose["position"][:2], position[:2]
        path = self.planner.plan(start, goal)
        if path is None:
            return {"status": "blocked", "pose": pose, "distance_m": 0.0,
                    "path_length_remaining": None}
        total = self.planner.path_length(path)
        traveled = 0.0
        reached = total == 0
        if reached:
            pose = self._update("LOOK", self.env.look_at_yaw, yaw)
        for index, endpoint in enumerate(path[1:], 1):
            current = pose["position"][:2]
            length = math.dist(current, endpoint)
            budget = max_distance - traveled
            if length == 0:
                continue
            if budget <= 0:
                break
            distance = min(length, budget)
            final_segment = index == len(path) - 1 and distance == length
            destination = endpoint if distance == length else tuple(
                a + (b - a) * distance / length for a, b in zip(current, endpoint))
            heading = yaw if final_segment else math.atan2(endpoint[1] - current[1], endpoint[0] - current[0])
            face = (destination[0] + math.cos(heading), destination[1] + math.sin(heading))
            pose = self._update("MOVE", self.env.teleport_to, destination, face_direction_xy=face)
            actual = pose["position"][:2]
            traveled += math.dist(current, actual)
            if math.dist(actual, destination) > 1e-8:
                raise RuntimeError("simulator did not execute the verified teleport endpoint")
            reached = final_segment
            if distance < length:
                break
        return {"status": "reached" if reached else "running", "pose": pose,
                "distance_m": traveled, "path_length_remaining": max(0.0, total - traveled)}
