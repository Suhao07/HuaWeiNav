"""Exercise geometry and the JSON boundary without importing Isaac or opening a server."""
import builtins
import ctypes
import io
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call
import uuid

import numpy as np
import pytest

from backends.isaac.environment import (CommandError, FreemapPlanner, IsaacEnvironment,
                                        MAX_BODY_BYTES, decode_json)
from scripts import serve_intentionnav as server


class FakeMap:
    """Mimic official nearest-cell indexing and deliberately permissive LOS."""

    def __init__(self, grid=None, x_coords=None, y_coords=None, scene_id="scene", rooms=None):
        self.grid = np.ones((17, 17), dtype=np.uint8) if grid is None else np.asarray(grid)
        self.x_coords = np.linspace(-2, 2, self.grid.shape[1]) if x_coords is None else np.asarray(x_coords)
        self.y_coords = np.linspace(-2, 2, self.grid.shape[0]) if y_coords is None else np.asarray(y_coords)
        self.rooms = {} if rooms is None else rooms
        self.explored_positions = []

    def mark_explored(self, x, y):
        self.explored_positions.append((x, y))

    def explored_density_at(self, x, y, radius_m=0.5):
        return float(any(math.dist((x, y), old) <= radius_m for old in self.explored_positions))

    def world_to_cell(self, x, y):
        """Use the official row-column convention, including out-of-bounds clamping."""
        return int(np.argmin(abs(self.y_coords - y))), int(np.argmin(abs(self.x_coords - x)))

    def cell_to_world(self, row, col):
        """Return actual coordinate centers, including descending axes."""
        return float(self.x_coords[col]), float(self.y_coords[row])

    def is_walkable(self, x, y):
        """Intentionally omit bounds checks to exercise the bridge's independent guard."""
        return bool(self.grid[self.world_to_cell(x, y)])

    def line_of_sight(self, *args):
        """Model a permissive official LOS that is not a collision certificate."""
        return True

    @classmethod
    def load(cls, *args, **kwargs):
        """Fail if code attempts to load room or target sidecars."""
        raise AssertionError("metadata-bearing loader must not be called")


class FakeEnv:
    """Record every camera update while providing only public pose and frontier methods."""

    def __init__(self, position=(-0.5, 0, 1.5), gpu=0):
        self.position = list(position)
        self.yaw = 0.0
        self.calls = []
        self.waypoints = []
        self.closed = False

    def get_pose(self):
        """Return an independent snapshot of the actual camera pose."""
        return {"position": list(self.position), "yaw": self.yaw,
                "look_dir": [math.cos(self.yaw), math.sin(self.yaw), 0.0]}

    def look_at_yaw(self, yaw):
        """Rotate in place without translating."""
        self.yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self.calls.append(("look", self.get_pose()))

    def teleport_to(self, xy, face_direction_xy):
        """Match official face_direction_xy's target-point rather than vector semantics."""
        self.position = [float(xy[0]), float(xy[1]), 1.5]
        self.yaw = math.atan2(face_direction_xy[1] - xy[1], face_direction_xy[0] - xy[0])
        self.calls.append(("move", self.get_pose()))

    def sample_frontier_waypoints(self, wm, K=8, seed=0):
        """Return unsafe proposals too, so the bridge must filter them."""
        assert K == 8 and seed == 0
        return self.waypoints

    def load_scene(self, scene_id, path):
        """Record the single scene geometry input."""
        self.scene = scene_id, path

    def place_agent(self, position, quat):
        """Record the public start pose."""
        self.position = list(position)
        self.start_quat = quat

    def close(self):
        """Mark the fake application closed."""
        self.closed = True


def fake_capture(env, root, frame_id):
    """Write a portable fake frame without looking up any scene metadata."""
    record = {"frame_id": frame_id, "rgb_path": f"rgb/{frame_id}.png",
              "depth_path": f"depth/{frame_id}.npy", "intrinsics": np.eye(3).tolist(),
              "camera_to_map": np.eye(4).tolist()}
    with (root / "frames.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def bridge_at(tmp_path, wm=None, env=None, capture=fake_capture):
    """Build an isolated bridge in an already-existing test run directory."""
    return IsaacEnvironment(env or FakeEnv(), wm or FakeMap(), tmp_path, capture)


def records(path):
    """Read JSONL artifacts to compare reported versus actual execution."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_thin_obstacle_and_out_of_bounds():
    """Block a one-cell wall even when the official map reports permissive LOS."""
    wm = FakeMap(grid=np.ones((5, 101)), x_coords=np.linspace(0, 1, 101),
                 y_coords=np.linspace(0, 0.04, 5))
    wm.grid[:, 53] = 0
    planner = FreemapPlanner(wm)
    assert wm.line_of_sight(0, 0, 1, 0)
    assert not planner.segment_is_clear((0, 0.02), (1, 0.02))
    assert planner.plan((0, 0.02), (1, 0.02)) is None
    assert not planner.is_walkable((-0.1, 0.02))
    assert not planner.is_walkable((float("inf"), 0.02))


def test_supercover_catches_short_obstacle_sliver_and_boundary():
    """Catch obstacle contact shorter than the half-cell sampling interval."""
    wm = FakeMap(np.ones((3, 3)), [0, 1, 2], [0, 1, 2])
    wm.grid[0, 1] = 0
    planner = FreemapPlanner(wm)
    assert not planner.segment_is_clear((0, 0), (1, 0.9999))
    assert not planner.segment_is_clear((0, 0), (1, 1))
    assert not planner.segment_is_clear((0.6, 0.5), (1.4, 0.5))


@pytest.mark.parametrize("reverse_x,reverse_y", [(False, False), (True, False), (False, True), (True, True)])
def test_astar_detour_exact_endpoints_and_axis_order(reverse_x, reverse_y):
    """Keep exact endpoints and safe detours for either coordinate-axis ordering."""
    wm = FakeMap()
    wm.grid[4:13, 8] = 0
    if reverse_x:
        wm.x_coords = wm.x_coords[::-1]
        wm.grid = wm.grid[:, ::-1]
    if reverse_y:
        wm.y_coords = wm.y_coords[::-1]
        wm.grid = wm.grid[::-1, :]
    planner = FreemapPlanner(wm)
    start, goal = (-0.57, 0.03), (0.61, -0.02)
    path = planner.plan(start, goal)
    assert path[0] == start and path[-1] == goal
    assert planner.path_length(path) > math.dist(start, goal)
    assert all(planner.segment_is_clear(a, b) for a, b in zip(path, path[1:]))


def test_no_diagonal_corner_cutting(tmp_path):
    """Disconnected diagonal free cells must produce blocked with no motion."""
    wm = FakeMap([[1, 0], [0, 1]], [0, 1], [0, 1])
    bridge = bridge_at(tmp_path, wm, FakeEnv((0, 0, 1.5)))
    pose = bridge.env.get_pose()
    result = bridge.command({"action": "move", "position": [1, 1, 1.5], "yaw": 0})
    assert result == {"status": "blocked", "pose": pose, "distance_m": 0.0,
                      "path_length_remaining": None}
    assert bridge.env.calls == []


@pytest.mark.parametrize("goal", [[0, 0, 1.5], [100, 0, 1.5], [-100, 0, 1.5]])
def test_invalid_goal_never_snaps_or_moves(tmp_path, goal):
    """Reject occupied and out-of-map goals rather than relocating them to free cells."""
    wm = FakeMap()
    wm.grid[8, 8] = 0
    bridge = bridge_at(tmp_path, wm)
    before = bridge.env.get_pose()
    result = bridge.command({"action": "move", "position": goal, "yaw": 2})
    assert result["status"] == "blocked"
    assert result["distance_m"] == 0 and result["pose"] == before
    assert bridge.env.calls == []
    assert len(records(tmp_path / "trajectory.jsonl")) == 1


@pytest.mark.parametrize("limit", [0.07, 0.8, 1.7, 10.0])
def test_movement_budget_follows_actual_path_and_travel_yaw(tmp_path, limit):
    """Count every traveled subsegment and cap a detour at the requested or hard limit."""
    wm = FakeMap()
    wm.grid[4:13, 8] = 0
    bridge = bridge_at(tmp_path, wm)
    goal = [0.5, 0, 1.5]
    path = bridge.planner.plan(bridge.env.position[:2], goal[:2])
    result = bridge.command({"action": "move", "position": goal, "yaw": -2,
                             "max_distance": limit})
    rows = records(tmp_path / "trajectory.jsonl")
    traveled = sum(math.dist(a["position"][:2], b["position"][:2]) for a, b in zip(rows, rows[1:]))
    assert result["status"] == "running"
    assert traveled == pytest.approx(min(limit, 1.7))
    assert traveled <= min(limit, 1.7) + 1e-12
    assert result["distance_m"] == pytest.approx(traveled)
    assert result["path_length_remaining"] == pytest.approx(bridge.planner.path_length(path) - traveled)
    assert len(rows) == len(bridge.env.calls) + 1
    assert [row["step"] for row in rows] == list(range(len(rows)))
    for previous, current in zip(rows, rows[1:]):
        a, b = previous["position"][:2], current["position"][:2]
        assert bridge.planner.segment_is_clear(a, b)
        assert current["yaw"] == pytest.approx(math.atan2(b[1] - a[1], b[0] - a[0]))
    if limit >= 1.7:
        assert traveled > math.dist(rows[0]["position"][:2], rows[-1]["position"][:2])


def test_repeated_subcell_moves_make_progress(tmp_path):
    """Replan from exact partial endpoints without oscillating back to a cell center."""
    bridge = bridge_at(tmp_path)
    goal = [0.6, 0, 1.5]
    previous = bridge.env.position[0]
    for _ in range(20):
        result = bridge.command({"action": "move", "position": goal, "yaw": 1,
                                 "max_distance": 0.07})
        assert result["distance_m"] <= 0.07 + 1e-12
        assert bridge.env.position[0] > previous
        previous = bridge.env.position[0]
        if result["status"] == "reached":
            break
    assert result["status"] == "reached" and bridge.env.position == goal


def test_exact_goal_final_yaw_and_zero_translation_move(tmp_path):
    """Apply the requested yaw only at the exact final endpoint, including an in-place goal."""
    bridge = bridge_at(tmp_path)
    goal = [-0.07, 0.04, 1.5]
    result = bridge.command({"action": "move", "position": goal, "yaw": 1.2})
    assert result["status"] == "reached"
    assert result["pose"]["position"] == goal
    assert result["pose"]["yaw"] == pytest.approx(1.2)
    assert result["path_length_remaining"] == pytest.approx(0)
    again = bridge.command({"action": "move", "position": goal, "yaw": -0.4})
    assert again["status"] == "reached" and again["distance_m"] == 0
    assert again["pose"]["yaw"] == pytest.approx(-0.4)


def test_exploration_tracks_motion_and_finish_freezes_scored_trace(tmp_path):
    import hashlib

    bridge = bridge_at(tmp_path)
    wm = bridge.planner.wm
    bridge.command({"action": "observe"})
    bridge.command({"action": "observe"})
    assert wm.explored_positions == [(-0.5, 0)]
    bridge.command({"action": "STOP"})
    bridge.command({"action": "move", "position": [0.5, 0, 1.5], "yaw": 0})
    assert wm.explored_positions[-1] == (0.5, 0)
    receipt = bridge.command({"action": "finish"})
    frozen = (tmp_path / "trajectory.jsonl").read_bytes()
    assert receipt["trajectory_sha256"] == hashlib.sha256(frozen).hexdigest()
    assert receipt["trajectory_steps"] == len(frozen.splitlines())
    assert json.loads(frozen.splitlines()[-1])["action"] == "STOP"
    for action in ("finish", "STOP", "shutdown"):
        assert bridge.command({"action": action}) == receipt
    bridge.command({"action": "observe"})
    with pytest.raises(CommandError, match="finalized"):
        bridge.command({"action": "look", "yaw": 1})
    assert (tmp_path / "trajectory.jsonl").read_bytes() == frozen


def test_look_stop_and_shutdown_never_translate(tmp_path):
    """STOP and shutdown preserve the pose, while LOOK rotates and records its actual yaw."""
    bridge = bridge_at(tmp_path)
    start = bridge.env.get_pose()["position"]
    looked = bridge.command({"action": "look", "yaw": 7.0})
    assert looked["pose"]["position"] == start
    assert not (tmp_path / "capture" / "frames.jsonl").exists()
    calls = list(bridge.env.calls)
    for action in ("STOP", "STOP", "shutdown"):
        result = bridge.command({"action": action})
        assert result == {"status": "reached", "pose": looked["pose"], "zero_velocity": True}
    assert bridge.env.calls == calls
    trace = records(tmp_path / "trajectory.jsonl")
    assert [row["action"] for row in trace] == ["START", "LOOK", "STOP", "STOP", "STOP"]
    assert all(row["position"] == start for row in trace)
    assert [row["event"] for row in records(tmp_path / "actions.jsonl")] == ["begin", "complete"] * 4


def test_observe_filters_official_proposals_and_preserves_capture(tmp_path):
    """Export only captured evidence, actual pose, and strictly reachable official proposals."""
    wm, env = FakeMap(), FakeEnv()
    wm.grid[:, 8] = 0
    env.waypoints = [(-1, 0, 0), (0.5, 0, 0), (999, 0, 0), (float("nan"), 0, 0)]
    bridge = bridge_at(tmp_path, wm, env)
    for frame_id in range(2):
        result = bridge.command({"action": "observe"})
        assert set(result) == {"frame", "capture_root", "pose", "frontiers"}
        assert result["capture_root"] == str((tmp_path / "capture").resolve())
        assert result["frame"]["frame_id"] == frame_id
        assert result["frontiers"] == [{"uid": f"frontier:{frame_id}:0", "position": [-1, 0, 1.5],
                                        "score": 0.5, "metadata": {"frame_id": "map",
                                        "explored_density": 1.0, "source": "official_freemap_action_proposer"}}]
        assert result["frame"] == records(tmp_path / "capture" / "frames.jsonl")[-1]
    assert env.calls == []


def test_viewpoints_use_only_observed_estimate_and_reachable_standoffs(tmp_path):
    """Use the observed estimate, not target metadata or an object-centroid navigation goal."""
    wm = FakeMap()
    wm.grid[:, 8] = 0
    bridge = bridge_at(tmp_path, wm)
    estimate = [0, 0, 15]  # Occupied object center; navigation remains on the camera XY plane.
    before = bridge.env.get_pose()
    result = bridge.command({"action": "viewpoints", "position": estimate})
    assert 0 < len(result) <= 4
    assert [item["path_length"] for item in result] == sorted(item["path_length"] for item in result)
    for item in result:
        point = item["position"]
        assert point[0] < 0 and point[2] == 1.5
        assert min(abs(math.dist(point[:2], estimate[:2]) - radius) for radius in (1, 1.5)) < 1e-10
        assert item["yaw"] == pytest.approx(math.atan2(estimate[1] - point[1], estimate[0] - point[0]))
        path = bridge.planner.plan(before["position"][:2], point[:2])
        assert item["path_length"] == pytest.approx(bridge.planner.path_length(path))
        assert item["source"] == "estimated_object_freemap_standoff"
    assert bridge.env.get_pose() == before and not bridge.env.calls
    assert bridge.command({"action": "viewpoints", "position": [100, 100, 0]}) == []
    assert "target" not in (tmp_path / "actions.jsonl").read_text()


def test_failed_capture_does_not_reuse_frame_id(tmp_path):
    """Journal simulator failures and keep subsequent commands usable with fresh frame IDs."""
    def capture(env, root, frame_id):
        if frame_id == 0:
            raise RuntimeError("capture unavailable")
        return fake_capture(env, root, frame_id)

    bridge = bridge_at(tmp_path, capture=capture)
    with pytest.raises(RuntimeError):
        bridge.command({"action": "observe"})
    assert bridge.command({"action": "observe"})["frame"]["frame_id"] == 1
    assert [row["event"] for row in records(tmp_path / "actions.jsonl")] == ["begin", "error", "begin", "complete"]


def test_partial_simulator_failure_still_traces_actual_update(tmp_path):
    """Record a successful teleport even if the renderer subsequently raises."""
    env = FakeEnv()
    teleport = env.teleport_to

    def failing_teleport(*args, **kwargs):
        teleport(*args, **kwargs)
        raise RuntimeError("render failed after movement")

    env.teleport_to = failing_teleport
    bridge = bridge_at(tmp_path, env=env)
    with pytest.raises(RuntimeError):
        bridge.command({"action": "move", "position": [0.5, 0, 1.5], "yaw": 0})
    trace = records(tmp_path / "trajectory.jsonl")
    assert len(trace) == 2 and trace[-1]["position"] == env.position
    assert records(tmp_path / "actions.jsonl")[-1]["event"] == "error"


class MemoryConnection:
    """Feed BaseHTTPRequestHandler bytes without a socket, service, or worker thread."""

    def __init__(self, request, host):
        self.input = io.BytesIO(request)
        self.output = bytearray()
        self.host = host

    def settimeout(self, timeout):
        """Accept the production request timeout without any real IO."""
        assert timeout == 10

    def makefile(self, mode, *args):
        """Provide the request stream expected by the standard-library handler."""
        assert mode == "rb"
        return self.input

    def sendall(self, payload):
        """Verify shutdown is not requested before the response has been flushed."""
        assert not self.host.shutdown_requested
        self.output.extend(payload)


def http_request(bridge, body=b'{"action":"STOP"}', headers=None, path="/command", method="POST"):
    """Run the HTTP handler entirely in memory and decode its wire response."""
    if headers is None:
        headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    raw = f"{method} {path} HTTP/1.0\r\n".encode()
    raw += b"".join(f"{key}: {value}\r\n".encode() for key, value in headers) + b"\r\n" + body
    host = SimpleNamespace(bridge=bridge, shutdown_requested=False)
    connection = MemoryConnection(raw, host)
    server.CommandHandler(connection, ("127.0.0.1", 12345), host)
    response_headers, response = bytes(connection.output).split(b"\r\n\r\n", 1)
    return int(response_headers.split()[1]), json.loads(response), host


@pytest.mark.parametrize("body", [b"", b"{", b"[]", b"null", b"\xff", b'{"action":"STOP","action":"observe"}',
    b'{"action":"look","yaw":NaN}', b'{"action":"look","yaw":Infinity}',
    b'{"action":"look","yaw":1e309}', b'{"action":"look","yaw":true}',
    b'{"action":"look","yaw":"1"}', b'{"action":"look"}',
    b'{"action":"move","position":[0,0],"yaw":0}',
    b'{"action":"move","position":[0,0,0],"yaw":0,"max_distance":0}',
    b'{"action":"move","position":[0,0,0],"yaw":0,"max_distance":-1}',
    b'{"action":"move","position":[0,0,null],"yaw":0}',
    b'{"action":"viewpoints","position":[0,0,0],"target_category":"private"}',
    b'{"action":"observe","path":"/etc/passwd"}', b'{"action":"exec","code":"pass"}',
    b'{"action":{}}', b"[" * 1100 + b"]" * 1100, b" " * (MAX_BODY_BYTES + 1)])
def test_invalid_json_and_action_boundary_returns_400_without_motion(tmp_path, body):
    """Reject malformed, nonfinite, oversized, or schema-breaking requests without execution."""
    bridge = bridge_at(tmp_path)
    status, result, host = http_request(bridge, body)
    assert status == 400 and "error" in result
    assert not host.shutdown_requested and not bridge.env.calls
    assert len(records(tmp_path / "trajectory.jsonl")) == 1
    assert records(tmp_path / "actions.jsonl")[-1]["event"] == "rejected"
    assert "private" not in (tmp_path / "actions.jsonl").read_text()


@pytest.mark.parametrize("headers", [[], [("Content-Length", "-1")], [("Content-Length", "xyz")],
    [("Content-Length", "999999999999999999999")],
    [("Content-Length", "17"), ("Content-Length", "17")],
    [("Content-Length", "17"), ("Content-Type", "text/plain")],
    [("Content-Length", "17"), ("Content-Type", "application/json"), ("Transfer-Encoding", "chunked")],
    [("Content-Length", "999"), ("Content-Type", "application/json")]])
def test_http_framing_boundary(tmp_path, headers):
    """Reject ambiguous framing, wrong content types, and truncated bodies."""
    bridge = bridge_at(tmp_path)
    assert http_request(bridge, headers=headers)[0] == 400
    assert not bridge.env.calls


@pytest.mark.parametrize("path,method", [("/elsewhere", "POST"), ("/command?file=x", "POST"), ("/command", "GET")])
def test_http_has_no_arbitrary_routes(tmp_path, path, method):
    """Expose only the fixed POST command route, returning JSON for other HTTP requests."""
    assert http_request(bridge_at(tmp_path), path=path, method=method)[0] == 400


@pytest.mark.parametrize("raw_request", [b"invalid request line here\r\n\r\n", b"POST /command\r\n\r\n"])
def test_malformed_http_still_has_json_status(tmp_path, raw_request):
    """Return a framed JSON error even before a valid HTTP version has been parsed."""
    host = SimpleNamespace(bridge=bridge_at(tmp_path), shutdown_requested=False)
    connection = MemoryConnection(raw_request, host)
    server.CommandHandler(connection, ("127.0.0.1", 12345), host)
    headers, body = bytes(connection.output).split(b"\r\n\r\n", 1)
    assert int(headers.split()[1]) == 400 and "error" in json.loads(body)
    assert not host.bridge.env.calls


def test_http_simulator_failure_is_500_and_shutdown_flushes(tmp_path):
    """Keep handling commands after an execution error and acknowledge shutdown before exit."""
    def failing_capture(*args):
        raise ValueError("simulator failure, not invalid client input")

    bridge = bridge_at(tmp_path, capture=failing_capture)
    status, error, host = http_request(bridge, b'{"action":"observe"}')
    assert status == 500 and "simulator failure" in error["error"]
    assert not host.shutdown_requested
    status, result, host = http_request(bridge, b'{"action":"shutdown"}')
    assert status == 200 and result["zero_velocity"] and host.shutdown_requested
    assert result["pose"] == bridge.env.get_pose()


def test_capture_directory_is_exclusive(tmp_path):
    """Never reuse a prior episode's capture directory."""
    bridge_at(tmp_path)
    with pytest.raises(FileExistsError):
        bridge_at(tmp_path)


def public_episode():
    """Provide the complete public input and no evaluation metadata."""
    return {"selection_id": "SEL_001", "scene_id": "scene", "style": "natural",
            "instruction": "Find somewhere to rest.", "start_position": [-0.5, 0, 1.5],
            "start_rotation_quat_wxyz": [1, 0, 0, 0]}


@pytest.mark.parametrize("change", [{"target_category": "private"}, {"target_position": [1, 2, 3]},
    {"room": "private"}, {"scene_id": "../scene"}, {"scene_id": "/scene"},
    {"start_rotation_quat_wxyz": [0, 0, 0, 0]}, {"start_position": [True, 0, 0]}])
def test_public_episode_rejects_private_fields_and_invalid_start(tmp_path, change):
    """Do not accept evaluator metadata, path-like scene names, or malformed public poses."""
    path = tmp_path / "public.json"
    path.write_text(json.dumps(dict(public_episode(), **change)))
    with pytest.raises(CommandError):
        server.load_public_episode(path)


def test_freemap_loader_never_reads_rooms_or_metadata(tmp_path, monkeypatch):
    """Construct the official-compatible map by opening only the scene's freemap array."""
    scene = tmp_path / "metadata" / "metadata_train" / "scene"
    scene.mkdir(parents=True)
    matrix = np.zeros((4, 4))
    matrix[0, 1:] = [2, 1, 0]
    matrix[1:, 0] = [0, 1, 2]
    matrix[1:, 1:] = 1
    np.save(scene / "freemap.npy", matrix)
    original = np.load
    opened = []

    def load(path, **kwargs):
        opened.append(Path(path))
        assert kwargs == {"allow_pickle": False}
        return original(path, **kwargs)

    monkeypatch.setattr(np, "load", load)
    wm = server.load_freemap(tmp_path, "scene", FakeMap)
    assert opened == [scene / "freemap.npy"]
    assert wm.rooms == {} and not wm.explored_count.any()
    assert FreemapPlanner(wm).plan((0, 0), (2, 2)) is not None


def test_freemap_uses_official_exact_one_walkability(tmp_path):
    scene = tmp_path / "metadata" / "metadata_train" / "scene"
    scene.mkdir(parents=True)
    matrix = np.array([[0, 2, 1, 0], [0, 0, 1, 2], [1, 2, 1, 0], [2, 1, 2, 1]], dtype=float)
    np.save(scene / "freemap.npy", matrix)
    wm = server.load_freemap(tmp_path, "scene", FakeMap)
    np.testing.assert_array_equal(wm.grid, matrix[1:, 1:] == 1)
    assert wm.grid.dtype == np.uint8


def test_main_thread_guard_without_starting_a_thread(tmp_path, monkeypatch):
    """Reject simulated worker-thread invocation before any Isaac calls."""
    from backends.isaac import environment

    bridge = bridge_at(tmp_path)
    monkeypatch.setattr(environment.threading, "current_thread", lambda: object())
    with pytest.raises(RuntimeError, match="main thread"):
        bridge.command({"action": "look", "yaw": 1})
    assert not bridge.env.calls


@pytest.fixture
def simulator_gpu():
    """Use a nonzero physical GPU and a deterministic UUID without querying hardware."""
    return {"physical_index": 7, "uuid": "GPU-00112233-4455-6677-8899-aabbccddeeff",
            "pci_bus_id": "00000000:CA:00.0"}


@pytest.mark.parametrize("index", [0, 7])
def test_gpu_identity_binds_physical_index_without_shell(monkeypatch, index):
    """Resolve the exact physical ordinal with argv, regardless of inherited CUDA masking."""
    identity = "GPU-00112233-4455-6677-8899-aabbccddeeff"
    query = Mock(return_value=f"  {index},  {identity}, 00000000:CA:00.0 \n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unrelated-parent-device")
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(check_output=query))
    assert server.gpu_identity(index) == {
        "physical_index": index, "uuid": identity, "pci_bus_id": "00000000:CA:00.0"}
    query.assert_called_once_with([
        "nvidia-smi", "-i", str(index), "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits"], text=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "unrelated-parent-device"


@pytest.mark.parametrize("row", [
    "0, GPU-00112233-4455-6677-8899-aabbccddeeff, 00000000:CA:00.0",
    "7, not-a-gpu-uuid, 00000000:CA:00.0",
    "7, GPU-00112233-4455-6677-8899-aabbccddeefg, 00000000:CA:00.0",
])
def test_gpu_identity_rejects_wrong_ordinal_or_uuid(monkeypatch, row):
    """Never accept another physical GPU or an unrecognized UUID as an allocation."""
    query = Mock(return_value=row)
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(check_output=query))
    with pytest.raises(RuntimeError, match="requested simulator GPU"):
        server.gpu_identity(7)
    query.assert_called_once()


@pytest.fixture
def cuda_driver(monkeypatch, simulator_gpu):
    """Implement libcuda output pointers in CPU memory; never load a real CUDA library."""
    state = SimpleNamespace(count=1, identity=simulator_gpu["uuid"], device=71,
                            values={"/physics/cudaDevice": 0, "/renderer/activeGpu": 7})
    driver = Mock()

    def initialize(flags):
        assert flags == 0
        return 0

    def get_count(pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = state.count
        return 0

    def get_device(pointer, ordinal):
        assert ordinal == 0
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = state.device
        return 0

    def get_uuid(pointer, device):
        assert device.value == state.device
        raw = uuid.UUID(state.identity[4:]).bytes
        ctypes.memmove(pointer, raw, len(raw))
        return 0

    driver.cuInit.side_effect = initialize
    driver.cuDeviceGetCount.side_effect = get_count
    driver.cuDeviceGet.side_effect = get_device
    driver.cuDeviceGetUuid.side_effect = get_uuid
    state.driver = driver
    state.load = Mock(return_value=driver)
    state.settings = Mock()
    state.settings.get.side_effect = state.values.__getitem__
    state.get_settings = Mock(return_value=state.settings)
    monkeypatch.setattr(server.ctypes, "CDLL", state.load)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(
        settings=SimpleNamespace(get_settings=state.get_settings)))
    return state


@pytest.mark.parametrize("uppercase", [False, True])
def test_validate_simulator_device_returns_verified_receipt(cuda_driver, simulator_gpu, uppercase):
    """Bind CUDA logical zero and Vulkan physical seven to the same case-insensitive UUID."""
    expected = dict(simulator_gpu)
    if uppercase:
        expected["uuid"] = expected["uuid"].upper()
    before = dict(expected)
    receipt = server.validate_simulator_device(expected)
    assert receipt == {**expected, "cuda_device_count": 1, "cuda_uuid": simulator_gpu["uuid"],
                       "physics_cuda_device": 0, "renderer_active_gpu": 7}
    assert expected == before
    cuda_driver.load.assert_called_once_with("libcuda.so.1")
    assert [entry[0] for entry in cuda_driver.driver.mock_calls] == [
        "cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid"]
    cuda_driver.get_settings.assert_called_once_with()
    assert cuda_driver.settings.get.call_args_list == [
        call("/physics/cudaDevice"), call("/renderer/activeGpu")]


@pytest.mark.parametrize("count", [0, 2, 8])
def test_validate_simulator_device_requires_exactly_one_cuda_device(cuda_driver, simulator_gpu, count):
    """Reject absent or multiple visible GPUs before querying a device or Carb settings."""
    cuda_driver.count = count
    with pytest.raises(RuntimeError, match=f"exactly one CUDA device, got {count}"):
        server.validate_simulator_device(simulator_gpu)
    assert [entry[0] for entry in cuda_driver.driver.mock_calls] == ["cuInit", "cuDeviceGetCount"]
    cuda_driver.get_settings.assert_not_called()


@pytest.mark.parametrize("function,stage", [
    ("cuInit", "initialization"), ("cuDeviceGetCount", "initialization"),
    ("cuDeviceGet", "identity query"), ("cuDeviceGetUuid", "identity query"),
])
def test_validate_simulator_device_rejects_cuda_errors(cuda_driver, simulator_gpu, function, stage):
    """Stop at every failed driver call rather than interpreting uninitialized output memory."""
    getattr(cuda_driver.driver, function).side_effect = lambda *args: 100
    with pytest.raises(RuntimeError, match=f"CUDA {stage} failed: 100"):
        server.validate_simulator_device(simulator_gpu)
    order = ["cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid"]
    assert [entry[0] for entry in cuda_driver.driver.mock_calls] == order[:order.index(function) + 1]
    cuda_driver.get_settings.assert_not_called()


def test_validate_simulator_device_rejects_uuid_mismatch(cuda_driver, simulator_gpu):
    """Do not accept a single CUDA device when its identity belongs to another card."""
    cuda_driver.identity = "GPU-ffeeddcc-bbaa-9988-7766-554433221100"
    with pytest.raises(RuntimeError, match="CUDA device differs from the allocated GPU"):
        server.validate_simulator_device(simulator_gpu)
    cuda_driver.driver.cuDeviceGetUuid.assert_called_once()
    cuda_driver.get_settings.assert_not_called()


@pytest.mark.parametrize("physics,renderer", [(7, 7), (1, 7), (0, 0), (0, 5), (None, 7), (0, None)])
def test_validate_simulator_device_rejects_wrong_settings(cuda_driver, simulator_gpu, physics, renderer):
    """Reject physical CUDA ordinals and renderer remapping to another physical card."""
    cuda_driver.values.update({"/physics/cudaDevice": physics, "/renderer/activeGpu": renderer})
    with pytest.raises(RuntimeError, match="CUDA and Vulkan device settings disagree"):
        server.validate_simulator_device(simulator_gpu)
    cuda_driver.driver.cuDeviceGetUuid.assert_called_once()
    assert cuda_driver.settings.get.call_args_list == [
        call("/physics/cudaDevice"), call("/renderer/activeGpu")]


@pytest.mark.parametrize("prior_observations", [0, 1])
def test_ready_and_single_loop_with_fake_httpserver(tmp_path, monkeypatch, simulator_gpu, prior_observations):
    """Publish a verified frame and GPU receipt only after bind and a complete observation."""
    bridge = bridge_at(tmp_path)
    for _ in range(prior_observations):
        bridge.command({"action": "observe"})
    events = []
    gpu = {**simulator_gpu, "cuda_device_count": 1, "cuda_uuid": simulator_gpu["uuid"],
           "physics_cuda_device": 0, "renderer_active_gpu": 7}

    def capture(env, root, frame_id):
        assert events == ["bind"]
        assert not (tmp_path / "server_ready.json").exists()
        events.append("observe")
        return fake_capture(env, root, frame_id)

    class FakeHTTPServer:
        """Emulate only the context manager and synchronous request loop."""

        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 9466) and handler is server.CommandHandler
            self.server_address = address
            events.append("bind")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("close")

        def handle_request(self):
            """Check ready data and emulate a single acknowledged shutdown."""
            ready = json.loads((tmp_path / "server_ready.json").read_text())
            assert ready["port"] == 9466 and ready["pid"] > 0
            assert ready["isaac_version"] == "test-VERSION"
            assert ready["start_pose"] == bridge.env.get_pose()
            assert ready["gpu"] == gpu
            assert ready["initial_observation_frame_id"] == prior_observations
            assert ready["initial_observation_frame_id"] == records(bridge.capture_root / "frames.jsonl")[-1]["frame_id"]
            assert records(tmp_path / "actions.jsonl")[-1]["event"] == "complete"
            assert self.bridge is bridge
            events.append("request")
            self.shutdown_requested = True

    monkeypatch.setattr(bridge, "_capture", capture)
    monkeypatch.setattr(server, "HTTPServer", FakeHTTPServer)
    server.serve(bridge, 9466, "test-VERSION", gpu=gpu)
    assert events == ["bind", "observe", "request", "close"]
    assert len(records(bridge.capture_root / "frames.jsonl")) == prior_observations + 1


@pytest.mark.parametrize("failure", ["capture", "frontiers"])
def test_ready_failure_preserves_observation_error_without_ready(tmp_path, monkeypatch, simulator_gpu, failure):
    """Keep failed initial-observation evidence and close the bound socket without serving."""
    events = []
    error = RuntimeError("initial RGB-D observation failed")

    def fail(*args, **kwargs):
        assert events == ["bind"]
        assert not (tmp_path / "server_ready.json").exists()
        events.append("failure")
        raise error

    bridge = bridge_at(tmp_path, capture=fail if failure == "capture" else fake_capture)
    if failure == "frontiers":
        monkeypatch.setattr(bridge.env, "sample_frontier_waypoints", fail)
    trajectory = (tmp_path / "trajectory.jsonl").read_bytes()

    class FakeHTTPServer:
        """Record context-manager cleanup and reject any request-loop entry."""

        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 9466) and handler is server.CommandHandler
            self.server_address = address
            events.append("bind")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            assert exc_type is RuntimeError and exc is error
            events.append("close")

        def handle_request(self):
            raise AssertionError("must not serve before a successful observation")

    monkeypatch.setattr(server, "HTTPServer", FakeHTTPServer)
    with pytest.raises(RuntimeError, match="initial RGB-D observation failed") as raised:
        server.serve(bridge, 9466, "test-VERSION", gpu=simulator_gpu)
    assert raised.value is error
    assert events == ["bind", "failure", "close"]
    assert not (tmp_path / "server_ready.json").exists()
    assert records(tmp_path / "actions.jsonl") == [
        {"command_id": 0, "event": "begin", "command": {"action": "observe"}},
        {"command_id": 0, "event": "error", "error": str(error)}]
    assert (tmp_path / "trajectory.jsonl").read_bytes() == trajectory
    if failure == "frontiers":
        assert records(bridge.capture_root / "frames.jsonl")[0]["frame_id"] == 0


@pytest.mark.parametrize("options", [
    ["--observation-mode", "single"],
    ["--observation-mode", "multi"],
    ["--observation-mode=multi"],
    ["--panorama-views", "4"],
])
def test_server_cli_rejects_policy_observation_options_before_resources(tmp_path, monkeypatch, capsys, options):
    """The simulator bridge accepts no navigation observation configuration."""
    resources = ["load_public_episode", "gpu_identity", "IsaacEnvironment", "serve"]
    factories = [Mock(side_effect=AssertionError("must not acquire resources")) for _ in resources]
    for name, factory in zip(resources, factories):
        monkeypatch.setattr(server, name, factory)
    monkeypatch.setattr(sys, "argv", ["serve_intentionnav.py", "--public-episode", str(tmp_path / "absent.json"),
        "--data-root", str(tmp_path), "--output", str(tmp_path / "absent"), *options])
    with pytest.raises(SystemExit) as error:
        server.main()
    assert error.value.code == 2
    assert "unrecognized arguments:" in capsys.readouterr().err
    for factory in factories:
        factory.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", [None, "validate", "load_scene", "place_agent", "bridge", "version", "serve"])
def test_cli_only_public_inputs_strips_argv_and_always_closes(tmp_path, monkeypatch, simulator_gpu, failure):
    """Isolate CUDA before official imports while retaining the Vulkan ordinal and cleanup."""
    path = tmp_path / "public.json"
    path.write_text(json.dumps(public_episode()))
    output = tmp_path / "run"
    output.mkdir()
    env = FakeEnv()
    seen, imports, lifecycle = [], [], []
    receipt = {**simulator_gpu, "cuda_device_count": 1, "cuda_uuid": simulator_gpu["uuid"],
               "physics_cuda_device": 0, "renderer_active_gpu": 7}
    error = RuntimeError("injected failure")

    def assert_isolated():
        assert sys.argv == ["serve_intentionnav.py", "--/physics/cudaDevice=0"]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == simulator_gpu["uuid"]

    original_import = builtins.__import__

    def isolated_import(name, *args, **kwargs):
        if name in {"iss_env", "walkable_map", "backends.isaac.capture", "isaacsim"}:
            assert_isolated()
            imports.append(name)
        return original_import(name, *args, **kwargs)

    def identity(index):
        assert index == 7
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-device-marker"
        assert sys.argv[-2:] == ["--gpu", "7"]
        assert imports == [] and lifecycle == []
        return dict(simulator_gpu)

    def factory(**kwargs):
        assert_isolated()
        assert kwargs == {"gpu": 7}
        assert imports == ["iss_env", "walkable_map", "backends.isaac.capture"]
        lifecycle.append("initialize")
        return env

    def fail(*args, **kwargs):
        raise error

    def validate(expected):
        assert expected == simulator_gpu
        assert_isolated()
        assert lifecycle == ["initialize"] and not env.closed
        assert not hasattr(env, "scene")
        lifecycle.append("validate")
        if failure == "validate":
            fail()
        return receipt

    def serve(bridge, port, version, *, gpu):
        assert port == 9466 and version == "test-VERSION"
        assert gpu is receipt
        assert bridge.env.scene == ("scene", str(tmp_path / "scenes" / "scene" / "start_result_navigation.usd"))
        assert bridge.env.position == public_episode()["start_position"]
        assert bridge.env.start_quat == public_episode()["start_rotation_quat_wxyz"]
        seen.append("serve")
        if failure == "serve":
            fail()

    close = Mock(wraps=env.close)
    resolve = Mock(side_effect=identity)
    check = Mock(side_effect=validate)
    monkeypatch.setenv("INTENTIONNAV_ROOT", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-device-marker")
    monkeypatch.setattr(sys, "argv", ["serve_intentionnav.py", "--public-episode", str(path),
                                     "--data-root", str(tmp_path), "--output", str(output), "--gpu", "7"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "iss_env", SimpleNamespace(IsaacSimEnv=factory))
    monkeypatch.setitem(sys.modules, "walkable_map", SimpleNamespace(WalkableMap=FakeMap))
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "backends.isaac.capture", SimpleNamespace(capture_frame=fake_capture))
    monkeypatch.setattr(builtins, "__import__", isolated_import)
    monkeypatch.setattr(server, "gpu_identity", resolve)
    monkeypatch.setattr(server, "validate_simulator_device", check)
    monkeypatch.setattr(server, "load_freemap", lambda *args: FakeMap())
    monkeypatch.setattr(server, "isaac_version_file", fail if failure == "version" else lambda module: "test-VERSION")
    monkeypatch.setattr(server, "serve", serve)
    monkeypatch.setattr(env, "close", close)
    if failure in {"load_scene", "place_agent"}:
        monkeypatch.setattr(env, failure, fail)
    if failure == "bridge":
        monkeypatch.setattr(server, "IsaacEnvironment", fail)
    if failure:
        with pytest.raises(RuntimeError, match="injected failure") as raised:
            server.main()
        assert raised.value is error
    else:
        server.main()
        assert seen == ["serve"]
    resolve.assert_called_once_with(7)
    check.assert_called_once_with(simulator_gpu)
    close.assert_called_once_with()
    assert env.closed
    assert not (output / "server_ready.json").exists()


def test_version_is_read_from_installed_version_file(tmp_path, monkeypatch):
    """Publish the installation's exact VERSION value, not an assumed Isaac release."""
    (tmp_path / "VERSION").write_text("6.0.1-rc.test+build\n")
    monkeypatch.setenv("ISAACSIM_ROOT", str(tmp_path))
    assert server.isaac_version_file(SimpleNamespace()) == "6.0.1-rc.test+build"


def test_json_large_int_rejected_as_coordinate():
    """Reject finite-looking integers that overflow when converted to simulator coordinates."""
    from backends.isaac.environment import validate_command

    command = decode_json(json.dumps({"action": "look", "yaw": 10 ** 400}).encode())
    with pytest.raises(CommandError):
        validate_command(command)
