"""Serve one public IntentionNav episode through a main-thread loopback JSON bridge."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends.isaac.environment import (CommandError, IsaacEnvironment, MAX_BODY_BYTES,
                                        decode_json, finite_vector, require_main_thread)


PUBLIC_FIELDS = {"selection_id", "scene_id", "style", "instruction", "start_position",
                 "start_rotation_quat_wxyz"}


def load_public_episode(path):
    """Read only a frozen, six-field public episode and reject metadata-bearing inputs."""
    with Path(path).open("rb") as stream:
        episode = decode_json(stream.read(MAX_BODY_BYTES + 1))
    if not isinstance(episode, dict) or set(episode) != PUBLIC_FIELDS:
        raise CommandError("public episode must contain exactly the six public fields")
    for field in ("selection_id", "scene_id", "style", "instruction"):
        if not isinstance(episode[field], str) or not episode[field].strip():
            raise CommandError(f"public episode {field} must be nonempty text")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", episode["scene_id"]):
        raise CommandError("scene_id must be a safe scene name")
    episode["start_position"] = finite_vector(episode["start_position"], "start_position")
    quat = finite_vector(episode["start_rotation_quat_wxyz"], "start_rotation_quat_wxyz", 4)
    if not math.isclose(math.hypot(*quat), 1.0, abs_tol=1e-4, rel_tol=0):
        raise CommandError("start_rotation_quat_wxyz must be a unit quaternion")
    episode["start_rotation_quat_wxyz"] = quat
    return episode


def load_freemap(data_root, scene_id, map_class):
    """Load only binary freemap geometry, never WalkableMap.load's room sidecar."""
    import numpy as np

    path = Path(data_root) / "metadata" / "metadata_train" / scene_id / "freemap.npy"
    matrix = np.load(path, allow_pickle=False)
    if matrix.ndim != 2 or min(matrix.shape) < 3:
        raise ValueError("invalid freemap matrix")
    grid = (matrix[1:, 1:] == 1).astype(np.uint8)
    wm = map_class(scene_id=scene_id, x_coords=matrix[0, 1:].astype(np.float64),
                   y_coords=matrix[1:, 0].astype(np.float64), grid=grid.astype(np.uint8), rooms={})
    wm.explored_count = np.zeros_like(wm.grid, dtype=np.uint16)
    return wm


class CommandHandler(BaseHTTPRequestHandler):
    """Handle one bounded JSON command synchronously, without any filesystem RPC."""

    def setup(self):
        """Bound request reads so an incomplete client cannot stall the main loop forever."""
        self.request.settimeout(10)
        super().setup()

    def log_message(self, format, *args):
        """Keep the action journal authoritative instead of logging untrusted requests."""

    def _send_json(self, status, result):
        payload = json.dumps(result, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)
        self.wfile.flush()

    def send_error(self, code, message=None, explain=None):
        """Return JSON even for malformed HTTP or unsupported verbs."""
        if self.request_version == "HTTP/0.9":
            self.request_version = "HTTP/1.0"
        self._send_json(400, {"error": "invalid HTTP request"})

    def _read_command(self):
        if self.path != "/command":
            raise CommandError("only POST /command is supported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,8}", lengths[0]):
            raise CommandError("one valid Content-Length is required")
        size = int(lengths[0])
        if not 0 < size <= MAX_BODY_BYTES:
            raise CommandError("JSON body is empty or oversized")
        if self.headers.get("Transfer-Encoding") is not None:
            raise CommandError("Transfer-Encoding is unsupported")
        types = self.headers.get_all("Content-Type", [])
        if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise CommandError("Content-Type must be application/json")
        try:
            body = self.rfile.read(size)
        except (TimeoutError, OSError) as exc:
            raise CommandError("incomplete request body") from exc
        if len(body) != size:
            raise CommandError("incomplete request body")
        return decode_json(body)

    def do_POST(self):
        """Execute on the serving main thread and acknowledge shutdown before exiting."""
        shutdown = False
        try:
            command = self._read_command()
            result = self.server.bridge.command(command)
            shutdown = command["action"] == "shutdown"
            status = 200
        except CommandError as exc:
            status, result = 400, {"error": str(exc)}
            self.server.bridge.record_rejection(exc)
        except Exception as exc:
            status, result = 500, {"error": f"{type(exc).__name__}: {exc}"}
        try:
            self._send_json(status, result)
        finally:
            # No HTTPServer.shutdown(): it deadlocks when called on the serving thread.
            if shutdown:
                self.server.shutdown_requested = True


def serve(bridge, port, isaac_version):
    """Publish readiness after binding loopback, then handle requests on this thread."""
    require_main_thread()
    with HTTPServer(("127.0.0.1", port), CommandHandler) as server:
        server.bridge = bridge
        server.shutdown_requested = False
        ready = {"port": server.server_address[1], "pid": os.getpid(),
                 "isaac_version": isaac_version, "start_pose": bridge.env.get_pose()}
        with (bridge.output / "server_ready.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(ready, allow_nan=False) + "\n")
        while not server.shutdown_requested:
            server.handle_request()


def isaac_version_file(isaacsim_module):
    """Read the installed Isaac VERSION rather than inferring a release from a path."""
    if os.environ.get("ISAACSIM_ROOT"):
        return (Path(os.environ["ISAACSIM_ROOT"]) / "VERSION").read_text(encoding="utf-8").strip()
    for parent in Path(isaacsim_module.__file__).resolve().parents:
        path = parent / "VERSION"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Isaac VERSION not found; set ISAACSIM_ROOT")


def main():
    """Load only public episode geometry and close Isaac after the synchronous loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-episode", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9466)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("gpu must be nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    require_main_thread()
    episode = load_public_episode(args.public_episode.resolve(strict=True))
    output = args.output.resolve(strict=True)
    if not output.is_dir():
        parser.error("output must be an existing run directory")
    data_root = args.data_root.resolve(strict=True)
    official_root = Path(os.environ["INTENTIONNAV_ROOT"]).resolve(strict=True)
    # Kit must never see the bridge's CLI flags; all official imports stay here.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(official_root / "eval" / "simulator"))
    from iss_env import IsaacSimEnv
    from walkable_map import WalkableMap
    from backends.isaac.capture import capture_frame

    wm = load_freemap(data_root, episode["scene_id"], WalkableMap)
    env = IsaacSimEnv(gpu=args.gpu)
    try:
        env.load_scene(episode["scene_id"], str(data_root / "scenes" / episode["scene_id"]
                                               / "start_result_navigation.usd"))
        env.place_agent(episode["start_position"], episode["start_rotation_quat_wxyz"])
        bridge = IsaacEnvironment(env, wm, output, capture_frame)
        import isaacsim
        serve(bridge, args.port, isaac_version_file(isaacsim))
    finally:
        env.close()


if __name__ == "__main__":
    main()
