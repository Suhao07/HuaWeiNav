"""Serve one public IntentionNav episode through a main-thread loopback JSON bridge."""
import argparse
import ctypes
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import traceback
import uuid

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
            if isinstance(command, dict) and command.get("action") == "shutdown":
                token = command.pop("run_token", None)
                expected = getattr(self.server, "run_token", None)
                if expected is not None and token != expected:
                    raise CommandError("shutdown run token mismatch")
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


def process_identity(pid=None):
    """Read Linux PID lifetime/session evidence without trusting a reused PID number."""
    pid = os.getpid() if pid is None else pid
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("Invalid process PID")
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in {"Z", "X"}:
        raise ProcessLookupError("Process is no longer alive")
    return {"pid": pid, "parent_pid": int(fields[1]), "session_id": int(fields[3]),
            "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def source_manifest(source):
    """Hash copied source bytes, excluding interpreter caches, not a claimed git checkout."""
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(source).rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def gpu_inventory():
    """Query only public GPU identity fields, without initializing CUDA or Warp."""
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
        text=True, timeout=10).strip().splitlines()
    devices = []
    for row in rows:
        index, identity, bus = (part.strip() for part in row.split(","))
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", identity) or not re.fullmatch(
                r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bus):
            raise RuntimeError("Invalid nvidia-smi GPU identity")
        identity = "GPU-" + str(uuid.UUID(identity[4:]))
        devices.append({"physical_index": int(index), "uuid": identity, "pci_bus_id": bus})
    if not devices or len({d["uuid"] for d in devices}) != len(devices):
        raise RuntimeError("Missing or ambiguous GPU inventory")
    return devices


def gpu_identity(index, inventory=None):
    """Resolve an nvidia-smi physical ordinal, never a CUDA logical ordinal."""
    matches = [d for d in (gpu_inventory() if inventory is None else inventory)
               if d["physical_index"] == index]
    if len(matches) != 1:
        raise RuntimeError("Unable to resolve requested physical GPU")
    return matches[0]


def cuda_identity():
    """Read the UUID of the sole visible device from the CUDA driver itself."""
    driver = ctypes.CDLL("libcuda.so.1")
    count, device = ctypes.c_int(), ctypes.c_int()
    identity = (ctypes.c_ubyte * 16)()
    # CUdevice is an int; CUuuid is a 16-byte struct. Declare the driver ABI explicitly.
    for name, argtypes in (("cuInit", [ctypes.c_uint]),
                           ("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)]),
                           ("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
                           ("cuDeviceGetUuid", [ctypes.c_void_p, ctypes.c_int])):
        function = getattr(driver, name)
        function.argtypes, function.restype = argtypes, ctypes.c_int
    for function, arguments in ((driver.cuInit, (0,)),
                                (driver.cuDeviceGetCount, (ctypes.byref(count),))):
        if function(*arguments) != 0:
            raise RuntimeError("CUDA initialization/identity query failed")
    if count.value != 1:
        raise RuntimeError(f"Expected one UUID-isolated CUDA device, got {count.value}")
    for function, arguments in ((driver.cuDeviceGet, (ctypes.byref(device), 0)),
                                (driver.cuDeviceGetUuid, (ctypes.byref(identity), device))):
        if function(*arguments) != 0:
            raise RuntimeError("CUDA identity query failed")
    return {"device_count": count.value, "logical_index": 0,
            "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity))), "source": "CUDA driver cuDeviceGetUuid"}


def renderer_identity(log_path, inventory, *, byte_length=None):
    """Resolve Kit's active Vulkan row via unique UUID prefix AND PCI bus.

    Supported Kit tables abbreviate UUIDs and use hexadecimal bus IDs. Unknown
    layouts fail closed; this parser is not a substitute for deployment ABI testing.
    An allocation or Carb setting alone is not evidence of the active renderer.
    """
    with Path(log_path).open("rb") as stream:
        payload = stream.read() if byte_length is None else stream.read(byte_length)
    if byte_length is not None and len(payload) != byte_length:
        raise RuntimeError("Truncated Kit renderer receipt")
    text = payload.decode("utf-8", errors="replace")
    tables = text.split("| GPU |")
    if len(tables) < 2:
        raise RuntimeError("Kit did not report an active renderer device table")
    lines = tables[-1].splitlines()
    active = []
    for index, line in enumerate(lines):
        cells = [part.strip() for part in line.split("|")[1:-1]]
        if len(cells) != 7 or not cells[0].isdigit() or not re.fullmatch(r"Yes(?:\s*:\s*\d+)?", cells[2]):
            continue
        try:
            prefix = lines[index + 1].split("|")[-2].strip().removesuffix("..").replace("-", "").lower()
            bus = int(lines[index + 2].split("|")[-3].strip(), 16)
            if not re.fullmatch(r"[0-9a-f]{8,32}", prefix) or not 0 <= bus <= 255:
                raise ValueError("invalid UUID prefix or PCI bus")
        except (ValueError, IndexError) as exc:
            raise RuntimeError("Malformed Kit renderer identity receipt") from exc
        matches = [d for d in inventory if len(prefix) >= 8
                   and d["uuid"][4:].replace("-", "").startswith(prefix)
                   and int(d["pci_bus_id"].split(":")[1], 16) == bus]
        if len(matches) != 1:
            raise RuntimeError("Ambiguous/unmatched Kit renderer UUID and PCI bus")
        active.append({**matches[0], "renderer_index": int(cells[0])})
    if len(active) != 1:
        raise RuntimeError("Expected exactly one active Kit renderer")
    return {**active[0], "source": "Kit active Vulkan device table",
            "receipt_path": str(Path(log_path).resolve()), "receipt_bytes": len(payload),
            "receipt_sha256": hashlib.sha256(payload).hexdigest()}


def validate_startup_device(expected, log_path):
    """Reject wrong CUDA/Vulkan UUIDs before loading the episode scene."""
    import carb

    cuda = cuda_identity()
    renderer = renderer_identity(log_path, gpu_inventory())
    if cuda["uuid"] != expected["uuid"] or renderer["uuid"] != expected["uuid"]:
        raise RuntimeError("Actual CUDA/renderer GPU differs from the allocated UUID")
    settings = carb.settings.get_settings()
    physics_setting = settings.get("/physics/cudaDevice")
    render_setting = settings.get("/renderer/activeGpu")
    if physics_setting != 0 or render_setting != renderer["renderer_index"]:
        raise RuntimeError("CUDA logical/renderer settings disagree with device receipts")
    return {"requested": expected, "cuda": cuda, "renderer": renderer,
            "settings": {"physics_cuda_device": physics_setting, "renderer_active_gpu": render_setting}}


def validate_simulator_device(env, expected, log_path):
    """Recheck actual CUDA/render identities and attest physics once the World exists."""
    receipt = validate_startup_device(expected, log_path)
    cuda = receipt["cuda"]
    context = env._world.get_physics_context()
    dynamics = context.is_gpu_dynamics_enabled()
    broadphase = context.get_broadphase_type()
    if not isinstance(dynamics, bool) or broadphase not in {"GPU", "MBP", "SAP"}:
        raise RuntimeError("Physics execution device cannot be established")
    from isaacsim.core.simulation_manager import SimulationManager
    from omni.physx import get_physx_interface
    view = SimulationManager.get_physics_simulation_view()
    if view is None or not view.is_valid:
        raise RuntimeError("Physics simulation view is not initialized and valid")
    ordinal = view.device_ordinal
    override = get_physx_interface().get_overwrite_gpu_setting()
    if type(override) is not int or override not in {-1, 0, 1}:
        raise RuntimeError("Unknown PhysX execution override")
    physics = {"gpu_dynamics": dynamics, "broadphase": broadphase, "cuda_ordinal": ordinal,
               "gpu_override": override,
               "source": "SimulationView.device_ordinal + PhysX.get_overwrite_gpu_setting + World.get_physics_context"}
    gpu_execution = override == 1 or (override == -1 and (dynamics or broadphase == "GPU"))
    if gpu_execution:
        if type(ordinal) is not int or ordinal != cuda["logical_index"]:
            raise RuntimeError("PhysX CUDA device disagrees with isolated CUDA device")
        physics.update(device="cuda:0", uuid=cuda["uuid"], logical_index=ordinal)
    else:
        if type(ordinal) is not int or ordinal != -1:
            raise RuntimeError("PhysX did not confirm CPU physics execution")
        physics.update(device="cpu")
    return {**receipt, "physics": physics}


def validate_initial_observation(bridge, initial):
    """Check the live pose and persisted RGBD before publishing readiness."""
    if initial["pose"] != bridge.env.get_pose():
        raise RuntimeError("Initial observation differs from the live simulator pose")
    return validate_rgbd(initial, bridge.capture_root)


def validate_rgbd(initial, capture_root):
    """Independently decode persisted initial RGBD and validate its capture journal."""
    import numpy as np
    from PIL import Image
    from prior_map.observations import load_depth

    frame = initial["frame"]
    root = Path(initial["capture_root"]).resolve(strict=True)
    if (root != Path(capture_root).resolve(strict=True) or frame["depth_unit"] != "m"
            or frame["depth_convention"] != "optical_z" or type(frame["frame_id"]) is not int
            or frame["frame_id"] != 0):
        raise RuntimeError("Initial RGBD capture source/convention mismatch")
    paths = {}
    for key, expected in (("rgb_path", "rgb/000000.png"), ("depth_path", "depth/000000.npy")):
        path = (root / frame[key]).resolve(strict=True)
        if frame[key] != expected or path != root / expected:
            raise RuntimeError("Initial RGBD path escapes or substitutes capture source")
        paths[key] = path
    depth = load_depth(paths["depth_path"])
    with Image.open(paths["rgb_path"]) as image:
        rgb = np.asarray(image)
    if (depth.ndim != 2 or not depth.size or depth.dtype != np.float32 or not np.isfinite(depth).all()
            or (depth < 0).any() or not (depth > 0).any()):
        raise RuntimeError("Initial depth has no valid finite measurements")
    if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8 or list(depth.shape) != frame["size_hw"]:
        raise RuntimeError("Initial RGB/depth alignment mismatch")
    # Reject blank/constant fallbacks (including constant magenta). A semantic
    # placeholder cannot in general be recognized from pixels; source receipts
    # below bind this frame to the owned capture implementation as well.
    magenta = np.all(rgb == [255, 0, 255], axis=2)
    black = np.all(rgb == 0, axis=2)
    if not np.any(rgb != rgb[0, 0]) or (magenta.any() and np.all(magenta | black)):
        raise RuntimeError("Initial RGB is blank or a renderer placeholder")
    fraction = float((depth > 0).mean())
    reported = frame["valid_depth_fraction"]
    if (isinstance(reported, bool) or not isinstance(reported, (int, float))
            or not math.isfinite(reported) or not math.isclose(fraction, reported, abs_tol=1e-6)):
        raise RuntimeError("Initial depth validity metadata mismatch")
    pose = initial["pose"]
    position = finite_vector(pose["position"], "initial position")
    yaw = pose["yaw"]
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise RuntimeError("Initial pose yaw is invalid")
    transform = np.asarray(frame["camera_to_map"], dtype=float)
    intrinsic = np.asarray(frame["intrinsics"], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)
            or not np.allclose(transform[:3, 3], position, atol=1e-5)
            or not np.allclose(transform[:3, 2], [math.cos(yaw), math.sin(yaw), 0], atol=1e-5, rtol=0)):
        raise RuntimeError("Initial camera transform and actual pose disagree")
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not 0 <= intrinsic[0, 2] < depth.shape[1]
            or not 0 <= intrinsic[1, 2] < depth.shape[0]
            or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], [0, 0], rtol=0, atol=1e-8)
            or not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-8)):
        raise RuntimeError("Initial camera intrinsics are invalid")
    # Both files must belong to the actual capture journal, not a substituted record.
    journal_path = root / "frames.jsonl"
    if journal_path.resolve(strict=True) != journal_path:
        raise RuntimeError("Initial capture journal is substituted")
    journal = journal_path.read_bytes()
    if len(journal.splitlines()) != 1 or json.loads(journal) != frame:
        raise RuntimeError("Initial frame differs from capture journal")
    return {"frame_id": frame["frame_id"], "valid_depth_fraction": fraction, "frame": frame,
            "size_hw": list(depth.shape), "pose": pose,
            "source": "Isaac RGB + distance_to_image_plane annotators; USD camera transform",
            "files": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for key, path in paths.items()},
            "frames_sha256": hashlib.sha256(journal).hexdigest()}


def serve(bridge, port, isaac_version, *, expected_gpu, device_log, ready_timeout=120,
          run_token=None, provenance=None):
    """Publish readiness after a bounded real RGBD/device preflight, on the main thread."""
    require_main_thread()
    if not 0 < ready_timeout <= 120:
        raise ValueError("ready_timeout must be in (0, 120]")

    def expired(signum, frame):
        raise TimeoutError("Isaac RGBD/device readiness deadline exceeded")

    for name in ("server_ready.json", "server_ready.pending"):
        if (bridge.output / name).exists():
            raise FileExistsError("Refusing stale simulator readiness")
    with HTTPServer(("127.0.0.1", port), CommandHandler) as server:
        server.bridge = bridge
        server.run_token = run_token
        server.shutdown_requested = False
        previous = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, ready_timeout)
        try:
            initial = bridge.command({"action": "observe"})
            observation = validate_initial_observation(bridge, initial)
            gpu = validate_simulator_device(bridge.env, expected_gpu, device_log)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        ready = {"port": server.server_address[1], "pid": os.getpid(),
                 "process": process_identity(), "run_token": run_token,
                 "output_root": str(bridge.output), "capture_root": str(bridge.capture_root),
                 "source_provenance": provenance or {},
                 "isaac_version": isaac_version, "start_pose": initial["pose"],
                 "initial_observation_frame_id": initial["frame"]["frame_id"],
                 "initial_observation": observation, "gpu": gpu}
        # Atomic publication prevents the launcher from reading a partially written JSON.
        temporary = bridge.output / "server_ready.pending"
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(ready, allow_nan=False) + "\n")
        os.link(temporary, bridge.output / "server_ready.json")
        temporary.unlink()
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
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi physical GPU index")
    parser.add_argument("--gpu-uuid", help="expected physical UUID from the launcher")
    parser.add_argument("--run-token", help="unique launcher attempt identifier")
    parser.add_argument("--renderer-gpu", type=int, help="Kit/Vulkan index hint (verified against actual receipt)")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("gpu must be nonnegative")
    if args.renderer_gpu is not None and args.renderer_gpu < 0:
        parser.error("renderer-gpu must be nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    require_main_thread()
    episode = load_public_episode(args.public_episode.resolve(strict=True))
    output = args.output.resolve(strict=True)
    if not output.is_dir():
        parser.error("output must be an existing run directory")
    data_root = args.data_root.resolve(strict=True)
    official_root = Path(os.environ["INTENTIONNAV_ROOT"]).resolve(strict=True)
    device = gpu_identity(args.gpu)
    if args.gpu_uuid is not None and args.gpu_uuid.lower() != device["uuid"].lower():
        raise RuntimeError("Requested physical GPU index/UUID changed before launch")
    # Before ANY Isaac/Warp import. CUDA sees logical 0; Vulkan keeps its own enumeration.
    os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_log = output / "kit-device.log"
    if device_log.exists():
        raise FileExistsError("Refusing a stale Kit device receipt")
    # The deployed SimulationApp must honor this logical physics override.
    # gpu= is only a Vulkan hint; the actual renderer/physics receipts remain mandatory.
    sys.argv = [sys.argv[0], "--/physics/cudaDevice=0", f"--/log/file={device_log}"]
    sys.path.insert(0, str(official_root / "eval" / "simulator"))
    from iss_env import IsaacSimEnv
    from walkable_map import WalkableMap
    from backends.isaac.capture import capture_frame

    provenance = {}
    if args.run_token:
        source = Path(__file__).resolve().parents[1]
        manifest = output / "source_manifest.json"
        if source != output / "source" or source_manifest(source) != json.loads(manifest.read_bytes()):
            raise RuntimeError("Simulator source differs from copied source manifest")
        import iss_env
        import walkable_map
        external = {}
        for key, module, filename in (("simulator_source", iss_env, "iss_env.py"),
                                      ("walkable_map_source", walkable_map, "walkable_map.py")):
            path = Path(module.__file__).resolve(strict=True)
            if path != (official_root / "eval/simulator" / filename).resolve(strict=True):
                raise RuntimeError("Simulator imported an unexpected external source")
            external[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        provenance = {"source_root": str(source),
                      "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "public_episode_sha256": hashlib.sha256(args.public_episode.read_bytes()).hexdigest(),
                      **external}
    wm = load_freemap(data_root, episode["scene_id"], WalkableMap)
    env = IsaacSimEnv(gpu=args.gpu if args.renderer_gpu is None else args.renderer_gpu)
    try:
        validate_startup_device(device, device_log)
        env.load_scene(episode["scene_id"], str(data_root / "scenes" / episode["scene_id"]
                                               / "start_result_navigation.usd"))
        env.place_agent(episode["start_position"], episode["start_rotation_quat_wxyz"])
        bridge = IsaacEnvironment(env, wm, output, capture_frame)
        import isaacsim
        serve(bridge, args.port, isaac_version_file(isaacsim), expected_gpu=device, device_log=device_log,
              run_token=args.run_token, provenance=provenance)
    except BaseException:
        # SimulationApp's fast shutdown can terminate before Python prints the exception.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
