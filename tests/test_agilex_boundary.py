from pathlib import Path
from collections import deque
import importlib.util
import json
import struct
import sys
import threading
import types
from unittest.mock import patch

import pytest


MOTION_ROOT = Path("real_robot/ros2_ws/src/strive_sysnav_motion")


class _Parameter:
    def __init__(self, value):
        self.value = value


class _Logger:
    def warning(self, message):
        self.message = message

    def error(self, message):
        self.message = message


class _FakeSocket:
    def __init__(self, responses=()):
        self.sent = []
        self.responses = deque(responses)

    def send(self, message):
        self.sent.append(message)

    def receive(self, _timeout):
        response = self.responses.popleft()
        if isinstance(response, bool):
            request = self.sent[-1]
            return {
                "op": "service_response",
                "id": request["id"],
                "result": response,
                "values": {},
            }
        return response


def _load_bridge_module():
    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = object
    geometry = types.ModuleType("geometry_msgs.msg")
    geometry.TwistStamped = type("TwistStamped", (), {})
    std_msgs = types.ModuleType("std_msgs.msg")
    std_msgs.Bool = types.SimpleNamespace
    std_msgs.String = types.SimpleNamespace
    std_srvs = types.ModuleType("std_srvs.srv")
    std_srvs.SetBool = type("SetBool", (), {
        "Request": type("Request", (), {}),
        "Response": type("Response", (), {}),
    })
    contract = types.ModuleType("real_robot.control.controller_contract")
    contract.ControllerContractError = type("ControllerContractError", (ValueError,), {})
    contract.load_controller_contract = lambda _path: None
    contract.validate_controller_contract = lambda *args, **kwargs: None
    modules = {
        "rclpy": rclpy,
        "rclpy.node": rclpy.node,
        "geometry_msgs": types.ModuleType("geometry_msgs"),
        "geometry_msgs.msg": geometry,
        "std_msgs": types.ModuleType("std_msgs"),
        "std_msgs.msg": std_msgs,
        "std_srvs": types.ModuleType("std_srvs"),
        "std_srvs.srv": std_srvs,
        "real_robot": types.ModuleType("real_robot"),
        "real_robot.control": types.ModuleType("real_robot.control"),
        "real_robot.control.controller_contract": contract,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        path = MOTION_ROOT / "strive_sysnav_motion/agilex_platform.py"
        spec = importlib.util.spec_from_file_location("agilex_platform_fake_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("agilex_platform_fake_test", None)
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _fake_platform(module, remote_control_enabled=True):
    platform = module.AgileXPlatform.__new__(module.AgileXPlatform)
    values = {
        "auto_topic": "/navflow_cmd_vel",
        "manual_topic": "/run_management/virtual_joy",
        "mode_request_timeout_s": 1.0,
        "selected_topic": "/mux_vel/selected",
        "telemetry_topics": ["/ranger_status", "/mux_vel/selected"],
    }
    platform.get_parameter = lambda name: _Parameter(values[name])
    platform.get_logger = lambda: _Logger()
    platform._mode_lock = threading.Lock()
    platform._command_lock = threading.Lock()
    platform._latest_command_at = 123.0
    platform._automatic = False
    platform._selected = ""
    platform._telemetry_publishers = {}
    platform._stop_event = threading.Event()
    platform._remote_control_enabled = remote_control_enabled
    # 合成许可值只验证配置语义，不代表本车硬件枚举。
    platform._hardware_contract = {
        "allowed_control_modes": [71], "ready_base_states": [72],
        "ready_fault_codes": [0], "telemetry_timeout_s": 1.0,
    }
    platform._hardware_state = (71, 72, 0)
    platform._hardware_received_at = module.time.monotonic()
    platform._hardware_fault_epoch = 0
    return platform


def test_hardware_fault_revokes_authorization_and_requires_explicit_reenable() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    platform._automatic = True
    socket = _FakeSocket()
    platform._consume_remote_message({"topic": "/robot_status", "msg": {
        "control_mode": 71, "base_state": 72, "fault_code": 4,
    }})
    platform._check_hardware_authorization(socket)
    assert platform._automatic is False
    assert platform._latest_command_at == 0.0
    assert socket.sent == [platform._zero_command()]
    platform._consume_remote_message({"topic": "/robot_status", "msg": {
        "control_mode": 71, "base_state": 72, "fault_code": 0,
    }})
    assert platform._hardware_ready()
    assert platform._automatic is False


def test_hardware_timeout_and_reconnect_require_fresh_status() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    platform._automatic = True
    platform._hardware_received_at = module.time.monotonic() - 2.0
    socket = _FakeSocket()
    platform._check_hardware_authorization(socket)
    assert platform._automatic is False
    assert socket.sent == [platform._zero_command()]
    platform._set_connected(False)
    platform._set_connected(True)
    assert platform._hardware_state is None
    assert platform._hardware_ready() is False
    request = module.ModeRequest(True, threading.Event())
    platform._apply_mode(socket, request)
    assert request.result is False
    assert len(socket.sent) == 1


def test_hardware_takeover_during_selection_cancels_even_after_recovery() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    fault = {"topic": "/robot_status", "msg": {
        "control_mode": 73, "base_state": 72, "fault_code": 0,
    }}
    recovery = {"topic": "/robot_status", "msg": {
        "control_mode": 71, "base_state": 72, "fault_code": 0,
    }}
    selected = {"topic": "/mux_vel/selected", "msg": {"data": "/navflow_cmd_vel"}}
    manual = {"topic": "/mux_vel/selected", "msg": {"data": "/run_management/virtual_joy"}}
    socket = _FakeSocket([True, fault, recovery, selected, True, True, manual])
    request = module.ModeRequest(True, threading.Event())
    platform._apply_mode(socket, request)
    assert request.result is False
    assert "hardware state changed" in request.message
    assert platform._automatic is False
    assert platform._selected == "/run_management/virtual_joy"


def test_agilex_bridge_owns_the_twiststamped_to_twist_boundary() -> None:
    source = (MOTION_ROOT / "strive_sysnav_motion/agilex_platform.py").read_text(encoding="utf-8")

    assert "TwistStamped" in source
    assert 'msg.twist.linear.x' in source
    assert '"topic": str(self.get_parameter("auto_topic").value)' in source
    assert '"type": "geometry_msgs/Twist"' in source
    assert 'self.declare_parameter("remote_control_enabled", False)' in source
    assert 'if not self._remote_control_enabled:' in source
    assert "self._validate_controller_contract()" in source
    assert "AgileX remote-control contract gate failed" in source
    assert 'contract.get("final_cmd_topic", default="")' in source
    assert "approved agilex_rosbridge" in source
    assert "must be positive in physical mode" in source


def test_agilex_mode_switch_selects_remote_source_before_local_enable() -> None:
    source = (MOTION_ROOT / "strive_sysnav_motion/agilex_platform.py").read_text(encoding="utf-8")

    select_call = source.index('"service": "/mux_vel/select"')
    local_mode_update = source.index("self._automatic = request.automatic", select_call)
    assert select_call < local_mode_update
    assert '"service": "/mux_vel/add"' in source
    assert '"service": "/mux_vel/select"' in source
    assert "self._zero_command()" in source
    assert "operation.cancelled = True" in source


def test_real_robot_launch_exposes_agilex_and_separate_lower_cloud_topic() -> None:
    launch = Path(
        "real_robot/ros2_ws/src/strive_sysnav_bringup/launch/strive_real_robot_stack.launch.py"
    ).read_text(encoding="utf-8")
    runner = Path("scripts/run_sysnav_detection_mapping.sh").read_text(encoding="utf-8")
    docker_entry = Path("docker_en.sh").read_text(encoding="utf-8")
    lower_launch = Path(
        "real_robot/ros2_ws/src/strive_sysnav_motion/launch/sysnav_lower_stack.launch.py"
    ).read_text(encoding="utf-8")
    terrain = Path(
        "real_robot/ros2_ws/src/terrain_analysis/src/terrainAnalysis.cpp"
    ).read_text(encoding="utf-8")

    assert '"start_agilex_platform": "false"' in launch
    assert "agilex_platform.launch.py" in launch
    assert '"agilex_remote_control_enabled": "false"' in launch
    assert '"controller_contract_file": LaunchConfiguration("controller_contract_file")' in launch
    assert '"command_timeout_s": LaunchConfiguration("agilex_command_timeout_s")' in launch
    assert '"cloud_topic": LaunchConfiguration("lower_cloud_topic")' in launch
    assert '"terrain_frame": LaunchConfiguration("terrain_frame")' in launch
    assert '"lower_cloud_topic:=${LOWER_CLOUD_TOPIC:-/cloud_registered}"' in runner
    assert '"terrain_frame:=${LOWER_TERRAIN_FRAME:-map}"' in runner
    assert '"agilex_command_timeout_s:=${AGILEX_COMMAND_TIMEOUT_S:-1.0}"' in runner
    assert "Shadow lower stack requires STRIVE_DRY_RUN=true" in runner
    assert "Physical lower stack requires ALLOW_REAL_MOTION=true" in runner
    assert '"require_controller_contract:=${require_controller_contract}"' in runner
    assert '"ALLOW_REAL_MOTION=${ALLOW_REAL_MOTION}"' in docker_entry
    assert '"START_AGILEX_PLATFORM=${START_AGILEX_PLATFORM:-false}"' in docker_entry
    assert '"AGILEX_COMMAND_TIMEOUT_S=${AGILEX_COMMAND_TIMEOUT_S:-1.0}"' in docker_entry
    assert 'declare_parameter<string>("terrainFrame", terrainFrame)' in terrain
    assert 'terrainCloud2.header.frame_id = terrainFrame' in terrain
    assert '"chassis_ready_topic"' in lower_launch
    assert '"require_chassis_ready"' in lower_launch
    assert '"chassis_ready_timeout_s"' in lower_launch


def test_safety_mux_starts_a_new_autonomy_candidate_cycle_after_reenable() -> None:
    source = (
        MOTION_ROOT / "strive_sysnav_motion/safety_velocity_mux.py"
    ).read_text(encoding="utf-8")

    enable_edge = source.index("if enabled and not self._autonomy_enabled:")
    cache_clear = source.index("self._latest_autonomy = None", enable_edge)
    manual_edge = source.index("if self._manual_takeover:")
    manual_cache_clear = source.index("self._latest_autonomy = None", manual_edge)
    chassis_fault = source.index("SafetyState.CONTROLLER_FAULT")
    chassis_cache_clear = source.index("self._latest_autonomy = None", chassis_fault - 220)
    assert enable_edge < cache_clear
    assert manual_edge < manual_cache_clear
    assert chassis_cache_clear < chassis_fault


def test_fake_rosbridge_verifies_mode_order_and_subscriptions() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    socket = _FakeSocket(
        responses=[True, True, {
            "op": "publish", "topic": "/mux_vel/selected",
            "msg": {"data": "/navflow_cmd_vel"},
        }]
    )
    request = module.ModeRequest(True, threading.Event())
    platform._apply_mode(socket, request)

    assert request.result is True
    assert platform._automatic is True
    assert platform._selected == "/navflow_cmd_vel"
    assert platform._latest_command_at == 0.0
    assert [message["op"] for message in socket.sent] == [
        "advertise", "publish", "call_service", "call_service",
    ]
    assert socket.sent[2]["service"] == "/mux_vel/add"
    assert socket.sent[3]["service"] == "/mux_vel/select"
    assert socket.sent[3]["args"] == {"topic": "/navflow_cmd_vel"}

    subscribe_socket = _FakeSocket()
    platform._subscribe(subscribe_socket)
    assert subscribe_socket.sent[0] == {
        "op": "subscribe", "topic": "/mux_vel/selected", "type": "std_msgs/String"
    }
    assert {message["topic"] for message in subscribe_socket.sent} == {
        "/mux_vel/selected", "/ranger_status",
    }


def test_mode_cleanup_selects_manual_and_telemetry_keeps_raw_message() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    socket = _FakeSocket(responses=[True, {
        "op": "publish", "topic": "/mux_vel/selected",
        "msg": {"data": "/run_management/virtual_joy"},
    }])
    platform._restore_manual_source(socket)

    assert socket.sent[0]["op"] == "publish"
    assert socket.sent[1]["service"] == "/mux_vel/select"
    assert socket.sent[1]["args"] == {"topic": "/run_management/virtual_joy"}
    assert platform._automatic is False
    assert platform._selected == "/run_management/virtual_joy"

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    publisher = Publisher()
    platform._telemetry_publishers = {"/ranger_status": publisher}
    raw = {"op": "publish", "topic": "/ranger_status", "msg": {"header": {"stamp": 3}}}
    platform._consume_remote_message(raw)
    stored = json.loads(publisher.messages[0].data)
    assert stored["msg"] == raw["msg"]
    assert stored["raw"] == raw


def test_cancelled_auto_request_restores_manual_source() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    request = module.ModeRequest(True, threading.Event())

    class CancelAfterSelect(_FakeSocket):
        def receive(self, timeout):
            if len(self.sent) == 4:
                request.cancelled = True
            return super().receive(timeout)

    socket = CancelAfterSelect(responses=[True, True, True, {
        "op": "publish", "topic": "/mux_vel/selected",
        "msg": {"data": "/run_management/virtual_joy"},
    }])
    platform._apply_mode(socket, request)

    assert request.result is False
    assert socket.sent[-1]["service"] == "/mux_vel/select"
    assert socket.sent[-1]["args"] == {"topic": "/run_management/virtual_joy"}
    assert platform._automatic is False
    assert platform._selected == "/run_management/virtual_joy"


def test_selected_feedback_timeout_preserves_observation() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    platform._selected = "/run_management/virtual_joy"
    request = module.ModeRequest(True, threading.Event())
    with patch.object(module.time, "monotonic", side_effect=[0.0, 2.0]):
        with pytest.raises(module.WebSocketError, match="waiting for /mux_vel/selected"):
            platform._wait_for_selected(_FakeSocket(), "/navflow_cmd_vel", request)
    assert platform._selected == "/run_management/virtual_joy"
    assert platform._automatic is False


def test_selected_feedback_before_service_response_and_repeated_selection() -> None:
    module = _load_bridge_module()
    platform = _fake_platform(module)
    selected = {"op": "publish", "topic": "/mux_vel/selected",
                "msg": {"data": "/navflow_cmd_vel"}}
    for responses in ([True, selected, True], [True, True]):
        socket = _FakeSocket(responses)
        request = module.ModeRequest(True, threading.Event())
        platform._apply_mode(socket, request)
        assert request.result is True
        assert platform._automatic is True


def test_fake_rosbridge_decodes_a_text_frame_and_emits_masked_json() -> None:
    module = _load_bridge_module()

    class RawSocket:
        def __init__(self, incoming=b""):
            self.incoming = incoming
            self.outgoing = b""

        def sendall(self, data):
            self.outgoing += data

        def settimeout(self, _timeout):
            return None

        def recv(self, size):
            chunk, self.incoming = self.incoming[:size], self.incoming[size:]
            return chunk

    payload = json.dumps({"op": "publish", "topic": "/ranger_status"}).encode()
    incoming = struct.pack("!BB", 0x81, len(payload)) + payload
    raw = RawSocket(incoming)
    receiver = module.RosbridgeSocket.__new__(module.RosbridgeSocket)
    receiver._socket = raw
    assert receiver.receive(0.1) == json.loads(payload)

    ping_raw = RawSocket(struct.pack("!BB", 0x89, 1) + b"p" + incoming)
    ping_receiver = module.RosbridgeSocket.__new__(module.RosbridgeSocket)
    ping_receiver._socket = ping_raw
    assert ping_receiver.receive(0.1) == json.loads(payload)
    assert ping_raw.outgoing[0] == 0x8A

    sender = module.RosbridgeSocket.__new__(module.RosbridgeSocket)
    sender._socket = RawSocket()
    sender.send({"op": "ping", "id": "test"})
    assert sender._socket.outgoing[0] == 0x81
    assert sender._socket.outgoing[1] & 0x80
