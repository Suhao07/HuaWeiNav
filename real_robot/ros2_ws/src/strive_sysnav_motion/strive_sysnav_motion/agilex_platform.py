"""ROS2 bridge and mode controller for the AgileX rosbridge interface.

The node owns one connection to the chassis rosbridge.  It accepts the final
``TwistStamped`` produced by ``SafetyVelocityMux``, sends the contained
``Twist`` to the selected automatic source, records chassis telemetry, and
switches the remote mux through one ROS service.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import socket
import struct
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool


class WebSocketError(RuntimeError):
    """Report a rosbridge transport or protocol failure."""


class RosbridgeSocket:
    """Small synchronous WebSocket client for the rosbridge text protocol."""

    def __init__(self, uri: str, interface: str = "") -> None:
        """Open a WebSocket connection.

        Args:
            uri: ``ws://`` or ``wss://`` rosbridge endpoint.
            interface: Optional local interface used for the TCP socket.

        Raises:
            WebSocketError: If connection or handshake fails.
        """

        parsed = urlparse(uri)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise WebSocketError(f"invalid rosbridge URI: {uri}")
        self._secure = parsed.scheme == "wss"
        self._host = parsed.hostname
        self._port = parsed.port or (443 if self._secure else 80)
        self._path = parsed.path or "/"
        if parsed.query:
            self._path += "?" + parsed.query
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.settimeout(5.0)
        try:
            if interface:
                raw.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode() + b"\0")
            raw.connect((self._host, self._port))
            if self._secure:
                import ssl

                raw = ssl.create_default_context().wrap_socket(raw, server_hostname=self._host)
            self._socket = raw
            self._handshake()
        except Exception as exc:
            raw.close()
            raise WebSocketError(f"rosbridge connection failed: {exc}") from exc

    def _handshake(self) -> None:
        """Complete the RFC 6455 client handshake."""

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._socket.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self._socket.recv(4096)
            if not chunk:
                break
            response += chunk
        header = response.decode("latin1", errors="replace")
        if " 101 " not in header.split("\r\n", 1)[0]:
            raise WebSocketError(f"rosbridge handshake rejected: {header.splitlines()[0] if header else 'empty response'}")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if f"Sec-WebSocket-Accept: {expected}" not in header:
            raise WebSocketError("rosbridge handshake returned an invalid accept key")

    def send(self, payload: dict[str, Any]) -> None:
        """Send one JSON rosbridge operation."""

        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_frame(0x1, data)

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        """Send one masked client WebSocket frame."""

        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, length)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self._socket.sendall(header + mask + masked)

    def receive(self, timeout_s: float) -> Optional[dict[str, Any]]:
        """Read one JSON text frame, returning ``None`` on a short timeout."""

        self._socket.settimeout(timeout_s)
        try:
            while True:
                opcode, payload = self._receive_frame()
                if opcode == 0x1:
                    return json.loads(payload.decode("utf-8"))
                if opcode == 0x8:
                    raise WebSocketError("rosbridge closed the WebSocket")
                if opcode == 0x9:
                    self._send_frame(0xA, payload)
        except socket.timeout:
            return None
        except json.JSONDecodeError as exc:
            raise WebSocketError(f"rosbridge returned invalid JSON: {exc}") from exc

    def _receive_frame(self) -> tuple[int, bytes]:
        """Read one WebSocket frame."""

        first, second = self._read_exact(2)
        opcode = first & 0x0F
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if second & 0x80 else b""
        payload = self._read_exact(length) if length else b""
        if mask:
            payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        return opcode, payload

    def _read_exact(self, size: int) -> bytes:
        """Read exactly ``size`` bytes or report a closed socket."""

        chunks: list[bytes] = []
        while size:
            chunk = self._socket.recv(size)
            if not chunk:
                raise WebSocketError("rosbridge socket closed")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        """Close the transport."""

        self._socket.close()


@dataclass
class ModeRequest:
    """One serialized remote mux operation."""

    automatic: bool
    done: threading.Event
    result: Optional[bool] = None
    message: str = ""
    cancelled: bool = False


class AgileXPlatform(Node):
    """Bridge HuaweiNav's final command and own the chassis mode transition."""

    def __init__(self) -> None:
        """Create ROS interfaces and the single rosbridge worker."""

        super().__init__("strive_agilex_platform")
        self.declare_parameter("input_cmd_topic", "/cmd_vel")
        self.declare_parameter("uri", "ws://192.168.1.102:9090")
        self.declare_parameter("interface", "")
        self.declare_parameter("remote_control_enabled", False)
        self.declare_parameter("controller_contract_file", "")
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("action_name", "/strive/execute_waypoint")
        self.declare_parameter("auto_topic", "/navflow_cmd_vel")
        self.declare_parameter("manual_topic", "/run_management/virtual_joy")
        self.declare_parameter("selected_topic", "/mux_vel/selected")
        self.declare_parameter("selected_output_topic", "/platform/chassis_selected")
        self.declare_parameter(
            "telemetry_topics",
            [
                "/ranger_status",
                "/robot_status",
                "/mux_vel/selected",
                "/navflow_cmd_vel",
                "/base_cmd_vel",
            ],
        )
        self.declare_parameter("telemetry_output_prefix", "/platform/chassis")
        self.declare_parameter("autonomy_enable_topic", "/platform/autonomy_enable")
        self.declare_parameter("manual_takeover_topic", "/platform/manual_takeover")
        self.declare_parameter("chassis_ready_topic", "/platform/chassis_ready")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("command_timeout_s", 1.0)
        self.declare_parameter("max_linear_mps", 0.5)
        self.declare_parameter("max_angular_rps", 0.8)
        self.declare_parameter("mode_request_timeout_s", 5.0)

        self._latest_command = TwistStamped()
        self._latest_command_at = 0.0
        self._hardware_contract = None
        self._hardware_state = None
        self._hardware_received_at = 0.0
        self._hardware_fault_epoch = 0
        self._remote_control_enabled = _as_bool(self.get_parameter("remote_control_enabled").value)
        self._validate_controller_contract()
        self._command_lock = threading.Lock()
        self._mode_lock = threading.Lock()
        self._automatic = False
        self._connected = False
        self._selected = ""
        self._request_queue: queue.Queue[ModeRequest] = queue.Queue()
        self._stop_event = threading.Event()

        self._autonomy_publisher = self.create_publisher(Bool, str(self.get_parameter("autonomy_enable_topic").value), 10)
        self._manual_publisher = self.create_publisher(Bool, str(self.get_parameter("manual_takeover_topic").value), 10)
        self._ready_publisher = self.create_publisher(Bool, str(self.get_parameter("chassis_ready_topic").value), 10)
        self._selected_publisher = self.create_publisher(String, str(self.get_parameter("selected_output_topic").value), 10)
        self._telemetry_publishers = {
            topic: self.create_publisher(
                String,
                f"{str(self.get_parameter('telemetry_output_prefix').value).rstrip('/')}/{_telemetry_suffix(topic)}",
                10,
            )
            for topic in self._telemetry_topics()
        }
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("input_cmd_topic").value),
            self._update_command,
            10,
        )
        self.create_service(SetBool, "/platform/set_autonomy", self._set_autonomy)
        period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(period, self._publish_mode_state)
        self._worker = threading.Thread(target=self._run_worker, name="agilex-rosbridge", daemon=True)
        self._worker.start()

    def _telemetry_topics(self) -> list[str]:
        """Return configured remote telemetry topics."""

        value = self.get_parameter("telemetry_topics").value
        return [str(topic) for topic in value]

    def _validate_controller_contract(self) -> None:
        """Require the reviewed motion contract before enabling remote control."""

        if not self._remote_control_enabled:
            return
        from real_robot.control.controller_contract import (
            ControllerContractError,
            load_controller_contract,
            validate_controller_contract,
        )

        try:
            contract = load_controller_contract(
                str(self.get_parameter("controller_contract_file").value or "")
            )
            validate_controller_contract(
                contract,
                waypoint_topic=str(self.get_parameter("waypoint_topic").value),
                world_frame=str(self.get_parameter("world_frame").value),
                action_name=str(self.get_parameter("action_name").value),
            )
            hardware = contract.get("agilex_rosbridge", "hardware_state")
            if not hardware or any(not hardware.get(key) for key in (
                "allowed_control_modes", "ready_base_states", "ready_fault_codes",
            )) or not float(hardware.get("telemetry_timeout_s", 0.0)) > 0.0:
                raise ControllerContractError(
                    "agilex_rosbridge.hardware_state requires confirmed allowed_control_modes, "
                    "ready_base_states, ready_fault_codes and positive telemetry_timeout_s"
                )
            self._hardware_contract = hardware
            runtime_input = _normalize_topic(str(self.get_parameter("input_cmd_topic").value))
            contract_input = _normalize_topic(str(contract.get("final_cmd_topic", default="")))
            if runtime_input != contract_input:
                raise ControllerContractError(
                    f"AgileX bridge input_cmd_topic={runtime_input!r} does not match "
                    f"approved final_cmd_topic={contract_input!r}"
                )
            for parameter_name, contract_name in (
                ("auto_topic", "automatic_topic"),
                ("manual_topic", "manual_topic"),
            ):
                approved_topic = str(contract.get("agilex_rosbridge", contract_name, default="")).strip()
                runtime_topic = _normalize_topic(str(self.get_parameter(parameter_name).value))
                if approved_topic and runtime_topic != _normalize_topic(approved_topic):
                    raise ControllerContractError(
                        f"AgileX bridge {parameter_name} does not match approved agilex_rosbridge.{contract_name}"
                    )
            limits = (
                ("max_linear_mps", "max_linear_speed_mps"),
                ("max_angular_rps", "max_angular_speed_rps"),
                ("command_timeout_s", "command_watchdog_timeout_s"),
            )
            for parameter_name, contract_name in limits:
                configured = float(self.get_parameter(parameter_name).value)
                if configured <= 0.0:
                    raise ControllerContractError(
                        f"AgileX bridge {parameter_name} must be positive in physical mode"
                    )
                approved = float(contract.get("safety", contract_name, default=0.0))
                if configured > approved:
                    raise ControllerContractError(
                        f"AgileX bridge {parameter_name}={configured} exceeds approved {approved}"
                    )
        except ControllerContractError as exc:
            raise RuntimeError(f"AgileX remote-control contract gate failed: {exc}") from exc

    def _update_command(self, msg: TwistStamped) -> None:
        """Cache the final safety-mux command using local receive time."""

        with self._command_lock:
            self._latest_command = msg
            self._latest_command_at = time.monotonic()

    def _set_autonomy(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """Select the remote source and publish the matching local mode."""

        if not self._remote_control_enabled:
            response.success = False
            response.message = "remote control is disabled; start the node with remote_control_enabled=true"
            return response
        operation = ModeRequest(bool(request.data), threading.Event())
        self._request_queue.put(operation)
        if not operation.done.wait(float(self.get_parameter("mode_request_timeout_s").value)):
            operation.cancelled = True
            response.success = False
            response.message = "mode request timed out while waiting for rosbridge"
            return response
        response.success = bool(operation.result)
        response.message = operation.message
        return response

    def _publish_mode_state(self) -> None:
        """Publish authorization and current software/hardware readiness."""

        with self._mode_lock:
            automatic = self._automatic
            connected = self._connected
            selected = self._selected
        self._autonomy_publisher.publish(Bool(data=automatic))
        self._manual_publisher.publish(Bool(data=self._remote_control_enabled and not automatic))
        self._ready_publisher.publish(Bool(data=connected and automatic and self._hardware_ready() and selected == str(self.get_parameter("auto_topic").value)))
        self._selected_publisher.publish(String(data=selected))

    def _hardware_ready(self) -> bool:
        """Evaluate confirmed robot_status values and monotonic receive age.

        就绪条件为模式、基座状态、故障码属于本车合同的许可集合，且
        0 <= t_now - t_receive <= T。合同记录实际枚举，代码保持消息原值；
        selected 的事件发布频率与该周期遥测超时独立。
        """

        with self._mode_lock:
            state = self._hardware_state
            received_at = self._hardware_received_at
        contract = self._hardware_contract
        if contract is None or state is None:
            return False
        return (
            state[0] in contract["allowed_control_modes"]
            and state[1] in contract["ready_base_states"]
            and state[2] in contract["ready_fault_codes"]
            and 0.0 <= time.monotonic() - received_at <= float(contract["telemetry_timeout_s"])
        )

    def _check_hardware_authorization(self, socket_client: RosbridgeSocket) -> None:
        """Withdraw active authorization and send zero after hardware invalidation."""

        if self._automatic and not self._hardware_ready():
            with self._mode_lock:
                self._automatic = False
            with self._command_lock:
                self._latest_command_at = 0.0
            socket_client.send(self._zero_command())
            self.get_logger().error("AgileX hardware state invalid or stale; automatic authorization revoked")

    def _run_worker(self) -> None:
        """Run the sole owner of WebSocket operations and mode transitions."""

        retry_s = 1.0
        while not self._stop_event.is_set():
            socket_client: Optional[RosbridgeSocket] = None
            try:
                socket_client = RosbridgeSocket(
                    str(self.get_parameter("uri").value),
                    str(self.get_parameter("interface").value),
                )
                self._subscribe(socket_client)
                self._set_connected(True)
                if self._remote_control_enabled:
                    # 每次连接从人工源开始；自动授权必须由本次连接中的显式服务请求产生。
                    initial_mode = ModeRequest(False, threading.Event())
                    self._apply_mode(socket_client, initial_mode)
                    if not initial_mode.result:
                        raise WebSocketError(initial_mode.message or "could not select the manual source")
                self._worker_session(socket_client)
            except (WebSocketError, OSError) as exc:
                self._set_connected(False)
                self.get_logger().error(f"AgileX rosbridge connection failed: {exc}")
                self._fail_pending_requests(str(exc))
                if self._stop_event.wait(retry_s):
                    break
            except Exception:
                self._set_connected(False)
                self._fail_pending_requests("unexpected AgileX bridge failure")
                self.get_logger().error(
                    "unexpected AgileX bridge failure:\n" + traceback.format_exc()
                )
                break
            finally:
                if socket_client is not None:
                    if self._remote_control_enabled:
                        try:
                            self._restore_manual_source(socket_client)
                        except (WebSocketError, OSError) as exc:
                            self.get_logger().warning(f"could not restore the remote manual source: {exc}")
                    socket_client.close()

    def _subscribe(self, socket_client: RosbridgeSocket) -> None:
        """Subscribe to selected-source and raw chassis telemetry topics."""

        socket_client.send({
            "op": "subscribe",
            "topic": str(self.get_parameter("selected_topic").value),
            "type": "std_msgs/String",
        })
        for topic in self._telemetry_topics():
            if topic != str(self.get_parameter("selected_topic").value):
                socket_client.send({"op": "subscribe", "topic": topic})

    def _worker_session(self, socket_client: RosbridgeSocket) -> None:
        """Process mode requests, telemetry and automatic commands."""

        last_publish = 0.0
        last_ping = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            self._check_hardware_authorization(socket_client)
            request = self._next_request()
            if request is not None:
                self._apply_mode(socket_client, request)
            if self._remote_control_enabled and self._automatic and now - last_publish >= 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value)):
                socket_client.send(self._automatic_command())
                last_publish = now
            if now - last_ping >= 2.0:
                socket_client.send({"op": "ping", "id": "strive_agilex_platform", "timeStamp": int(time.time() * 1000)})
                last_ping = now
            message = socket_client.receive(0.02)
            if message is not None:
                self._consume_remote_message(message)

    def _next_request(self) -> Optional[ModeRequest]:
        """Return one pending mode request without blocking the worker."""

        try:
            return self._request_queue.get_nowait()
        except queue.Empty:
            return None

    def _apply_mode(self, socket_client: RosbridgeSocket, request: ModeRequest) -> None:
        """Perform one ordered remote mux transition."""

        target = str(self.get_parameter("auto_topic").value if request.automatic else self.get_parameter("manual_topic").value)
        try:
            if request.cancelled:
                request.result = False
                request.message = "mode request expired before rosbridge became available"
                return
            if request.automatic and not self._hardware_ready():
                request.result = False
                request.message = "hardware state is invalid or stale"
                return
            hardware_epoch = self._hardware_fault_epoch
            with self._mode_lock:
                self._automatic = False
            with self._command_lock:
                self._latest_command_at = 0.0
            socket_client.send({
                "op": "advertise",
                "topic": str(self.get_parameter("auto_topic").value),
                "type": "geometry_msgs/Twist",
            })
            socket_client.send(self._zero_command())
            if request.automatic:
                add_id = f"add:{target}:{time.time():.6f}"
                socket_client.send({
                    "op": "call_service",
                    "service": "/mux_vel/add",
                    "id": add_id,
                    "args": {"topic": target},
                })
                add_response = self._wait_for_service_id(socket_client, add_id, request)
                if not bool(add_response.get("result")):
                    # 松灵 mux 保留已注册 source；select 是最终可用性判据。
                    self.get_logger().warning(
                        f"/mux_vel/add rejected an existing or invalid source {target}: "
                        f"{add_response.get('values')}"
                    )
                if request.cancelled:
                    raise WebSocketError("mode request expired before selecting the automatic source")
            select_id = f"select:{target}:{time.time():.6f}"
            socket_client.send({
                "op": "call_service",
                "service": "/mux_vel/select",
                "id": select_id,
                "args": {"topic": target},
            })
            response = self._wait_for_service_id(socket_client, select_id, request)
            if not bool(response.get("result")):
                raise WebSocketError(f"/mux_vel/select rejected {target}: {response.get('values')}")
            if request.cancelled:
                raise WebSocketError("mode request expired after selecting the remote source")
            self._wait_for_selected(socket_client, target, request)
            if request.automatic and (
                not self._hardware_ready() or self._hardware_fault_epoch != hardware_epoch
            ):
                raise WebSocketError("hardware state changed during automatic selection")
            # 新授权只接收切换完成后的速度；selected 始终保留底盘实际反馈。
            with self._command_lock:
                self._latest_command_at = 0.0
            with self._mode_lock:
                self._automatic = request.automatic
            request.result = True
            request.message = f"selected {target}"
        except (WebSocketError, OSError) as exc:
            cleanup_error = ""
            try:
                self._restore_manual_source(socket_client)
            except (WebSocketError, OSError) as restore_exc:
                cleanup_error = f"; manual source restore failed: {restore_exc}"
            with self._mode_lock:
                self._automatic = False
            request.result = False
            request.message = str(exc) + cleanup_error
        finally:
            request.done.set()

    def _restore_manual_source(self, socket_client: RosbridgeSocket) -> None:
        """Stop automatic input and confirm the remote manual source."""

        with self._mode_lock:
            self._automatic = False
        with self._command_lock:
            self._latest_command_at = 0.0
        socket_client.send(self._zero_command())
        target = str(self.get_parameter("manual_topic").value)
        request_id = f"select:{target}:{time.time():.6f}"
        socket_client.send({
            "op": "call_service",
            "service": "/mux_vel/select",
            "id": request_id,
            "args": {"topic": target},
        })
        cleanup_request = ModeRequest(False, threading.Event())
        response = self._wait_for_service_id(
            socket_client,
            request_id,
            cleanup_request,
            respect_stop_event=False,
        )
        if not bool(response.get("result")):
            raise WebSocketError(f"/mux_vel/select rejected {target}: {response.get('values')}")
        self._wait_for_selected(socket_client, target, cleanup_request, respect_stop_event=False)

    def _wait_for_selected(
        self,
        socket_client: RosbridgeSocket,
        target: str,
        request: ModeRequest,
        *,
        respect_stop_event: bool = True,
    ) -> None:
        """Confirm mux feedback while processing telemetry on the same connection.

        selected 按选源事件更新；当前连接已有的同源反馈也有效，周期遥测的
        新鲜度规则应单独处理。服务响应只确认调用结果，反馈确认实际选源。
        """

        deadline = time.monotonic() + float(self.get_parameter("mode_request_timeout_s").value)
        while not request.cancelled and (not respect_stop_event or not self._stop_event.is_set()):
            with self._mode_lock:
                if self._selected == target:
                    return
            if time.monotonic() >= deadline:
                break
            message = socket_client.receive(0.02)
            if message is not None:
                self._consume_remote_message(message)
        raise WebSocketError(f"timed out waiting for /mux_vel/selected={target}")

    def _wait_for_service_id(
        self,
        socket_client: RosbridgeSocket,
        request_id: str,
        request: ModeRequest,
        *,
        respect_stop_event: bool = True,
    ) -> dict[str, Any]:
        """Wait for one service response while continuing telemetry handling."""

        deadline = time.monotonic() + float(self.get_parameter("mode_request_timeout_s").value)
        while (
            not request.cancelled
            and (not respect_stop_event or not self._stop_event.is_set())
            and time.monotonic() < deadline
        ):
            message = socket_client.receive(0.05)
            if message is None:
                continue
            if message.get("op") == "service_response" and str(message.get("id", "")) == request_id:
                return message
            self._consume_remote_message(message)
        raise WebSocketError(f"timed out waiting for service response {request_id}")

    def _consume_remote_message(self, message: dict[str, Any]) -> None:
        """Publish remote status and update the observed selected source."""

        if message.get("op") == "status" and message.get("level") == "error":
            raise WebSocketError(f"rosbridge reported an error: {message.get('msg', message)}")
        topic = str(message.get("topic", ""))
        payload = message.get("msg")
        if topic == "/robot_status":
            # 真实消息定义：robot_status_core/RobotStatus，三个字段均为整数。
            with self._mode_lock:
                self._hardware_state = (payload["control_mode"], payload["base_state"], payload["fault_code"])
                self._hardware_received_at = time.monotonic()
            if not self._hardware_ready():
                self._hardware_fault_epoch += 1
        publisher = self._telemetry_publishers.get(topic)
        if publisher is not None:
            publisher.publish(String(data=json.dumps({
                "source_topic": topic,
                "received_time_ns": time.time_ns(),
                "received_monotonic_ns": time.monotonic_ns(),
                "msg": payload,
                "raw": message,
            }, ensure_ascii=False)))
        if topic == str(self.get_parameter("selected_topic").value):
            selected = str(payload.get("data", "")) if isinstance(payload, dict) else str(payload)
            with self._mode_lock:
                self._selected = selected
                if selected != str(self.get_parameter("auto_topic").value):
                    self._automatic = False
            return

    def _automatic_command(self) -> dict[str, Any]:
        """Build a rosbridge Twist publish operation from the final command."""

        with self._command_lock:
            msg = self._latest_command
            received_at = self._latest_command_at
        timeout = float(self.get_parameter("command_timeout_s").value)
        if received_at <= 0.0 or time.monotonic() - received_at > timeout:
            return self._zero_command()
        max_linear = float(self.get_parameter("max_linear_mps").value)
        max_angular = float(self.get_parameter("max_angular_rps").value)
        # 这里只做逐分量限幅；坐标系、运动学和加速度约束由上游安全 mux 负责。
        return {
            "op": "publish",
            "topic": str(self.get_parameter("auto_topic").value),
            "msg": {
                "linear": {
                    "x": _clip(msg.twist.linear.x, max_linear),
                    "y": _clip(msg.twist.linear.y, max_linear),
                    "z": _clip(msg.twist.linear.z, max_linear),
                },
                "angular": {
                    "x": _clip(msg.twist.angular.x, max_angular),
                    "y": _clip(msg.twist.angular.y, max_angular),
                    "z": _clip(msg.twist.angular.z, max_angular),
                },
            },
        }

    def _zero_command(self) -> dict[str, Any]:
        """Build a zero-speed command for the automatic source."""

        return {
            "op": "publish",
            "topic": str(self.get_parameter("auto_topic").value),
            "msg": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}},
        }

    def _set_connected(self, connected: bool) -> None:
        """Update transport state and revoke automatic mode on disconnect."""

        with self._mode_lock:
            self._connected = connected
            if not connected:
                self._automatic = False
                self._selected = ""
                self._hardware_state = None
                self._hardware_received_at = 0.0

    def _fail_pending_requests(self, message: str) -> None:
        """Complete queued mode requests with the connection failure."""

        while True:
            try:
                request = self._request_queue.get_nowait()
            except queue.Empty:
                return
            request.result = False
            request.message = message
            request.done.set()

    def destroy_node(self) -> bool:
        """Stop the worker before destroying ROS resources."""

        self._stop_event.set()
        if self._worker.is_alive():
            timeout = float(self.get_parameter("mode_request_timeout_s").value) + 1.0
            self._worker.join(timeout=max(2.0, timeout))
        return super().destroy_node()


def _clip(value: float, limit: float) -> float:
    """Clip a command component to a configured absolute limit.

    The bridge applies the component-wise bound
    ``v_out = max(-v_max, min(v_max, v_in))``.  方向、坐标变换和加速度限制由
    上游安全 mux 负责。
    """

    value = float(value)
    return max(-limit, min(limit, value)) if limit > 0.0 else value


def _telemetry_suffix(topic: str) -> str:
    """Keep FWTW recorder names for command and mux observations."""

    return {
        "/mux_vel/selected": "mux_selected",
        "/navflow_cmd_vel": "navflow_cmd_vel_received",
        "/base_cmd_vel": "base_cmd_vel_received",
    }.get(topic, topic.strip("/").replace("/", "_"))


def _normalize_topic(value: str) -> str:
    """Normalize a ROS topic for contract comparisons."""

    text = str(value or "").strip()
    return "/" + text.strip("/") if text else ""


def _as_bool(value: object) -> bool:
    """Parse a launch or command-line boolean parameter."""

    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def main(args: Optional[list[str]] = None) -> None:
    """Run the AgileX platform node."""

    rclpy.init(args=args)
    node = AgileXPlatform()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
