"""ROS2 safety velocity mux for the migrated SysNav path follower."""

from __future__ import annotations

import time
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Empty
from strive_motion_msgs.msg import SafetyState as SafetyStateMessage

from real_robot.control.controller_contract import (
    ControllerContractError,
    load_controller_contract,
    validate_controller_contract,
)
from real_robot.motion_safety import SafetyState, SafetyVelocityPolicy, VelocityCommand, VelocityLimits


class SafetyVelocityMux(Node):
    """Own the final ``/cmd_vel`` output after safety arbitration."""

    def __init__(self) -> None:
        """Create command subscriptions, state inputs, and the output timer."""

        super().__init__("strive_safety_velocity_mux")
        self.declare_parameter("autonomy_cmd_topic", "/cmd_vel/autonomy")
        self.declare_parameter("manual_cmd_topic", "/cmd_vel/manual")
        self.declare_parameter("output_cmd_topic", "/cmd_vel")
        self.declare_parameter("autonomy_enable_topic", "/platform/autonomy_enable")
        self.declare_parameter("manual_takeover_topic", "/platform/manual_takeover")
        self.declare_parameter("estop_topic", "/platform/estop_active")
        self.declare_parameter("estop_reset_topic", "/platform/estop_reset")
        self.declare_parameter("hold_topic", "/platform/safe_hold")
        self.declare_parameter("safety_state_topic", "/platform/safety_state")
        self.declare_parameter("odom_topic", "/aft_mapped_to_init")
        self.declare_parameter("pointcloud_topic", "/cloud_registered")
        self.declare_parameter("require_sensor_freshness", True)
        self.declare_parameter("sensor_watchdog_timeout_s", 0.5)
        self.declare_parameter("output_rate_hz", 20.0)
        self.declare_parameter("max_linear_speed_mps", 0.5)
        self.declare_parameter("max_angular_speed_rps", 1.0)
        self.declare_parameter("max_linear_accel_mps2", 0.5)
        self.declare_parameter("max_angular_accel_rps2", 1.0)
        self.declare_parameter("command_watchdog_timeout_s", 0.25)
        self.declare_parameter("start_autonomy_enabled", False)
        self.declare_parameter("controller_contract_file", "")
        self.declare_parameter("require_controller_contract", True)
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("action_name", "/strive/execute_waypoint")
        self.declare_parameter("planner_status_topic", "/local_planner/status")

        self._validate_controller_contract()

        self._policy = SafetyVelocityPolicy(
            VelocityLimits(
                max_linear_mps=float(self.get_parameter("max_linear_speed_mps").value),
                max_angular_rps=float(self.get_parameter("max_angular_speed_rps").value),
                max_linear_accel_mps2=float(self.get_parameter("max_linear_accel_mps2").value),
                max_angular_accel_rps2=float(self.get_parameter("max_angular_accel_rps2").value),
            ),
            watchdog_timeout_s=float(self.get_parameter("command_watchdog_timeout_s").value),
        )
        self._autonomy_enabled = bool(self.get_parameter("start_autonomy_enabled").value)
        self._manual_takeover = False
        self._estop_active = False
        self._last_odom_received_at: Optional[float] = None
        self._last_pointcloud_received_at: Optional[float] = None
        self._latest_autonomy: Optional[TwistStamped] = None
        self._latest_manual: Optional[TwistStamped] = None
        self._latest_autonomy_received_at: Optional[float] = None
        self._latest_manual_received_at: Optional[float] = None
        self._last_tick = time.monotonic()

        queue_size = 10
        self._publisher = self.create_publisher(TwistStamped, str(self.get_parameter("output_cmd_topic").value), queue_size)
        self._state_publisher = self.create_publisher(
            SafetyStateMessage,
            str(self.get_parameter("safety_state_topic").value),
            queue_size,
        )
        self.create_subscription(TwistStamped, str(self.get_parameter("autonomy_cmd_topic").value), self._update_autonomy, queue_size)
        self.create_subscription(TwistStamped, str(self.get_parameter("manual_cmd_topic").value), self._update_manual, queue_size)
        self.create_subscription(Bool, str(self.get_parameter("autonomy_enable_topic").value), self._update_autonomy_enable, queue_size)
        self.create_subscription(Bool, str(self.get_parameter("manual_takeover_topic").value), self._update_manual_takeover, queue_size)
        self.create_subscription(Bool, str(self.get_parameter("estop_topic").value), self._update_estop, queue_size)
        self.create_subscription(Bool, str(self.get_parameter("estop_reset_topic").value), self._reset_estop, queue_size)
        self.create_subscription(Empty, str(self.get_parameter("hold_topic").value), self._hold, queue_size)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._update_odometry,
            queue_size,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("pointcloud_topic").value),
            self._update_pointcloud,
            queue_size,
        )
        period = 1.0 / max(1.0, float(self.get_parameter("output_rate_hz").value))
        self.create_timer(period, self._publish_once)
        self.get_logger().info("SafetyVelocityMux owns final /cmd_vel output; autonomy starts disabled")

    def _validate_controller_contract(self) -> None:
        """Fail closed unless the approved contract matches the safety mux.

        The contract is the reviewed upper bound for the live platform.  The
        launch configuration may choose stricter limits, but it must not
        silently exceed the approved velocity, acceleration, watchdog, topic,
        or emergency-takeover boundaries.

        Raises:
            RuntimeError: If live safety configuration is missing or unsafe.
        """

        if not _as_bool(self.get_parameter("require_controller_contract").value):
            return
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
            self._validate_runtime_limits(contract)
        except ControllerContractError as exc:
            raise RuntimeError(f"safety mux contract gate failed: {exc}") from exc

    def _validate_runtime_limits(self, contract) -> None:
        """Check that this mux instance does not exceed reviewed limits."""

        topic_checks = (
            ("autonomy_cmd_topic", ("autonomy_cmd_topic",)),
            ("manual_cmd_topic", ("manual_cmd_topic",)),
            ("output_cmd_topic", ("final_cmd_topic",)),
            ("manual_takeover_topic", ("safety", "manual_takeover_topic")),
            ("estop_topic", ("safety", "emergency_stop_topic")),
        )
        for parameter_name, contract_path in topic_checks:
            runtime_topic = _normalize_topic(str(self.get_parameter(parameter_name).value))
            contract_topic = _normalize_topic(str(contract.get(*contract_path, default="")))
            if runtime_topic != contract_topic:
                raise ControllerContractError(
                    f"safety mux {parameter_name}={runtime_topic!r} does not match "
                    f"approved contract {'.'.join(contract_path)}={contract_topic!r}"
                )
        planner_status = _normalize_topic(str(self.get_parameter("planner_status_topic").value))
        contract_status = _normalize_topic(str(contract.get("planner_status_topic", default="")))
        feedback_status = _normalize_topic(str(contract.get("feedback", "status_topic", default="")))
        if contract_status and planner_status != contract_status:
            raise ControllerContractError("planner status topic does not match approved contract")
        if feedback_status and planner_status != feedback_status:
            raise ControllerContractError("feedback status topic does not match approved contract")

        limits = {
            "max_linear_speed_mps": ("max_linear_speed_mps", "max_linear_speed_mps"),
            "max_angular_speed_rps": ("max_angular_speed_rps", "max_angular_speed_rps"),
            "max_linear_accel_mps2": ("max_linear_accel_mps2", "max_linear_accel_mps2"),
            "max_angular_accel_rps2": ("max_angular_accel_rps2", "max_angular_accel_rps2"),
            "command_watchdog_timeout_s": ("command_watchdog_timeout_s", "command_watchdog_timeout_s"),
        }
        for parameter_name, contract_key in limits.values():
            configured = float(self.get_parameter(parameter_name).value)
            approved = float(contract.get("safety", contract_key, default=0.0))
            if configured > approved:
                raise ControllerContractError(
                    f"safety mux {parameter_name}={configured} exceeds approved {approved}"
                )

        if _normalize_topic(str(self.get_parameter("estop_topic").value)) != _normalize_topic(
            str(contract.get("safety", "emergency_stop_topic", default=""))
        ):
            raise ControllerContractError("estop topic does not match approved contract")

    def _update_autonomy(self, msg: TwistStamped) -> None:
        """Cache the latest local-planner command."""

        self._latest_autonomy = msg
        self._latest_autonomy_received_at = time.monotonic()

    def _update_manual(self, msg: TwistStamped) -> None:
        """Cache the latest manually controlled command."""

        self._latest_manual = msg
        self._latest_manual_received_at = time.monotonic()

    def _update_autonomy_enable(self, msg: Bool) -> None:
        """Enable or disable autonomous output after external approval."""

        self._autonomy_enabled = bool(msg.data)

    def _update_manual_takeover(self, msg: Bool) -> None:
        """Switch final command ownership to or from manual control."""

        self._manual_takeover = bool(msg.data)

    def _update_estop(self, msg: Bool) -> None:
        """Latch an external emergency-stop assertion until explicit reset."""

        if bool(msg.data):
            self._estop_active = True

    def _reset_estop(self, msg: Bool) -> None:
        """Clear the software estop latch only after an explicit reset request."""

        if bool(msg.data) and not self._manual_takeover:
            self._estop_active = False

    def _update_odometry(self, _: Odometry) -> None:
        """Record fresh localization input for the final command watchdog."""

        self._last_odom_received_at = time.monotonic()

    def _update_pointcloud(self, _: PointCloud2) -> None:
        """Record fresh registered point-cloud input for the final command watchdog."""

        self._last_pointcloud_received_at = time.monotonic()

    def _sensors_are_fresh(self, now: float) -> bool:
        """Return whether required localization and obstacle inputs are current."""

        if not bool(self.get_parameter("require_sensor_freshness").value):
            return True
        timeout = max(0.0, float(self.get_parameter("sensor_watchdog_timeout_s").value))
        return all(
            stamp is not None and now - stamp <= timeout
            for stamp in (self._last_odom_received_at, self._last_pointcloud_received_at)
        )

    def _hold(self, _: Empty) -> None:
        """Latch a safe hold until explicit autonomy enable is received."""

        self._autonomy_enabled = False
        self._latest_autonomy = None
        self._latest_autonomy_received_at = None
        self._policy.set_state(SafetyState.HOLD)

    def _publish_once(self) -> None:
        """Evaluate and publish one bounded final command."""

        if self._estop_active:
            self._policy.set_state(SafetyState.ESTOP)
        elif self._manual_takeover:
            self._policy.set_state(SafetyState.MANUAL_TAKEOVER)
        elif not self._autonomy_enabled:
            self._policy.set_state(SafetyState.HOLD)
        elif not self._sensors_are_fresh(time.monotonic()):
            # 中文说明：底盘不能继续沿用“最后一条有效速度”穿过定位或点云断流。
            # 传感器新鲜度由 mux 统一门控，避免 pathFollower 自己无法感知断流。
            self._policy.set_state(SafetyState.STALE_INPUT)
        else:
            self._policy.set_state(SafetyState.CLEAR)

        source = "manual" if self._manual_takeover else "autonomy"
        msg = self._latest_manual if self._manual_takeover else self._latest_autonomy
        received_at = self._latest_manual_received_at if self._manual_takeover else self._latest_autonomy_received_at
        command = VelocityCommand()
        stamp = None
        if msg is not None:
            command = VelocityCommand(msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z)
            # Header stamps may use ROS/sim time; the watchdog uses monotonic
            # receipt time so a clock-domain mismatch cannot disable safety.
            stamp = received_at
        now = time.monotonic()
        decision = self._policy.evaluate(command, now_s=now, command_stamp_s=stamp, dt_s=max(1e-3, now - self._last_tick), source=source)
        self._last_tick = now
        output = TwistStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "base_link"
        output.twist.linear.x = decision.command.linear_x_mps
        output.twist.linear.y = decision.command.linear_y_mps
        output.twist.angular.z = decision.command.angular_z_rps
        self._publisher.publish(output)
        state_message = SafetyStateMessage()
        state_message.state = _safety_state_value(decision.state)
        state_message.reason_code = str(decision.reason)
        state_message.autonomy_enabled = bool(self._autonomy_enabled)
        state_message.manual_takeover = bool(self._manual_takeover)
        state_message.estop_active = bool(self._estop_active)
        self._state_publisher.publish(state_message)


def _safety_state_value(state: SafetyState) -> int:
    """Map the platform-neutral safety state to the ROS message constant."""

    return {
        SafetyState.CLEAR: SafetyStateMessage.CLEAR,
        SafetyState.HOLD: SafetyStateMessage.HOLD,
        SafetyState.MANUAL_TAKEOVER: SafetyStateMessage.MANUAL_TAKEOVER,
        SafetyState.ESTOP: SafetyStateMessage.ESTOP,
        SafetyState.STALE_INPUT: SafetyStateMessage.STALE_INPUT,
        SafetyState.CONTROLLER_FAULT: SafetyStateMessage.CONTROLLER_FAULT,
    }[state]


def _as_bool(value: object) -> bool:
    """Parse ROS boolean parameters consistently across CLI and launch files."""

    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_topic(value: str) -> str:
    """Normalize a ROS topic name for contract comparisons."""

    text = str(value or "").strip()
    return "/" + text.strip("/") if text else ""


def main(args: Optional[list[str]] = None) -> None:
    """Run the safety velocity mux node."""

    rclpy.init(args=args)
    node = SafetyVelocityMux()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
