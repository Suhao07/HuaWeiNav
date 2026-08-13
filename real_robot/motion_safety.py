"""Platform-neutral velocity safety primitives.

The module deliberately does not import ROS or a chassis SDK.  It provides the
small deterministic core used by a ROS safety mux, a bag-replay harness, and
unit tests.  The final velocity command must still be emitted by one platform
owner after this policy has approved it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from real_robot.contracts import Vector3


class SafetyState(str, Enum):
    """Safety state that controls whether autonomy may produce motion."""

    CLEAR = "clear"
    HOLD = "hold"
    MANUAL_TAKEOVER = "manual_takeover"
    ESTOP = "estop"
    STALE_INPUT = "stale_input"
    CONTROLLER_FAULT = "controller_fault"


@dataclass(frozen=True)
class VelocityLimits:
    """Maximum velocity and acceleration accepted by the safety policy.

    Args:
        max_linear_mps: Maximum planar linear speed.
        max_angular_rps: Maximum yaw rate.
        max_linear_accel_mps2: Maximum planar linear acceleration.
        max_angular_accel_rps2: Maximum yaw acceleration.
    """

    max_linear_mps: float
    max_angular_rps: float
    max_linear_accel_mps2: float
    max_angular_accel_rps2: float

    def __post_init__(self) -> None:
        """Reject non-physical negative limits."""

        if min(
            self.max_linear_mps,
            self.max_angular_rps,
            self.max_linear_accel_mps2,
            self.max_angular_accel_rps2,
        ) < 0:
            raise ValueError("velocity and acceleration limits must be non-negative")


@dataclass(frozen=True)
class VelocityCommand:
    """Planar velocity command in the chassis base frame."""

    linear_x_mps: float = 0.0
    linear_y_mps: float = 0.0
    angular_z_rps: float = 0.0

    def planar_speed(self) -> float:
        """Return the magnitude of the planar linear velocity."""

        return (self.linear_x_mps**2 + self.linear_y_mps**2) ** 0.5


@dataclass(frozen=True)
class SafetyDecision:
    """Result of applying state, freshness, and rate limits to a command."""

    command: VelocityCommand
    state: SafetyState
    reason: str
    watchdog_expired: bool = False


class SafetyVelocityPolicy:
    """Apply safety ownership, watchdog, magnitude, and acceleration limits.

    The policy is intentionally memoryful only for the previous accepted
    command.  It does not select a path and it does not publish a command.
    A ROS ``SafetyVelocityMux`` can call :meth:`evaluate` and publish the
    returned command on the single chassis output topic.
    """

    def __init__(self, limits: VelocityLimits, watchdog_timeout_s: float = 0.25) -> None:
        """Initialize the policy.

        Args:
            limits: Platform-approved velocity and acceleration limits.
            watchdog_timeout_s: Maximum age of an autonomous command.
        """

        if watchdog_timeout_s <= 0:
            raise ValueError("watchdog_timeout_s must be positive")
        self.limits = limits
        self.watchdog_timeout_s = float(watchdog_timeout_s)
        self.state = SafetyState.HOLD
        self._previous = VelocityCommand()

    def set_state(self, state: SafetyState) -> None:
        """Set the externally owned safety state."""

        next_state = SafetyState(state)
        if next_state != self.state and next_state != SafetyState.CLEAR:
            # 核心安全不变量：任何安全态都把加速度限制器的历史速度清零，
            # 解除 HOLD/急停后必须从静止重新爬升，不能恢复上一条旧速度。
            self._previous = VelocityCommand()
        self.state = next_state

    def evaluate(
        self,
        command: VelocityCommand,
        *,
        now_s: float,
        command_stamp_s: Optional[float],
        dt_s: float,
        source: str = "autonomy",
    ) -> SafetyDecision:
        """Return the only command the output mux may publish.

        Args:
            command: Candidate command from local path tracking.
            now_s: Monotonic time used by the watchdog.
            command_stamp_s: Timestamp of the candidate command.
            dt_s: Time since the previous output command.
            source: ``autonomy`` or ``manual`` command ownership.

        Returns:
            A bounded command and the reason for the decision.

        Raises:
            ValueError: If ``dt_s`` is not positive.
        """

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if self.state in {SafetyState.ESTOP, SafetyState.HOLD, SafetyState.STALE_INPUT, SafetyState.CONTROLLER_FAULT}:
            self._previous = VelocityCommand()
            return self._zero(self.state, f"motion denied by safety state: {self.state.value}")
        if self.state == SafetyState.MANUAL_TAKEOVER and source != "manual":
            self._previous = VelocityCommand()
            return self._zero(self.state, "autonomy command denied during manual takeover")
        if command_stamp_s is None or now_s - command_stamp_s > self.watchdog_timeout_s:
            self._previous = VelocityCommand()
            return SafetyDecision(
                command=VelocityCommand(),
                state=SafetyState.STALE_INPUT,
                reason="velocity command watchdog expired",
                watchdog_expired=True,
            )

        bounded = self._limit_magnitude(command)
        bounded = self._limit_acceleration(bounded, dt_s)
        self._previous = bounded
        return SafetyDecision(bounded, self.state, "command accepted within safety limits")

    def _limit_magnitude(self, command: VelocityCommand) -> VelocityCommand:
        """Clip linear and angular magnitudes without changing direction."""

        speed = command.planar_speed()
        if speed > self.limits.max_linear_mps and speed > 0:
            scale = self.limits.max_linear_mps / speed
            linear_x = command.linear_x_mps * scale
            linear_y = command.linear_y_mps * scale
        else:
            linear_x, linear_y = command.linear_x_mps, command.linear_y_mps
        angular = max(-self.limits.max_angular_rps, min(self.limits.max_angular_rps, command.angular_z_rps))
        return VelocityCommand(linear_x, linear_y, angular)

    def _limit_acceleration(self, command: VelocityCommand, dt_s: float) -> VelocityCommand:
        """Limit the change from the previous output command."""

        max_delta_linear = self.limits.max_linear_accel_mps2 * dt_s
        previous_speed = self._previous.planar_speed()
        current_speed = command.planar_speed()
        if current_speed > previous_speed + max_delta_linear and current_speed > 0:
            desired_speed = previous_speed + max_delta_linear
            scale = desired_speed / current_speed
            linear_x = command.linear_x_mps * scale
            linear_y = command.linear_y_mps * scale
        else:
            linear_x, linear_y = command.linear_x_mps, command.linear_y_mps

        max_delta_angular = self.limits.max_angular_accel_rps2 * dt_s
        delta = command.angular_z_rps - self._previous.angular_z_rps
        angular = self._previous.angular_z_rps + max(-max_delta_angular, min(max_delta_angular, delta))
        return VelocityCommand(linear_x, linear_y, angular)

    @staticmethod
    def _zero(state: SafetyState, reason: str) -> SafetyDecision:
        """Build a zero-velocity safety decision."""

        return SafetyDecision(VelocityCommand(), state, reason)
