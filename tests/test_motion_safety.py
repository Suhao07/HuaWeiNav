from real_robot.motion_safety import SafetyState, SafetyVelocityPolicy, VelocityCommand, VelocityLimits


def _policy():
    return SafetyVelocityPolicy(
        VelocityLimits(
            max_linear_mps=1.0,
            max_angular_rps=1.0,
            max_linear_accel_mps2=2.0,
            max_angular_accel_rps2=2.0,
        ),
        watchdog_timeout_s=0.25,
    )


def test_safety_policy_requires_clear_state_before_autonomy() -> None:
    policy = _policy()

    denied = policy.evaluate(
        VelocityCommand(linear_x_mps=0.5),
        now_s=1.0,
        command_stamp_s=1.0,
        dt_s=0.1,
    )

    assert denied.state == SafetyState.HOLD
    assert denied.command == VelocityCommand()


def test_safety_policy_clips_speed_and_acceleration() -> None:
    policy = _policy()
    policy.set_state(SafetyState.CLEAR)

    first = policy.evaluate(
        VelocityCommand(linear_x_mps=4.0, angular_z_rps=4.0),
        now_s=1.0,
        command_stamp_s=1.0,
        dt_s=0.1,
    )
    second = policy.evaluate(
        VelocityCommand(linear_x_mps=4.0, angular_z_rps=4.0),
        now_s=1.1,
        command_stamp_s=1.1,
        dt_s=0.1,
    )

    assert first.command.planar_speed() <= 0.2
    assert abs(first.command.angular_z_rps) <= 0.2
    assert second.command.planar_speed() <= 0.4
    assert abs(second.command.angular_z_rps) <= 0.4


def test_safety_policy_watchdog_enters_stale_input() -> None:
    policy = _policy()
    policy.set_state(SafetyState.CLEAR)

    decision = policy.evaluate(
        VelocityCommand(linear_x_mps=0.1),
        now_s=2.0,
        command_stamp_s=1.0,
        dt_s=0.1,
    )

    assert decision.state == SafetyState.STALE_INPUT
    assert decision.watchdog_expired is True
    assert decision.command == VelocityCommand()


def test_safety_policy_manual_takeover_denies_autonomy() -> None:
    policy = _policy()
    policy.set_state(SafetyState.MANUAL_TAKEOVER)

    decision = policy.evaluate(
        VelocityCommand(linear_x_mps=0.1),
        now_s=1.0,
        command_stamp_s=1.0,
        dt_s=0.1,
        source="autonomy",
    )

    assert decision.state == SafetyState.MANUAL_TAKEOVER
    assert decision.command == VelocityCommand()


def test_safety_state_transition_resets_previous_velocity() -> None:
    policy = _policy()
    policy.set_state(SafetyState.CLEAR)
    policy.evaluate(
        VelocityCommand(linear_x_mps=0.8),
        now_s=1.0,
        command_stamp_s=1.0,
        dt_s=1.0,
    )
    policy.set_state(SafetyState.HOLD)
    policy.set_state(SafetyState.CLEAR)

    decision = policy.evaluate(
        VelocityCommand(linear_x_mps=0.8),
        now_s=2.0,
        command_stamp_s=2.0,
        dt_s=0.1,
    )

    assert decision.command.planar_speed() <= 0.2
