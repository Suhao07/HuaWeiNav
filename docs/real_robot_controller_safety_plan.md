# Real-robot controller safety plan

This plan defines how STRIVE may hand a semantic waypoint to an externally
owned chassis/local-planner stack. It is intentionally separate from the
Point-LIO and semantic-mapping bringup. The current Orin-26 contract remains
`unapproved` because controller ownership and the emergency-stop interface have
not been independently verified.

## Current observed interfaces

- STRIVE produces `/way_point` as `geometry_msgs/msg/PointStamped`.
- The observed external Urban-Nav-SR controller consumes `/waypoint` as
  `std_msgs/msg/Float32MultiArray`, flattened ego-frame `[x, y]`.
- The observed feedback is `/topoplan/reached_goal` as `std_msgs/msg/Bool`;
  only `true = reached` is confirmed. Blocked, timeout and preempted values are
  not yet defined.
- Historical controller limits were `max_v=1.5 m/s` and `max_w=0.5 rad/s`.
  These are not acceptable first-motion limits and must be clamped lower for
  acceptance.
- The observed AgileX bridge would eventually consume `/cmd_vel` and forward
  through `/navflow_cmd_vel`/mux. STRIVE must never publish `/cmd_vel`.
- No verified emergency-stop topic/service or controller owner is currently
  recorded. The existing observation contract therefore cannot be approved.

## Proposed ownership boundary

```text
STRIVE runtime
  -> /way_point (PointStamped, world frame)
  -> configurable waypoint adapter
  -> /waypoint (Float32MultiArray, ego-frame [x, y])
  -> external local planner / PD controller (owner)
  -> /cmd_vel (external owner only)
  -> chassis bridge + mux
```

The external owner starts, monitors and stops the lower controller. STRIVE owns
only the high-level waypoint and its own dry-run/test topics. The adapter may
transform frames and reject stale goals, but it cannot bypass the local planner
or publish velocity commands.

## Four-stage acceptance gate

### Stage 0 — static and shadow validation (no motion)

1. Confirm the controller owner, launch command, namespace and process list in
   writing.
2. Confirm message types, frame semantics, feedback values, watchdog timeout,
   speed/acceleration limits and an independently reachable emergency stop.
3. Run STRIVE with `dry_run=true`, `BLOCK_LOWER_CONTROLLER=1` and
   `output_enabled=false`.
4. Publish only to `/strive/test_way_point` or a non-subscribed shadow topic;
   record that `/cmd_vel` has zero STRIVE publishers.
5. Resolve the frame mismatch before enabling the adapter. Current LIO pose
   samples use `header.frame_id=camera_init`, while the adapter template expects
   `map`; a verified TF or an explicit profile frame change is required.

### Stage 1 — controller bench mode (no wheel-ground motion)

The external owner starts the controller with motor output disabled, brakes
applied, or wheels lifted. The adapter output may be connected to the real
`/waypoint` topic only if the owner confirms that the controller cannot energize
the chassis in this mode. Send one goal, a stale goal, a cancel/hold event and a
malformed-frame case. Record the exact feedback and watchdog behavior.

### Stage 2 — tethered low-speed motion

Use a clear test area, physical emergency stop held by a second person, and a
short tether/bumper observer. Start with limits much lower than the historical
controller configuration:

```text
max linear speed:       0.15 m/s
max angular speed:      0.15 rad/s
max linear accel:       0.10 m/s^2
max angular accel:      0.20 rad/s^2
goal distance:          <= 0.50 m
goal timeout:           10 s
waypoint stale timeout: 1 s
controller heartbeat:   <= 0.5 s
```

The test sequence is forward, stop/hold, reverse, rotate-in-place, blocked
obstacle, timeout and emergency-stop assertion. Any missing feedback, stale
pose, heartbeat loss, frame mismatch or e-stop uncertainty immediately fails
the stage and leaves the control gate closed.

### Stage 3 — supervised deployment limits

Only after Stage 2 passes may the owner propose higher limits. The contract must
record the exact limits, acceptance log, date, operator and rollback command.
The first deployment remains waypoint-only; velocity limits and mux ownership
stay with the external controller.

## Required contract approval fields

The robot-specific contract may be marked `approved` only when all of these are
filled and independently checked:

- controller owner and approval reference;
- `/waypoint` message type, coordinate frame/semantics and adapter transform;
- reached, blocked, timeout and preempted feedback values;
- heartbeat/watchdog behavior and stale-goal timeout;
- linear/angular speed and acceleration limits;
- emergency-stop topic/service, message type, asserted value and clear/reset
  procedure;
- manual takeover procedure and a tested rollback command;
- explicit `allow_strive_waypoint_handoff: true`, while
  `cmd_vel_direct_publish: false` remains mandatory.

The current Orin-26 observation contract deliberately fails these requirements:
owner is unconfirmed, frame semantics are unconfirmed, blocked/timeout feedback
is missing, and emergency stop is unverified. It must remain `unapproved`.

## Rollback

Rollback is a controller-owner action: assert the verified emergency stop,
disable waypoint handoff, stop only the deployment container, and leave external
Livox/Point-LIO and other workspaces untouched. Never use `git reset`, Docker
prune, or a blanket process kill as a motion-safety action.
