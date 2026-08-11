# Lower-controller contract gate

STRIVE never publishes `/cmd_vel` directly.  Real movement is allowed only as a
hand-off to an externally owned waypoint/local-planner controller, after its
topic, message type, frame, feedback, speed limits, e-stop and takeover
procedure have been reviewed.

Copy `controller_contract_template.yaml` to a robot-specific YAML in this
directory.  Keep that file Git-ignored.  After the review, set the profile's
`CONTROL_CONTRACT_FILE` to the container path, for example:

```bash
export CONTROL_CONTRACT_FILE="/workspace/STRIVE/real_robot/control/orin26_controller_contract.yaml"
```

The profile and runtime refuse live lower-controller startup unless this file
exists and has all of the following exact approved gate values:

```yaml
  approval_status: approved
  allow_strive_waypoint_handoff: true
  cmd_vel_direct_publish: false
  emergency_stop_verified: true
```

This directory is mounted read-only into the deployment container.  Do not use
this mechanism to start, change, or take ownership of another project's lower
controller.
