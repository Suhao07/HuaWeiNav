# SysNav Lower-Stack Migration

This overlay vendors the motion components needed by STRIVE from the SysNav
workspace:

```text
SysNav localPlanner      -> local_planner/src/localPlanner.cpp
SysNav pathFollower      -> local_planner/src/pathFollower.cpp
SysNav terrainAnalysis   -> terrain_analysis/src/terrainAnalysis.cpp
```

The path sampling, local collision checking, terrain voxel update and
look-ahead path following algorithms are retained. The migration deliberately
changes only the platform boundary:

```text
original SysNav pathFollower -> /cmd_vel + optional serial
STRIVE overlay pathFollower -> /cmd_vel/autonomy
                            -> SafetyVelocityMux -> /cmd_vel
```

Direct serial writes were removed from the vendored `pathFollower`. Chassis
ownership belongs to the robot driver and the safety mux, not to the semantic
navigation process. The source remains BSD licensed; the original SysNav
repository is the provenance for the vendored files.

The lower stack consumes these interfaces:

```text
/state_estimation  nav_msgs/Odometry
/registered_scan   sensor_msgs/PointCloud2
/terrain_map       sensor_msgs/PointCloud2
/way_point         geometry_msgs/PointStamped
```

The local planner publishes `/path` in its `vehicle` frame. The STRIVE status
provider therefore computes remaining local path length from the vehicle origin
and never concatenates it with a `map`-frame odometry position. Goal distance
remains a separate `map`-frame quantity.

The Orin defaults are remapped from `/aft_mapped_to_init` and
`/cloud_registered` by `sysnav_lower_stack.launch.py`. The action server owns
the task lifecycle, while SysNav remains the local path generator/tracker.

The safety mux is the single-owner final velocity gate and starts in `HOLD`. A separate, externally approved enable signal
must be published to `/platform/autonomy_enable` before non-zero autonomous
velocity can reach `/cmd_vel`. Hardware emergency stop and manual takeover
remain independent safety inputs.

Cancellation is a two-layer operation:

```text
ExecuteWaypoint cancel
  -> /local_planner/cancel (std_msgs/Empty)
  -> localPlanner clears its current goal and publishes a one-point path
  -> /platform/safe_hold (std_msgs/Empty)
  -> SafetyVelocityMux disables autonomous output and publishes zero velocity
```

The planner-side clear is required because a velocity hold alone does not
remove the old waypoint from a running local planner. A later, explicitly
approved goal is the only operation that should create new motion.

Planner status and safety ownership are explicit:

```text
localPlanner -> /local_planner/status
  waiting_for_sensor | tracking | no_feasible_path | cancelled

pathFollower -> /cmd_vel/autonomy
manual input  -> /cmd_vel/manual
SafetyVelocityMux -> /cmd_vel
```

The status provider consumes planner messages by goal generation, so a stale
`no_feasible_path` from a previous waypoint cannot terminate a new goal. The
mux also requires fresh odometry and registered point-cloud messages before
passing non-zero velocity. Manual mode in the vendored path follower never
overwrites the autonomy command stream; the external manual-takeover signal
must explicitly authorize the manual stream.
