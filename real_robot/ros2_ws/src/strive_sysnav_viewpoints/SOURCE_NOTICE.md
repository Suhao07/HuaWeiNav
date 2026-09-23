# SysNav viewpoint model provenance

Source workspace on SSH host `ubuntu`:
`/home/ubuntu/WorkSpace/project/Huawei Nav/Code/SysNav/src/exploration_planner/tare_planner`.
Source repository HEAD: `0fa15cc8bc18be7409272fb65fa514a9a5bca6b0`.
The copied working-tree files retain their original author notices.
Trailing whitespace is normalized; SOURCE_SHA256SUMS records the original files.

This package contains the original grid, rolling occupancy, point-cloud storage,
representation and utility modules needed for viewpoint generation. It does not
include the original sensor_coverage_planner executable or its TSP binary.
The utility include in `src/utils/misc_utils.cpp` uses the package include root.
The representation include uses Humble's `cv_bridge/cv_bridge.h` header.
The unused, unimplemented `Representation::ToJSON` declaration and body are omitted.

`viewpoint_provider.cpp` extracts the observation flow from RegisteredScanCallback,
PlanningEnv::UpdateRobotPosition, UpdateObjectVisibility, UpdateViewpointRep and
UpdateViewpointObjectVisibility. The original occupancy/raycasting and 2-metre
representation deduplication implementations are reused. Room classification and
motion-goal generation remain outside this observation node.

Boundary adaptations: input frames must match the configured world frame; cloud
and odometry timestamps must differ by at most max_odom_offset_s. Camera height
offset is a parameter with the original 0.265-metre default, pending confirmation
against the actual sensor configuration. Empty coverage has no ratio trigger;
the original object-score trigger is still evaluated.

On 2026-09-23 the executable linked on Orin/Humble and passed the network-isolated
native-provider/bridge smoke: events [0, 1], stationary duplicate suppression, and
object 42 associated with viewpoint 0. This verifies synthetic inputs only.
The aggregate launch includes the provider; an external provider deployment can
set start_viewpoint_provider:=false to retain one owner of the event topic.

Original-source deprecated Odometry::ConstPtr warnings remain visible.
