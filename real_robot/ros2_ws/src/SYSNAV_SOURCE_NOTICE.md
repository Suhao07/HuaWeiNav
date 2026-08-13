# SysNav Source Notice

The following files are vendored from the local SysNav workspace:

```text
/home/ubuntu/WorkSpace/project/Huawei Nav/Code/SysNav/src/base_autonomy/local_planner
/home/ubuntu/WorkSpace/project/Huawei Nav/Code/SysNav/src/base_autonomy/terrain_analysis
```

They retain the SysNav BSD license and algorithm provenance. STRIVE changes are
limited to the deployment boundary:

- remove direct serial/chassis writes from `pathFollower`;
- publish candidate velocity on `/cmd_vel/autonomy`;
- connect the candidate stream to STRIVE `SafetyVelocityMux`;
- expose task lifecycle through `strive_motion_msgs/ExecuteWaypoint`.

The vendored local planner is still responsible for SysNav path sampling,
terrain checks and path following. It is not a semantic planner and does not
own STRIVE instruction state.
