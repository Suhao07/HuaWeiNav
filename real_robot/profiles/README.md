# Real-robot profiles

Each `*.env` profile is a sourceable, credential-free hardware binding.  It
selects only the new workspace's image, container, model paths, output topics,
and runtime switches; it must never start, stop, or modify another project.

Run a profile through the guarded helper:

```bash
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb check
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb build
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb smoke
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb lio-diagnostics
bash scripts/run_real_robot_profile.sh orin26_livox_mid360_generic_rgb runtime-smoke
```

For a container-only dependency check while the shared LIO graph is unavailable,
source the profile from **Bash** (or use the helper above), then run the smoke
with the generic env file disabled:

```bash
bash -lc 'source real_robot/profiles/orin26_livox_mid360_generic_rgb.env; \
  SYSNAV_ENV_FILE=/dev/null CHECK_LIO_SAMPLES=0 REQUIRE_LIO=0 bash docker_en.sh smoke'
```

Profiles are Bash source files.  This mode does not create `ros2 topic echo`/`hz`
subscribers and does not prove sensor integration; it is only appropriate for
validating the isolated image and GPU stack.

`runtime-smoke` starts a unique short-lived container with detector/mapping,
camera, host-LIO management, waypoint publishing, and lower control all
disabled.  It proves that the high-level runtime writes a persisted dry-run
`WAIT` `RuntimeDecision` under `output/runtime/<profile>/`; it does not count
as a sensor or motion acceptance test.

`lio-diagnostics` is host-side but read-only: it snapshots the existing Livox
and Point-LIO endpoint QoS, performs bounded header samples, and writes a
Markdown report only under `logs/diagnostics/`.  It never invokes the external
LIO start/stop helper.  A failed sample is a gate failure, not permission to
restart an externally owned LIO session.

The profile helper refuses semantic fusion with an uncalibrated projection and
refuses real motion unless the explicit three-switch gate is satisfied.  Copy a
profile to add another robot.  Keep API credentials in `.env.realworld`, which
is ignored by Git, rather than in a profile.
