# AgileX boundary source notice

`agilex_platform.py` implements the small rosbridge boundary used by HuaweiNav.
Its protocol operations follow the FWTW working-tree implementation inspected at
`/Users/ricksu/Documents/ChatGPT/FWTW ICRA2027/work/control_update/tools/agilex_api/`.
That path is a local reference, not a deployment dependency:

- rosbridge WebSocket advertise/publish and application ping;
- `/mux_vel/add` and `/mux_vel/select` service calls;
- `/mux_vel/selected`, `/ranger_status`, `/robot_status`, `/navflow_cmd_vel`,
  and `/base_cmd_vel` subscriptions;
- zero command on mode change and transport shutdown.

The command observation suffixes keep the FWTW recorder convention:
`navflow_cmd_vel_received`, `base_cmd_vel_received`, and `mux_selected`.

Reference hashes recorded on 2026-09-21:

- `ros2_cmd_vel_to_agilex_ws.py`: `420cd2bf6b257a11ec0af8d98df393e8a3251dbd20f4298cd2c4c2e8bd9cf9cd`
- `keyboard_mux_switch.py`: `32b253d28cbab9789644c518dae5ed14803feb8723dd471f1b421c23bd72ccfc`
- HuaweiNav implementation: `f1919c9b03c4201f6bad1fbd1c649a94966c3bbddd6b2a118366c8f4515e4433`

Provenance snapshot:

- HuaweiNav base repository HEAD: `55bf3676e1ac4aad6d6308f5c2110b9cf66cfd00`; the implementation files are local working-tree changes.
- The FWTW reference directory is not a Git checkout in this workspace; both referenced files are untracked there, so their file hashes are the authoritative source identity.

The HuaweiNav implementation owns the ROS2 `TwistStamped` boundary and the
single local mode service. It does not copy the FWTW keyboard process or run a
second bridge. `remote_control_enabled=false` keeps the node in a read-only
shadow session; the physical profile must enable it explicitly. The source
worktree contains uncommitted files, so the deployer must record the exact
source hash used for a robot build.
