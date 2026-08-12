# Workspace-local camera calibration

Put the RGB camera's calibrated `camera_info` YAML here, for example
`orin26_generic_rgb.yaml`.  The real-robot container mounts this directory
read-only at `/workspace/STRIVE/real_robot/calibration`.

`camera_x001_intrinsics.yaml` is a read-only import of the Orin-26 Generic
USB-camera calibration (`1920x1080`, radial-3, offline reprojection error
`0.69 px`). It is not a D435i calibration and must not use the D435i extrinsic.
The current USB device advertises MJPEG only up to 1280x960; at 1920x1080 it
uses YUYV (5 Hz). Select a matching mode before using these intrinsics.

The Orin-26 deployment also has an imported, read-only extrinsics asset:
`orin26_d435i_mid360_targetless_v009_r009_extrinsics.json`. It preserves the
source project's 4x4 transforms and frame names. The asset is marked
`extrinsics_only` because it does not contain RGB intrinsics, distortion,
time-offset, or reprojection validation; it must not be treated as an
approved semantic-fusion profile by itself.

The D435i projection profile is kept separately at
`real_robot/ros2_ws/src/semantic_mapping/config/projection_orin26_d435i_mid360.yaml`.
It imports only `T_camera_from_lidar`; live D435i `CameraInfo` and a measured
RGB-LiDAR time offset are still required before semantic fusion is approved.

After the calibration file is present, set the selected USB-camera profile to:

```bash
export USB_CAMERA_INFO_URL="file:///workspace/STRIVE/real_robot/calibration/orin26_generic_rgb.yaml"
```

Then run the normal profile check and detector-only camera validation.  Do not
set this value until the file is present.  Calibration files are intentionally
Git-ignored; retain the calibration report and values with the robot's
deployment records. The imported extrinsics file records the source path and
SHA-256 for auditability without modifying the source project.
