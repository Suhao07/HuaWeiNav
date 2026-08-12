# Workspace-local camera calibration

Put the RGB camera's calibrated `camera_info` YAML here, for example
`orin26_generic_rgb.yaml`.  The real-robot container mounts this directory
read-only at `/workspace/STRIVE/real_robot/calibration`.

`camera_x001_intrinsics.yaml` is a read-only import of the Orin-26 camera
calibration (`1920x1080`, radial-3, offline reprojection error `0.69 px`). It
is not bound to the current Generic USB `1280x720` profile automatically;
select it only when the camera topic and resolution match.

The Orin-26 deployment also has an imported, read-only extrinsics asset:
`orin26_d435i_mid360_targetless_v009_r009_extrinsics.json`. It preserves the
source project's 4x4 transforms and frame names. The asset is marked
`extrinsics_only` because it does not contain RGB intrinsics, distortion,
time-offset, or reprojection validation; it must not be treated as an
approved semantic-fusion profile by itself.

After the calibration file is present, set the selected robot profile to:

```bash
export USB_CAMERA_INFO_URL="file:///workspace/STRIVE/real_robot/calibration/orin26_generic_rgb.yaml"
```

Then run the normal profile check and detector-only camera validation.  Do not
set this value until the file is present.  Calibration files are intentionally
Git-ignored; retain the calibration report and values with the robot's
deployment records. The imported extrinsics file records the source path and
SHA-256 for auditability without modifying the source project.
