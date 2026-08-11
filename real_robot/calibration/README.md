# Workspace-local camera calibration

Put the RGB camera's calibrated `camera_info` YAML here, for example
`orin26_generic_rgb.yaml`.  The real-robot container mounts this directory
read-only at `/workspace/STRIVE/real_robot/calibration`.

After the calibration file is present, set the selected robot profile to:

```bash
export USB_CAMERA_INFO_URL="file:///workspace/STRIVE/real_robot/calibration/orin26_generic_rgb.yaml"
```

Then run the normal profile check and detector-only camera validation.  Do not
set this value until the file is present.  Calibration files are intentionally
Git-ignored; retain the calibration report and values with the robot's
deployment records.
