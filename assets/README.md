# Deployment assets

This directory is intentionally empty in Git.  The Orin bootstrap copies the
approved detector and foundation-model files here, verifies SHA256 values, and
mounts them read-only into the container.  Do not symlink or edit model files
from another robot workspace.

The optional `orin26_livox_mid360_d435i_yoloworld_candidate` profile requires
`yolov8l-world.pt` with SHA256
`8bdfaef999116760247d6fb0b0f8fca064b43e94598b3d4a807ebae9bcf0cdd5`.
It reuses `ViT-B-32.pt` for text encoding. The container and smoke runners mount
that file into both CLIP's default cache and Ultralytics' default
`/workspace/STRIVE/weights/clip/` cache, so initialization needs no download.
The candidate image tag is separate and must be built before use. This profile
does not replace the default model or authorize physical motion; see
`docs/real_robot_chair_detection_diagnosis_20261105.md` for its evidence limits.
