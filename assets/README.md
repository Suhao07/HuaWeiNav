# Deployment assets

This directory is intentionally empty in Git.  The Orin bootstrap copies the
approved detector and foundation-model files here, verifies SHA256 values, and
mounts them read-only into the container.  Do not symlink or edit model files
from another robot workspace.
