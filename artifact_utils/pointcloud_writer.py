"""Point-cloud debug writers for STRIVE.

These helpers are intentionally thin wrappers around Open3D IO. They centralize
directory creation and filenames while leaving all geometry construction in the
caller modules.
"""

from __future__ import annotations

import os
from typing import Any

import open3d as o3d


def write_line_set(line_set: Any, directory: str, filename: str) -> None:
    """Write an Open3D line set after creating the destination directory."""

    os.makedirs(directory, exist_ok=True)
    o3d.io.write_line_set(os.path.join(directory, filename), line_set)


def write_point_cloud(point_cloud: Any, directory: str, filename: str) -> None:
    """Write an Open3D point cloud after creating the destination directory."""

    os.makedirs(directory, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(directory, filename), point_cloud)
