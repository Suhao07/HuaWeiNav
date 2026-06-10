"""Path builders for STRIVE debug artifacts.

The runtime writes many images and point clouds while building maps and
verifying targets. Keeping path construction here prevents navigation and
mapping code from duplicating episode/step directory conventions.
"""

from __future__ import annotations

import os


def episode_dir(save_dir: str, episode_idx: int) -> str:
    """Return and create the root debug directory for one episode."""

    path = os.path.join(save_dir, f"episode-{episode_idx}")
    os.makedirs(path, exist_ok=True)
    return path


def episode_subdir(save_dir: str, episode_idx: int, name: str) -> str:
    """Return and create a named debug subdirectory under an episode."""

    path = os.path.join(episode_dir(save_dir, episode_idx), name)
    os.makedirs(path, exist_ok=True)
    return path


def detection_step_dir(save_dir: str, episode_idx: int, step: int) -> str:
    """Return and create the detection directory for one episode step."""

    path = os.path.join(episode_subdir(save_dir, episode_idx, "detection"), f"step_{step}")
    os.makedirs(path, exist_ok=True)
    return path
