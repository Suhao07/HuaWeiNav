"""Small ROS message helpers used by the motion action server."""

from __future__ import annotations

import copy
from typing import Any


def replace_message(message: Any) -> Any:
    """Return an independent mutable copy of a ROS request message."""

    return copy.deepcopy(message)
