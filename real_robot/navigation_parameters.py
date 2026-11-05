"""Validate the formal navigation entry's coupled distance parameters offline."""
import math
import os
import sys

DEFAULTS = {
    "NAV_GOAL_TOLERANCE_M": 1.0,
    "NAV_FOLLOWER_STOP_DISTANCE_M": 0.5,
    "NAV_FOLLOWER_LOOKAHEAD_M": 1.0,
    "NAV_FOLLOWER_SLOWDOWN_DISTANCE_M": 1.0,
}


def validate_parameters(environ):
    """Return positive finite distances; reject a follower stop outside arrival tolerance.

    This is a necessary configuration check, not a guarantee that any local path
    ends at the Action goal. Keep margin for path discretization and localization.
    """
    values = {}
    for name, default in DEFAULTS.items():
        try:
            value = float(environ.get(name, default))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite positive number") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
        values[name] = value
    stop = values["NAV_FOLLOWER_STOP_DISTANCE_M"]
    if stop >= values["NAV_GOAL_TOLERANCE_M"]:
        raise ValueError("NAV_FOLLOWER_STOP_DISTANCE_M must be smaller than NAV_GOAL_TOLERANCE_M")
    if stop >= values["NAV_FOLLOWER_LOOKAHEAD_M"]:
        raise ValueError("NAV_FOLLOWER_LOOKAHEAD_M must be larger than NAV_FOLLOWER_STOP_DISTANCE_M")
    return values


if __name__ == "__main__":
    try:
        for key, value in validate_parameters(os.environ).items():
            print(f"{key}={value:g}")
    except ValueError as error:
        print(f"Invalid navigation parameters: {error}", file=sys.stderr)
        sys.exit(2)
