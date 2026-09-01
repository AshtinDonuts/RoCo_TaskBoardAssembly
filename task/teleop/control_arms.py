"""Helpers for which DexMate arm(s) teleop drives."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .schema import gripper_ratio, pack_action


def merge_joint_position_actions(
    *actions: Any,
    n_dof: int,
) -> Any:
    """Overlay non-None joint targets from ArticulationAction-like objects.

    Later actions win on conflicts. Returns a plain list when Isaac's
    ArticulationAction is unavailable (unit tests).
    """
    full = [None] * int(n_dof)
    for action in actions:
        if action is None:
            continue
        positions = getattr(action, "joint_positions", action)
        if positions is None:
            continue
        for i, val in enumerate(list(positions)):
            if i >= n_dof:
                break
            if val is not None:
                full[i] = float(val)
    try:
        from isaacsim.core.utils.types import ArticulationAction

        return ArticulationAction(joint_positions=full)
    except Exception:
        return full


def pack_teleop_action(
    *,
    control_arms: str,
    last_left_pos,
    last_left_quat,
    last_left_grip: float,
    last_right_pos,
    last_right_quat,
    last_right_grip: float,
    home_left_pos,
    home_left_quat,
    home_left_grip: float,
) -> np.ndarray:
    """Build the challenge 14-D action for right-only or dual teleop."""
    if control_arms == "dual":
        return pack_action(
            last_left_pos,
            last_left_quat,
            gripper_ratio(last_left_grip),
            last_right_pos,
            last_right_quat,
            gripper_ratio(last_right_grip),
        )
    # right: left slice is frozen home; right slice is live targets
    return pack_action(
        home_left_pos,
        home_left_quat,
        gripper_ratio(home_left_grip),
        last_right_pos,
        last_right_quat,
        gripper_ratio(last_right_grip),
    )


def prefer_cmd(*cmds: Optional[str]) -> str:
    """Pick the first non-none operator/recording command."""
    for cmd in cmds:
        if cmd not in (None, "", "none"):
            return str(cmd)
    return "none"