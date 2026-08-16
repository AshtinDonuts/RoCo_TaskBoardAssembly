"""Unit tests for teleop control-arm action packing / merge helpers."""
from __future__ import annotations

import numpy as np

from teleop.control_arms import (
    merge_joint_position_actions,
    pack_teleop_action,
    prefer_cmd,
)
from teleop.schema import gripper_ratio


def test_prefer_cmd_picks_first_nonzero():
    assert prefer_cmd("none", "save_episode") == "save_episode"
    assert prefer_cmd("pause", "save_episode") == "pause"
    assert prefer_cmd(None, "none", None) == "none"


def test_pack_teleop_action_right_uses_home_left():
    home_pos = [0.1, 0.2, 0.3]
    home_quat = [1.0, 0.0, 0.0, 0.0]
    live_pos = [0.4, 0.5, 0.6]
    live_quat = [1.0, 0.0, 0.0, 0.0]
    home_grip = 0.25
    live_grip = 0.5
    action = pack_teleop_action(
        control_arms="right",
        last_left_pos=[9.0, 9.0, 9.0],
        last_left_quat=home_quat,
        last_left_grip=0.9,
        last_right_pos=live_pos,
        last_right_quat=live_quat,
        last_right_grip=live_grip,
        home_left_pos=home_pos,
        home_left_quat=home_quat,
        home_left_grip=home_grip,
    )
    assert action.shape == (14,)
    np.testing.assert_allclose(action[0:3], home_pos)
    np.testing.assert_allclose(action[7:10], live_pos)
    np.testing.assert_allclose(action[6], gripper_ratio(home_grip), rtol=1e-5)
    np.testing.assert_allclose(action[13], gripper_ratio(live_grip), rtol=1e-5)


def test_pack_teleop_action_dual_uses_both_live():
    left_pos = [0.1, 0.0, 0.2]
    right_pos = [0.3, 0.0, 0.4]
    quat = [1.0, 0.0, 0.0, 0.0]
    action = pack_teleop_action(
        control_arms="dual",
        last_left_pos=left_pos,
        last_left_quat=quat,
        last_left_grip=0.1,
        last_right_pos=right_pos,
        last_right_quat=quat,
        last_right_grip=0.8,
        home_left_pos=[0.0, 0.0, 0.0],
        home_left_quat=quat,
        home_left_grip=0.0,
    )
    np.testing.assert_allclose(action[0:3], left_pos)
    np.testing.assert_allclose(action[7:10], right_pos)
    np.testing.assert_allclose(action[6], gripper_ratio(0.1), rtol=1e-5)
    np.testing.assert_allclose(action[13], gripper_ratio(0.8), rtol=1e-5)


def test_merge_joint_position_actions_overlays_non_none():
    class _A:
        def __init__(self, positions):
            self.joint_positions = positions

    merged = merge_joint_position_actions(
        _A([1.0, None, None, None]),
        _A([None, 2.0, None, 4.0]),
        n_dof=4,
    )
    positions = getattr(merged, "joint_positions", merged)
    assert positions[0] == 1.0
    assert positions[1] == 2.0
    assert positions[2] is None
    assert positions[3] == 4.0
