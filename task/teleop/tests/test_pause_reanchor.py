"""Policy-level pause / resume reanchor without a full Isaac stack."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from teleop.retarget import CartesianRetargeter, RetargetConfig
from policies.aloha_teleop import AlohaTeleopPolicy


def _identity():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _make_stub_policy(*, dual: bool = False):
    """Minimal stand-in with pause/resume helpers bound from AlohaTeleopPolicy."""
    p = SimpleNamespace()
    p.active_arms = ("L", "R") if dual else ("R",)
    p._paused = False
    p._resume_hold_pending = False
    p._skip_record_once = False
    p._next_record_wall = None
    p._next_record_sim = None
    p._frozen_right = None
    p._frozen_left = None
    p._last_right_pos = np.array([0.40, 0.05, 0.25], dtype=np.float64)
    p._last_right_quat = _identity()
    p._last_right_grip = 0.7
    p._last_left_pos = np.array([0.40, -0.05, 0.25], dtype=np.float64)
    p._last_left_quat = _identity()
    p._last_left_grip = 0.7
    p._retarget_R = CartesianRetargeter(
        RetargetConfig(max_lin_vel=10.0, max_ang_vel=10.0, max_lin_acc=0.0)
    )
    p._retarget_L = (
        CartesianRetargeter(
            RetargetConfig(max_lin_vel=10.0, max_ang_vel=10.0, max_lin_acc=0.0)
        )
        if dual
        else None
    )
    p._session = SimpleNamespace(set_clock_paused=lambda *_a, **_k: None)
    p.R = SimpleNamespace(
        current_cspace_q=lambda: np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]),
        forward_raw_q=lambda q, g: {"q": list(q), "grip": g},
        forward=lambda pos, quat, g: {"pos": list(pos), "grip": g},
    )
    p.L = SimpleNamespace(
        current_cspace_q=lambda: np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0]),
        forward_raw_q=lambda q, g: {"q": list(q), "grip": g},
        forward=lambda pos, quat, g: {"pos": list(pos), "grip": g},
    )
    p._n_dof = 20

    # Bind helpers from the real policy class.
    for name in (
        "_disengage_retargeters",
        "_controller_cspace_q",
        "_latch_frozen_arm",
        "_enter_pause",
        "_resume_and_reanchor",
        "_emit_resume_hold_action",
    ):
        setattr(p, name, getattr(AlohaTeleopPolicy, name).__get__(p, AlohaTeleopPolicy))
    return p


def test_pause_resume_keeps_frozen_virtual_pose_when_leader_stationary():
    p = _make_stub_policy()
    # Actual EE differs from last Cartesian command (IK tracking error).
    actual_pos = np.array([0.41, 0.04, 0.26], dtype=np.float64)
    actual_quat = _identity()
    old_cmd = p._last_right_pos.copy()
    assert not np.allclose(actual_pos, old_cmd)

    now = time.monotonic()
    p._enter_pause(
        now,
        left_pos=None,
        left_quat=None,
        right_pos=actual_pos,
        right_quat=actual_quat,
    )
    assert p._paused is True
    np.testing.assert_allclose(p._last_right_pos, actual_pos)
    np.testing.assert_allclose(p._frozen_right["pos"], actual_pos)

    # Leader moved during pause to a far absolute pose.
    resume_leader = {
        "ee_pos": [0.9, -0.4, 0.1],
        "ee_quat_wxyz": _identity().tolist(),
        "gripper_norm": 0.2,
        "clutch": True,
        "cmd": "resume",
        "deadman": True,
    }
    p._resume_and_reanchor(now + 1.0, sample_R=resume_leader, sample_L=None)
    assert p._paused is False
    assert p._resume_hold_pending is True
    np.testing.assert_allclose(p._last_right_pos, actual_pos)
    np.testing.assert_allclose(p._retarget_R.state.dex_origin_pos, actual_pos)
    np.testing.assert_allclose(
        p._retarget_R.state.leader_origin_pos, resume_leader["ee_pos"]
    )

    # Stationary leader after resume → target unchanged.
    pos, quat, grip, info = p._retarget_R.step(
        leader_pos=resume_leader["ee_pos"],
        leader_quat=resume_leader["ee_quat_wxyz"],
        gripper_norm=resume_leader["gripper_norm"],
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=actual_pos,
        current_dex_quat=actual_quat,
    )
    assert info["reason"] == "tracking"
    np.testing.assert_allclose(pos, actual_pos, atol=1e-9)

    # Later physical delta moves virtually from the frozen origin only.
    pos2, *_ = p._retarget_R.step(
        leader_pos=[0.95, -0.4, 0.1],
        leader_quat=resume_leader["ee_quat_wxyz"],
        gripper_norm=resume_leader["gripper_norm"],
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=actual_pos,
        current_dex_quat=actual_quat,
    )
    np.testing.assert_allclose(pos2, actual_pos + np.array([0.05, 0.0, 0.0]), atol=1e-6)


def test_dual_arm_pause_resume_reanchors_each_frozen_pose():
    p = _make_stub_policy(dual=True)
    right_actual = np.array([0.42, 0.05, 0.2], dtype=np.float64)
    left_actual = np.array([0.42, -0.05, 0.2], dtype=np.float64)
    now = time.monotonic()
    p._enter_pause(
        now,
        left_pos=left_actual,
        left_quat=_identity(),
        right_pos=right_actual,
        right_quat=_identity(),
    )
    sample_R = {
        "ee_pos": [0.1, 0.2, 0.0],
        "ee_quat_wxyz": _identity().tolist(),
        "gripper_norm": 1.0,
        "clutch": True,
        "cmd": "resume",
        "deadman": True,
    }
    sample_L = {
        "ee_pos": [-0.1, 0.2, 0.0],
        "ee_quat_wxyz": _identity().tolist(),
        "gripper_norm": 1.0,
        "clutch": True,
        "cmd": "resume",
        "deadman": True,
    }
    p._resume_and_reanchor(now + 0.5, sample_R=sample_R, sample_L=sample_L)
    np.testing.assert_allclose(p._last_right_pos, right_actual)
    np.testing.assert_allclose(p._last_left_pos, left_actual)
    np.testing.assert_allclose(p._retarget_R.state.leader_origin_pos, sample_R["ee_pos"])
    np.testing.assert_allclose(p._retarget_L.state.leader_origin_pos, sample_L["ee_pos"])
    np.testing.assert_allclose(p._retarget_R.state.dex_origin_pos, right_actual)
    np.testing.assert_allclose(p._retarget_L.state.dex_origin_pos, left_actual)
