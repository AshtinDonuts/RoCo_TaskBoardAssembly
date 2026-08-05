"""CPU-side tests for DiffusionLeRobotPolicy helpers (no Isaac / no GPU)."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TASK_DIR = Path(__file__).resolve().parents[1]
POLICIES_PARENT = TASK_DIR  # task/ contains policies/
if str(POLICIES_PARENT) not in sys.path:
    sys.path.insert(0, str(POLICIES_PARENT))
# Also support `python -m pytest` from repo root with PYTHONPATH=task.

# Stub policy_api / param_config so helpers import without Isaac.
sys.modules.setdefault(
    "policy_api",
    SimpleNamespace(
        EnvInfo=object,
        Observation=object,
        PartTarget=object,
        Policy=object,
    ),
)
sys.modules.setdefault("param_config", SimpleNamespace())

from policies.diffusion_lerobot import (  # noqa: E402
    CONTROL_PERIOD_TICKS,
    GRIPPER_OPEN_LIMIT,
    STATE_SLICES,
    _IMG_H,
    _IMG_W,
    _resize_rgb,
    camera_payload_from_obs,
    euler_xyz_to_quat_wxyz,
    left_action_to_ik_target,
)


def test_state_slice_layout_covers_44d():
    assert STATE_SLICES["left_ee_pose"] == slice(0, 7)
    assert STATE_SLICES["right_ee_pose"] == slice(7, 14)
    assert STATE_SLICES["left_joint_pos"] == slice(14, 21)
    assert STATE_SLICES["right_joint_pos"] == slice(21, 28)
    assert STATE_SLICES["left_joint_vel"] == slice(28, 35)
    assert STATE_SLICES["right_joint_vel"] == slice(35, 42)
    assert STATE_SLICES["left_gripper"] == slice(42, 43)
    assert STATE_SLICES["right_gripper"] == slice(43, 44)
    covered = 0
    for sl in STATE_SLICES.values():
        covered += sl.stop - sl.start
    assert covered == 44


def test_resize_rgb_shape_and_none():
    z = _resize_rgb(None)
    assert z.shape == (_IMG_H, _IMG_W, 3)
    big = np.zeros((480, 640, 3), dtype=np.uint8)
    big[100, 200] = (10, 20, 30)
    out = _resize_rgb(big)
    assert out.shape == (_IMG_H, _IMG_W, 3)
    assert out.dtype == np.uint8


def test_camera_key_mapping():
    obs = SimpleNamespace(
        rgb={
            "head": np.full((480, 640, 3), 1, np.uint8),
            "L_wrist": np.full((480, 640, 3), 2, np.uint8),
            "R_wrist": np.full((480, 640, 3), 3, np.uint8),
        }
    )
    cams = camera_payload_from_obs(obs)
    assert set(cams) == {"head", "left", "right"}
    assert cams["head"].shape == (_IMG_H, _IMG_W, 3)
    assert cams["left"].shape == (_IMG_H, _IMG_W, 3)
    assert cams["right"].shape == (_IMG_H, _IMG_W, 3)


def test_euler_xyz_identity_and_beyond_pi():
    q = euler_xyz_to_quat_wxyz(0.0, 0.0, 0.0)
    np.testing.assert_allclose(np.abs(q), [1, 0, 0, 0], atol=1e-6)

    # Unwrapped angle beyond ±π must still produce a valid unit quaternion.
    q2 = euler_xyz_to_quat_wxyz(0.0, 0.0, 2.0 * math.pi + 0.3)
    assert np.isfinite(q2).all()
    np.testing.assert_allclose(np.linalg.norm(q2), 1.0, atol=1e-6)

    q3 = euler_xyz_to_quat_wxyz(7.0, -4.0, 3.5)
    assert np.isfinite(q3).all()
    np.testing.assert_allclose(np.linalg.norm(q3), 1.0, atol=1e-6)


def test_left_action_gripper_clip_preserves_euler():
    a = np.zeros(14, dtype=np.float64)
    a[:3] = [0.1, 0.2, 0.3]
    a[3:6] = [4.5, -3.2, 6.8]  # unwrapped Euler
    a[6] = 1.5  # over-open -> clipped
    pos, quat, grip = left_action_to_ik_target(a)
    np.testing.assert_allclose(pos, [0.1, 0.2, 0.3])
    assert grip == pytest.approx(GRIPPER_OPEN_LIMIT)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)

    a[6] = -1.0
    _, _, grip0 = left_action_to_ik_target(a)
    assert grip0 == pytest.approx(0.0)


def test_control_period_is_20_ticks_at_200hz():
    assert CONTROL_PERIOD_TICKS == 20


def test_10hz_hold_behavior():
    """Gate on physics step IDs, not the number of rendered act() calls."""
    period = CONTROL_PERIOD_TICKS
    queries = []
    last_query_step = None
    # The rendered harness advances 20 physics ticks per outer-loop call.
    for step_idx in (60, 80, 100, 120):
        if last_query_step is None or step_idx - last_query_step >= period:
            queries.append(step_idx)
            last_query_step = step_idx
    assert queries == [60, 80, 100, 120]

    # A true one-physics-tick loop still queries only every 20 ticks.
    queries = []
    last_query_step = None
    for step_idx in range(50):
        if last_query_step is None or step_idx - last_query_step >= period:
            queries.append(step_idx)
            last_query_step = step_idx
    assert queries == [0, 20, 40]
