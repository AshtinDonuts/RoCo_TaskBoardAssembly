"""Retarget XYZ span must be identical under fix_orientation / fixed quat."""
from __future__ import annotations

import numpy as np

from teleop.diag_fix_orientation_sweep import _IDENTITY, _TOP_DOWN, _make_cfg, _sweep
from teleop import transforms as T


def test_fix_orientation_does_not_shrink_commanded_xyz_span():
    base = {
        "translation_gain": 2.0,
        "axes_map": (
            (0.0, 0.57358151, 0.81914849),
            (-1.0, 0.0, 0.0),
            (0.0, -0.81914849, 0.57358151),
        ),
        "workspace_min": (-1.5, -1.5, -1.5),
        "workspace_max": (1.5, 1.5, 1.5),
    }
    dex_origin = np.array([0.25, -0.15, 1.05], dtype=np.float64)
    engage_quat = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.4, -0.6, 0.2]))
    )
    xs = np.linspace(-0.05, 0.05, 5)
    ys = np.linspace(-0.05, 0.05, 5)
    leader_xy = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    free = _sweep(
        "free",
        _make_cfg(base, fix_orientation=False, fixed_orientation_wxyz=None),
        dex_origin,
        engage_quat,
        leader_xy,
    )
    fixed = _sweep(
        "fix",
        _make_cfg(base, fix_orientation=True, fixed_orientation_wxyz=None),
        dex_origin,
        engage_quat,
        leader_xy,
    )
    np.testing.assert_allclose(free["pos_span"], fixed["pos_span"], atol=1e-9)
    # Engage quat held.
    np.testing.assert_allclose(fixed["quat_last"], engage_quat, atol=1e-9)
    assert fixed["quat_angle_spread_rad"] < 1e-9

    # After slewing onto top-down, XYZ span still matches free.
    lock_cfg = _make_cfg(
        base, fix_orientation=False, fixed_orientation_wxyz=_TOP_DOWN
    )
    from teleop.retarget import CartesianRetargeter

    warm = CartesianRetargeter(lock_cfg)
    warm.capture_origins([0, 0, 0], _IDENTITY, dex_origin, engage_quat)
    pos, quat = dex_origin, engage_quat
    for _ in range(120):
        pos, quat, _, _ = warm.step(
            leader_pos=[0, 0, 0],
            leader_quat=_IDENTITY,
            gripper_norm=0.5,
            dt=0.05,
            clutch=True,
            deadman=True,
            current_dex_pos=pos,
            current_dex_quat=quat,
        )
    locked = _sweep("locked", lock_cfg, dex_origin, quat, leader_xy)
    np.testing.assert_allclose(free["pos_span"], locked["pos_span"], atol=1e-9)
    np.testing.assert_allclose(locked["quat_last"], _TOP_DOWN, atol=1e-6)
