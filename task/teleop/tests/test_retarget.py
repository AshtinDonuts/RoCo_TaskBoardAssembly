from __future__ import annotations

import numpy as np

from teleop.retarget import CartesianRetargeter, RetargetConfig


def _identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])


def test_hold_when_clutch_off():
    r = CartesianRetargeter(RetargetConfig())
    pos, quat, grip, info = r.step(
        leader_pos=[0.1, 0, 0],
        leader_quat=_identity_quat(),
        gripper_norm=0.0,
        dt=0.02,
        clutch=False,
        deadman=True,
        current_dex_pos=[0.2, 0.0, 0.3],
        current_dex_quat=_identity_quat(),
    )
    assert info["held"] is True
    np.testing.assert_allclose(pos, [0.2, 0.0, 0.3], atol=1e-9)


def test_clutch_off_freezes_gripper():
    r = CartesianRetargeter(RetargetConfig())
    origin = np.array([0.2, 0.0, 0.3])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    _, _, grip_open, _ = r.step(
        leader_pos=[0, 0, 0],
        leader_quat=_identity_quat(),
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    _, _, grip_held, info = r.step(
        leader_pos=[0.1, 0, 0],
        leader_quat=_identity_quat(),
        gripper_norm=0.0,
        dt=0.02,
        clutch=False,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    assert info["reason"] == "clutch_off"
    np.testing.assert_allclose(grip_held, grip_open, atol=1e-9)


def test_relative_translation_identity_map():
    r = CartesianRetargeter(RetargetConfig(max_lin_vel=10.0, max_ang_vel=10.0, max_lin_acc=0.0))
    origin = np.array([0.3, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    pos, quat, grip, info = r.step(
        leader_pos=[0.05, -0.02, 0.01],
        leader_quat=_identity_quat(),
        gripper_norm=0.5,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    assert info["reason"] == "tracking"
    np.testing.assert_allclose(pos, origin + np.array([0.05, -0.02, 0.01]), atol=1e-6)


def test_translation_gain_scales_meter_deltas():
    """Leader EE is meters; gain multiplies those deltas onto DexMate meters."""
    r = CartesianRetargeter(
        RetargetConfig(
            translation_gain=2.0,
            max_lin_vel=10.0,
            max_ang_vel=10.0,
            max_lin_acc=0.0,
        )
    )
    origin = np.array([0.3, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    pos, *_ = r.step(
        leader_pos=[0.05, 0.0, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=0.0,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    np.testing.assert_allclose(pos, origin + np.array([0.10, 0.0, 0.0]), atol=1e-6)


def test_axes_sign_and_perm():
    cfg = RetargetConfig(
        axes_perm=(1, 0, 2),
        axes_sign=(-1.0, 1.0, 1.0),
        max_lin_vel=10.0,
        max_ang_vel=10.0,
        max_lin_acc=0.0,
    )
    r = CartesianRetargeter(cfg)
    origin = np.array([0.0, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    pos, *_ = r.step(
        leader_pos=[0.10, 0.04, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=0.0,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    # perm (1,0,2) then signs (-1,1,1): [x,y,z] -> [-y, x, z]
    np.testing.assert_allclose(pos, origin + np.array([-0.04, 0.10, 0.0]), atol=1e-6)


def test_headcam_axes_map_leader_forward_into_image():
    """Leader +X (away from operator) must move DexMate into the headcam view."""
    cfg = RetargetConfig(
        axes_map=(
            (0.0, 0.57358151, 0.81914849),
            (-1.0, 0.0, 0.0),
            (0.0, -0.81914849, 0.57358151),
        ),
        translation_gain=1.0,
        max_lin_vel=10.0,
        max_ang_vel=10.0,
        max_lin_acc=0.0,
    )
    r = CartesianRetargeter(cfg)
    origin = np.array([0.2, 0.1, 0.9])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    pos, *_ = r.step(
        leader_pos=[0.10, 0.0, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=0.0,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    # Into headcam ≈ world -Y at INIT head pitch.
    np.testing.assert_allclose(pos, origin + np.array([0.0, -0.10, 0.0]), atol=1e-5)


def test_space_fixed_yaw_maps_through_axes_map():
    """Leader yaw about +Z becomes a DexMate rotation about mapped +Z."""
    from teleop import transforms as T

    axes_map = (
        (0.0, 0.57358151, 0.81914849),
        (-1.0, 0.0, 0.0),
        (0.0, -0.81914849, 0.57358151),
    )
    cfg = RetargetConfig(
        axes_map=axes_map,
        max_lin_vel=10.0,
        max_ang_vel=10.0,
        max_lin_acc=0.0,
    )
    r = CartesianRetargeter(cfg)
    dex0 = np.array([0.2, 0.1, 0.9])
    q0 = _identity_quat()
    r.capture_origins([0, 0, 0], q0, dex0, q0)

    yaw = 0.3
    q_leader = T.rotvec_to_quat_wxyz([0.0, 0.0, yaw])
    _, quat, _, info = r.step(
        leader_pos=[0.0, 0.0, 0.0],
        leader_quat=q_leader,
        gripper_norm=0.0,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=dex0,
        current_dex_quat=q0,
    )
    assert info["reason"] == "tracking"
    # Space-fixed: R_dex = R_map @ RotZ(yaw) @ R_map^T
    R_map = np.asarray(axes_map, dtype=np.float64)
    R_rel = T.quat_wxyz_to_matrix(q_leader)
    R_exp = R_map @ R_rel @ R_map.T
    R_got = T.quat_wxyz_to_matrix(quat)
    np.testing.assert_allclose(R_got, R_exp, atol=1e-6)
    # Axis of the mapped yaw should align with mapped leader +Z (image-up).
    axis_exp = R_map @ np.array([0.0, 0.0, 1.0])
    axis_exp = axis_exp / np.linalg.norm(axis_exp)
    rotvec = T.quat_wxyz_to_rotvec(quat)
    axis_got = rotvec / np.linalg.norm(rotvec)
    np.testing.assert_allclose(axis_got, axis_exp, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(rotvec), yaw, atol=1e-6)


def test_workspace_clamp_and_rate_limit():
    cfg = RetargetConfig(
        workspace_min=(0.0, -0.1, 0.0),
        workspace_max=(0.05, 0.1, 0.4),
        max_lin_vel=0.10,
        max_ang_vel=10.0,
        max_lin_acc=0.0,
    )
    r = CartesianRetargeter(cfg)
    origin = np.array([0.0, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    pos, *_ = r.step(
        leader_pos=[1.0, 0.0, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=0.0,
        dt=0.1,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    # 0.1 m/s * 0.1 s = 0.01 m, then clamp max x=0.05
    assert pos[0] <= 0.05 + 1e-9
    np.testing.assert_allclose(pos[0], 0.01, atol=1e-6)


def test_deadman_holds():
    r = CartesianRetargeter()
    origin = np.array([0.1, 0.2, 0.3])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    pos, *_ = r.step(
        leader_pos=[0.4, 0, 0],
        leader_quat=_identity_quat(),
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=False,
        current_dex_pos=origin,
        current_dex_quat=_identity_quat(),
    )
    np.testing.assert_allclose(pos, origin, atol=1e-9)


def test_gripper_hysteresis():
    r = CartesianRetargeter(RetargetConfig(gripper_hysteresis=0.2, gripper_open_limit=1.0, gripper_close_norm=0.0))
    a = r.map_gripper(0.0)
    b = r.map_gripper(0.05)
    c = r.map_gripper(0.5)
    assert a == b
    assert c > a


def test_gripper_close_norm_early_full_close():
    """Almost-closed leader (below close_norm) must fully close DexMate."""
    r = CartesianRetargeter(
        RetargetConfig(
            gripper_close=0.0,
            gripper_open_limit=1.0,
            gripper_close_norm=0.20,
            gripper_hysteresis=0.0,
        )
    )
    assert r.map_gripper(0.20) == 0.0
    assert r.map_gripper(0.10) == 0.0
    # Midway between close_norm and open → half aperture.
    mid = r.map_gripper(0.60)
    np.testing.assert_allclose(mid, 0.5, atol=1e-9)
    np.testing.assert_allclose(r.map_gripper(1.0), 1.0, atol=1e-9)


def test_accel_limit_does_not_overshoot_stopped_target():
    """Coast/overshoot after the leader stops caused keyboard EE drift."""
    cfg = RetargetConfig(
        max_lin_vel=1.0,
        max_ang_vel=10.0,
        max_lin_acc=0.5,
    )
    r = CartesianRetargeter(cfg)
    origin = np.array([0.0, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, _identity_quat())
    # Build up velocity toward a far target.
    for _ in range(20):
        r.step(
            leader_pos=[0.5, 0.0, 0.0],
            leader_quat=_identity_quat(),
            gripper_norm=0.0,
            dt=0.05,
            clutch=True,
            deadman=True,
            current_dex_pos=origin,
            current_dex_quat=_identity_quat(),
        )
    assert float(np.linalg.norm(r.state.last_lin_vel)) > 0.05
    stop = r.state.last_pos.copy()
    # Leader freezes at the current commanded pose: must not keep coasting.
    for _ in range(30):
        pos, *_ = r.step(
            leader_pos=stop - origin,  # relative map: dex = origin + leader
            leader_quat=_identity_quat(),
            gripper_norm=0.0,
            dt=0.05,
            clutch=True,
            deadman=True,
            current_dex_pos=stop,
            current_dex_quat=_identity_quat(),
        )
    np.testing.assert_allclose(pos, stop, atol=1e-6)
    np.testing.assert_allclose(r.state.last_lin_vel, 0.0, atol=1e-9)


def test_reanchor_after_clutch_off_keeps_dex_and_tracks_new_origin():
    """Freeze, move leader, re-engage: DexMate stays put then tracks deltas."""
    r = CartesianRetargeter(
        RetargetConfig(max_lin_vel=10.0, max_ang_vel=10.0, max_lin_acc=0.0)
    )
    dex0 = np.array([0.5, 0.0, 0.3])
    pos, _, _, info = r.step(
        leader_pos=[0.0, 0.0, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=dex0,
        current_dex_quat=_identity_quat(),
    )
    assert info["reason"] == "clutch_engage"
    np.testing.assert_allclose(pos, dex0)

    pos, _, _, info = r.step(
        leader_pos=[0.05, 0.0, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=dex0,
        current_dex_quat=_identity_quat(),
    )
    assert info["reason"] == "tracking"
    held = pos.copy()

    pos, _, _, info = r.step(
        leader_pos=[0.20, 0.10, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=1.0,
        dt=0.02,
        clutch=False,
        deadman=True,
        current_dex_pos=held,
        current_dex_quat=_identity_quat(),
    )
    assert info["reason"] == "clutch_off"
    np.testing.assert_allclose(pos, held)

    r.disengage()
    pos, _, _, info = r.step(
        leader_pos=[0.20, 0.10, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=held,
        current_dex_quat=_identity_quat(),
    )
    assert info["reason"] == "clutch_engage"
    np.testing.assert_allclose(pos, held)
    np.testing.assert_allclose(r.state.leader_origin_pos, [0.20, 0.10, 0.0])
    np.testing.assert_allclose(r.state.dex_origin_pos, held)

    pos, _, _, info = r.step(
        leader_pos=[0.25, 0.10, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=held,
        current_dex_quat=_identity_quat(),
    )
    assert info["reason"] == "tracking"
    np.testing.assert_allclose(pos, held + np.array([0.05, 0.0, 0.0]), atol=1e-6)

