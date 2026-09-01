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
    r = CartesianRetargeter(
        RetargetConfig(
            gripper_mode="continuous",
            gripper_hysteresis=0.2,
            gripper_open_limit=1.0,
            gripper_close_norm=0.0,
        )
    )
    a = r.map_gripper(0.0)
    b = r.map_gripper(0.05)
    c = r.map_gripper(0.5)
    assert a == b
    assert c > a


def test_gripper_close_norm_early_full_close():
    """Almost-closed leader (below close_norm) must fully close DexMate."""
    r = CartesianRetargeter(
        RetargetConfig(
            gripper_mode="continuous",
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


def test_gripper_binary_map_endpoints():
    r = CartesianRetargeter(
        RetargetConfig(
            gripper_compliance_enabled=True,
            gripper_mode="binary",
            gripper_close=0.0,
            gripper_open_limit=1.0,
            gripper_close_norm=0.20,
            gripper_hysteresis=0.05,
        )
    )
    assert r.map_gripper(0.10) == 0.0
    assert r.map_gripper(0.50) == 1.0


def test_gripper_compliance_disabled_uses_linear_map():
    r = CartesianRetargeter(
        RetargetConfig(
            gripper_compliance_enabled=False,
            gripper_mode="binary",  # ignored for mapping when disabled
            gripper_close=0.0,
            gripper_open_limit=1.0,
            gripper_close_norm=0.20,
            gripper_hysteresis=0.0,
        )
    )
    mid = r.map_gripper(0.60)
    np.testing.assert_allclose(mid, 0.5, atol=1e-9)


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


def test_proximity_band_scale_curve():
    from teleop.retarget import ProximityScaleConfig, proximity_band_scale

    cfg = ProximityScaleConfig(
        enabled=True, depth_outer_m=0.40, depth_inner_m=0.20, scale_min=0.20
    )
    assert proximity_band_scale(None, cfg) == 1.0
    assert proximity_band_scale(-1.0, cfg) == 1.0
    assert proximity_band_scale(0.50, cfg) == 1.0
    assert proximity_band_scale(0.40, cfg) == 1.0
    assert proximity_band_scale(0.20, cfg) == 0.20
    assert proximity_band_scale(0.10, cfg) == 0.20
    mid = proximity_band_scale(0.30, cfg)
    np.testing.assert_allclose(mid, 0.60, atol=1e-9)
    assert proximity_band_scale(0.30, ProximityScaleConfig(enabled=False)) == 1.0


def test_motion_scale_throttles_lin_and_ang_rate_limits():
    """Explicit motion_scale multiplies vel caps; absolute map unchanged."""
    from teleop import transforms as T

    cfg = RetargetConfig(
        translation_gain=1.0,
        rotation_gain=1.0,
        max_lin_vel=1.0,
        max_ang_vel=1.0,
        max_lin_acc=0.0,
        workspace_min=(-1.5, -1.5, -1.5),
        workspace_max=(1.5, 1.5, 1.5),
    )
    origin = np.array([0.0, 0.0, 0.0])
    q0 = _identity_quat()
    leader_pos = np.array([1.0, 0.0, 0.0])
    leader_quat = T.rotvec_to_quat_wxyz([0.0, 0.0, 1.5])

    r_full = CartesianRetargeter(cfg)
    r_full.capture_origins([0, 0, 0], q0, origin, q0)
    pos_full, quat_full, _, info_full = r_full.step(
        leader_pos=leader_pos,
        leader_quat=leader_quat,
        gripper_norm=0.0,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=q0,
        motion_scale=1.0,
    )

    r_slow = CartesianRetargeter(cfg)
    r_slow.capture_origins([0, 0, 0], q0, origin, q0)
    pos_slow, quat_slow, _, info_slow = r_slow.step(
        leader_pos=leader_pos,
        leader_quat=leader_quat,
        gripper_norm=0.0,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=q0,
        motion_scale=0.20,
    )

    assert info_full["rate_limit_scale"] == 1.0
    assert info_slow["rate_limit_scale"] == 0.20
    step_full = float(np.linalg.norm(pos_full - origin))
    step_slow = float(np.linalg.norm(pos_slow - origin))
    assert step_full > 1e-6
    np.testing.assert_allclose(step_slow / step_full, 0.20, atol=1e-6)

    ang_full = float(np.linalg.norm(T.quat_wxyz_to_rotvec(
        T.quat_multiply_wxyz(T.quat_conjugate_wxyz(q0), quat_full)
    )))
    ang_slow = float(np.linalg.norm(T.quat_wxyz_to_rotvec(
        T.quat_multiply_wxyz(T.quat_conjugate_wxyz(q0), quat_slow)
    )))
    assert ang_full > 1e-6
    np.testing.assert_allclose(ang_slow / ang_full, 0.20, atol=1e-6)


def test_retarget_config_parses_proximity_keys():
    from teleop.retarget import ProximityScaleConfig

    cfg = RetargetConfig.from_dict(
        {
            "translation_gain": 2.0,
            "proximity_slowdown": {  # legacy alias → rate_limit
                "enabled": True,
                "depth_outer_m": 0.4,
                "depth_inner_m": 0.2,
                "scale_min": 0.2,
            },
            "proximity_delta_gain": {
                "enabled": True,
                "depth_outer_m": 0.4,
                "depth_inner_m": 0.25,
                "scale_min": 0.05,
            },
        }
    )
    assert cfg.translation_gain == 2.0
    assert cfg.proximity_rate_limit.enabled is True
    assert cfg.proximity_rate_limit.scale_min == 0.2
    assert cfg.proximity_delta_gain.enabled is True
    assert cfg.proximity_delta_gain.scale_min == 0.05
    assert isinstance(cfg.proximity_rate_limit, ProximityScaleConfig)


def test_delta_gain_scales_incremental_target_no_catchup():
    """Near-band delta gain shrinks motion; holding leader does not catch up."""
    from teleop.retarget import ProximityScaleConfig

    cfg = RetargetConfig(
        translation_gain=1.0,
        max_lin_vel=10.0,
        max_ang_vel=10.0,
        max_lin_acc=0.0,
        workspace_min=(-2, -2, -2),
        workspace_max=(2, 2, 2),
        proximity_delta_gain=ProximityScaleConfig(
            enabled=True,
            depth_outer_m=0.40,
            depth_inner_m=0.20,
            scale_min=0.20,
        ),
        proximity_rate_limit=ProximityScaleConfig(enabled=False),
    )
    q0 = _identity_quat()
    origin = np.zeros(3)
    r = CartesianRetargeter(cfg)
    r.capture_origins([0, 0, 0], q0, origin, q0)

    # One 5 cm leader step at full scale (depth far).
    pos_far, _, _, info_far = r.step(
        leader_pos=[0.05, 0, 0],
        leader_quat=q0,
        gripper_norm=0.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=q0,
        proximity_depth_m=0.50,
    )
    assert info_far["delta_gain_scale"] == 1.0
    np.testing.assert_allclose(pos_far, [0.05, 0, 0], atol=1e-6)

    # Another 5 cm at near-band scale_min=0.2 → only +1 cm DexMate.
    pos_near, _, _, info_near = r.step(
        leader_pos=[0.10, 0, 0],
        leader_quat=q0,
        gripper_norm=0.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=pos_far,
        current_dex_quat=q0,
        proximity_depth_m=0.20,
    )
    assert info_near["delta_gain_scale"] == 0.20
    np.testing.assert_allclose(pos_near, [0.06, 0, 0], atol=1e-6)

    # Hold leader still — must not catch up toward a 0.10 absolute map.
    pos_hold, _, _, _ = r.step(
        leader_pos=[0.10, 0, 0],
        leader_quat=q0,
        gripper_norm=0.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=pos_near,
        current_dex_quat=q0,
        proximity_depth_m=0.20,
    )
    np.testing.assert_allclose(pos_hold, pos_near, atol=1e-9)


def test_delta_gain_scale_drop_does_not_snap():
    """Changing proximity mid-hold must not yank the clutched pose."""
    from teleop.retarget import ProximityScaleConfig

    cfg = RetargetConfig(
        translation_gain=2.0,
        max_lin_vel=10.0,
        max_ang_vel=10.0,
        max_lin_acc=0.0,
        workspace_min=(-2, -2, -2),
        workspace_max=(2, 2, 2),
        proximity_delta_gain=ProximityScaleConfig(
            enabled=True, depth_outer_m=0.4, depth_inner_m=0.2, scale_min=0.05
        ),
    )
    q0 = _identity_quat()
    origin = np.array([0.3, 0.0, 1.0])
    r = CartesianRetargeter(cfg)
    r.capture_origins([0, 0, 0], q0, origin, q0)
    pos, _, _, _ = r.step(
        leader_pos=[0.05, 0, 0],
        leader_quat=q0,
        gripper_norm=0.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=q0,
        proximity_depth_m=0.50,
    )
    # Depth collapses while leader holds — pose stays put.
    pos2, _, _, info = r.step(
        leader_pos=[0.05, 0, 0],
        leader_quat=q0,
        gripper_norm=0.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=pos,
        current_dex_quat=q0,
        proximity_depth_m=0.20,
    )
    assert info["delta_gain_scale"] == 0.05
    np.testing.assert_allclose(pos2, pos, atol=1e-9)


def test_fix_orientation_holds_dex_origin_quat():
    """Leader wrist tilts must not change the commanded DexMate quat."""
    from teleop import transforms as T

    dex_quat = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.0, 0.3, 0.0]))
    )
    r = CartesianRetargeter(
        RetargetConfig(
            fix_orientation=True,
            translation_gain=2.0,
            max_lin_vel=10.0,
            max_ang_vel=10.0,
            max_lin_acc=0.0,
        )
    )
    origin = np.array([0.3, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, dex_quat)

    tilted = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.4, -0.2, 0.5]))
    )
    pos, quat, _, info = r.step(
        leader_pos=[0.05, 0.0, 0.0],
        leader_quat=tilted,
        gripper_norm=0.5,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=dex_quat,
    )
    assert info["reason"] == "tracking"
    np.testing.assert_allclose(pos, origin + np.array([0.10, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(quat, dex_quat, atol=1e-9)

    # Still fixed after another large tilt + translation.
    tilted2 = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([-0.8, 0.1, 0.2]))
    )
    pos2, quat2, _, _ = r.step(
        leader_pos=[0.08, 0.02, 0.0],
        leader_quat=tilted2,
        gripper_norm=0.5,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=pos,
        current_dex_quat=quat,
    )
    np.testing.assert_allclose(pos2, origin + np.array([0.16, 0.04, 0.0]), atol=1e-6)
    np.testing.assert_allclose(quat2, dex_quat, atol=1e-9)


def test_retarget_config_parses_fix_orientation():
    cfg = RetargetConfig.from_dict({"fix_orientation": True, "rotation_gain": 0.5})
    assert cfg.fix_orientation is True
    assert cfg.rotation_gain == 0.5


def test_retarget_config_parses_fixed_orientation_wxyz():
    cfg = RetargetConfig.from_dict(
        {"fixed_orientation_wxyz": [0.0, 1.0, 0.0, 0.0], "fix_orientation": False}
    )
    assert cfg.fixed_orientation_wxyz == (0.0, 1.0, 0.0, 0.0)
    assert cfg.fix_orientation is False
    cfg_none = RetargetConfig.from_dict({"fixed_orientation_wxyz": None})
    assert cfg_none.fixed_orientation_wxyz is None
    cfg_false = RetargetConfig.from_dict({"fixed_orientation_wxyz": False})
    assert cfg_false.fixed_orientation_wxyz is None
    cfg_empty = RetargetConfig.from_dict({"fixed_orientation_wxyz": []})
    assert cfg_empty.fixed_orientation_wxyz is None


def test_fixed_orientation_wxyz_ignores_leader_wrist_relative():
    """Constant world quat; XYZ still tracks under relative (non-delta) map."""
    from teleop import transforms as T

    top_down = (0.0, 1.0, 0.0, 0.0)
    start_quat = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.0, 0.4, 0.0]))
    )
    r = CartesianRetargeter(
        RetargetConfig(
            fixed_orientation_wxyz=top_down,
            translation_gain=2.0,
            max_lin_vel=10.0,
            # Large enough that one step reaches top-down from start_quat.
            max_ang_vel=10.0,
            max_lin_acc=0.0,
        )
    )
    origin = np.array([0.3, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, start_quat)

    # Slew onto the lock before asserting tilt-ignore (angle to top-down ≫
    # one frame at max_ang_vel=10).
    pos, quat = origin, start_quat
    for _ in range(80):
        pos, quat, _, _ = r.step(
            leader_pos=[0.0, 0.0, 0.0],
            leader_quat=_identity_quat(),
            gripper_norm=0.5,
            dt=0.05,
            clutch=True,
            deadman=True,
            current_dex_pos=pos,
            current_dex_quat=quat,
        )
    np.testing.assert_allclose(quat, top_down, atol=1e-6)
    np.testing.assert_allclose(pos, origin, atol=1e-6)

    tilted = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.4, -0.2, 0.5]))
    )
    pos, quat, _, info = r.step(
        leader_pos=[0.05, 0.0, 0.0],
        leader_quat=tilted,
        gripper_norm=0.5,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=pos,
        current_dex_quat=quat,
    )
    assert info["reason"] == "tracking"
    np.testing.assert_allclose(pos, origin + np.array([0.10, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(quat, top_down, atol=1e-9)

    tilted2 = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([-0.8, 0.1, 0.2]))
    )
    pos2, quat2, _, _ = r.step(
        leader_pos=[0.08, 0.02, 0.0],
        leader_quat=tilted2,
        gripper_norm=0.5,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=pos,
        current_dex_quat=quat,
    )
    np.testing.assert_allclose(pos2, origin + np.array([0.16, 0.04, 0.0]), atol=1e-6)
    np.testing.assert_allclose(quat2, top_down, atol=1e-9)


def test_fixed_orientation_wxyz_ignores_leader_wrist_incremental():
    """Constant world quat under proximity_delta_gain incremental path."""
    from teleop import transforms as T
    from teleop.retarget import ProximityScaleConfig

    top_down = (0.0, 1.0, 0.0, 0.0)
    start_quat = _identity_quat()
    r = CartesianRetargeter(
        RetargetConfig(
            fixed_orientation_wxyz=top_down,
            translation_gain=1.0,
            max_lin_vel=10.0,
            max_ang_vel=10.0,
            max_lin_acc=0.0,
            workspace_min=(-1.5, -1.5, -1.5),
            proximity_delta_gain=ProximityScaleConfig(
                enabled=True,
                depth_outer_m=0.40,
                depth_inner_m=0.20,
                scale_min=1.0,
            ),
        )
    )
    origin = np.array([0.0, 0.0, 0.0])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, start_quat)

    # Reach the lock first (identity → top-down needs several rate-limited steps).
    pos, quat = origin, start_quat
    for _ in range(80):
        pos, quat, _, _ = r.step(
            leader_pos=[0.0, 0.0, 0.0],
            leader_quat=_identity_quat(),
            gripper_norm=0.0,
            dt=0.05,
            clutch=True,
            deadman=True,
            current_dex_pos=pos,
            current_dex_quat=quat,
            proximity_depth_m=0.50,
        )
    np.testing.assert_allclose(quat, top_down, atol=1e-6)

    tilted = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.5, -0.3, 0.2]))
    )
    pos, quat, _, info = r.step(
        leader_pos=[0.05, 0.0, 0.0],
        leader_quat=tilted,
        gripper_norm=0.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=pos,
        current_dex_quat=quat,
        proximity_depth_m=0.50,
    )
    assert info["reason"] == "tracking"
    np.testing.assert_allclose(pos, [0.05, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(quat, top_down, atol=1e-9)


def test_fixed_orientation_wxyz_rate_limit_slews_not_snap():
    """From a non-top-down home, slew toward lock at max_ang_vel (no snap)."""
    from teleop import transforms as T

    top_down = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    # ~90 deg about Y from identity — far from top-down.
    start_quat = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.0, np.pi / 2, 0.0]))
    )
    # Small ang vel so one 50 ms step cannot reach top-down.
    max_ang = 0.5  # rad/s → 0.025 rad per 0.05 s
    r = CartesianRetargeter(
        RetargetConfig(
            fixed_orientation_wxyz=tuple(top_down.tolist()),
            translation_gain=1.0,
            max_lin_vel=10.0,
            max_ang_vel=max_ang,
            max_lin_acc=0.0,
        )
    )
    origin = np.array([0.3, 0.0, 0.2])
    r.capture_origins([0, 0, 0], _identity_quat(), origin, start_quat)

    pos, quat, _, info = r.step(
        leader_pos=[0.0, 0.0, 0.0],
        leader_quat=_identity_quat(),
        gripper_norm=0.0,
        dt=0.05,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=start_quat,
    )
    assert info["reason"] == "tracking"
    np.testing.assert_allclose(pos, origin, atol=1e-9)
    # Must have moved toward top-down, but not arrived in one step.
    assert not np.allclose(quat, start_quat, atol=1e-6)
    assert not np.allclose(quat, top_down, atol=1e-3)
    q_rel = T.quat_multiply_wxyz(T.quat_conjugate_wxyz(start_quat), quat)
    angle = float(np.linalg.norm(T.quat_wxyz_to_rotvec(q_rel)))
    np.testing.assert_allclose(angle, max_ang * 0.05, atol=1e-6)

    # After enough steps, should reach the lock.
    for _ in range(200):
        pos, quat, _, _ = r.step(
            leader_pos=[0.0, 0.0, 0.0],
            leader_quat=_identity_quat(),
            gripper_norm=0.0,
            dt=0.05,
            clutch=True,
            deadman=True,
            current_dex_pos=pos,
            current_dex_quat=quat,
        )
    np.testing.assert_allclose(quat, top_down, atol=1e-6)


def test_spike_filter_rejects_large_leader_jump():
    """One-frame leader glitch must not move the DexMate target."""
    from teleop.retarget import ProximityScaleConfig

    origin = np.array([0.1, 0.2, 0.3])
    r = CartesianRetargeter(
        RetargetConfig(
            translation_gain=1.0,
            max_lin_vel=10.0,
            max_ang_vel=10.0,
            max_lin_acc=0.0,
            spike_filter_enabled=True,
            spike_max_lin_vel=1.0,  # 0.02 m/frame at dt=0.02
            spike_max_ang_vel=100.0,
            spike_max_gripper_vel=100.0,
            proximity_delta_gain=ProximityScaleConfig(enabled=False),
        )
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    pos, _, _, info = r.step(
        leader_pos=[0.0, 0.0, 0.0],
        leader_quat=identity,
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=identity,
    )
    assert info["reason"] == "clutch_engage"
    np.testing.assert_allclose(pos, origin)

    pos, _, _, info = r.step(
        leader_pos=[0.01, 0.0, 0.0],
        leader_quat=identity,
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=origin,
        current_dex_quat=identity,
    )
    assert info["spike_rejected"] is False
    assert float(np.linalg.norm(pos - origin)) > 1e-6
    good = pos.copy()

    pos2, _, _, info2 = r.step(
        leader_pos=[0.21, 0.0, 0.0],
        leader_quat=identity,
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=good,
        current_dex_quat=identity,
    )
    assert info2["spike_rejected"] is True
    assert info2["spike_lin"] is True
    np.testing.assert_allclose(pos2, good, atol=1e-9)

    pos3, _, _, info3 = r.step(
        leader_pos=[0.015, 0.0, 0.0],
        leader_quat=identity,
        gripper_norm=1.0,
        dt=0.02,
        clutch=True,
        deadman=True,
        current_dex_pos=good,
        current_dex_quat=identity,
    )
    assert info3["spike_rejected"] is False


def test_spike_filter_config_from_dict():
    cfg = RetargetConfig.from_dict(
        {
            "spike_filter_enabled": True,
            "spike_max_lin_vel": 2.0,
            "spike_max_ang_vel": 4.0,
            "spike_max_gripper_vel": 3.0,
        }
    )
    assert cfg.spike_filter_enabled is True
    assert cfg.spike_max_lin_vel == 2.0
    assert cfg.spike_max_ang_vel == 4.0
    assert cfg.spike_max_gripper_vel == 3.0

