"""Unit tests for slow-close + stall hold gripper compliance."""
from __future__ import annotations

import numpy as np

from controllers.gripper_compliance import (
    GripperCompliance,
    GripperComplianceConfig,
    GripperPhase,
    binary_intent_from_norm,
)


def _cfg(**kwargs) -> GripperComplianceConfig:
    base = dict(
        mode="binary",
        open_limit=1.0,
        close=0.0,
        close_norm=0.20,
        hysteresis=0.05,
        close_speed_rad_s=0.5,
        open_speed_rad_s=2.0,
        stall_qd=0.02,
        stall_err=0.02,
        stall_dq=0.005,
        stall_min_close_rad=0.03,
        hold_margin=0.01,
        # Allow classic lag stall in most unit tests (production uses 0.01).
        max_close_lag=0.05,
        stall_hold_ticks=1,
        stall_progress=0.002,
    )
    base.update(kwargs)
    return GripperComplianceConfig(**base)


def test_binary_intent_hysteresis():
    assert binary_intent_from_norm(0.10, close_norm=0.20, hysteresis=0.05, prev_intent="open") == "close"
    assert binary_intent_from_norm(0.50, close_norm=0.20, hysteresis=0.05, prev_intent="close") == "open"
    # In the deadband, keep previous intent.
    assert binary_intent_from_norm(0.22, close_norm=0.20, hysteresis=0.05, prev_intent="close") == "close"
    assert binary_intent_from_norm(0.22, close_norm=0.20, hysteresis=0.05, prev_intent="open") == "open"


def test_slew_respects_close_speed():
    g = GripperCompliance(_cfg(close_speed_rad_s=0.5))
    q = 1.0
    dt = 0.02
    cmd = g.update(q_meas=q, qd_meas=-0.5, dt=dt, intent="close")
    # One step at most 0.5 * 0.02 = 0.01 rad.
    np.testing.assert_allclose(cmd, 1.0 - 0.01, atol=1e-9)
    assert g.phase == GripperPhase.CLOSING


def test_stall_enters_hold():
    g = GripperCompliance(_cfg(close_speed_rad_s=1.0, stall_min_close_rad=0.03))
    # Seed open, then close while fingers follow for a bit.
    q = 1.0
    for _ in range(5):
        cmd = g.update(q_meas=q, qd_meas=-0.5, dt=0.02, intent="close")
        q = float(cmd)
    assert g.phase == GripperPhase.CLOSING
    # Fingers stall at a fixed aperture while command walks further closed.
    q_stall = float(q)
    for _ in range(20):
        cmd = g.update(q_meas=q_stall, qd_meas=0.0, dt=0.02, intent="close")
        if g.phase == GripperPhase.HOLDING:
            break
    assert g.phase == GripperPhase.HOLDING
    np.testing.assert_allclose(cmd, q_stall - 0.01, atol=1e-9)
    # Further close ticks keep the hold aperture.
    cmd2 = g.update(q_meas=q_stall, qd_meas=0.0, dt=0.02, intent="close")
    np.testing.assert_allclose(cmd2, cmd, atol=1e-9)


def test_max_close_lag_blocks_catchup():
    """Leader-0 catch-up must not command far past measured contact."""
    g = GripperCompliance(
        _cfg(
            mode="continuous",
            close_speed_rad_s=10.0,
            max_close_lag=0.01,
            stall_err=0.008,
            stall_hold_ticks=50,  # don't latch yet; inspect lag cap
            stall_min_close_rad=0.03,
        )
    )
    q_contact = 0.40
    # Seed so close_start is past min close.
    g.reset(q_meas=0.50)
    for _ in range(5):
        cmd = g.update(
            q_meas=q_contact,
            qd_meas=0.0,
            dt=0.02,
            gripper_norm=0.0,  # full close demand
        )
        assert cmd >= q_contact - 0.01 - 1e-9
        assert g.phase != GripperPhase.HOLDING


def test_lag_saturated_stall_holds():
    """Contact: lag at cap + frozen fingers → hold (not want_more_close alone)."""
    g = GripperCompliance(
        _cfg(
            mode="continuous",
            close_speed_rad_s=0.5,
            max_close_lag=0.01,
            stall_err=0.008,
            stall_hold_ticks=3,
            hold_margin=0.002,
            stall_min_close_rad=0.03,
            stall_progress=0.002,
        )
    )
    q = 0.55
    g.reset(q_meas=0.60)
    for _ in range(8):
        cmd = g.update(q_meas=q, qd_meas=-0.2, dt=0.02, gripper_norm=0.0)
        q = float(cmd)
    q_stall = float(q)
    for _ in range(10):
        cmd = g.update(q_meas=q_stall, qd_meas=0.0, dt=0.02, gripper_norm=0.0)
        if g.phase == GripperPhase.HOLDING:
            break
    assert g.phase == GripperPhase.HOLDING
    np.testing.assert_allclose(cmd, q_stall - 0.002, atol=1e-9)
    cmd2 = g.update(q_meas=q_stall, qd_meas=0.0, dt=0.02, gripper_norm=0.0)
    np.testing.assert_allclose(cmd2, cmd, atol=1e-9)


def test_no_false_hold_slow_free_air_creep():
    """Soft-drive free-air creep must reach full close, not mid-hold."""
    g = GripperCompliance(
        _cfg(
            mode="continuous",
            close_speed_rad_s=0.25,
            max_close_lag=0.01,
            stall_err=0.008,
            stall_dq=0.005,
            stall_hold_ticks=12,
            stall_progress=0.002,
            stall_min_close_rad=0.03,
            hold_margin=0.002,
        )
    )
    q = 0.60
    g.reset(q_meas=q)
    # Creep ~0.001 rad/tick with lag saturated (typical soft free-air).
    for _ in range(800):
        cmd = g.update(q_meas=q, qd_meas=-0.01, dt=0.02, gripper_norm=0.0)
        assert g.phase != GripperPhase.HOLDING
        step = min(0.001, abs(float(cmd) - q))
        q = q + np.sign(float(cmd) - q) * step
        if q <= 1e-6 and abs(cmd) <= 1e-6:
            break
    assert q <= 1e-5
    assert abs(cmd) <= 1e-5
    assert g.phase != GripperPhase.HOLDING


def test_open_clears_hold():
    g = GripperCompliance(_cfg())
    g.phase = GripperPhase.HOLDING
    g._intent = "close"
    g._q_cmd = 0.4
    g._q_hold = 0.4
    g._q_meas_prev = 0.41
    g._close_start_q = 1.0
    cmd = g.update(q_meas=0.41, qd_meas=0.0, dt=0.02, intent="open")
    assert g.phase == GripperPhase.OPENING
    assert g._q_hold is None
    assert cmd > 0.4


def test_no_false_hold_in_free_motion():
    g = GripperCompliance(_cfg(close_speed_rad_s=0.5, stall_min_close_rad=0.05))
    q = 1.0
    for _ in range(4):
        # Fingers track the command with closing velocity.
        cmd = g.update(q_meas=q, qd_meas=-0.5, dt=0.02, intent="close")
        q = cmd
        assert g.phase == GripperPhase.CLOSING


def test_continuous_mode_from_norm():
    g = GripperCompliance(_cfg(mode="continuous", close_speed_rad_s=10.0))
    # Mid aperture target; fingers track with non-zero qd so no false hold.
    cmd = 1.0
    for _ in range(50):
        cmd = g.update(
            q_meas=cmd,
            qd_meas=-0.5,
            dt=0.02,
            gripper_norm=0.6,  # close_norm=0.2 → mid of [0.2,1] → 0.5
        )
    np.testing.assert_allclose(cmd, 0.5, atol=1e-6)


def test_config_from_retarget_aliases():
    cfg = GripperComplianceConfig.from_dict(
        {
            "gripper_compliance_enabled": True,
            "gripper_mode": "binary",
            "gripper_close_speed_rad_s": 0.07,
            "gripper_hold_margin": 0.02,
            "gripper_max_close_lag": 0.015,
            "gripper_stall_hold_ticks": 4,
            "gripper_stall_progress": 0.003,
        }
    )
    assert cfg.enabled is True
    assert cfg.mode == "binary"
    assert cfg.close_speed_rad_s == 0.07
    assert cfg.hold_margin == 0.02
    assert cfg.max_close_lag == 0.015
    assert cfg.stall_hold_ticks == 4
    assert cfg.stall_progress == 0.003


def test_disabled_snaps_without_slew_or_hold():
    g = GripperCompliance(_cfg(enabled=False, close_speed_rad_s=0.01))
    cmd = g.update(q_meas=1.0, qd_meas=0.0, dt=0.02, intent="close")
    np.testing.assert_allclose(cmd, 0.0, atol=1e-9)
    # No stall hold latch when disabled.
    cmd2 = g.update(q_meas=0.5, qd_meas=0.0, dt=0.02, intent="close")
    np.testing.assert_allclose(cmd2, 0.0, atol=1e-9)
    assert g._q_hold is None


def test_freeze_prefers_measured_aperture_over_lagged_command():
    g = GripperCompliance(
        _cfg(hold_margin=0.002, max_close_lag=0.01, stall_hold_ticks=50)
    )
    g.reset(q_meas=0.12)
    g._q_cmd = 0.06
    hold = g.freeze(q_meas=0.07)
    np.testing.assert_allclose(hold, 0.068, atol=1e-9)
    assert g.phase == GripperPhase.HOLDING
    np.testing.assert_allclose(g.q_cmd, hold, atol=1e-9)


def test_production_defaults_latch_hold_when_stall_err_le_lag():
    """Regression: stall_err > max_close_lag never latches (keeps squeezing)."""
    contact = 0.07

    def _run(cfg: GripperComplianceConfig):
        g = GripperCompliance(cfg)
        q = 0.12
        g.reset(q_meas=q)
        for _ in range(2000):
            q_meas = max(contact, float(g.q_cmd) if g.q_cmd is not None else q)
            qd = 0.0 if q_meas <= contact + 1e-9 else -0.05
            if q_meas <= contact + 1e-9:
                q_meas = contact
            g.update(q_meas=q_meas, qd_meas=qd, dt=0.005, intent="close")
            if g.phase == GripperPhase.HOLDING:
                return True
        return False

    bad_cfg = GripperComplianceConfig(stall_err=0.02, max_close_lag=0.01)
    production_cfg = GripperComplianceConfig()
    assert production_cfg.stall_err <= production_cfg.max_close_lag
    assert _run(bad_cfg) is False
    assert _run(production_cfg) is True
