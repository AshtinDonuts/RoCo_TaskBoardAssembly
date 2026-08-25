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
        }
    )
    assert cfg.enabled is True
    assert cfg.mode == "binary"
    assert cfg.close_speed_rad_s == 0.07
    assert cfg.hold_margin == 0.02


def test_disabled_snaps_without_slew_or_hold():
    g = GripperCompliance(_cfg(enabled=False, close_speed_rad_s=0.01))
    cmd = g.update(q_meas=1.0, qd_meas=0.0, dt=0.02, intent="close")
    np.testing.assert_allclose(cmd, 0.0, atol=1e-9)
    # No stall hold latch when disabled.
    cmd2 = g.update(q_meas=0.5, qd_meas=0.0, dt=0.02, intent="close")
    np.testing.assert_allclose(cmd2, 0.0, atol=1e-9)
    assert g._q_hold is None
