"""Unit tests for claw-machine soft orientation cone samples."""
from __future__ import annotations

import numpy as np

from controllers.soft_orientation import cone_orientation_samples, _quat_angle
from teleop.retarget import RetargetConfig


_TOP_DOWN = (0.0, 1.0, 0.0, 0.0)


def test_cone_zero_is_preferred_only():
    samples = cone_orientation_samples(_TOP_DOWN, 0.0)
    assert len(samples) == 1
    np.testing.assert_allclose(samples[0], _TOP_DOWN, atol=1e-9)


def test_cone_samples_stay_within_cone():
    cone = 0.40
    samples = cone_orientation_samples(_TOP_DOWN, cone)
    assert len(samples) >= 2
    np.testing.assert_allclose(samples[0], _TOP_DOWN, atol=1e-9)
    for q in samples:
        assert _quat_angle(_TOP_DOWN, q) <= cone + 1e-5


def test_cone_prefers_last_achieved_early():
    cone = 0.40
    # Build a tilt at half-cone about X.
    ang = 0.5 * cone
    s = np.sin(ang / 2.0)
    q_rel = np.array([np.cos(ang / 2.0), s, 0.0, 0.0])
    # q = q_rel * top_down
    w1, x1, y1, z1 = q_rel
    w2, x2, y2, z2 = _TOP_DOWN
    last = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    samples = cone_orientation_samples(
        _TOP_DOWN, cone, last_achieved_wxyz=last
    )
    assert len(samples) >= 2
    # Second sample should be the last achieved (hysteresis), within cone.
    assert _quat_angle(samples[1], last) < 1e-4


def test_retarget_config_parses_orientation_cone_rad():
    cfg = RetargetConfig.from_dict(
        {
            "fixed_orientation_wxyz": [0.0, 1.0, 0.0, 0.0],
            "orientation_cone_rad": 0.40,
        }
    )
    assert cfg.fixed_orientation_wxyz == (0.0, 1.0, 0.0, 0.0)
    assert cfg.orientation_cone_rad == 0.40
    cfg_off = RetargetConfig.from_dict({"orientation_cone_rad": None})
    assert cfg_off.orientation_cone_rad is None
    cfg_false = RetargetConfig.from_dict({"orientation_cone_rad": False})
    assert cfg_false.orientation_cone_rad is None
