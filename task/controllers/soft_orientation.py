"""Soft orientation helpers for claw-machine teleop (top-down + free tilts).

Generate a small set of world EE quaternions (wxyz) inside a cone around a
preferred orientation. Used by LulaIKController when ``orientation_cone_rad``
is set: try preferred first, then tilts, so translation is not blocked when
exact top-down is IK-infeasible.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-9


def _normalize_quat_wxyz(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    if q[0] < 0.0:
        q = -q
    return q


def _quat_mul(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = _normalize_quat_wxyz(q1)
    w2, x2, y2, z2 = _normalize_quat_wxyz(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_angle(q1: Sequence[float], q2: Sequence[float]) -> float:
    a = _normalize_quat_wxyz(q1)
    b = _normalize_quat_wxyz(q2)
    dot = float(np.clip(np.abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _rotvec_to_quat(rotvec: Sequence[float]) -> np.ndarray:
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle <= _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = v / angle
    s = np.sin(angle * 0.5)
    return _normalize_quat_wxyz(
        [np.cos(angle * 0.5), axis[0] * s, axis[1] * s, axis[2] * s]
    )


def cone_orientation_samples(
    preferred_wxyz: Sequence[float],
    cone_rad: float,
    *,
    last_achieved_wxyz: Optional[Sequence[float]] = None,
) -> List[np.ndarray]:
    """Return preferred + in-cone tilts, nearest-first (unique up to sign).

    ``cone_rad <= 0`` → only the preferred quat (hard lock).
    """
    preferred = _normalize_quat_wxyz(preferred_wxyz)
    cone = float(cone_rad)
    out: List[np.ndarray] = [preferred]
    if cone <= _EPS:
        return out

    candidates: List[Tuple[float, np.ndarray]] = []
    if last_achieved_wxyz is not None:
        last = _normalize_quat_wxyz(last_achieved_wxyz)
        ang = _quat_angle(preferred, last)
        if ang <= cone + 1e-6 and ang > 1e-6:
            candidates.append((ang, last))

    # Half- and full-cone tilts about world X/Y/Z (space-fixed relative to preferred).
    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    fractions = (0.5, 1.0)
    for axis in axes:
        for frac in fractions:
            ang = cone * float(frac)
            if ang <= _EPS:
                continue
            for sign in (1.0, -1.0):
                q_rel = _rotvec_to_quat(axis * (sign * ang))
                q = _quat_mul(q_rel, preferred)
                a = _quat_angle(preferred, q)
                if a <= cone + 1e-6:
                    candidates.append((a, q))

    candidates.sort(key=lambda t: t[0])
    for _, q in candidates:
        if all(_quat_angle(q, existing) > 1e-4 for existing in out):
            out.append(q)
    return out
