"""Minimal SE(3) helpers that do not depend on scipy."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

EPS = 1e-9


def as_vec(values: Iterable[float], n: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size != n:
        raise ValueError(f"expected {n} values, got {arr.size}")
    return arr


def normalize_quat_wxyz(quat: Sequence[float]) -> np.ndarray:
    q = as_vec(quat, 4)
    n = float(np.linalg.norm(q))
    if n <= EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    if q[0] < 0.0:
        q = -q
    return q


def quat_multiply_wxyz(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quat_wxyz(q1)
    w2, x2, y2, z2 = normalize_quat_wxyz(q2)
    return normalize_quat_wxyz(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def quat_conjugate_wxyz(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_slerp_wxyz(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    a = normalize_quat_wxyz(q0)
    b = normalize_quat_wxyz(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    t = float(np.clip(t, 0.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_wxyz(a + t * (b - a))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = np.sin(theta) / sin_0
    return normalize_quat_wxyz(s0 * a + s1 * b)


def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quat_wxyz((w, x, y, z))


def pose_from_matrix(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T[:3, 3].copy(), matrix_to_quat_wxyz(T[:3, :3])


def matrix_from_pose(pos: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    T[:3, 3] = as_vec(pos, 3)
    return T


def quat_wxyz_to_rotvec(quat: Sequence[float]) -> np.ndarray:
    q = normalize_quat_wxyz(quat)
    w = float(np.clip(q[0], -1.0, 1.0))
    xyz = q[1:]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = xyz / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    return axis * angle


def rotvec_to_quat_wxyz(rotvec: Sequence[float]) -> np.ndarray:
    v = as_vec(rotvec, 3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = v / angle
    half = 0.5 * angle
    s = np.sin(half)
    return normalize_quat_wxyz((np.cos(half), *(axis * s)))


def apply_axes_map(vec: Sequence[float], perm: Sequence[int], signs: Sequence[float]) -> np.ndarray:
    v = as_vec(vec, 3)
    p = np.asarray(list(perm), dtype=np.int64)
    s = as_vec(signs, 3)
    if p.size != 3 or set(int(i) for i in p) != {0, 1, 2}:
        raise ValueError(f"perm must be a permutation of 0,1,2; got {perm}")
    return s * v[p]


def clamp_vec(vec: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    return np.minimum(as_vec(hi, 3), np.maximum(as_vec(lo, 3), as_vec(vec, 3)))


def limit_delta(delta: Sequence[float], max_abs: float) -> np.ndarray:
    v = as_vec(delta, 3)
    n = float(np.linalg.norm(v))
    if n <= max_abs or n <= EPS:
        return v
    return v * (max_abs / n)
