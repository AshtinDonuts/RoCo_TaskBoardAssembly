"""Pixel ↔ world-XY helpers for the ±1 cm board-plane domain."""
from __future__ import annotations

import numpy as np

from .constants import NUMERICAL_CLAMP_EPS_M, XY_MAX_M, XY_MIN_M


def as_xy(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"expected at least two values, got {value!r}")
    return np.array([float(arr[0]), float(arr[1])], dtype=np.float64)


def clamp_xy(offset_xy, *, min_m=XY_MIN_M, max_m=XY_MAX_M,
             numerical_eps=NUMERICAL_CLAMP_EPS_M) -> np.ndarray:
    """Clamp tiny numerical overshoot into the official XY square.

    Values far outside the square are still clipped — the estimator should
    not propose a 5 cm correction — but ``numerical_eps`` documents that
    only millimetre-scale FFT/NCC error is expected.
    """
    xy = as_xy(offset_xy)
    lo = float(min_m) - float(numerical_eps)
    hi = float(max_m) + float(numerical_eps)
    if xy[0] < lo or xy[0] > hi or xy[1] < lo or xy[1] > hi:
        # Still clip: a failed match must not command a huge waypoint jump.
        pass
    return np.clip(xy, min_m, max_m).astype(np.float64)


def pixel_delta_to_world_xy(du_dv, jacobian_2x2) -> np.ndarray:
    """Map a pixel displacement ``(du, dv)`` to world XY metres."""
    delta = np.asarray(du_dv, dtype=np.float64).reshape(2)
    jac = np.asarray(jacobian_2x2, dtype=np.float64).reshape(2, 2)
    return jac @ delta


def world_xy_to_pixel_delta(offset_xy, jacobian_2x2) -> np.ndarray:
    jac = np.asarray(jacobian_2x2, dtype=np.float64).reshape(2, 2)
    try:
        inv = np.linalg.inv(jac)
    except np.linalg.LinAlgError as exc:
        raise ValueError("pixel-to-world Jacobian is singular") from exc
    return inv @ as_xy(offset_xy)


def roi_half_extent_px(jacobian_2x2, template_shape, xy_limit_m=XY_MAX_M,
                       margin_px=6) -> tuple[int, int]:
    """Search half-size that covers ±xy_limit plus template and margin."""
    jac = np.asarray(jacobian_2x2, dtype=np.float64).reshape(2, 2)
    try:
        inv = np.linalg.inv(jac)
    except np.linalg.LinAlgError:
        inv = np.array([[1000.0, 0.0], [0.0, 1000.0]], dtype=np.float64)
    corners = np.array(
        [[xy_limit_m, xy_limit_m],
         [xy_limit_m, -xy_limit_m],
         [-xy_limit_m, xy_limit_m],
         [-xy_limit_m, -xy_limit_m]],
        dtype=np.float64,
    )
    pix = np.abs(corners @ inv.T)
    extra_u = int(np.ceil(float(np.max(pix[:, 0]))))
    extra_v = int(np.ceil(float(np.max(pix[:, 1]))))
    th, tw = (int(template_shape[0]), int(template_shape[1]))
    return extra_u + tw // 2 + int(margin_px), extra_v + th // 2 + int(margin_px)


def rot_from_wxyz(quat) -> np.ndarray:
    w, x, y, z = [float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4)]
    n = w * w + x * x + y * y + z * z
    if n <= 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pinhole_project(world_xyz, K, R_world_from_cam, t_world_from_cam):
    """Project world points with OpenCV pinhole (Z forward in camera)."""
    pts = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R_world_from_cam, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_from_cam, dtype=np.float64).reshape(3)
    R_cam_from_world = R.T
    t_cam = -R_cam_from_world @ t
    cam = (R_cam_from_world @ pts.T).T + t_cam
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    z = cam[:, 2]
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    valid = z > 1e-8
    uv[valid, 0] = K[0, 0] * cam[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * cam[valid, 1] / z[valid] + K[1, 2]
    return uv


def jacobian_at_uv(K, R_world_from_cam, t_world_from_cam, uv, plane_z):
    """Finite-difference d(world_xy)/d(pixel) on a horizontal plane."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    R = np.asarray(R_world_from_cam, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t_world_from_cam, dtype=np.float64).reshape(3)

    def plane_xy(u, v):
        ray = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=np.float64)
        origin = t
        dir_w = R @ ray
        if abs(dir_w[2]) < 1e-12:
            raise ValueError("ray parallel to board plane")
        s = (float(plane_z) - origin[2]) / dir_w[2]
        p = origin + s * dir_w
        return p[:2]

    eps = 1.0
    c = plane_xy(uv[0], uv[1])
    du = plane_xy(uv[0] + eps, uv[1]) - c
    dv = plane_xy(uv[0], uv[1] + eps) - c
    return np.column_stack([du, dv]).astype(np.float64)
