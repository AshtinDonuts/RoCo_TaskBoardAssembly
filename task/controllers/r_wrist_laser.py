"""Right-wrist TCP approach laser for visual guidance.

Casts a ray from the right EE tool frame along gripper local +Z (approach),
terminating at the first non-robot collider hit. Distance is logged on a
throttle; the beam is drawn only as a 2D overlay on R-wrist camera RGB
(no world USD prim / debugdraw). Live preview uses an ``omni.ui`` window —
OpenCV HighGUI (``cv2.imshow``) fights Kit's windowing and aborts the app.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import carb
import numpy as np
import omni.usd
from omni.physx import get_physx_scene_query_interface
from pxr import UsdGeom

from .ee_pose_controller import _approach_axis_world
from .pick_place_task import ROBOT_PRIM_PATH

_OVERLAY_WINDOW = "R Wrist Laser"
_UI_MIN_PERIOD_S = 0.05  # ~20 Hz preview; avoids flooding ByteImageProvider


def _as_float3(v: Sequence[float]) -> carb.Float3:
    return carb.Float3(float(v[0]), float(v[1]), float(v[2]))


def _quat_wxyz_to_rot_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _path_excluded(path: str, prefixes: Iterable[str]) -> bool:
    if not path:
        return False
    return any(path == p or path.startswith(p + "/") for p in prefixes)


def raycast_beam_end(
    origin: np.ndarray,
    direction: np.ndarray,
    max_length: float,
    exclude_prefixes: Sequence[str],
) -> Tuple[np.ndarray, float, bool]:
    """Return (end_point, length, hit_object).

    Skips colliders under ``exclude_prefixes`` (robot self-hits) by walking
    past them along the ray.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(direction))
    if n < 1e-12 or max_length <= 0.0:
        return origin.copy(), 0.0, False
    direction = direction / n

    iface = get_physx_scene_query_interface()
    traveled = 0.0
    query_origin = origin.copy()
    for _ in range(24):
        remaining = float(max_length - traveled)
        if remaining <= 1e-4:
            break
        hit = iface.raycast_closest(
            _as_float3(query_origin),
            _as_float3(direction),
            remaining,
        )
        if not hit or not hit.get("hit"):
            end = origin + direction * max_length
            return end, float(max_length), False

        dist = float(hit.get("distance", 0.0))
        body = str(hit.get("rigidBody") or "")
        coll = str(hit.get("collision") or "")
        if _path_excluded(body, exclude_prefixes) or _path_excluded(
            coll, exclude_prefixes
        ):
            traveled += dist + 1e-3
            query_origin = origin + direction * traveled
            continue

        total = traveled + dist
        pos = hit.get("position")
        if pos is not None:
            end = np.array(
                [float(pos[0]), float(pos[1]), float(pos[2])],
                dtype=np.float64,
            )
        else:
            end = origin + direction * total
        return end, float(total), True

    end = origin + direction * max_length
    return end, float(max_length), False


def _camera_world_Rt(
    camera_prim_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (R_world_from_cam 3x3, t_world 3,) for the camera prim."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim or not prim.IsValid():
        return None
    mat = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    rot = mat.ExtractRotationQuat()
    imag = rot.GetImaginary()
    quat_wxyz = np.array(
        [rot.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64
    )
    R = _quat_wxyz_to_rot_matrix(quat_wxyz)
    t_w = np.array([t[0], t[1], t[2]], dtype=np.float64)
    return R, t_w


def _world_to_camera(
    point_w: np.ndarray, R_wc: np.ndarray, t_w: np.ndarray
) -> np.ndarray:
    """Transform world point into USD camera frame (looks down local -Z)."""
    return R_wc.T @ (np.asarray(point_w, dtype=np.float64).reshape(3) - t_w)


def _project_point(
    point_w: np.ndarray,
    R_wc: np.ndarray,
    t_w: np.ndarray,
    K: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Return (u, v, depth_along_view) or None if behind the camera.

    USD / Isaac cameras look along local -Z; depth = -Z_cam (>0 in front).
    """
    p_cam = _world_to_camera(point_w, R_wc, t_w)
    depth = float(-p_cam[2])
    if depth <= 1e-6:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (float(p_cam[0]) / depth) + cx
    v = fy * (float(p_cam[1]) / depth) + cy
    return u, v, depth


def _clip_segment_to_image(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    width: int,
    height: int,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Cohen–Sutherland clip; returns integer pixel endpoints or None."""
    x_min, y_min, x_max, y_max = 0.0, 0.0, float(width - 1), float(height - 1)

    def _code(x: float, y: float) -> int:
        c = 0
        if x < x_min:
            c |= 1
        elif x > x_max:
            c |= 2
        if y < y_min:
            c |= 4
        elif y > y_max:
            c |= 8
        return c

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    c0, c1 = _code(x0, y0), _code(x1, y1)
    for _ in range(8):
        if not (c0 | c1):
            return (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))
        if c0 & c1:
            return None
        c_out = c0 or c1
        if c_out & 8:
            x = x0 + (x1 - x0) * (y_max - y0) / (y1 - y0 + 1e-12)
            y = y_max
        elif c_out & 4:
            x = x0 + (x1 - x0) * (y_min - y0) / (y1 - y0 + 1e-12)
            y = y_min
        elif c_out & 2:
            y = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0 + 1e-12)
            x = x_max
        else:
            y = y0 + (y1 - y0) * (x_min - x0) / (x1 - x0 + 1e-12)
            x = x_min
        if c_out == c0:
            x0, y0 = x, y
            c0 = _code(x0, y0)
        else:
            x1, y1 = x, y
            c1 = _code(x1, y1)
    return None


class RWristLaser:
    """Per-frame right-wrist approach laser (raycast + RGB overlay)."""

    def __init__(
        self,
        *,
        max_length: float = 2.0,
        origin_offset_m: float = 0.02,
        tip_offset_ee: Optional[Sequence[float]] = None,
        camera_prim_path: Optional[str] = None,
        exclude_prefixes: Optional[Sequence[str]] = None,
        line_thickness: int = 2,
        show_window: bool = True,
    ):
        self.max_length = float(max_length)
        self.origin_offset_m = float(origin_offset_m)
        self.tip_offset_ee = (
            np.asarray(tip_offset_ee, dtype=np.float64).reshape(3)
            if tip_offset_ee is not None
            else None
        )
        self.camera_prim_path = camera_prim_path
        self.exclude_prefixes = list(
            exclude_prefixes if exclude_prefixes is not None else (ROBOT_PRIM_PATH,)
        )
        self.line_thickness = int(line_thickness)
        self.show_window = bool(show_window)

        self.last_origin: Optional[np.ndarray] = None
        self.last_end: Optional[np.ndarray] = None
        self.last_hit: bool = False
        self.last_length: float = 0.0
        self._last_log_s: float = -1e9
        self._last_ui_s: float = -1e9
        self._window_created = False
        self._ui_window: Any = None
        self._ui_provider: Any = None
        self._ui_failed = False

    def update(self, ee_pos, ee_quat_wxyz) -> None:
        """Raycast along gripper +Z. Call once per rendered sim frame."""
        pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(ee_quat_wxyz, dtype=np.float64).reshape(-1)
        direction = _approach_axis_world(quat)
        n = float(np.linalg.norm(direction))
        if n < 1e-12:
            return
        direction = direction / n

        if self.tip_offset_ee is not None:
            origin = pos + _quat_wxyz_to_rot_matrix(quat) @ self.tip_offset_ee
        else:
            origin = pos + direction * self.origin_offset_m

        end, length, hit = raycast_beam_end(
            origin,
            direction,
            self.max_length,
            self.exclude_prefixes,
        )
        self.last_origin = origin
        self.last_end = end
        self.last_hit = bool(hit)
        self.last_length = float(length)

    def maybe_log_distance(self, now_s: float, period_s: float = 0.5) -> None:
        if now_s - self._last_log_s < float(period_s):
            return
        self._last_log_s = float(now_s)
        tag = "hit" if self.last_hit else "max"
        print(
            f"[r_wrist_laser] dist={self.last_length:.3f} m ({tag})",
            flush=True,
        )

    def overlay_rgb(self, rgb, camera) -> Optional[np.ndarray]:
        """Draw the last beam onto a copy of ``rgb`` (HxWx3). Returns RGB uint8."""
        if (
            rgb is None
            or self.last_origin is None
            or self.last_end is None
            or camera is None
        ):
            return None

        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = np.ascontiguousarray(arr[..., :3].copy())
        h, w = out.shape[:2]

        try:
            K = camera.get_intrinsics_matrix()
        except Exception:
            K = None
        if K is None:
            return out
        K = np.asarray(K, dtype=np.float64)

        cam_path = self.camera_prim_path or getattr(camera, "prim_path", None)
        if not cam_path:
            return out
        Rt = _camera_world_Rt(str(cam_path))
        if Rt is None:
            return out
        R_wc, t_w = Rt

        # Subsample the beam so a long ray still projects if endpoints clip.
        n_pts = 16
        pts_uv = []
        for i in range(n_pts + 1):
            alpha = i / float(n_pts)
            pw = self.last_origin * (1.0 - alpha) + self.last_end * alpha
            proj = _project_point(pw, R_wc, t_w, K)
            if proj is not None:
                pts_uv.append((proj[0], proj[1]))

        try:
            import cv2
        except ImportError:
            return out

        if len(pts_uv) >= 2:
            for a, b in zip(pts_uv[:-1], pts_uv[1:]):
                clipped = _clip_segment_to_image(a, b, w, h)
                if clipped is None:
                    continue
                # Draw in RGB; Kit ByteImageProvider / video writers expect RGB.
                cv2.line(out, clipped[0], clipped[1], (255, 32, 32), self.line_thickness)
            end_uv = pts_uv[-1]
            if 0 <= end_uv[0] < w and 0 <= end_uv[1] < h:
                cv2.circle(
                    out,
                    (int(round(end_uv[0])), int(round(end_uv[1]))),
                    max(3, self.line_thickness + 1),
                    (255, 255, 0),
                    -1,
                )
        return out

    def show_overlay(self, overlay_rgb: Optional[np.ndarray], now_s: Optional[float] = None) -> None:
        """Show overlaid RGB in a Kit ``omni.ui`` window (never OpenCV HighGUI)."""
        if not self.show_window or overlay_rgb is None or self._ui_failed:
            return
        t = float(now_s) if now_s is not None else self._last_log_s
        if t - self._last_ui_s < _UI_MIN_PERIOD_S and self._window_created:
            return
        self._last_ui_s = t

        arr = np.asarray(overlay_rgb)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return
        h, w = int(arr.shape[0]), int(arr.shape[1])
        rgb = np.ascontiguousarray(arr[..., :3], dtype=np.uint8)
        rgba = np.dstack(
            [rgb, np.full((h, w), 255, dtype=np.uint8)]
        )
        # ByteImageProvider wants a flat sequence of RGBA bytes.
        pixels = rgba.reshape(-1).tolist()

        try:
            import omni.ui as ui
        except Exception as exc:
            self._ui_failed = True
            print(f"[r_wrist_laser] omni.ui unavailable; preview disabled: {exc}", flush=True)
            return

        try:
            if self._ui_provider is None or self._ui_window is None:
                self._ui_provider = ui.ByteImageProvider()
                self._ui_window = ui.Window(
                    _OVERLAY_WINDOW,
                    width=max(320, w),
                    height=max(240, h),
                    visible=True,
                )
                with self._ui_window.frame:
                    ui.ImageWithProvider(
                        self._ui_provider,
                        fill_policy=ui.IwpFillPolicy.IWP_STRETCH,
                    )
                self._window_created = True
                print(
                    f"[r_wrist_laser] Kit preview window '{_OVERLAY_WINDOW}' "
                    f"({w}x{h})",
                    flush=True,
                )
            self._ui_provider.set_bytes_data(pixels, [w, h])
        except Exception as exc:
            self._ui_failed = True
            print(f"[r_wrist_laser] preview failed; disabling: {exc}", flush=True)
            self.close()

    def close(self) -> None:
        if self._ui_window is not None:
            try:
                self._ui_window.visible = False
                self._ui_window = None
            except Exception:
                pass
        self._ui_provider = None
        self._window_created = False
