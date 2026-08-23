"""Right-wrist TCP approach laser for visual guidance.

Casts a ray from the right EE tool frame along gripper local +Z (approach),
terminating at the first non-robot collider hit.

Drawing strategy:
  1. Prefer ``omni.debugdraw`` (viewport overlay; not in camera RGB).
  2. If debugdraw is unavailable, fall back to a thin USD cylinder.
  3. When ``record`` is True, always keep the USD cylinder visible so
     head/wrist camera frames (datasets / --record-video) include the beam.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import carb
import numpy as np
import omni.usd
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, UsdGeom

from .ee_pose_controller import _approach_axis_world
from .pick_place_task import ROBOT_PRIM_PATH

_LASER_PRIM_PATH = "/World/_r_wrist_laser"
_LASER_COLOR_ARGB = 0xFFFF2020  # bright red (AARRGGBB)
_DEBUGDRAW_EXT = "omni.debugdraw"


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


def _rotation_aligning_z_to(direction: np.ndarray) -> Gf.Quatd:
    """Unit quaternion (Gf) rotating local +Z onto ``direction``."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    d = d / n
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
    if dot < -1.0 + 1e-8:
        return Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))
    axis = np.cross(z, d)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle = float(np.arccos(dot))
    half = 0.5 * angle
    s = float(np.sin(half))
    return Gf.Quatd(float(np.cos(half)), Gf.Vec3d(*(axis * s)))


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


def _try_get_debug_draw():
    """Enable omni.debugdraw if needed and return its interface, or None."""
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        if app is not None:
            ext_mgr = app.get_extension_manager()
            if ext_mgr is not None and not ext_mgr.is_extension_enabled(_DEBUGDRAW_EXT):
                ext_mgr.set_extension_enabled_immediate(_DEBUGDRAW_EXT, True)
    except Exception:
        pass
    try:
        from omni.debugdraw import get_debug_draw_interface

        iface = get_debug_draw_interface()
        if iface is None:
            return None
        return iface
    except Exception:
        return None


class RWristLaser:
    """Per-frame right-wrist approach laser."""

    def __init__(
        self,
        *,
        max_length: float = 2.0,
        radius: float = 0.0015,
        origin_offset_m: float = 0.02,
        tip_offset_ee: Optional[Sequence[float]] = None,
        record: bool = False,
        color_argb: int = _LASER_COLOR_ARGB,
        line_width: float = 2.0,
        exclude_prefixes: Optional[Sequence[str]] = None,
    ):
        self.max_length = float(max_length)
        self.radius = float(radius)
        self.origin_offset_m = float(origin_offset_m)
        self.tip_offset_ee = (
            np.asarray(tip_offset_ee, dtype=np.float64).reshape(3)
            if tip_offset_ee is not None
            else None
        )
        self.record = bool(record)
        self.color_argb = int(color_argb)
        self.line_width = float(line_width)
        self.exclude_prefixes = list(
            exclude_prefixes
            if exclude_prefixes is not None
            else (ROBOT_PRIM_PATH, _LASER_PRIM_PATH)
        )
        self._dd = None
        self._dd_checked = False
        self._use_usd_guidance = False
        self._cyl = None
        self._xform_ops = None
        self.last_origin: Optional[np.ndarray] = None
        self.last_end: Optional[np.ndarray] = None
        self.last_hit: bool = False
        self.last_length: float = 0.0

    def set_record(self, record: bool) -> None:
        self.record = bool(record)
        if not self.record and not self._use_usd_guidance:
            self._set_usd_visible(False)

    def _ensure_debug_draw(self) -> None:
        if self._dd_checked:
            return
        self._dd_checked = True
        self._dd = _try_get_debug_draw()
        if self._dd is None:
            # Viewport guidance without the Kit extension: keep a USD beam.
            self._use_usd_guidance = True
            print(
                "[r_wrist_laser] omni.debugdraw unavailable; "
                "using USD cylinder for viewport guidance "
                "(will appear in camera RGB whenever visible).",
                flush=True,
            )

    def _ensure_usd(self) -> None:
        if self._cyl is not None:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        if stage.GetPrimAtPath(_LASER_PRIM_PATH).IsValid():
            stage.RemovePrim(_LASER_PRIM_PATH)
        cyl = UsdGeom.Cylinder.Define(stage, _LASER_PRIM_PATH)
        cyl.CreateAxisAttr(UsdGeom.Tokens.z)
        cyl.CreateRadiusAttr(self.radius)
        cyl.CreateHeightAttr(self.max_length)
        cyl.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.12, 0.12)])
        cyl.CreateDisplayOpacityAttr([0.95])
        prim = cyl.GetPrim()
        imageable = UsdGeom.Imageable(prim)
        imageable.CreateVisibilityAttr().Set("inherited")
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        t_op = xform.AddTranslateOp()
        o_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self._cyl = cyl
        self._xform_ops = (t_op, o_op)
        self._set_usd_visible(False)

    def _set_usd_visible(self, visible: bool) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(_LASER_PRIM_PATH)
        if not prim or not prim.IsValid():
            return
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _update_usd_beam(self, origin: np.ndarray, end: np.ndarray) -> None:
        self._ensure_usd()
        if self._cyl is None or self._xform_ops is None:
            return
        delta = end - origin
        length = float(np.linalg.norm(delta))
        if length < 1e-4:
            self._set_usd_visible(False)
            return
        direction = delta / length
        mid = origin + 0.5 * delta
        t_op, o_op = self._xform_ops
        self._cyl.GetHeightAttr().Set(length)
        self._cyl.GetRadiusAttr().Set(self.radius)
        t_op.Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
        o_op.Set(_rotation_aligning_z_to(direction))
        self._set_usd_visible(True)

    def update(self, ee_pos, ee_quat_wxyz) -> None:
        """Raycast and draw. Call once per rendered sim frame."""
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

        self._ensure_debug_draw()
        if self._dd is not None:
            w = self.line_width
            self._dd.draw_line(
                _as_float3(origin),
                self.color_argb,
                w,
                _as_float3(end),
                self.color_argb,
                w,
            )

        # USD beam: required when recording into cameras, or as guidance
        # fallback when debugdraw is missing.
        if self.record or self._use_usd_guidance:
            self._update_usd_beam(origin, end)
        else:
            self._set_usd_visible(False)
