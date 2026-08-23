"""Right-wrist TCP approach laser for visual guidance.

Terminates the beam at the first surface along the wrist-cam line of sight
(center depth) when available, with PhysX raycasts (tool +Z and camera look)
as fallback. Table/board visuals often lack PhysX colliders, which previously
left the overlay stuck on MAX length. Overlay follows AimBot: tool-forward
shooting line + stop reticle; distance is also shown via HUD text / range bar.
Live preview uses an ``omni.ui`` window — OpenCV HighGUI fights Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class BeamCastResult:
    """Outcome of a single approach-axis PhysX cast."""

    end: np.ndarray
    length: float
    hit: bool
    collision_path: str = ""
    rigid_body_path: str = ""
    skipped_robot_hits: int = 0
    total_hits: int = 0
    nearest_any_distance: float = -1.0  # closest hit incl. robot; -1 if none
    source: str = ""  # "closest_walk" | "raycast_all" | ""
    hit_summaries: Tuple[str, ...] = ()


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


def _hit_paths(hit: Any) -> Tuple[str, str]:
    """Extract (collision, rigid_body) paths from a RaycastHit or dict."""
    if isinstance(hit, dict):
        return str(hit.get("collision") or ""), str(
            hit.get("rigidBody") or hit.get("rigid_body") or ""
        )
    coll = str(getattr(hit, "collision", "") or "")
    body = str(getattr(hit, "rigid_body", "") or "")
    return coll, body


def _hit_distance(hit: Any) -> float:
    if isinstance(hit, dict):
        return float(hit.get("distance", 0.0))
    return float(getattr(hit, "distance", 0.0) or 0.0)


def _hit_position(hit: Any) -> Optional[np.ndarray]:
    pos = hit.get("position") if isinstance(hit, dict) else getattr(hit, "position", None)
    if pos is None:
        return None
    return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)


def _snapshot_hit(hit: Any) -> dict:
    """Copy hit fields immediately — PhysX may reuse the callback object."""
    coll, body = _hit_paths(hit)
    pos = _hit_position(hit)
    return {
        "distance": _hit_distance(hit),
        "collision": coll,
        "rigidBody": body,
        "position": None if pos is None else pos.copy(),
    }


def _summarize_hits(hits: Sequence[dict], limit: int = 3) -> Tuple[str, ...]:
    ordered = sorted(hits, key=lambda h: float(h["distance"]))[: int(limit)]
    out: list = []
    for h in ordered:
        path = str(h.get("collision") or h.get("rigidBody") or "?")
        # Keep log lines short: last 2 path components.
        tail = "/".join(path.strip("/").split("/")[-2:]) if path else "?"
        out.append(f"{float(h['distance']):.4f}:{tail}")
    return tuple(out)


def _pick_first_non_excluded(
    origin: np.ndarray,
    direction: np.ndarray,
    max_length: float,
    hits: Sequence[dict],
    exclude_prefixes: Sequence[str],
    *,
    source: str = "",
) -> BeamCastResult:
    """Sort snapshotted hits by distance; keep first non-robot collider."""
    ordered = sorted(hits, key=lambda h: float(h["distance"]))
    nearest_any = float(ordered[0]["distance"]) if ordered else -1.0
    summaries = _summarize_hits(ordered, limit=3)
    skipped = 0
    for hit in ordered:
        coll = str(hit.get("collision") or "")
        body = str(hit.get("rigidBody") or "")
        if _path_excluded(body, exclude_prefixes) or _path_excluded(
            coll, exclude_prefixes
        ):
            skipped += 1
            continue
        dist = float(hit["distance"])
        # Accept near-contact hits (dist ~= 0); do not apply a min-distance gate.
        pos = hit.get("position")
        end = (
            np.asarray(pos, dtype=np.float64).reshape(3)
            if pos is not None
            else (origin + direction * dist)
        )
        return BeamCastResult(
            end=end,
            length=float(dist),
            hit=True,
            collision_path=coll,
            rigid_body_path=body,
            skipped_robot_hits=skipped,
            total_hits=len(ordered),
            nearest_any_distance=nearest_any,
            source=source,
            hit_summaries=summaries,
        )

    end = origin + direction * max_length
    return BeamCastResult(
        end=end,
        length=float(max_length),
        hit=False,
        skipped_robot_hits=skipped,
        total_hits=len(ordered),
        nearest_any_distance=nearest_any,
        source=source,
        hit_summaries=summaries,
    )


def _raycast_all_snapshots(
    origin: np.ndarray,
    direction: np.ndarray,
    max_length: float,
    both_sides: bool,
) -> list:
    """Gather all ray hits with field copies (safe across PhysX callbacks)."""
    hits: list = []

    def _report(hit) -> bool:
        hits.append(_snapshot_hit(hit))
        return True

    iface = get_physx_scene_query_interface()
    iface.raycast_all(
        _as_float3(origin),
        _as_float3(direction),
        float(max_length),
        _report,
        bool(both_sides),
    )
    return hits


def _raycast_closest_walk_snapshots(
    origin: np.ndarray,
    direction: np.ndarray,
    max_length: float,
    exclude_prefixes: Sequence[str],
    both_sides: bool,
) -> list:
    """Walk raycast_closest, snapshotting every reported hit along the ray."""
    iface = get_physx_scene_query_interface()
    hits: list = []
    traveled = 0.0
    query_origin = origin.copy()
    for _ in range(32):
        remaining = float(max_length - traveled)
        if remaining <= 1e-6:
            break
        hit = iface.raycast_closest(
            _as_float3(query_origin),
            _as_float3(direction),
            remaining,
            bool(both_sides),
        )
        if not hit or not hit.get("hit"):
            break
        snap = _snapshot_hit(hit)
        # Convert to distance along the original ray.
        snap["distance"] = traveled + float(snap["distance"])
        if snap["position"] is None:
            snap["position"] = origin + direction * snap["distance"]
        hits.append(snap)

        coll, body = snap["collision"], snap["rigidBody"]
        step = float(hit.get("distance", 0.0)) + 1e-3
        traveled += step
        query_origin = origin + direction * traveled
        if not (
            _path_excluded(body, exclude_prefixes)
            or _path_excluded(coll, exclude_prefixes)
        ):
            break
    return hits


def raycast_beam_end(
    origin: np.ndarray,
    direction: np.ndarray,
    max_length: float,
    exclude_prefixes: Sequence[str],
    *,
    both_sides: bool = True,
) -> BeamCastResult:
    """Return the first non-excluded collider along the ray.

    Primary: walk ``raycast_closest`` (stable first-hit). Fallback / cross-check:
    copied ``raycast_all`` if the walk finds nothing.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(direction))
    if n < 1e-12 or max_length <= 0.0:
        return BeamCastResult(end=origin.copy(), length=0.0, hit=False)
    direction = direction / n

    hits_walk = _raycast_closest_walk_snapshots(
        origin, direction, max_length, exclude_prefixes, both_sides
    )
    if hits_walk:
        return _pick_first_non_excluded(
            origin,
            direction,
            max_length,
            hits_walk,
            exclude_prefixes,
            source="closest_walk",
        )

    hits_all = _raycast_all_snapshots(origin, direction, max_length, both_sides)
    return _pick_first_non_excluded(
        origin,
        direction,
        max_length,
        hits_all,
        exclude_prefixes,
        source="raycast_all",
    )


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


def _world_to_usd_camera(
    point_w: np.ndarray, R_wc: np.ndarray, t_w: np.ndarray
) -> np.ndarray:
    """World point → USD camera frame (+Y up, looks along local -Z)."""
    return R_wc.T @ (np.asarray(point_w, dtype=np.float64).reshape(3) - t_w)


def _usd_cam_to_cv(p_usd: np.ndarray) -> np.ndarray:
    """USD camera (+Y up, -Z look) → OpenCV camera (+Y down, +Z look)."""
    p = np.asarray(p_usd, dtype=np.float64).reshape(3)
    return np.array([p[0], -p[1], -p[2]], dtype=np.float64)


def _world_to_cv_camera(
    point_w: np.ndarray, R_wc: np.ndarray, t_w: np.ndarray
) -> np.ndarray:
    return _usd_cam_to_cv(_world_to_usd_camera(point_w, R_wc, t_w))


def _project_cv(
    p_cv: np.ndarray, K: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """Project OpenCV-camera point → (u, v, Z). None if behind near plane."""
    z = float(p_cv[2])
    if z <= 1e-4:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (float(p_cv[0]) / z) + cx
    v = fy * (float(p_cv[1]) / z) + cy
    return u, v, z


def _clip_segment_to_near(
    p0_cv: np.ndarray, p1_cv: np.ndarray, z_near: float = 1e-3
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a camera-frame segment to Z >= z_near. None if fully behind."""
    a = np.asarray(p0_cv, dtype=np.float64).reshape(3)
    b = np.asarray(p1_cv, dtype=np.float64).reshape(3)
    za, zb = float(a[2]), float(b[2])
    a_in, b_in = za >= z_near, zb >= z_near
    if a_in and b_in:
        return a, b
    if (not a_in) and (not b_in):
        return None
    # One endpoint behind: intersect Z = z_near.
    t = (z_near - za) / (zb - za + 1e-12)
    mid = a + t * (b - a)
    mid[2] = z_near
    if a_in:
        return a, mid
    return mid, b


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


def _draw_crosshair(
    img: np.ndarray,
    center: Tuple[int, int],
    *,
    arm_len: int,
    gap: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """AimBot-style scope reticle at the projected stop point."""
    try:
        import cv2
    except ImportError:
        return
    u, v = int(center[0]), int(center[1])
    h, w = img.shape[:2]
    g = max(1, int(gap))
    L = max(g + 1, int(arm_len))

    def _seg(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        clipped = _clip_segment_to_image(a, b, w, h)
        if clipped is not None:
            cv2.line(img, clipped[0], clipped[1], color, thickness)

    _seg((u + g, v), (u + L, v))
    _seg((u - g, v), (u - L, v))
    _seg((u, v + g), (u, v + L))
    _seg((u, v - g), (u, v - L))


def _camera_look_world(R_wc: np.ndarray) -> np.ndarray:
    """USD camera forward in world (local −Z)."""
    look = -np.asarray(R_wc, dtype=np.float64).reshape(3, 3)[:, 2]
    n = float(np.linalg.norm(look))
    if n < 1e-12:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return look / n


def _depth_center_meters(
    depth: np.ndarray, *, patch: int = 5
) -> Optional[float]:
    """Median finite depth (m) in a small patch around the image center."""
    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0:
        return None
    h, w = int(arr.shape[0]), int(arr.shape[1])
    cy, cx = h // 2, w // 2
    r = max(0, int(patch) // 2)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch_vals = arr[y0:y1, x0:x1].reshape(-1)
    finite = patch_vals[np.isfinite(patch_vals) & (patch_vals > 1e-4)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


class RWristLaser:
    """Per-frame right-wrist approach laser (raycast + RGB overlay)."""

    def __init__(
        self,
        *,
        max_length: float = 2.0,
        origin_offset_m: float = 0.0,
        # 0 by default: a positive offset can start the query past a near
        # contact (gripper already touching table/object) and miss the hit.
        raycast_origin_offset_m: float = 0.0,
        tip_offset_ee: Optional[Sequence[float]] = None,
        camera_prim_path: Optional[str] = None,
        exclude_prefixes: Optional[Sequence[str]] = None,
        both_sides: bool = True,
        prefer_depth: bool = True,
        line_thickness: int = 2,
        show_window: bool = True,
        debug_log: bool = False,
    ):
        self.max_length = float(max_length)
        # Visual emission at TCP by default (AimBot shooting-line start).
        self.origin_offset_m = float(origin_offset_m)
        # PhysX cast may start slightly ahead of the emit point; keep 0 unless
        # self-hits are noisy (robot hits are already filtered by prefix).
        self.raycast_origin_offset_m = float(raycast_origin_offset_m)
        self.tip_offset_ee = (
            np.asarray(tip_offset_ee, dtype=np.float64).reshape(3)
            if tip_offset_ee is not None
            else None
        )
        self.camera_prim_path = camera_prim_path
        self.exclude_prefixes = list(
            exclude_prefixes if exclude_prefixes is not None else (ROBOT_PRIM_PATH,)
        )
        self.both_sides = bool(both_sides)
        self.prefer_depth = bool(prefer_depth)
        self.line_thickness = int(line_thickness)
        self.show_window = bool(show_window)
        self.debug_log = bool(debug_log)

        self.last_origin: Optional[np.ndarray] = None  # visual / emit origin
        self.last_ray_origin: Optional[np.ndarray] = None
        self.last_end: Optional[np.ndarray] = None
        self.last_hit: bool = False
        self.last_length: float = 0.0
        self.last_collision_path: str = ""
        self.last_rigid_body_path: str = ""
        self.last_skipped_robot_hits: int = 0
        self.last_total_hits: int = 0
        self.last_nearest_any_distance: float = -1.0
        self.last_cast_source: str = ""
        self.last_hit_summaries: Tuple[str, ...] = ()
        self.last_stop_uv: Optional[Tuple[float, float, float]] = None
        self.last_direction: Optional[np.ndarray] = None
        self.last_depth_m: float = -1.0
        self._last_log_s: float = -1e9
        self._last_ui_s: float = -1e9
        self._window_created = False
        self._ui_window: Any = None
        self._ui_provider: Any = None
        self._ui_failed = False

    def _apply_result(
        self,
        *,
        emit: np.ndarray,
        direction: np.ndarray,
        ray_origin: np.ndarray,
        result: BeamCastResult,
        length: float,
    ) -> None:
        self.last_origin = emit
        self.last_ray_origin = ray_origin
        self.last_direction = direction
        self.last_end = result.end
        self.last_hit = bool(result.hit)
        self.last_length = float(length)
        self.last_collision_path = result.collision_path
        self.last_rigid_body_path = result.rigid_body_path
        self.last_skipped_robot_hits = int(result.skipped_robot_hits)
        self.last_total_hits = int(result.total_hits)
        self.last_nearest_any_distance = float(result.nearest_any_distance)
        self.last_cast_source = str(result.source or "")
        self.last_hit_summaries = tuple(result.hit_summaries or ())

    def update(
        self,
        ee_pos,
        ee_quat_wxyz,
        depth_m: Optional[np.ndarray] = None,
    ) -> None:
        """Update beam end from depth (LOS) and/or PhysX.

        Prefer wrist-cam center depth when available — table/board meshes often
        lack PhysX collision, which made the overlay stuck on MAX 2.0 m.
        PhysX along tool +Z (and camera look) remains as fallback / cross-check.
        """
        pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(ee_quat_wxyz, dtype=np.float64).reshape(-1)
        direction = _approach_axis_world(quat)
        n = float(np.linalg.norm(direction))
        if n < 1e-12:
            return
        direction = direction / n

        if self.tip_offset_ee is not None:
            emit = pos + _quat_wxyz_to_rot_matrix(quat) @ self.tip_offset_ee
        else:
            emit = pos + direction * self.origin_offset_m

        ray_origin = emit + direction * self.raycast_origin_offset_m
        remaining = max(0.0, self.max_length - self.raycast_origin_offset_m)

        # --- Depth along wrist-cam optical axis (true line of sight) ---
        self.last_depth_m = -1.0
        depth_hit: Optional[BeamCastResult] = None
        depth_length = float(self.max_length)
        Rt = None
        if self.camera_prim_path:
            Rt = _camera_world_Rt(str(self.camera_prim_path))
        if depth_m is not None and Rt is not None:
            z = _depth_center_meters(depth_m)
            if z is not None and 1e-4 < z <= self.max_length + 0.5:
                self.last_depth_m = float(z)
                R_wc, t_cam = Rt
                look = _camera_look_world(R_wc)
                end = t_cam + look * float(z)
                # Report length along tool approach from TCP (AimBot muzzle).
                along = float(np.dot(end - emit, direction))
                if along < 0.0:
                    along = float(np.linalg.norm(end - emit))
                along = float(min(max(along, 0.0), self.max_length))
                depth_length = along
                depth_hit = BeamCastResult(
                    end=emit + direction * along,
                    length=along,
                    hit=True,
                    collision_path="wrist_depth",
                    rigid_body_path="",
                    source="wrist_depth",
                    hit_summaries=(f"{z:.4f}:depth_center",),
                    total_hits=1,
                    nearest_any_distance=float(z),
                )

        # --- PhysX: tool approach from TCP ---
        physx_tcp = raycast_beam_end(
            ray_origin,
            direction,
            remaining,
            self.exclude_prefixes,
            both_sides=self.both_sides,
        )
        if physx_tcp.hit:
            physx_tcp = BeamCastResult(
                end=physx_tcp.end,
                length=float(physx_tcp.length + self.raycast_origin_offset_m),
                hit=True,
                collision_path=physx_tcp.collision_path,
                rigid_body_path=physx_tcp.rigid_body_path,
                skipped_robot_hits=physx_tcp.skipped_robot_hits,
                total_hits=physx_tcp.total_hits,
                nearest_any_distance=physx_tcp.nearest_any_distance,
                source="physx_tcp",
                hit_summaries=physx_tcp.hit_summaries,
            )

        # --- PhysX: camera look (matches overlay LOS better) ---
        physx_cam: Optional[BeamCastResult] = None
        if Rt is not None:
            R_wc, t_cam = Rt
            look = _camera_look_world(R_wc)
            cam_cast = raycast_beam_end(
                t_cam,
                look,
                self.max_length,
                self.exclude_prefixes,
                both_sides=self.both_sides,
            )
            if cam_cast.hit:
                along = float(np.dot(cam_cast.end - emit, direction))
                if along < 0.0:
                    along = float(cam_cast.length)
                along = float(min(max(along, 0.0), self.max_length))
                physx_cam = BeamCastResult(
                    end=emit + direction * along,
                    length=along,
                    hit=True,
                    collision_path=cam_cast.collision_path,
                    rigid_body_path=cam_cast.rigid_body_path,
                    skipped_robot_hits=cam_cast.skipped_robot_hits,
                    total_hits=cam_cast.total_hits,
                    nearest_any_distance=cam_cast.nearest_any_distance,
                    source="physx_cam",
                    hit_summaries=cam_cast.hit_summaries,
                )

        # Choose result: prefer depth (LOS) when enabled, else nearest PhysX hit.
        candidates: list = []
        if depth_hit is not None and self.prefer_depth:
            candidates.append(depth_hit)
        for c in (physx_tcp, physx_cam):
            if c is not None and c.hit:
                candidates.append(c)
        if not candidates and depth_hit is not None:
            candidates.append(depth_hit)

        if candidates:
            best = min(candidates, key=lambda r: float(r.length))
            self._apply_result(
                emit=emit,
                direction=direction,
                ray_origin=ray_origin,
                result=best,
                length=float(best.length),
            )
            # Keep PhysX diagnostics even when depth wins.
            if physx_tcp.total_hits and not self.last_hit_summaries:
                self.last_hit_summaries = physx_tcp.hit_summaries
            if physx_tcp.total_hits:
                self.last_total_hits = max(
                    self.last_total_hits, int(physx_tcp.total_hits)
                )
                self.last_skipped_robot_hits = int(physx_tcp.skipped_robot_hits)
                self.last_nearest_any_distance = float(
                    physx_tcp.nearest_any_distance
                )
            return

        # Miss: full beam.
        miss = BeamCastResult(
            end=emit + direction * self.max_length,
            length=float(self.max_length),
            hit=False,
            skipped_robot_hits=int(physx_tcp.skipped_robot_hits),
            total_hits=int(physx_tcp.total_hits),
            nearest_any_distance=float(physx_tcp.nearest_any_distance),
            source=physx_tcp.source or "miss",
            hit_summaries=physx_tcp.hit_summaries,
        )
        self._apply_result(
            emit=emit,
            direction=direction,
            ray_origin=ray_origin,
            result=miss,
            length=float(self.max_length),
        )

    def maybe_log_distance(self, now_s: float, period_s: float = 0.5) -> None:
        if now_s - self._last_log_s < float(period_s):
            return
        self._last_log_s = float(now_s)
        tag = "hit" if self.last_hit else "max"
        frac = float(self.last_length) / max(float(self.max_length), 1e-9)
        end = self.last_end
        emit = self.last_origin
        ray_o = self.last_ray_origin
        end_err = -1.0
        if end is not None and emit is not None:
            end_err = float(np.linalg.norm(end - emit) - self.last_length)
        msg = (
            f"[r_wrist_laser] dist={self.last_length:.4f} m ({tag}) "
            f"frac={frac:.3f} src={self.last_cast_source or '?'} "
            f"depth={self.last_depth_m:.4f} "
            f"end_vs_len={end_err:+.4f}"
        )
        if self.last_hit:
            path = self.last_collision_path or self.last_rigid_body_path or "?"
            msg += f" coll={path}"
        msg += (
            f" nearest_any={self.last_nearest_any_distance:.4f}"
            f" skipped_robot={self.last_skipped_robot_hits}"
            f" physx_hits={self.last_total_hits}"
        )
        if self.last_hit_summaries:
            msg += " top=[" + ", ".join(self.last_hit_summaries) + "]"
        if self.last_stop_uv is not None:
            u, v, z = self.last_stop_uv
            msg += f" stop_uv=({u:.1f},{v:.1f}) Zcv={z:.3f}"
        if self.debug_log and emit is not None and end is not None and ray_o is not None:
            msg += (
                f" emit=({emit[0]:.3f},{emit[1]:.3f},{emit[2]:.3f})"
                f" ray0=({ray_o[0]:.3f},{ray_o[1]:.3f},{ray_o[2]:.3f})"
                f" end=({end[0]:.3f},{end[1]:.3f},{end[2]:.3f})"
            )
        print(msg, flush=True)

    def overlay_rgb(self, rgb, camera) -> Optional[np.ndarray]:
        """Draw AimBot-style beam: TCP→stop shooting line + stop reticle.

        The R wrist cam sits ~10 cm along approach ahead of the TCP, so the
        true emission point is often behind the camera; we clip the 3D
        segment to the near plane so the visible beam still starts where the
        ray enters the image (muzzle), then runs to the stop / hit.
        """
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

        try:
            import cv2
        except ImportError:
            return out

        origin_cv = _world_to_cv_camera(self.last_origin, R_wc, t_w)
        end_cv = _world_to_cv_camera(self.last_end, R_wc, t_w)
        clipped = _clip_segment_to_near(origin_cv, end_cv, z_near=1e-3)
        if clipped is None:
            return out
        start_cv, stop_cv = clipped

        # Dense sample along the visible (near-clipped) beam so a long ray
        # still draws when endpoints leave the image.
        n_pts = 32
        pts_uv: list = []
        for i in range(n_pts + 1):
            alpha = i / float(n_pts)
            p_cv = start_cv * (1.0 - alpha) + stop_cv * alpha
            proj = _project_cv(p_cv, K)
            if proj is not None:
                pts_uv.append((proj[0], proj[1], proj[2]))

        line_rgb = (0, 220, 0)  # AimBot-ish green shooting line (RGB)
        emit_rgb = (255, 48, 48)  # red muzzle / TCP marker
        hit_rgb = (255, 255, 0) if self.last_hit else (80, 180, 255)

        stop_proj = _project_cv(stop_cv, K)
        emit_proj = _project_cv(origin_cv, K)
        if emit_proj is None:
            emit_proj = _project_cv(start_cv, K)

        span = 0.0
        if len(pts_uv) >= 2:
            for a, b in zip(pts_uv[:-1], pts_uv[1:]):
                clipped_uv = _clip_segment_to_image(
                    (a[0], a[1]), (b[0], b[1]), w, h
                )
                if clipped_uv is None:
                    continue
                cv2.line(
                    out,
                    clipped_uv[0],
                    clipped_uv[1],
                    line_rgb,
                    max(2, self.line_thickness),
                )
            span = float(
                np.hypot(
                    pts_uv[0][0] - pts_uv[-1][0],
                    pts_uv[0][1] - pts_uv[-1][1],
                )
            )

        # Wrist cam is coaxial with approach, so the projected 3D beam often
        # collapses to ~1 px. Draw an AimBot-style HUD shooting line from the
        # bottom of the frame (gripper side) to the *true* projected stop —
        # that point lies on the tool approach axis in the image. Do NOT lerp
        # the aim by dist/max (that pulls the beam off "forward out of TCP").
        # Distance is cued by reticle size + hit color instead.
        if span < 6.0 and stop_proj is not None:
            muzzle = (int(w * 0.5), int(h * 0.98))
            stop_px = (int(round(stop_proj[0])), int(round(stop_proj[1])))
            clipped_uv = _clip_segment_to_image(muzzle, stop_px, w, h)
            if clipped_uv is not None:
                cv2.line(
                    out,
                    clipped_uv[0],
                    clipped_uv[1],
                    line_rgb,
                    max(2, self.line_thickness),
                )
                emit_proj = (
                    float(clipped_uv[0][0]),
                    float(clipped_uv[0][1]),
                    0.0,
                )

        # Emission / muzzle marker.
        if emit_proj is not None:
            eu, ev = int(round(emit_proj[0])), int(round(emit_proj[1]))
            if 0 <= eu < w and 0 <= ev < h:
                cv2.circle(
                    out,
                    (eu, ev),
                    max(4, self.line_thickness + 2),
                    emit_rgb,
                    -1,
                )

        # Stop point + AimBot wrist reticle at the tool-forward aim point.
        # Coaxial optics keep (u,v) nearly fixed with range — distance is
        # shown via reticle size, HUD text, and the range bar (not by moving aim).
        if stop_proj is not None:
            su, sv = int(round(stop_proj[0])), int(round(stop_proj[1]))
            self.last_stop_uv = (
                float(stop_proj[0]),
                float(stop_proj[1]),
                float(stop_proj[2]),
            )
            if 0 <= su < w and 0 <= sv < h:
                z_cue = float(self.last_length) if self.last_length > 1e-6 else 1.0
                arm = int(
                    np.clip(
                        18.0 + 40.0 * (1.0 - min(z_cue, 1.2) / 1.2),
                        14,
                        56,
                    )
                )
                _draw_crosshair(
                    out,
                    (su, sv),
                    arm_len=arm,
                    gap=max(3, self.line_thickness + 1),
                    color=line_rgb,
                    thickness=max(1, self.line_thickness),
                )
                cv2.circle(
                    out,
                    (su, sv),
                    max(3, self.line_thickness + 1),
                    hit_rgb,
                    -1,
                )
        else:
            self.last_stop_uv = None

        # Phase-3 distance cues (tool-forward aim unchanged).
        tag = "HIT" if self.last_hit else "MAX"
        label = f"{tag} {self.last_length:.3f} m"
        cv2.putText(
            out,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            hit_rgb,
            2,
            cv2.LINE_AA,
        )
        # Range bar: filled fraction = dist/max (short fill = close).
        bar_x0, bar_y0 = 12, h - 22
        bar_w, bar_h = max(40, w - 24), 10
        frac = float(
            np.clip(self.last_length / max(self.max_length, 1e-9), 0.0, 1.0)
        )
        cv2.rectangle(
            out,
            (bar_x0, bar_y0),
            (bar_x0 + bar_w, bar_y0 + bar_h),
            (40, 40, 40),
            -1,
        )
        fill_w = int(round(bar_w * frac))
        if fill_w > 0:
            cv2.rectangle(
                out,
                (bar_x0, bar_y0),
                (bar_x0 + fill_w, bar_y0 + bar_h),
                hit_rgb,
                -1,
            )
        cv2.rectangle(
            out,
            (bar_x0, bar_y0),
            (bar_x0 + bar_w, bar_y0 + bar_h),
            (200, 200, 200),
            1,
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
