"""Right-wrist TCP approach laser for visual guidance.

Aims along gripper local +Z from the EE tool center (jaw midline), not along
the wrist-cam optical axis. Range prefers depth marched along that tool ray
in the wrist image, with PhysX along tool +Z as fallback. Overlay projects
TCP→stop into the wrist RGB and draws a distal-tip aperture ruler (mm) for
grasp calibration. Live preview uses an ``omni.ui`` window — OpenCV HighGUI
fights Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import carb
import numpy as np
import omni.usd
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, Usd, UsdGeom

from .ee_pose_controller import _approach_axis_world
from .pick_place_task import ROBOT_PRIM_PATH

_OVERLAY_WINDOW = "R Wrist Laser"
_UI_MIN_PERIOD_S = 0.05  # ~20 Hz preview; avoids flooding ByteImageProvider

# Distal inner-jaw tips (same frame/threshold as aperture_calibration).
_R_GRIPPER_LINK = f"{ROBOT_PRIM_PATH}/R_ee_link/gripper_link"
_R_ACTIVE_MESH = (
    f"{ROBOT_PRIM_PATH}/R_ee_link/gripper_active_link/gripper_active_link"
)
_R_PASSIVE_MESH = (
    f"{ROBOT_PRIM_PATH}/R_ee_link/gripper_passive_link/"
    "left_gripper_passive_link"
)
_DISTAL_Z_MIN_M = 0.18


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


def _sample_depth_m(depth: np.ndarray, u: float, v: float) -> float:
    """Bilinear-ish depth sample; returns +inf if invalid."""
    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0:
        return float("inf")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or y < 0 or x >= w or y >= h:
        return float("inf")
    z = float(arr[y, x])
    if not np.isfinite(z) or z <= 1e-4:
        return float("inf")
    return z


def _find_depth_cutoff_uv(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    depth: np.ndarray,
    hit_z: float,
    *,
    n: int = 96,
    match_tol_m: float = 0.05,
) -> Tuple[float, float]:
    """Walk p0→p1; stop where scene depth first meets the laser hit.

    Prefer a pixel whose depth matches ``hit_z`` (the collision surface), so
    closer clutter (e.g. gripper in the lower frame) does not truncate the
    beam early. Fallback: first pixel with depth <= hit_z.
    """
    hit_z = float(hit_z)
    if hit_z <= 1e-4 or not np.isfinite(hit_z):
        return float(p1[0]), float(p1[1])
    match_uv = None
    close_uv = None
    for i in range(int(n) + 1):
        t = i / float(n)
        u = float(p0[0]) * (1.0 - t) + float(p1[0]) * t
        v = float(p0[1]) * (1.0 - t) + float(p1[1]) * t
        z = _sample_depth_m(depth, u, v)
        if not np.isfinite(z):
            continue
        if match_uv is None and abs(z - hit_z) <= float(match_tol_m):
            match_uv = (u, v)
            break
        if close_uv is None and z <= hit_z + 0.02:
            close_uv = (u, v)
    if match_uv is not None:
        return match_uv
    if close_uv is not None:
        return close_uv
    return float(p1[0]), float(p1[1])


def _draw_polyline_uv(
    img: np.ndarray,
    pts: Sequence[Tuple[float, float]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    try:
        import cv2
    except ImportError:
        return
    h, w = img.shape[:2]
    for a, b in zip(pts[:-1], pts[1:]):
        clipped = _clip_segment_to_image((a[0], a[1]), (b[0], b[1]), w, h)
        if clipped is None:
            continue
        cv2.line(img, clipped[0], clipped[1], color, thickness)


def _mesh_points_local(stage: Usd.Stage, mesh_path: str) -> Optional[np.ndarray]:
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsValid():
        return None
    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    if not points:
        return None
    return np.asarray(
        [[float(p[0]), float(p[1]), float(p[2])] for p in points],
        dtype=np.float64,
    )


def _xform_points_to_frame(
    stage: Usd.Stage,
    mesh_path: str,
    frame_path: str,
    points_local: np.ndarray,
) -> Optional[np.ndarray]:
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    frame_prim = stage.GetPrimAtPath(frame_path)
    if not mesh_prim or not mesh_prim.IsValid():
        return None
    if not frame_prim or not frame_prim.IsValid():
        return None
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_to_world = np.asarray(
        cache.GetLocalToWorldTransform(mesh_prim), dtype=np.float64
    )
    frame_to_world = np.asarray(
        cache.GetLocalToWorldTransform(frame_prim), dtype=np.float64
    )
    world_to_frame = np.linalg.inv(frame_to_world)
    hom = np.concatenate(
        [points_local, np.ones((len(points_local), 1), dtype=np.float64)],
        axis=1,
    )
    return (hom @ mesh_to_world @ world_to_frame)[:, :3]


def _pick_distal_inner_tip_locals(
    *,
    distal_z_min_m: float = _DISTAL_Z_MIN_M,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Finger-mesh-local distal tip verts (active min-X, passive max-X)."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    active_local = _mesh_points_local(stage, _R_ACTIVE_MESH)
    passive_local = _mesh_points_local(stage, _R_PASSIVE_MESH)
    if active_local is None or passive_local is None:
        return None
    active_in_link = _xform_points_to_frame(
        stage, _R_ACTIVE_MESH, _R_GRIPPER_LINK, active_local
    )
    passive_in_link = _xform_points_to_frame(
        stage, _R_PASSIVE_MESH, _R_GRIPPER_LINK, passive_local
    )
    if active_in_link is None or passive_in_link is None:
        return None
    a_mask = active_in_link[:, 2] >= float(distal_z_min_m)
    p_mask = passive_in_link[:, 2] >= float(distal_z_min_m)
    if not np.any(a_mask) or not np.any(p_mask):
        return None
    a_idx = int(np.where(a_mask)[0][np.argmin(active_in_link[a_mask, 0])])
    p_idx = int(np.where(p_mask)[0][np.argmax(passive_in_link[p_mask, 0])])
    return active_local[a_idx].copy(), passive_local[p_idx].copy()


def _tip_world_from_local(
    mesh_path: str, point_local: np.ndarray
) -> Optional[np.ndarray]:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mat = cache.GetLocalToWorldTransform(prim)
    p = mat.Transform(
        Gf.Vec3d(
            float(point_local[0]), float(point_local[1]), float(point_local[2])
        )
    )
    return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)


def measure_r_gripper_distal_tips_world(
    tip_locals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    distal_z_min_m: float = _DISTAL_Z_MIN_M,
) -> Optional[
    Tuple[np.ndarray, np.ndarray, float, Tuple[np.ndarray, np.ndarray]]
]:
    """Return (tip_active_w, tip_passive_w, gap_m, tip_locals) or None."""
    locals_pair = tip_locals
    if locals_pair is None:
        locals_pair = _pick_distal_inner_tip_locals(distal_z_min_m=distal_z_min_m)
        if locals_pair is None:
            return None
    a_local, p_local = locals_pair
    a_w = _tip_world_from_local(_R_ACTIVE_MESH, a_local)
    p_w = _tip_world_from_local(_R_PASSIVE_MESH, p_local)
    if a_w is None or p_w is None:
        return None
    gap = float(np.linalg.norm(a_w - p_w))
    return a_w, p_w, gap, (a_local, p_local)


def _draw_aperture_ruler(
    img: np.ndarray,
    uv_a: Tuple[float, float],
    uv_b: Tuple[float, float],
    width_m: float,
    *,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    tick_px: int = 10,
    label: Optional[str] = None,
) -> None:
    """Draw a tip-to-tip ruler with end ticks and a mm label."""
    try:
        import cv2
    except ImportError:
        return
    h, w = img.shape[:2]
    clipped = _clip_segment_to_image(uv_a, uv_b, w, h)
    if clipped is None:
        return
    (x0, y0), (x1, y1) = clipped
    cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

    dx = float(x1 - x0)
    dy = float(y1 - y0)
    seg_len = float(np.hypot(dx, dy))
    if seg_len < 1e-3:
        return
    nx, ny = -dy / seg_len, dx / seg_len
    t = float(max(4, tick_px))
    for x, y in ((x0, y0), (x1, y1)):
        a = (int(round(x - nx * t)), int(round(y - ny * t)))
        b = (int(round(x + nx * t)), int(round(y + ny * t)))
        tick = _clip_segment_to_image(a, b, w, h)
        if tick is not None:
            cv2.line(img, tick[0], tick[1], color, thickness, cv2.LINE_AA)
        cv2.circle(
            img, (int(round(x)), int(round(y))), max(3, thickness + 1), color, -1
        )

    text = label if label is not None else f"{width_m * 1000.0:.1f} mm"
    mx = int(round(0.5 * (x0 + x1)))
    my = int(round(0.5 * (y0 + y1)))
    lx = int(np.clip(mx + nx * 16.0, 4, w - 8))
    ly = int(np.clip(my + ny * 16.0, 16, h - 8))
    cv2.putText(
        img,
        text,
        (lx, ly),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


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


def _depth_along_tool_ray(
    emit: np.ndarray,
    direction: np.ndarray,
    max_length: float,
    depth: np.ndarray,
    R_wc: np.ndarray,
    t_w: np.ndarray,
    K: np.ndarray,
    *,
    n: int = 80,
    match_tol_m: float = 0.03,
    min_t_m: float = 0.02,
) -> Optional[Tuple[float, np.ndarray, float]]:
    """March along tool +Z; return (along_m, end_world, z_cam) at first hit.

    Projects ``emit + t * direction`` into the wrist image and compares that
    sample's camera-frame depth to the scene depth at the same pixel. The
    first ``t`` where scene depth is at/closer than the tool-ray point is the
    collision along tool-forward (not along the camera boresight).
    """
    emit = np.asarray(emit, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64)
    t0 = max(float(min_t_m), 1e-3)
    t1 = max(float(max_length), t0)
    for i in range(1, int(n) + 1):
        t = t0 + (t1 - t0) * (i / float(n))
        pw = emit + direction * t
        p_cv = _world_to_cv_camera(pw, R_wc, t_w)
        proj = _project_cv(p_cv, K)
        if proj is None:
            continue
        u, v, z_cam = proj
        z_scene = _sample_depth_m(depth, u, v)
        if not np.isfinite(z_scene):
            continue
        if float(z_scene) <= float(z_cam) + float(match_tol_m):
            return float(t), pw, float(z_cam)
    return None


def _parse_vec3_env(raw: str) -> Optional[np.ndarray]:
    parts = [p.strip() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        return None
    try:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    except ValueError:
        return None


class RWristLaser:
    """Per-frame right-wrist approach laser (raycast + RGB overlay)."""

    def __init__(
        self,
        *,
        max_length: float = 2.0,
        origin_offset_m: float = 0.0,
        raycast_origin_offset_m: float = 0.0,
        tip_offset_ee: Optional[Sequence[float]] = None,
        camera_prim_path: Optional[str] = None,
        exclude_prefixes: Optional[Sequence[str]] = None,
        both_sides: bool = True,
        prefer_depth: bool = True,
        show_aim_debug: bool = False,
        line_thickness: int = 2,
        show_window: bool = True,
        debug_log: bool = False,
    ):
        self.max_length = float(max_length)
        self.origin_offset_m = float(origin_offset_m)
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
        self.show_aim_debug = bool(show_aim_debug)
        self.line_thickness = int(line_thickness)
        self.show_window = bool(show_window)
        self.debug_log = bool(debug_log)

        self.last_origin: Optional[np.ndarray] = None  # TCP emit (tool midline)
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
        self.last_aperture_m: float = -1.0
        self.last_aperture_calib_m: float = -1.0
        self._tip_locals: Optional[Tuple[np.ndarray, np.ndarray]] = None
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
        camera=None,
    ) -> None:
        """Update beam end along tool +Z from the TCP (jaw midline).

        Aim direction is always gripper local +Z from the EE pose. Range uses
        depth marched along that tool ray (projected into the wrist image),
        then PhysX along tool +Z. Camera-boresight depth is only a last-resort
        length estimate; the stop point stays on the tool ray.
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

        self.last_depth_m = -1.0
        tool_depth_hit: Optional[BeamCastResult] = None
        Rt = None
        K = None
        if self.camera_prim_path:
            Rt = _camera_world_Rt(str(self.camera_prim_path))
        if camera is not None:
            try:
                K = camera.get_intrinsics_matrix()
            except Exception:
                K = None
            if K is not None:
                K = np.asarray(K, dtype=np.float64)

        # --- Depth marched along tool +Z (primary when prefer_depth) ---
        if (
            self.prefer_depth
            and depth_m is not None
            and Rt is not None
            and K is not None
        ):
            marched = _depth_along_tool_ray(
                emit,
                direction,
                self.max_length,
                depth_m,
                Rt[0],
                Rt[1],
                K,
            )
            if marched is not None:
                along, end_w, z_cam = marched
                self.last_depth_m = float(z_cam)
                tool_depth_hit = BeamCastResult(
                    end=np.asarray(end_w, dtype=np.float64).reshape(3),
                    length=float(along),
                    hit=True,
                    collision_path="tool_depth",
                    rigid_body_path="",
                    source="tool_depth",
                    hit_summaries=(f"{along:.4f}:tool_ray",),
                    total_hits=1,
                    nearest_any_distance=float(along),
                )

        # --- PhysX along tool +Z ---
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

        # --- Last resort: cam-center depth → length only, end on tool ray ---
        cam_depth_hit: Optional[BeamCastResult] = None
        if tool_depth_hit is None and depth_m is not None and Rt is not None:
            z = _depth_center_meters(depth_m)
            if z is not None and 1e-4 < z <= self.max_length + 0.5:
                self.last_depth_m = float(z)
                # Approximate standoff: cam is usually ahead of TCP on approach.
                R_wc, t_cam = Rt
                cam_ahead = float(np.dot(t_cam - emit, direction))
                along = float(min(max(z + max(cam_ahead, 0.0), 0.0), self.max_length))
                cam_depth_hit = BeamCastResult(
                    end=emit + direction * along,
                    length=along,
                    hit=True,
                    collision_path="wrist_depth_fallback",
                    rigid_body_path="",
                    source="wrist_depth_fallback",
                    hit_summaries=(f"{z:.4f}:depth_center_fb",),
                    total_hits=1,
                    nearest_any_distance=float(z),
                )

        candidates: list = []
        if tool_depth_hit is not None:
            candidates.append(tool_depth_hit)
        if physx_tcp.hit:
            candidates.append(physx_tcp)
        if not candidates and cam_depth_hit is not None:
            candidates.append(cam_depth_hit)

        if candidates:
            # Prefer tool_depth over physx when both exist and prefer_depth.
            if self.prefer_depth and tool_depth_hit is not None:
                best = tool_depth_hit
            else:
                best = min(candidates, key=lambda r: float(r.length))
            self._apply_result(
                emit=emit,
                direction=direction,
                ray_origin=ray_origin,
                result=best,
                length=float(best.length),
            )
            if physx_tcp.total_hits:
                self.last_total_hits = max(
                    self.last_total_hits, int(physx_tcp.total_hits)
                )
                self.last_skipped_robot_hits = int(physx_tcp.skipped_robot_hits)
                if self.last_nearest_any_distance < 0.0:
                    self.last_nearest_any_distance = float(
                        physx_tcp.nearest_any_distance
                    )
            return

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

    def overlay_rgb(
        self,
        rgb,
        camera,
        depth_m: Optional[np.ndarray] = None,
        gripper_q_rad: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Draw tool-forward beam + distal-tip aperture ruler.

        Aim is the gripper +Z ray from the TCP (jaw midline), projected into
        the wrist image. The stop-point crosshair is omitted; a tip-to-tip
        ruler shows the live distal inner-jaw gap (mm) for grasp calibration.
        Optional ``gripper_q_rad`` adds the Design D model gap (q→gap).
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

        # Tool-ray segment in camera frame (TCP → stop on approach axis).
        origin_cv = _world_to_cv_camera(self.last_origin, R_wc, t_w)
        end_cv = _world_to_cv_camera(self.last_end, R_wc, t_w)
        clipped = _clip_segment_to_near(origin_cv, end_cv, z_near=1e-3)
        if clipped is None:
            return out
        start_cv, stop_cv = clipped

        depth_arr = None
        if depth_m is not None:
            depth_arr = np.asarray(depth_m, dtype=np.float64)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[..., 0]

        n_pts = 48
        pts_uv: list = []
        for i in range(n_pts + 1):
            alpha = i / float(n_pts)
            p_cv = start_cv * (1.0 - alpha) + stop_cv * alpha
            proj = _project_cv(p_cv, K)
            if proj is None:
                continue
            u, v, z_laser = proj
            if depth_arr is not None:
                z_scene = _sample_depth_m(depth_arr, u, v)
                if np.isfinite(z_scene) and z_scene < float(z_laser) - 0.01:
                    break
            pts_uv.append((u, v, z_laser))

        line_rgb = (0, 220, 0)
        emit_rgb = (255, 48, 48)
        hit_rgb = (255, 255, 0) if self.last_hit else (80, 180, 255)
        ruler_rgb = (0, 255, 255)
        dbg_cam_rgb = (80, 80, 255)

        stop_proj = _project_cv(stop_cv, K)
        emit_proj = _project_cv(origin_cv, K)
        if emit_proj is None:
            emit_proj = _project_cv(start_cv, K)

        # Aim point = projected tool-ray stop (no crosshair).
        aim_uv: Optional[Tuple[float, float]] = None
        if stop_proj is not None:
            aim_uv = (float(stop_proj[0]), float(stop_proj[1]))
            self.last_stop_uv = (
                float(stop_proj[0]),
                float(stop_proj[1]),
                float(stop_proj[2]),
            )
        else:
            self.last_stop_uv = None
        if pts_uv:
            # If occlusion trimmed samples, use last kept point on tool ray.
            aim_uv = (float(pts_uv[-1][0]), float(pts_uv[-1][1]))

        if len(pts_uv) >= 2:
            _draw_polyline_uv(
                out,
                [(p[0], p[1]) for p in pts_uv],
                line_rgb,
                max(2, self.line_thickness),
            )
        elif emit_proj is not None and aim_uv is not None:
            # Near-coaxial: still connect projected muzzle → tool aim.
            clipped_uv = _clip_segment_to_image(
                (emit_proj[0], emit_proj[1]),
                (aim_uv[0], aim_uv[1]),
                w,
                h,
            )
            if clipped_uv is not None:
                cv2.line(
                    out,
                    clipped_uv[0],
                    clipped_uv[1],
                    line_rgb,
                    max(2, self.line_thickness),
                )

        # TCP / muzzle marker on the tool ray.
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

        if aim_uv is not None:
            su, sv = int(round(aim_uv[0])), int(round(aim_uv[1]))
            if 0 <= su < w and 0 <= sv < h:
                cv2.circle(
                    out,
                    (su, sv),
                    max(3, self.line_thickness + 1),
                    hit_rgb,
                    -1,
                )

        # Distal-tip aperture ruler (live mesh gap between inner jaw tips).
        self.last_aperture_m = -1.0
        tips = measure_r_gripper_distal_tips_world(self._tip_locals)
        if tips is not None:
            tip_a, tip_b, gap_m, self._tip_locals = tips
            self.last_aperture_m = float(gap_m)
            a_cv = _world_to_cv_camera(tip_a, R_wc, t_w)
            b_cv = _world_to_cv_camera(tip_b, R_wc, t_w)
            a_proj = _project_cv(a_cv, K)
            b_proj = _project_cv(b_cv, K)
            if a_proj is not None and b_proj is not None:
                _draw_aperture_ruler(
                    out,
                    (float(a_proj[0]), float(a_proj[1])),
                    (float(b_proj[0]), float(b_proj[1])),
                    gap_m,
                    color=ruler_rgb,
                    thickness=max(2, self.line_thickness),
                    label=f"{gap_m * 1000.0:.1f} mm",
                )

        self.last_aperture_calib_m = -1.0
        if gripper_q_rad is not None:
            try:
                from teleop.grasp_aperture import close_rad_to_aperture_m

                self.last_aperture_calib_m = float(
                    close_rad_to_aperture_m(float(gripper_q_rad))
                )
            except Exception:
                self.last_aperture_calib_m = -1.0

        # Optional: show camera principal point so cam vs tool offset is visible.
        if self.show_aim_debug:
            cx, cy = float(K[0, 2]), float(K[1, 2])
            cv2.drawMarker(
                out,
                (int(round(cx)), int(round(cy))),
                dbg_cam_rgb,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=18,
                thickness=2,
            )

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
        y_txt = 54
        if self.last_aperture_m >= 0.0:
            cv2.putText(
                out,
                f"aperture {self.last_aperture_m * 1000.0:.1f} mm",
                (12, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                ruler_rgb,
                2,
                cv2.LINE_AA,
            )
            y_txt += 24
        if self.last_aperture_calib_m >= 0.0:
            cv2.putText(
                out,
                f"q→gap {self.last_aperture_calib_m * 1000.0:.1f} mm",
                (12, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 80),
                2,
                cv2.LINE_AA,
            )
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
