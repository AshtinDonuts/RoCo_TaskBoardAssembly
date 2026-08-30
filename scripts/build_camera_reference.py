"""Build a submission-legal camera reference bundle from a nominal NPZ.

Usage (after capturing the unrandomized head snapshot):

  python3 scripts/build_camera_reference.py \\
      --observation artifacts/randomization-final-frames/reference/nominal-observation.npz \\
      --output task/policies/camera_reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from policies.camera_offset.constants import (  # noqa: E402
    BUNDLE_VERSION,
    DEFAULT_BUFFER_FRAMES,
    DEFAULT_TEMPLATE_HALF_PX,
    SUPPORT_COUPLED_PARTS,
    XY_MAX_M,
)
from policies.camera_offset.geometry import (  # noqa: E402
    jacobian_at_uv,
    pinhole_project,
    rot_from_wxyz,
)
from policies.camera_offset.reference import (  # noqa: E402
    PartTemplate,
    make_bundle,
)
import param_config as pc  # noqa: E402


# Columns express the OpenCV optical axes (x right, y down, z forward)
# in each recorded frame convention.
_AXES_FROM_OPTICAL = {
    # isaacsim.sensors.camera.Camera.get_world_pose() default: +X forward,
    # +Z up, +Y left ("world" camera axes). Verified against the nominal
    # capture: depth back-projects to a flat z≈1.02 table plane and all
    # parts land in-frame (see scripts/diagnose_camera_pose.py).
    "world": np.array([[0.0, 0.0, 1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]]),
    # USD camera prim pose: -Z forward, +Y up.
    "usd": np.diag([1.0, -1.0, -1.0]),
    # Already OpenCV optical.
    "optical": np.eye(3),
}


def _optical_rotation(R_world_from_frame: np.ndarray, camera_axes: str) -> np.ndarray:
    try:
        T = _AXES_FROM_OPTICAL[camera_axes]
    except KeyError:
        raise ValueError(
            f"camera_axes must be one of {sorted(_AXES_FROM_OPTICAL)}, "
            f"got {camera_axes!r}"
        ) from None
    return np.asarray(R_world_from_frame, dtype=np.float64) @ T


def _load_snapshot(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    needed = ("head_rgb", "head_depth", "head_intrinsics")
    missing = [k for k in needed if k not in data.files]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    return {k: data[k] for k in data.files}


def _crop_window(shape_hw, centre_uv, half: int):
    h, w = shape_hw
    cu, cv = float(centre_uv[0]), float(centre_uv[1])
    x0 = int(round(cu - half))
    y0 = int(round(cv - half))
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        x0 = max(0, min(x0, w - 2 * half))
        y0 = max(0, min(y0, h - 2 * half))
    sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
    # Centre of the actual crop (may differ from the requested uv when
    # clamped at an image border) so template and search origin agree.
    centre = np.array([x0 + half, y0 + half], dtype=np.float64)
    return sl, centre


def _flood_component(elevated: np.ndarray, seed_rc) -> np.ndarray:
    """4-connected component of ``elevated`` containing ``seed_rc``."""
    from collections import deque

    h, w = elevated.shape
    out = np.zeros_like(elevated, dtype=bool)
    queue = deque([seed_rc])
    out[seed_rc] = True
    while queue:
        r, c = queue.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and elevated[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                queue.append((rr, cc))
    return out


def _part_template_mask(wz_crop, wx_crop, wy_crop, z_plate, own_xy,
                        other_xys, radius_m: float = 0.025,
                        min_pixels: int = 20) -> np.ndarray:
    """Pixels of the part itself.

    Elevated above the plate, connected to the blob nearest the crop
    centre, within ``radius_m`` of the part's world XY, and closer to
    this part than to any other part (Voronoi) so touching neighbours
    (meshed gears, adjacent batteries, the raised tray) don't leak in.
    Falls back to all-ones when depth gives too little signal.
    """
    elevated = np.isfinite(wz_crop) & (wz_crop > z_plate + 0.004)
    d_own2 = (wx_crop - own_xy[0]) ** 2 + (wy_crop - own_xy[1]) ** 2
    elevated &= d_own2 <= radius_m ** 2
    for oxy in other_xys:
        d_other2 = (wx_crop - oxy[0]) ** 2 + (wy_crop - oxy[1]) ** 2
        elevated &= d_own2 < d_other2
    if int(elevated.sum()) < min_pixels:
        return np.ones_like(wz_crop, dtype=bool)
    h, w = wz_crop.shape
    rows, cols = np.nonzero(elevated)
    d2 = (rows - (h - 1) / 2.0) ** 2 + (cols - (w - 1) / 2.0) ** 2
    seed = (int(rows[np.argmin(d2)]), int(cols[np.argmin(d2)]))
    component = _flood_component(elevated, seed)
    if int(component.sum()) < min_pixels:
        return np.ones_like(wz_crop, dtype=bool)
    return component


def build_reference(observation_path, output_dir, *, camera_axes="world",
                    template_half=DEFAULT_TEMPLATE_HALF_PX,
                    uv_overrides=None):
    snap = _load_snapshot(Path(observation_path))
    rgb = np.asarray(snap["head_rgb"])
    depth = np.asarray(snap["head_depth"], dtype=np.float64)
    K = np.asarray(snap["head_intrinsics"], dtype=np.float64)
    h, w = rgb.shape[:2]

    R = t = None
    if "head_camera_pos" in snap and "head_camera_orn" in snap:
        t = np.asarray(snap["head_camera_pos"], dtype=np.float64).reshape(3)
        R_frame = rot_from_wxyz(snap["head_camera_orn"])
        R = _optical_rotation(R_frame, camera_axes)

    pick_xy = {}
    pick_xyz = {}
    for name in pc.part_order:
        cfg = pc.get_part_config(name)
        pos = cfg.get("pick_pos")
        if pos is None:
            pose = pc.PART_INIT_POSES.get(name)
            pos = None if pose is None else pose.get("pos")
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float64).reshape(3)
        pick_xyz[name] = arr
        pick_xy[name] = arr[:2]

    uv_overrides = dict(uv_overrides or {})
    centres = {}
    if R is not None and t is not None and pick_xyz:
        names = list(pick_xyz)
        pts = np.vstack([pick_xyz[n] for n in names])
        uvs = pinhole_project(pts, K, R, t)
        for name, uv in zip(names, uvs):
            if np.all(np.isfinite(uv)):
                centres[name] = uv
    centres.update({k: np.asarray(v, dtype=np.float64) for k, v in uv_overrides.items()})
    if not centres:
        raise RuntimeError(
            "cannot project parts: snapshot has no head_camera_pos/orn and "
            "no --uv-json was given"
        )

    out_of_frame = {
        name: uv for name, uv in centres.items()
        if not (0 <= uv[0] < w and 0 <= uv[1] < h)
    }
    if out_of_frame:
        details = ", ".join(
            f"{n}=({uv[0]:.0f},{uv[1]:.0f})" for n, uv in sorted(out_of_frame.items())
        )
        raise RuntimeError(
            f"projected part centres fall outside the {w}x{h} frame: {details}. "
            "The camera pose convention is likely wrong — run "
            "scripts/diagnose_camera_pose.py and pass the matching --camera-axes."
        )
    for name, uv in sorted(centres.items()):
        print(f"[reference] {name:<16} centre uv=({uv[0]:7.1f}, {uv[1]:7.1f})")

    board_uvs = [centres[n] for n in centres]
    board_center = np.mean(np.vstack(board_uvs), axis=0)

    zs = [float(pick_xyz[n][2]) for n in pick_xyz if n in centres]
    plane_z = float(np.median(zs)) if zs else 1.04

    if R is not None and t is not None:
        jac = jacobian_at_uv(K, R, t, board_center, plane_z)
    else:
        raise RuntimeError("camera pose required to build the pixel-to-world Jacobian")

    # Back-project every pixel to world once; used by the board mask and
    # by the per-part template masks below.
    finite = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    yy_f, xx_f = np.mgrid[0:h, 0:w]
    rays = np.linalg.inv(K) @ np.stack(
        [xx_f.ravel(), yy_f.ravel(), np.ones(h * w)], axis=0
    )
    world = (R @ (rays * np.where(finite, depth, 1.0).ravel())).T + t
    wx = np.where(finite, world[:, 0].reshape(h, w), np.nan)
    wy = np.where(finite, world[:, 1].reshape(h, w), np.nan)
    wz = np.where(finite, world[:, 2].reshape(h, w), np.nan)

    # Board mask in world space: keep only pixels on the board plane
    # inside the board XY extent. A raw depth band would cut an oblique
    # stripe through the static background, which biases board
    # registration toward zero shift.
    board_mask = np.zeros((h, w), dtype=bool)
    part_pts = np.vstack([pick_xyz[n] for n in pick_xyz])
    margin = 0.06
    xy_box = (
        (wx >= part_pts[:, 0].min() - margin)
        & (wx <= part_pts[:, 0].max() + margin)
        & (wy >= part_pts[:, 1].min() - margin)
        & (wy <= part_pts[:, 1].max() + margin)
    )
    in_box = finite & xy_box
    z_plate = plane_z
    if np.any(in_box):
        # The plate's top face dominates the in-box area; centre the
        # z-band on its median rather than the grasp-height plane_z.
        z_plate = float(np.median(wz[in_box]))
        board_mask = in_box & (np.abs(wz - z_plate) < 0.012)
    yy, xx = np.mgrid[0:h, 0:w]
    try:
        inv = np.linalg.inv(jac)
    except np.linalg.LinAlgError:
        inv = np.array([[800.0, 0.0], [0.0, 800.0]])
    part_radius = float(np.linalg.norm(inv @ np.array([XY_MAX_M, 0.0]))) + template_half + 4
    for name, uv in centres.items():
        if name in SUPPORT_COUPLED_PARTS:
            continue
        disk = (xx - uv[0]) ** 2 + (yy - uv[1]) ** 2 <= part_radius ** 2
        board_mask &= ~disk

    parts = {}
    for name, uv in centres.items():
        sl, centre = _crop_window((h, w), uv, int(template_half))
        own_xy = pick_xyz[name][:2] if name in pick_xyz else None
        other_xys = [
            pick_xyz[o][:2] for o in pick_xyz if o != name
        ] if own_xy is not None else []
        if own_xy is not None:
            mask = _part_template_mask(
                wz[sl], wx[sl], wy[sl], z_plate, own_xy, other_xys
            )
        else:
            mask = np.ones((2 * int(template_half),) * 2, dtype=bool)
        part_z = float(pick_xyz[name][2]) if name in pick_xyz else plane_z
        part_jac = jacobian_at_uv(K, R, t, centre, part_z)
        parts[name] = PartTemplate(
            name=name,
            rgb=rgb[sl].copy(),
            depth=depth[sl].copy(),
            mask=mask,
            search_center_uv=centre,
            jacobian_xy_per_px=part_jac,
        )
        print(f"[reference] {name:<16} template mask px={int(mask.sum())}"
              f"/{mask.size}")

    bundle = make_bundle(
        rgb=rgb,
        depth=depth,
        intrinsics=K,
        board_mask=board_mask,
        jacobian_xy_per_px=jac,
        board_center_uv=board_center,
        parts=parts,
        buffer_frames=DEFAULT_BUFFER_FRAMES,
        camera_R_world_from_cam=R,
        camera_t_world=t,
        plane_z=plane_z,
        diagnostics={
            "source_observation": str(observation_path),
            "bundle_version": BUNDLE_VERSION,
        },
    )
    out = bundle.save(output_dir)
    print(f"[reference] wrote {out} hash={bundle.content_hash[:12]}")
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", required=True, help="nominal NPZ snapshot")
    parser.add_argument(
        "--output",
        default=str(TASK_DIR / "policies" / "camera_reference"),
        help="bundle directory (default: task/policies/camera_reference)",
    )
    parser.add_argument(
        "--uv-json",
        default=None,
        help="optional {part: [u, v]} overrides for template centres",
    )
    parser.add_argument(
        "--template-half",
        type=int,
        default=DEFAULT_TEMPLATE_HALF_PX,
    )
    parser.add_argument(
        "--camera-axes",
        choices=("world", "usd", "optical"),
        default="world",
        help="convention of the recorded head_camera_orn quaternion "
             "(default: world, the isaacsim Camera.get_world_pose() default)",
    )
    args = parser.parse_args(argv)
    overrides = None
    if args.uv_json:
        overrides = json.loads(Path(args.uv_json).read_text())
    build_reference(
        args.observation,
        args.output,
        camera_axes=args.camera_axes,
        template_half=args.template_half,
        uv_overrides=overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
