"""Diagnose which camera-axes convention the captured head pose uses.

Back-projects the nominal depth image to world coordinates under several
candidate rotation conventions and reports which one lands the tabletop on
a flat z-plane with plausible world XY. Also projects known part pick
positions and checks they fall inside the image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from policies.camera_offset.geometry import pinhole_project, rot_from_wxyz  # noqa: E402
import param_config as pc  # noqa: E402

npz_path = sys.argv[1] if len(sys.argv) > 1 else str(
    REPO_ROOT / "artifacts/randomization-final-frames/reference/nominal-observation.npz"
)
snap = np.load(npz_path)
print("NPZ keys:", sorted(snap.files))
rgb = snap["head_rgb"]
depth = np.asarray(snap["head_depth"], dtype=np.float64)
K = np.asarray(snap["head_intrinsics"], dtype=np.float64)
t = np.asarray(snap["head_camera_pos"], dtype=np.float64).reshape(3)
quat = np.asarray(snap["head_camera_orn"], dtype=np.float64).reshape(4)
h, w = depth.shape
print(f"image {w}x{h}  K=\n{K}")
print(f"camera_pos={t}  camera_orn(wxyz)={quat}")

R_prim = rot_from_wxyz(quat)

# Candidate mappings from "optical" axes (x right, y down, z forward)
# into the frame the recorded quaternion describes.
FLIP_X180 = np.diag([1.0, -1.0, -1.0])          # USD cam (-Z fwd, +Y up)
# Isaac 'world' camera axes: +X forward, +Z up, +Y left.
T_WORLD = np.array([[0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0]])
# ROS camera axes: +Z forward, +X right, +Y down == optical (identity).
CANDIDATES = {
    "raw (optical==prim)": R_prim,
    "usd_flip (prim @ diag(1,-1,-1))": R_prim @ FLIP_X180,
    "isaac_world (prim @ T_WORLD)": R_prim @ T_WORLD,
    "isaac_world_T (prim @ T_WORLD.T)": R_prim @ T_WORLD.T,
}

Kinv = np.linalg.inv(K)
us = np.linspace(0.15 * w, 0.85 * w, 12)
vs = np.linspace(0.15 * h, 0.85 * h, 12)
uu, vv = np.meshgrid(us, vs)
uv1 = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=0)
dd = np.array([depth[int(v), int(u)] for u, v in zip(uu.ravel(), vv.ravel())])
ok = np.isfinite(dd) & (dd > 0.05) & (dd < 5.0)

pick = {}
for name in pc.part_order:
    cfg = pc.get_part_config(name)
    if cfg.get("pick_pos") is not None:
        pick[name] = np.asarray(cfg["pick_pos"], dtype=np.float64)

for label, R in CANDIDATES.items():
    rays = Kinv @ uv1
    cam_pts = rays * dd  # z-depth scaling: distance_to_image_plane
    world = (R @ cam_pts).T + t
    world = world[ok]
    z_med = float(np.median(world[:, 2]))
    z_spread = float(np.percentile(world[:, 2], 90) - np.percentile(world[:, 2], 10))
    xy_med = np.median(world[:, :2], axis=0)
    uvp = pinhole_project(np.vstack(list(pick.values())), K, R, t)
    inb = np.sum(
        (uvp[:, 0] >= 0) & (uvp[:, 0] < w) & (uvp[:, 1] >= 0) & (uvp[:, 1] < h)
    )
    print(f"\n[{label}]")
    print(f"  backproj z median={z_med:+.4f} spread(p90-p10)={z_spread:.4f}")
    print(f"  backproj xy median=({xy_med[0]:+.3f}, {xy_med[1]:+.3f})")
    print(f"  parts projected in-bounds: {inb}/{len(pick)}")
    for name, uv in zip(pick, uvp):
        print(f"    {name:<16} uv=({uv[0]:8.1f}, {uv[1]:8.1f})")
