"""Compare R-arm IK reachability for several fixed world EE orientations.

Diagnoses why ``retarget.fix_orientation`` (hold clutch-engage / home quat)
can feel workspace-restricted vs ``fixed_orientation_wxyz: [0,1,0,0]``
(top-down) or free orientation.

For each named orientation, probes the same XYZ grid above the board AABBs
with Lula IK (R arm, arm-only descriptor matching OWNS_LIFT_R/OWNS_TORSO_R).
Prints a reachability table and writes per-mode PLYs.

Must run with the Isaac venv python (SimulationApp):

    OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_HEADLESS=1 \\
      .venv/bin/python task/find_reachable_r_arm_orn_compare.py

    # denser / board-focused:
    ... --xy-res 24 --z-steps 8 --xy-margin 0.05
"""
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pxr import Usd, UsdGeom
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import param_config as pc  # noqa: E402


ROBOT_PRIM_PATH = "/World/robotics/vega_1u_gripper"
R_EE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/R_ee_link/gripper_link"

IK_POS_TOL = 1e-3
IK_ORN_TOL = 5e-2

# R arm URDF↔USD offset is identity (see lula_ik_controller._STAGE_OFFSET_INV_R).
_STAGE_OFFSET_INV_R = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _orn_for_lula_r(orn):
    if orn is None:
        return None
    return _quat_mul(np.asarray(orn, dtype=np.float64), _STAGE_OFFSET_INV_R)


def _approach_axis_world(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gripper local +Z in world (top-down ≈ (0,0,-1))."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    # Rotate (0,0,1) by quat.
    # R * v for v=(0,0,1):
    return np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dtype=np.float64)


def get_world_pose(prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
    prim = get_prim_at_path(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    rot = mat.ExtractRotationQuat()
    imag = rot.GetImaginary()
    pos = np.array([t[0], t[1], t[2]], dtype=np.float64)
    quat_wxyz = np.array(
        [rot.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64
    )
    return pos, quat_wxyz


def _world_aabb(prim: Usd.Prim) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        return None
    mn = bbox.GetMin()
    mx = bbox.GetMax()
    return (
        np.array([mn[0], mn[1], mn[2]], dtype=np.float64),
        np.array([mx[0], mx[1], mx[2]], dtype=np.float64),
    )


def _autodetect_board_paths(stage: Usd.Stage) -> List[str]:
    out = []
    for p in stage.Traverse():
        path = p.GetPath().pathString
        name = p.GetName().lower()
        if path.startswith("/World/") and path.count("/") == 2 and "board" in name:
            out.append(path)
    return out


def write_ply(path: str, points, rgb=None) -> None:
    pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in points]
    n = len(pts)
    has_color = rgb is not None
    if has_color and isinstance(rgb, tuple) and len(rgb) == 3:
        rgb_list = [rgb] * n
    else:
        rgb_list = list(rgb) if has_color else None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write(
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            )
        f.write("end_header\n")
        for i, p in enumerate(pts):
            if has_color:
                r, g, b = rgb_list[i]
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(r)} {int(g)} {int(b)}\n"
                )
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"  wrote {n} pts -> {path}")


def _r_descriptor_path() -> str:
    owns_lift = bool(getattr(pc, "OWNS_LIFT_R", False))
    owns_torso = bool(getattr(pc, "OWNS_TORSO_R", False))
    if owns_torso:
        suffix = ""
    elif owns_lift:
        suffix = "_liftonly"
    else:
        suffix = "_armonly"
    return os.path.join(
        _HERE, "controllers", f"vega_1u_R_arm_description{suffix}.yaml"
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_scene = os.path.abspath(os.path.join(_HERE, pc.SCENE_USD))
    ap.add_argument("--scene", default=default_scene)
    ap.add_argument("--boards", nargs="+", default=None, metavar="PATH")
    ap.add_argument("--xy-res", type=int, default=16)
    ap.add_argument("--xy-margin", type=float, default=0.05)
    ap.add_argument("--z-min-above", type=float, default=0.02)
    ap.add_argument("--z-max-above", type=float, default=0.30)
    ap.add_argument("--z-steps", type=int, default=6)
    ap.add_argument(
        "--ee-offset",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Fingertip→EE offset (default PART_DEFAULTS ee_offset).",
    )
    ap.add_argument(
        "--target-is-ee-pos",
        action="store_true",
        help="Grid points are EE targets (no ee_offset).",
    )
    ap.add_argument(
        "--output-prefix",
        default=None,
        help="PLY prefix. Default: reachable_r_arm_orn next to this script.",
    )
    ap.add_argument(
        "--extra-orn",
        nargs=5,
        action="append",
        default=[],
        metavar=("NAME", "W", "X", "Y", "Z"),
        help="Extra named orientation to probe (repeatable).",
    )
    ap.add_argument(
        "--also-free",
        action="store_true",
        help="Also probe each XYZ with orientation=None (position-only IK) "
             "and with a small set of tilts around top-down, to quantify "
             "how much fix_orientation shrinks the reachable set vs free 6DoF.",
    )
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[info] ignoring unknown args (likely Isaac Sim runtime): {unknown}")
    return args


def _probe_grid(
    ik: LulaKinematicsSolver,
    ee_frame: str,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    ee_orn: np.ndarray,
    ee_offset: np.ndarray,
    target_is_ee_pos: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    target_orn_lula = _orn_for_lula_r(ee_orn)
    reachable: List[np.ndarray] = []
    unreachable: List[np.ndarray] = []
    for z in zs:
        for y in ys:
            for x in xs:
                fingertip = np.array([x, y, z], dtype=np.float64)
                target_pos = (
                    fingertip if target_is_ee_pos else fingertip + ee_offset
                )
                _, ok = ik.compute_inverse_kinematics(
                    frame_name=ee_frame,
                    target_position=target_pos,
                    target_orientation=target_orn_lula,
                    warm_start=None,
                    position_tolerance=IK_POS_TOL,
                    orientation_tolerance=IK_ORN_TOL,
                )
                if bool(ok):
                    reachable.append(fingertip)
                else:
                    unreachable.append(fingertip)
    return reachable, unreachable


def main() -> None:
    args = _parse_args()
    print(f"scene: {args.scene}")
    if not os.path.isfile(args.scene):
        raise SystemExit(f"scene USD not found: {args.scene}")
    open_stage(usd_path=args.scene)

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise SystemExit("failed to open stage")

    board_paths = args.boards or _autodetect_board_paths(stage)
    if not board_paths:
        raise SystemExit("no board prims found")
    print(f"boards: {board_paths}")

    board_aabbs: List[Tuple[np.ndarray, np.ndarray]] = []
    for bp in board_paths:
        prim = stage.GetPrimAtPath(bp)
        if not prim or not prim.IsValid():
            print(f"  WARNING: {bp} not in stage, skipping")
            continue
        aabb = _world_aabb(prim)
        if aabb is None:
            print(f"  WARNING: {bp} empty AABB, skipping")
            continue
        board_aabbs.append(aabb)
        print(
            f"  {bp} AABB: min={np.round(aabb[0], 4).tolist()} "
            f"max={np.round(aabb[1], 4).tolist()}"
        )
    if not board_aabbs:
        raise SystemExit("no valid board AABBs")

    combined_min = np.min(np.stack([a[0] for a in board_aabbs]), axis=0)
    combined_max = np.max(np.stack([a[1] for a in board_aabbs]), axis=0)
    combined_min[0] -= args.xy_margin
    combined_min[1] -= args.xy_margin
    combined_max[0] += args.xy_margin
    combined_max[1] += args.xy_margin

    xs = np.linspace(combined_min[0], combined_max[0], int(args.xy_res))
    ys = np.linspace(combined_min[1], combined_max[1], int(args.xy_res))
    z0 = combined_max[2] + float(args.z_min_above)
    z1 = combined_max[2] + float(args.z_max_above)
    zs = np.linspace(z0, z1, int(args.z_steps))
    n_probe = len(xs) * len(ys) * len(zs)
    print(
        f"grid: {len(xs)}x{len(ys)} XY × {len(zs)} Z = {n_probe} probes/mode"
    )
    print(f"  z values: {np.round(zs, 4).tolist()}")

    robot_pos, robot_orn = get_world_pose(ROBOT_PRIM_PATH)
    home_pos, home_quat = get_world_pose(R_EE_PRIM_PATH)
    home_quat = home_quat / max(np.linalg.norm(home_quat), 1e-12)
    approach = _approach_axis_world(home_quat)
    print(f"robot base pos: {np.round(robot_pos, 4).tolist()}")
    print(
        f"R EE home (USD authored) pos={np.round(home_pos, 4).tolist()} "
        f"quat_wxyz={np.round(home_quat, 4).tolist()}"
    )
    print(
        f"  home approach (+Z_ee in world)={np.round(approach, 4).tolist()} "
        f"(top-down would be ≈ [0,0,-1])"
    )

    R_desc = _r_descriptor_path()
    urdf_path = os.path.abspath(
        os.path.join(_HERE, "..", "robot", "vega_1u_gripper.urdf")
    )
    print(f"R descriptor : {R_desc}")
    print(f"URDF         : {urdf_path}")

    if args.ee_offset is not None:
        ee_offset = np.asarray(args.ee_offset, dtype=np.float64)
    else:
        ee_offset = np.asarray(pc.PART_DEFAULTS["ee_offset"], dtype=np.float64)
    print(
        f"ee_offset    : {np.round(ee_offset, 4).tolist()}  "
        f"(applied: {not args.target_is_ee_pos})"
    )

    top_down = np.asarray(pc.PART_DEFAULTS["ee_orientation"], dtype=np.float64)
    modes: Dict[str, np.ndarray] = {
        # What fix_orientation freezes at clutch if engaged at startup home.
        "home_engage": home_quat.copy(),
        # What fixed_orientation_wxyz / scripted left uses.
        "top_down": top_down.copy(),
        # Identity — sometimes people confuse stage/wxyz conventions.
        "identity": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
    for entry in args.extra_orn:
        name = str(entry[0])
        q = np.array([float(v) for v in entry[1:]], dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-12)
        modes[name] = q

    ik = LulaKinematicsSolver(
        robot_description_path=R_desc,
        urdf_path=urdf_path,
    )
    ik.set_robot_base_pose(
        robot_position=robot_pos,
        robot_orientation=robot_orn,
    )
    ee_frame = "R_ee_link_gripper_link"

    prefix = args.output_prefix or os.path.join(_HERE, "reachable_r_arm_orn")
    results = []
    print("\n=== R-arm fixed-orientation reachability ===")
    for name, orn in modes.items():
        approach = _approach_axis_world(orn)
        print(
            f"\n[{name}] orn={np.round(orn, 4).tolist()} "
            f"approach={np.round(approach, 4).tolist()}"
        )
        reachable, unreachable = _probe_grid(
            ik,
            ee_frame,
            xs,
            ys,
            zs,
            orn,
            ee_offset,
            args.target_is_ee_pos,
        )
        total = len(reachable) + len(unreachable)
        pct = 100.0 * len(reachable) / total if total else 0.0
        results.append((name, len(reachable), total, pct, orn, approach))
        print(f"  reachable: {len(reachable)}/{total} ({pct:.1f}%)")
        write_ply(
            f"{prefix}_{name}_reachable.ply",
            reachable,
            rgb=(0, 200, 0),
        )
        write_ply(
            f"{prefix}_{name}_unreachable.ply",
            unreachable,
            rgb=(200, 50, 50),
        )

    home_pct = next(r[3] for r in results if r[0] == "home_engage")
    top_pct = next(r[3] for r in results if r[0] == "top_down")

    if args.also_free:
        print("\n[also-free] position-only (orn=None) + top-down tilts…", flush=True)
        ok_none = 0
        ok_tilt = 0
        total_f = 0
        # Small tilt set around top-down (≈ free wrist with limited samples).
        tilt_orns = [top_down.copy()]
        for axis, ang in (
            ((1, 0, 0), 0.4),
            ((1, 0, 0), -0.4),
            ((0, 1, 0), 0.4),
            ((0, 1, 0), -0.4),
            ((0, 0, 1), 0.4),
            ((0, 0, 1), -0.4),
        ):
            rv = np.asarray(axis, dtype=np.float64) * float(ang)
            a = float(np.linalg.norm(rv))
            u = rv / a
            s = np.sin(a / 2.0)
            q_rel = np.array(
                [np.cos(a / 2.0), u[0] * s, u[1] * s, u[2] * s],
                dtype=np.float64,
            )
            tilt_orns.append(_quat_mul(q_rel, top_down))

        free_pts: List[np.ndarray] = []
        for z in zs:
            for y in ys:
                for x in xs:
                    fingertip = np.array([x, y, z], dtype=np.float64)
                    target_pos = (
                        fingertip if args.target_is_ee_pos
                        else fingertip + ee_offset
                    )
                    total_f += 1
                    _, s_none = ik.compute_inverse_kinematics(
                        frame_name=ee_frame,
                        target_position=target_pos,
                        target_orientation=None,
                        warm_start=None,
                        position_tolerance=IK_POS_TOL,
                        orientation_tolerance=IK_ORN_TOL,
                    )
                    if bool(s_none):
                        ok_none += 1
                    any_tilt = False
                    for o in tilt_orns:
                        _, s_t = ik.compute_inverse_kinematics(
                            frame_name=ee_frame,
                            target_position=target_pos,
                            target_orientation=_orn_for_lula_r(o),
                            warm_start=None,
                            position_tolerance=IK_POS_TOL,
                            orientation_tolerance=IK_ORN_TOL,
                        )
                        if bool(s_t):
                            any_tilt = True
                            break
                    if any_tilt:
                        ok_tilt += 1
                        free_pts.append(fingertip)
        none_pct = 100.0 * ok_none / total_f if total_f else 0.0
        tilt_pct = 100.0 * ok_tilt / total_f if total_f else 0.0
        results.append(
            (
                "orn_none",
                ok_none,
                total_f,
                none_pct,
                np.array([np.nan] * 4),
                np.array([np.nan] * 3),
            )
        )
        results.append(
            (
                "free_tilts",
                ok_tilt,
                total_f,
                tilt_pct,
                top_down.copy(),
                _approach_axis_world(top_down),
            )
        )
        print(
            f"  orn=None:     {ok_none}/{total_f} ({none_pct:.1f}%)",
            flush=True,
        )
        print(
            f"  free tilts:   {ok_tilt}/{total_f} ({tilt_pct:.1f}%)",
            flush=True,
        )
        write_ply(f"{prefix}_free_tilts_reachable.ply", free_pts, rgb=(50, 150, 255))

    summary_path = prefix + "_summary.txt"
    with open(summary_path, "w") as sf:
        sf.write("=== R-arm fixed-orientation reachability ===\n")
        for name, ok, total, pct, orn, approach in results:
            line = (
                f"{name:<16} {ok:8d}/{total:<8d} {pct:6.1f}%  "
                f"orn={np.round(orn, 4).tolist()}  "
                f"approach={np.round(approach, 3).tolist()}\n"
            )
            sf.write(line)
            print(line, end="", flush=True)
        sf.write(
            f"\nhome_engage={home_pct:.1f}%  top_down={top_pct:.1f}%\n"
        )
        sf.write(
            "fix_orientation freezes ≈ home_engage at clutch; "
            "fixed_orientation_wxyz=[0,1,0,0] freezes top_down.\n"
        )
    print(f"wrote summary -> {summary_path}", flush=True)
    print(
        "\nInterpretation: fix_orientation=true holds clutch-engage quat "
        "(≈ home_engage at start). fixed_orientation_wxyz=[0,1,0,0] holds "
        "top_down. If home_engage << top_down, the restricted workspace is "
        "IK infeasibility of that frozen orn — not a translation clamp in "
        "retarget.",
        flush=True,
    )
    print(
        f"  home_engage reachability {home_pct:.1f}% vs "
        f"top_down {top_pct:.1f}%",
        flush=True,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
