# ruff: noqa: E402
"""Headless PhysX validation of asset width -> gripper aperture calibration.

Loads the real Vega articulation (including the mimic finger), commands the
selected asset's Design D close angle, measures the distal inner jaw gap from
the simulated finger meshes, and compares it with the asset's actual mesh
geometry.

Example:
    OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y \
      .venv/bin/python task/validate_grasp_aperture_physics.py \
      --asset gear_60teeth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="gear_60teeth")
    parser.add_argument("--settle-steps", type=int, default=360)
    parser.add_argument("--physics-hz", type=float, default=120.0)
    parser.add_argument("--distal-z-min", type=float, default=0.18)
    parser.add_argument("--tolerance-mm", type=float, default=1.0)
    return parser.parse_args()


_ARGS = _arguments()
sys.argv = [sys.argv[0]]
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "task"))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "multi_gpu": False})

import numpy as np
from pxr import Usd, UsdGeom

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
import param_config as pc
from teleop.grasp_aperture import (
    close_rad_to_aperture_m,
    grasp_width_m,
    load_aabb_extents,
    resolve_grasp_close_rad,
)

_ROBOT_USD = _REPO / "robot" / "vega_1u_gripper_nocam.usda"
_ROBOT_ROOT = "/World/vega_1u"
_BASE = _ROBOT_ROOT + "/R_ee_link/gripper_link"
_ACTIVE = _ROBOT_ROOT + "/R_ee_link/gripper_active_link/gripper_active_link"
_PASSIVE = (
    _ROBOT_ROOT + "/R_ee_link/gripper_passive_link/left_gripper_passive_link"
)


def _points_in_frame(stage: Usd.Stage, mesh_path: str, frame_path: str) -> np.ndarray:
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    frame_prim = stage.GetPrimAtPath(frame_path)
    if not mesh_prim or not mesh_prim.IsValid():
        raise RuntimeError(f"missing mesh prim: {mesh_path}")
    if not frame_prim or not frame_prim.IsValid():
        raise RuntimeError(f"missing frame prim: {frame_path}")
    points = np.asarray(
        UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=np.float64
    )
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mesh_to_world = np.asarray(cache.GetLocalToWorldTransform(mesh_prim))
    frame_to_world = np.asarray(cache.GetLocalToWorldTransform(frame_prim))
    hom = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (hom @ mesh_to_world @ np.linalg.inv(frame_to_world))[:, :3]


def _distal_gap_m(stage: Usd.Stage, distal_z_min: float) -> float:
    active = _points_in_frame(stage, _ACTIVE, _BASE)
    passive = _points_in_frame(stage, _PASSIVE, _BASE)
    active = active[active[:, 2] >= float(distal_z_min)]
    passive = passive[passive[:, 2] >= float(distal_z_min)]
    if len(active) == 0 or len(passive) == 0:
        raise RuntimeError(
            f"distal-z threshold {distal_z_min:g} selected no finger vertices"
        )
    return float(active[:, 0].min() - passive[:, 0].max())


def _asset_mesh_extents_m(asset_path: Path) -> np.ndarray:
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"failed to open asset: {asset_path}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError(f"asset has no default prim: {asset_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_to_world = np.asarray(cache.GetLocalToWorldTransform(root))
    world_to_root = np.linalg.inv(root_to_world)
    clouds = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = np.asarray(
            UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64
        )
        mesh_to_world = np.asarray(cache.GetLocalToWorldTransform(prim))
        hom = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        clouds.append((hom @ mesh_to_world @ world_to_root)[:, :3])
    if not clouds:
        raise RuntimeError(f"asset contains no mesh points: {asset_path}")
    points = np.concatenate(clouds, axis=0)
    return points.max(axis=0) - points.min(axis=0)


def _width_from_rule(extents: np.ndarray, rule: str) -> float:
    if rule == "cylinder_yz_mean_diameter":
        return float(np.mean(extents[[1, 2]]))
    if rule == "xz_max":
        return float(np.max(extents[[0, 2]]))
    raise ValueError(f"unsupported grasp_rule {rule!r}")


def main() -> int:
    asset = str(_ARGS.asset)
    config = load_aabb_extents()
    part = (config.get("parts") or {}).get(asset)
    if part is None:
        raise KeyError(f"asset {asset!r} is absent from part_local_aabb_extents.json")
    asset_path = _REPO / "parts" / f"{asset}.usdc"
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    extents = _asset_mesh_extents_m(asset_path)
    mesh_width = _width_from_rule(extents, str(part["grasp_rule"]))
    configured_width = float(grasp_width_m(asset))
    target_q = float(resolve_grasp_close_rad(asset))
    target_gap = float(close_rad_to_aperture_m(target_q))
    open_q = float(pc.part_grasp_open_rad(asset))

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / float(_ARGS.physics_hz),
    )
    add_reference_to_stage(str(_ROBOT_USD), _ROBOT_ROOT)
    robot = world.scene.add(
        SingleArticulation(
            prim_path=_ROBOT_ROOT,
            name="vega_grasp_aperture_validation",
            reset_xform_properties=False,
        )
    )
    world.reset()
    dof_names = list(robot.dof_names)
    q_idx = dof_names.index("R_gripper_joint")
    q = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
    q[q_idx] = open_q
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros(len(q), dtype=np.float64))
    controller = robot.get_articulation_controller()
    action = [None] * len(q)
    action[q_idx] = target_q
    for _ in range(max(1, int(_ARGS.settle_steps))):
        controller.apply_action(ArticulationAction(joint_positions=action))
        world.step(render=False)

    measured_q = float(robot.get_joint_positions()[q_idx])
    measured_gap = _distal_gap_m(get_current_stage(), _ARGS.distal_z_min)
    tol = float(_ARGS.tolerance_mm) / 1000.0
    q_tol = 0.005
    checks = {
        "json_width_matches_mesh": abs(configured_width - mesh_width) <= tol,
        "open_target_is_wider_than_close": open_q > target_q,
        "joint_reached_target": abs(measured_q - target_q) <= q_tol,
        "measured_gap_matches_calibration": abs(measured_gap - target_gap) <= tol,
        "asset_fits_target_gap": measured_gap + tol >= mesh_width,
    }

    print(
        f"[grasp_physics] asset={asset} rule={part['grasp_rule']}\n"
        f"[grasp_physics] mesh_extents_mm={np.round(extents * 1000.0, 3).tolist()} "
        f"mesh_width_mm={mesh_width * 1000.0:.3f} "
        f"configured_width_mm={configured_width * 1000.0:.3f}\n"
        f"[grasp_physics] open_q_rad={open_q:.6f} "
        f"target_q_rad={target_q:.6f} "
        f"measured_q_rad={measured_q:.6f}\n"
        f"[grasp_physics] calibrated_gap_mm={target_gap * 1000.0:.3f} "
        f"measured_gap_mm={measured_gap * 1000.0:.3f} "
        f"clearance_over_mesh_mm={(measured_gap - mesh_width) * 1000.0:.3f}",
        flush=True,
    )
    for name, passed in checks.items():
        print(f"[grasp_physics] {'PASS' if passed else 'FAIL'} {name}", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    try:
        _result = main()
    finally:
        simulation_app.close()
    raise SystemExit(_result)
