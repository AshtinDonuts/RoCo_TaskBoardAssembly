# ruff: noqa: E402
"""Empty Isaac scene + ground; import gear_20teeth with its XZ face up."""
from __future__ import annotations

import os
import sys

from isaacsim import SimulationApp

_HEADLESS = os.getenv("ISAACSIM_HEADLESS", "").strip().lower() in {
    "1", "true", "yes", "on",
}
if not _HEADLESS and not os.environ.get("DISPLAY"):
    print("[preview_gear] ERROR: DISPLAY unset; need GUI or ISAACSIM_HEADLESS=1", flush=True)
    sys.exit(1)

simulation_app = SimulationApp(
    {
        "headless": _HEADLESS,
        "multi_gpu": False,
        "renderer": "RaytracedLighting",
    }
)

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TASK_DIR)
_GEAR_USD = os.path.join(_REPO, "parts", "gear_20teeth.usdc")
_PRIM_PATH = "/World/gear_20teeth"
_SAVE_USD = os.path.join(_REPO, "scene_gear_20teeth_xz_up.usd")


def _world_z_bounds(stage: Usd.Stage, prim_path: str) -> tuple[float, float]:
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    ).ComputeWorldBound(stage.GetPrimAtPath(prim_path))
    aligned = bbox.ComputeAlignedRange()
    return float(aligned.GetMin()[2]), float(aligned.GetMax()[2])


def main() -> None:
    if not os.path.isfile(_GEAR_USD):
        raise FileNotFoundError(_GEAR_USD)

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    GroundPlane(prim_path="/World/groundPlane", z_position=0.0)
    stage = get_current_stage()

    light = UsdLux.DistantLight.Define(stage, "/World/defaultLight")
    light.CreateIntensityAttr(3000.0)
    light.CreateAngleAttr(0.53)

    add_reference_to_stage(usd_path=_GEAR_USD, prim_path=_PRIM_PATH)
    gear = stage.GetPrimAtPath(_PRIM_PATH)
    if not gear or not gear.IsValid():
        raise RuntimeError(f"failed to reference {_GEAR_USD} at {_PRIM_PATH}")

    xform = UsdGeom.Xformable(gear)
    xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    # local +Y -> world +Z, so the asset's XZ plane faces upward.
    xform.AddRotateXOp().Set(90.0)

    z_min, z_max = _world_z_bounds(stage, _PRIM_PATH)
    clearance = 0.001
    z_lift = -z_min + clearance
    translate.Set(Gf.Vec3d(0.0, 0.0, z_lift))
    print(
        f"[preview_gear] rotated Z AABB=[{z_min*1000:.2f}, {z_max*1000:.2f}] mm; "
        f"lift z={z_lift*1000:.2f} mm so XZ face is up (+Z)",
        flush=True,
    )

    if not gear.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.CollisionAPI.Apply(gear)

    world.reset()
    set_camera_view(
        eye=np.array([0.12, 0.12, 0.10], dtype=np.float64),
        target=np.array([0.0, 0.0, z_lift + 0.01], dtype=np.float64),
    )

    stage.GetRootLayer().Export(_SAVE_USD)
    print(f"[preview_gear] saved {_SAVE_USD}", flush=True)
    print(
        "[preview_gear] gear_20teeth at /World/gear_20teeth; "
        "XZ face up (world +Z). Close the Kit window to exit.",
        flush=True,
    )

    if _HEADLESS:
        return
    while simulation_app.is_running():
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
