# ruff: noqa: E402
"""Load a configured USDC part into an Isaac Sim ground-plane scene."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/part_preview_xz_up.yaml",
        help="YAML scene configuration, relative to the repository root",
    )
    parser.add_argument(
        "--asset",
        help="Override config selected_asset (for example: gear_60teeth)",
    )
    return parser.parse_args()


_ARGS = _arguments()
# Do not forward this script's arguments to Kit/SimulationApp.
sys.argv = [sys.argv[0]]
_REPO = Path(__file__).resolve().parent.parent
_CONFIG_PATH = Path(_ARGS.config)
if not _CONFIG_PATH.is_absolute():
    _CONFIG_PATH = _REPO / _CONFIG_PATH
with _CONFIG_PATH.open(encoding="utf-8") as stream:
    _CONFIG = yaml.safe_load(stream)

_HEADLESS = os.getenv("ISAACSIM_HEADLESS", "").strip().lower() in {
    "1", "true", "yes", "on",
}
if not _HEADLESS and not os.environ.get("DISPLAY"):
    print("[part_preview] ERROR: DISPLAY unset; need GUI or ISAACSIM_HEADLESS=1")
    sys.exit(1)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": _HEADLESS,
        "multi_gpu": False,
        "renderer": _CONFIG["render"]["renderer"],
    }
)

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.viewports import set_camera_view


def _world_z_bounds(stage: Usd.Stage, prim_path: str) -> tuple[float, float]:
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    ).ComputeWorldBound(stage.GetPrimAtPath(prim_path))
    aligned = bbox.ComputeAlignedRange()
    return float(aligned.GetMin()[2]), float(aligned.GetMax()[2])


def main() -> None:
    asset_name = _ARGS.asset or _CONFIG["selected_asset"]
    assets = _CONFIG["assets"]
    if asset_name not in assets:
        choices = ", ".join(assets)
        raise ValueError(f"unknown asset {asset_name!r}; choose one of: {choices}")

    asset_path = Path(assets[asset_name])
    if not asset_path.is_absolute():
        asset_path = _REPO / asset_path
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    ground_cfg = _CONFIG["ground"]
    orientation = _CONFIG["orientation"]
    camera = _CONFIG["camera"]
    prim_path = f"/World/{asset_name}"

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
    GroundPlane(
        prim_path="/World/groundPlane",
        z_position=float(ground_cfg["z_position"]),
    )
    stage = get_current_stage()

    light = UsdLux.DistantLight.Define(stage, "/World/defaultLight")
    light.CreateIntensityAttr(float(_CONFIG["render"]["light_intensity"]))
    light.CreateAngleAttr(0.53)

    add_reference_to_stage(usd_path=str(asset_path), prim_path=prim_path)
    part = stage.GetPrimAtPath(prim_path)
    if not part or not part.IsValid():
        raise RuntimeError(f"failed to reference {asset_path} at {prim_path}")

    xform = UsdGeom.Xformable(part)
    xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddRotateXYZOp().Set(
        Gf.Vec3d(
            float(orientation["rotate_x_deg"]),
            float(orientation["rotate_y_deg"]),
            float(orientation["rotate_z_deg"]),
        )
    )

    z_min, z_max = _world_z_bounds(stage, prim_path)
    ground_z = float(ground_cfg["z_position"])
    z_lift = ground_z - z_min + float(ground_cfg["clearance"])
    translate.Set(Gf.Vec3d(0.0, 0.0, z_lift))

    if not part.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.CollisionAPI.Apply(part)

    world.reset()
    target_xy = camera["target_xy"]
    set_camera_view(
        eye=np.asarray(camera["eye"], dtype=np.float64),
        target=np.array(
            [
                float(target_xy[0]),
                float(target_xy[1]),
                z_lift + float(camera["target_height_above_part_origin"]),
            ],
            dtype=np.float64,
        ),
    )

    output_path = Path(_CONFIG["output_usd"])
    if not output_path.is_absolute():
        output_path = _REPO / output_path
    stage.GetRootLayer().Export(str(output_path))
    print(
        f"[part_preview] loaded {asset_name}: {asset_path}\n"
        f"[part_preview] rotation XYZ = "
        f"({orientation['rotate_x_deg']}, {orientation['rotate_y_deg']}, "
        f"{orientation['rotate_z_deg']}) degrees\n"
        f"[part_preview] rotated Z AABB = [{z_min:.6f}, {z_max:.6f}] m; "
        f"translated to z = {z_lift:.6f} m\n"
        f"[part_preview] saved {output_path}",
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
