# ruff: noqa: E402
"""Standalone Isaac Sim tool: calibrate L_arm_j1..j7 init joint targets.

Loads robot + ground only (no table / task board / parts). An omni.ui panel
drives each L-arm joint; Save writes param_config.INIT_JOINT_TARGETS and
keeps Lula descriptor YAMLs in sync.

Launch:
    ./scripts/calibrate_l_arm.sh
    # or: uv run python task/calibrate_l_arm_joints.py
"""
from __future__ import annotations

import os
import sys

from isaacsim import SimulationApp

_HEADLESS = os.getenv("ISAACSIM_HEADLESS", "").strip().lower() in {
    "1", "true", "yes", "on",
}
if _HEADLESS or not os.environ.get("DISPLAY"):
    print(
        "[l_arm_calib] ERROR: this tool needs a GUI display. "
        "Unset ISAACSIM_HEADLESS and ensure DISPLAY is set.",
        flush=True,
    )
    sys.exit(1)

simulation_app = SimulationApp(
    {
        "headless": False,
        "multi_gpu": False,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 0,
        "samples_per_pixel_per_frame": 1,
        "denoiser": False,
        "max_bounces": 2,
        "max_specular_transmission_bounces": 2,
        "max_volume_bounces": 2,
        "width": 960,
        "height": 540,
        "window_width": 1280,
        "window_height": 720,
    }
)

import numpy as np

import param_config as pc
from controllers.pick_place_task import ROBOT_PRIM_PATH
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from l_arm_calib_ui import (
    L_ARM_JOINTS,
    LArmCalibUI,
    print_targets,
    save_l_arm_init_targets,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROBOT_USD = os.path.join(_REPO_ROOT, "robot", "vega_1u_gripper.usda")


def _seed_l_targets_from_config() -> dict:
    targets = getattr(pc, "INIT_JOINT_TARGETS", {}) or {}
    out = {}
    for jname in L_ARM_JOINTS:
        if jname not in targets:
            raise KeyError(
                f"param_config.INIT_JOINT_TARGETS missing {jname!r}; "
                "cannot seed calibration UI"
            )
        out[jname] = float(targets[jname])
    return out


def _apply_full_init(robot: SingleArticulation, dof_names: list) -> np.ndarray:
    """Teleport all listed INIT_JOINT_TARGETS; return the resulting q vector."""
    q = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
    for jname, val in (getattr(pc, "INIT_JOINT_TARGETS", None) or {}).items():
        if jname in dof_names:
            q[dof_names.index(jname)] = float(val)
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros(len(dof_names), dtype=np.float64))
    return q


def main() -> None:
    if not os.path.isfile(_ROBOT_USD):
        raise FileNotFoundError(f"robot USD not found: {_ROBOT_USD}")

    my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)
    my_world.scene.add_default_ground_plane()

    add_reference_to_stage(usd_path=_ROBOT_USD, prim_path=ROBOT_PRIM_PATH)
    robot = my_world.scene.add(
        SingleArticulation(
            prim_path=ROBOT_PRIM_PATH,
            name="vega_1u_calib",
            reset_xform_properties=False,
        )
    )

    my_world.reset()
    dof_names = list(robot.dof_names)
    for jname in L_ARM_JOINTS:
        if jname not in dof_names:
            raise RuntimeError(
                f"articulation at {ROBOT_PRIM_PATH} has no DOF {jname!r}; "
                f"dof_names={dof_names}"
            )

    q0 = _apply_full_init(robot, dof_names)
    # Hold every non-L DOF at the post-init snapshot; L comes from the UI.
    hold = {
        i: float(q0[i])
        for i, jname in enumerate(dof_names)
        if jname not in L_ARM_JOINTS
    }
    l_indices = {jname: dof_names.index(jname) for jname in L_ARM_JOINTS}
    articulation_controller = robot.get_articulation_controller()

    # View from +Y so the left arm fills the viewport (mirror of R calibrator).
    set_camera_view(
        eye=np.array([1.8, 1.6, 1.2], dtype=np.float64),
        target=np.array([0.0, 0.0, 0.5], dtype=np.float64),
    )

    initial_l = _seed_l_targets_from_config()

    def _on_save(targets: dict) -> None:
        touched = save_l_arm_init_targets(targets)
        print_targets(targets)
        print("[l_arm_calib] wrote:", flush=True)
        for p in touched:
            print(f"[l_arm_calib]   {p}", flush=True)

    ui = LArmCalibUI(
        initial_targets_rad=initial_l,
        on_save=_on_save,
        on_print=print_targets,
    )
    print(
        "[l_arm_calib] ready — use the 'L Arm Init Calibration' window. "
        "Print dumps values; Save updates param_config + Lula YAMLs.",
        flush=True,
    )

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        # After Stop+Play, World.reset restores USD defaults — re-apply hold
        # baseline and teleport to current UI targets.
        if my_world.current_time_step_index <= 2:
            q0 = _apply_full_init(robot, dof_names)
            for i, jname in enumerate(dof_names):
                if jname not in L_ARM_JOINTS:
                    hold[i] = float(q0[i])
            ui.request_teleport()

        targets = ui.get_targets_rad()
        q = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
        for i, val in hold.items():
            q[i] = val
        for jname, idx in l_indices.items():
            q[idx] = float(targets[jname])

        if ui.consume_teleport_request():
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros(len(dof_names), dtype=np.float64))

        action = [None] * len(dof_names)
        for i, val in hold.items():
            action[i] = val
        for jname, idx in l_indices.items():
            action[idx] = float(targets[jname])
        articulation_controller.apply_action(
            ArticulationAction(joint_positions=action)
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
