# Setup for the vega_1u bimanual pick-and-place simulation.
#
# Scene strategy: load ../../scene_base.usd (relative to this file) as the
# stage. That USDA already contains the robot, ground, table, and the two pick
# objects -- this setup only wraps existing prims and builds the two
# per-arm PickPlaceControllers.
#
# The post-task plumbing (world.reset() -> per-arm controllers + cameras +
# viewports) is factored into `_finalize_pick_place_setup`.

import carb
import os
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from typing import Optional

import omni.kit.commands
import omni.usd
from omni.kit.viewport.utility import (
    create_viewport_window,
    get_active_viewport_window,
)

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.viewports import set_camera_view

from .ee_pose_controller import EEPoseController
from .pick_place_task import PickPlaceTask_scene_bimanual, ROBOT_PRIM_PATH

# Top-level dev-loop config: knobs for which descriptor each arm uses.
# Assumes Task_test/ is on sys.path (set up by run_pick_place.py at launch).
import param_config as pc


# ---------------------------------------------------------------------------
# Path to the pre-built scene. Source of truth is param_config.SCENE_USD,
# resolved relative to param_config.py's directory (task/).
# ---------------------------------------------------------------------------
TASK_NAME = "vega_1u_pick_place"


def _resolve_scene_path() -> str:
    pc_dir = os.path.dirname(os.path.abspath(pc.__file__))
    return os.path.abspath(os.path.join(pc_dir, pc.SCENE_USD))


def create_viewport_for_camera(
    viewport_name: str,
    camera_prim_path: str,
    width: int = 1280,
    height: int = 720,
    position_x: int = 0,
    position_y: int = 0,
):
    viewport_window = create_viewport_window(
        name=viewport_name, width=width, height=height, position_x=position_x, position_y=position_y
    )
    omni.kit.commands.execute(
        "SetViewportCamera", camera_path=camera_prim_path, viewport_api=viewport_window.viewport_api
    )
    carb.log_info(f"Added new viewport '{viewport_name}' for camera '{camera_prim_path}'")
    return viewport_window


def _set_camera_focal_length(stage, camera_prim_path: str, focal_length: float):
    prim = stage.GetPrimAtPath(camera_prim_path) if stage else None
    if not prim or not prim.IsValid():
        print(f"[setup] WARNING: cannot set focal length; missing camera {camera_prim_path!r}")
        return
    camera = UsdGeom.Camera(prim)
    camera.GetFocalLengthAttr().Set(float(focal_length))
    print(f"[setup] set {camera_prim_path} focalLength={float(focal_length):.6g}")


def find_ground_world_z(default: float = 0.0) -> float:
    """Walk the open stage and return the world-space z of the ground prim.

    Searches for the first prim whose name contains 'ground', 'plane', or
    'floor' (case-insensitive). Returns ``default`` if none is found. Used to
    apply a stage-level ground shift to Lula's robot base pose so the IK FK
    matches the actual articulation pose after the ground was moved.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return default
    xform_cache = UsdGeom.XformCache()
    needles = ("ground", "plane", "floor")
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if any(n in name for n in needles):
            try:
                mat = xform_cache.GetLocalToWorldTransform(prim)
                t = mat.ExtractTranslation()
                return float(t[2])
            except Exception:
                continue
    return default


def _apply_physx_determinism():
    """Enable PhysX enhanced determinism + single-threaded solver.

    Removes the two main run-to-run drift sources: contact-resolution
    nondeterminism (via the PhysicsScene `enableEnhancedDeterminism` flag)
    and thread-scheduling jitter (by pinning the solver to 1 thread).
    """
    carb.settings.get_settings().set("/persistent/physics/numThreads", 1)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsScene":
            attr = prim.GetAttribute("physxScene:enableEnhancedDeterminism")
            if not attr or not attr.IsValid():
                attr = prim.CreateAttribute(
                    "physxScene:enableEnhancedDeterminism", Sdf.ValueTypeNames.Bool
                )
            attr.Set(True)
            break


# Parts whose scene-authored pose is intentionally different from the JSON
# (e.g. permanently assembled on the rack). Skipped entirely by the verifier.
_VERIFY_SKIP = frozenset({"rod_16mm", "bolt_8mm"})

# Parts that get auto-shifted to match JSON on mismatch. For everything else
# (in JSON, in the scene, NOT in skip, NOT in autofix), we only warn.
_AUTOFIX_PARTS = frozenset({"gear_20teeth", "gear_60teeth"})

# XY mismatch above this triggers a verify warning / autofix shift.
_VERIFY_XY_TOL_M = 5e-3   # 5 mm


def _verify_and_autofix_scene_part_poses():
    """Compare each scene-resident part's mesh world XY against the JSON's
    ``pos.xy``. Three buckets:

      _VERIFY_SKIP   -> skipped entirely (rod, bolt: assembled on rack).
      _AUTOFIX_PARTS -> on mismatch > tol, shift the root prim's translate
                        by (delta_x, delta_y, 0) so the mesh lands at the
                        JSON pos. Works through rotation/scale on the root
                        because translation commutes through the outer
                        composition: shifting T_root by delta shifts the
                        mesh world by exactly delta.
      anything else  -> on mismatch, log a WARNING but don't touch the stage.

    Must run BEFORE ``World(...)`` is created. Otherwise the task's
    SingleRigidPrim wrappers snapshot the pre-shift pose, and World.reset()
    will revert the shift.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    poses = getattr(pc, "PART_INIT_POSES", {}) or {}
    cache = UsdGeom.XformCache()
    n_checked = 0
    n_match = 0
    n_warn = 0
    n_fix = 0
    for name, entry in poses.items():
        if name in _VERIFY_SKIP:
            continue
        if "pos" not in entry:
            continue
        prim_path = f"/World/parts/{name}"
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim or not root_prim.IsValid():
            continue
        # Deepest mesh = same "canonical mesh" extract_part_poses.py uses
        # to record JSON pos. World pose of THIS prim is what we compare.
        deepest_mesh = None
        deepest_depth = -1
        for p in Usd.PrimRange(root_prim):
            if p.GetTypeName() != "Mesh":
                continue
            d = p.GetPath().pathString.count("/")
            if d > deepest_depth:
                deepest_depth = d
                deepest_mesh = p
        if deepest_mesh is None:
            continue
        n_checked += 1

        m_world = cache.GetLocalToWorldTransform(deepest_mesh)
        cur = m_world.ExtractTranslation()
        json_x = float(entry["pos"][0])
        json_y = float(entry["pos"][1])
        dx = json_x - float(cur[0])
        dy = json_y - float(cur[1])
        if abs(dx) <= _VERIFY_XY_TOL_M and abs(dy) <= _VERIFY_XY_TOL_M:
            n_match += 1
            continue

        if name not in _AUTOFIX_PARTS:
            print(f"[setup] WARNING: {name} mesh world XY in scene = "
                  f"({float(cur[0]):+.5f}, {float(cur[1]):+.5f}); JSON "
                  f"expects ({json_x:+.5f}, {json_y:+.5f}); "
                  f"delta=({dx:+.5f}, {dy:+.5f}) m. Edit scene or JSON; "
                  f"or add {name!r} to _AUTOFIX_PARTS for automatic shift.")
            n_warn += 1
            continue

        # Apply delta-shift to the root prim's outermost translate.
        xform = UsdGeom.Xformable(root_prim)
        ops = xform.GetOrderedXformOps()
        translate_op = None
        transform_op = None
        for op in ops:
            t = op.GetOpType()
            if t == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            elif t == UsdGeom.XformOp.TypeTransform and transform_op is None:
                transform_op = op
        if translate_op is not None:
            cur_t = translate_op.Get()
            if cur_t is None:
                cur_t = Gf.Vec3d(0.0, 0.0, 0.0)
            translate_op.Set(Gf.Vec3d(
                float(cur_t[0]) + dx, float(cur_t[1]) + dy, float(cur_t[2])
            ))
            n_fix += 1
        elif transform_op is not None:
            mat = transform_op.Get()
            if mat is None:
                n_warn += 1
                continue
            cur_t = mat.ExtractTranslation()
            new_mat = Gf.Matrix4d(mat)
            new_mat.SetTranslateOnly(Gf.Vec3d(
                float(cur_t[0]) + dx, float(cur_t[1]) + dy, float(cur_t[2])
            ))
            transform_op.Set(new_mat)
            n_fix += 1
        else:
            op_names = [op.GetName() for op in ops]
            print(f"[setup] WARNING: {name} needs XY shift but has no "
                  f"xformOp:translate/transform on root (found: {op_names}).")
            n_warn += 1
    if n_warn:
        print(f"[setup] scene-pose verifier: {n_warn} warning(s); "
              f"check scene vs part_init_poses.json.")


def _translate_prim_xy(stage, prim_path: str, offset, name: str) -> bool:
    """Translate a prim's outer transform in XY while preserving Z/pose."""
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if not prim or not prim.IsValid():
        print(f"[setup] {name}: prim {prim_path} not in stage; no XY shift")
        return False
    delta = np.asarray(offset, dtype=np.float64).reshape(-1)
    if delta.size < 2:
        raise ValueError(f"{name} offset must contain X/Y, got {offset!r}")

    xform = UsdGeom.Xformable(prim)
    translate_op = None
    transform_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeTransform and transform_op is None:
            transform_op = op

    if translate_op is not None:
        cur = translate_op.Get()
        if cur is None:
            cur = Gf.Vec3d(0.0, 0.0, 0.0)
        translate_op.Set(Gf.Vec3d(
            float(cur[0]) + float(delta[0]),
            float(cur[1]) + float(delta[1]),
            float(cur[2]),
        ))
        return True
    if transform_op is not None:
        matrix = transform_op.Get()
        if matrix is not None:
            cur = matrix.ExtractTranslation()
            shifted = Gf.Matrix4d(matrix)
            shifted.SetTranslateOnly(Gf.Vec3d(
                float(cur[0]) + float(delta[0]),
                float(cur[1]) + float(delta[1]),
                float(cur[2]),
            ))
            transform_op.Set(shifted)
            return True
    print(f"[setup] {name}: no translate/transform op on {prim_path}; no XY shift")
    return False


def apply_xy_randomization(board_offset=None, part_offsets=None):
    """Apply trial XY offsets before World snapshots are created."""
    if board_offset is None and not part_offsets:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    if board_offset is not None:
        _translate_prim_xy(
            stage, "/World/task_board", board_offset, "task board"
        )
    for name, offset in (part_offsets or {}).items():
        _translate_prim_xy(
            stage, f"/World/parts/{name}", offset, f"part {name}"
        )


def _find_part_mesh(prim, deepest=False):
    """Return the first (or deepest) Mesh descendant below ``prim``."""
    selected = None
    selected_depth = -1
    if prim and prim.GetTypeName() == "Mesh":
        selected = prim
        selected_depth = prim.GetPath().pathString.count("/")
    if not prim:
        return None
    for desc in Usd.PrimRange(prim):
        if desc == prim or desc.GetTypeName() != "Mesh":
            continue
        if not deepest:
            return desc
        depth = desc.GetPath().pathString.count("/")
        if depth > selected_depth:
            selected = desc
            selected_depth = depth
    return selected


def _world_xform(stage, prim_path):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return cache.GetLocalToWorldTransform(stage.GetPrimAtPath(Sdf.Path(prim_path)))


def preplace_part_at_success_target(stage, placement):
    """Place one earlier part at its successful post-placement state.

    ``placement`` is prepared by ``run_pick_place`` and contains the part
    name, its randomized runtime config, and the pure endpoint specification.
    This runs after scene randomization but before the initial xform snapshot,
    so the contextual part is restored to this state on every iteration reset.
    """
    if not placement:
        return None
    part_name = placement["part"]
    config = placement["config"]
    spec = placement["spec"]
    snap = config.get("snap") or {}
    movable_path = snap.get("movable_path", f"/World/parts/{part_name}")
    root = stage.GetPrimAtPath(Sdf.Path(movable_path))
    if not root or not root.IsValid():
        raise RuntimeError(
            f"cannot pre-place predecessor {part_name!r}: "
            f"missing prim {movable_path}"
        )

    from snap_attach import (
        author_fixed_joint,
        build_world_matrix,
        resolve_rigid_body,
        snap_to_pose,
    )

    body_path = resolve_rigid_body(stage, movable_path)
    if not body_path:
        raise RuntimeError(
            f"cannot pre-place predecessor {part_name!r}: no rigid body "
            f"under {movable_path}"
        )
    mesh_path = spec.get("mesh_path")
    mesh = (stage.GetPrimAtPath(Sdf.Path(mesh_path))
            if mesh_path else _find_part_mesh(root))
    if not mesh or not mesh.IsValid() or mesh.GetTypeName() != "Mesh":
        raise RuntimeError(
            f"cannot pre-place predecessor {part_name!r}: no Mesh "
            f"descendant under {movable_path}"
        )

    m_body_cur = _world_xform(stage, body_path)
    m_mesh_cur = _world_xform(stage, str(mesh.GetPath()))
    mesh_local_in_body = m_mesh_cur * m_body_cur.GetInverse()
    target_position = Gf.Vec3d(*spec["position"])

    if spec["measure"] == "mesh_pose":
        m_mesh_target = build_world_matrix(
            spec["position"], spec["rotation"]
        )
        m_body_target = mesh_local_in_body.GetInverse() * m_mesh_target
    elif spec["measure"] == "aabb_midpoint":
        deepest_mesh = _find_part_mesh(root, deepest=True)
        if not deepest_mesh:
            raise RuntimeError(
                f"cannot pre-place predecessor {part_name!r}: no deepest Mesh"
            )
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        aabb = bbox_cache.ComputeWorldBound(deepest_mesh).ComputeAlignedRange()
        current_midpoint = aabb.GetMidpoint()
        delta = target_position - current_midpoint
        m_body_target = Gf.Matrix4d(m_body_cur)
        current_body_position = m_body_cur.ExtractTranslation()
        m_body_target.SetTranslateOnly(
            current_body_position + Gf.Vec3d(delta[0], delta[1], delta[2])
        )
    elif spec["measure"] == "mesh_translation":
        # Open-part grade_pos/place_pos is a mesh translation target. Preserve
        # the existing orientation, matching the position-only grader.
        m_mesh_target = Gf.Matrix4d(m_mesh_cur)
        m_mesh_target.SetTranslateOnly(target_position)
        m_body_target = mesh_local_in_body.GetInverse() * m_mesh_target
    else:
        raise ValueError(
            f"unknown successful endpoint measure {spec['measure']!r} "
            f"for {part_name!r}"
        )

    snap_to_pose(stage, body_path, m_body_target, set_kinematic=False)
    joint_path = f"/World/_context_success_joint_{part_name}"
    if spec["fixed_joint"]:
        joint, _, _ = author_fixed_joint(
            stage,
            body_path,
            spec.get("parent_body_path", ""),
            m_body_target,
            joint_path,
        )
        if joint is None:
            raise RuntimeError(
                f"failed to author contextual fixed joint for {part_name!r}"
            )

    print(
        f"[setup] preplaced prior part {part_name} at successful "
        f"{spec['measure']} target=({target_position[0]:+.5f}, "
        f"{target_position[1]:+.5f}, {target_position[2]:+.5f})"
        + (f" joint={joint_path}" if spec["fixed_joint"] else ""),
        flush=True,
    )
    return {
        "part": part_name,
        "measure": spec["measure"],
        "fixed_joint": bool(spec["fixed_joint"]),
        "joint_path": joint_path if spec["fixed_joint"] else None,
        "body_path": body_path,
        "mesh_path": str(mesh.GetPath()),
    }


# Snapshot of every scene-resident part's xformOp values, captured ONCE
# before the first physics step. Used by restore_scene_part_xforms() to
# undo the moved-by-PhysX poses on each iteration restart — without this,
# stop+play leaves the part wherever the previous run's snap+release left
# it (and removing the snap joint releases the body with stale velocity
# from the last run, sending it flying).
_SCENE_PART_XFORMOPS_SNAPSHOT = {}


def _snapshot_scene_part_xforms():
    """Record the current xformOp values for every part in
    pc.PART_INIT_POSES that's already in the scene. Stores a list of
    (op_name, value) per prim so we can write the same values back later.
    """
    _SCENE_PART_XFORMOPS_SNAPSHOT.clear()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    poses = getattr(pc, "PART_INIT_POSES", {}) or {}
    for name in poses:
        prim_path = f"/World/parts/{name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        ops_snap = []
        for op in xform.GetOrderedXformOps():
            val = op.Get()
            ops_snap.append((op.GetName(), val))
        _SCENE_PART_XFORMOPS_SNAPSHOT[prim_path] = ops_snap


def restore_scene_part_xforms():
    """Restore the snapshotted xformOp values for every scene-resident
    part. Called from run_pick_place._restart_iteration so a stop+play
    starts each part at its scene-authored (or autofix-shifted) pose
    instead of wherever PhysX left it.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    n_restored = 0
    for prim_path, ops_snap in _SCENE_PART_XFORMOPS_SNAPSHOT.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        xform = UsdGeom.Xformable(prim)
        ops = {op.GetName(): op for op in xform.GetOrderedXformOps()}
        for op_name, val in ops_snap:
            if op_name in ops and val is not None:
                ops[op_name].Set(val)
        n_restored += 1


def open_scene_and_world(
    board_offset=None, part_offsets=None, preplaced_success_parts=None
):
    """Open scene_base.usd into a fresh stage and create the World."""
    scene_path = _resolve_scene_path()
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(
            f"scene USD {pc.SCENE_USD!r} not found at {scene_path}"
        )
    open_stage(usd_path=scene_path)
    print(f"[setup] opened scene: {scene_path}")
    # Verify each scene-resident part's mesh world XY against
    # part_init_poses.json. Parts in _AUTOFIX_PARTS (gears) get a
    # delta-based root-translate shift on mismatch — translation commutes
    # through the outer composition, so shifting T_root by delta moves
    # the mesh by exactly delta regardless of any rotation/scale below.
    # Parts in _VERIFY_SKIP (rod_16mm, bolt_8mm) are intentionally
    # different from the JSON spawn pose and skipped. Everything else
    # gets a warning on mismatch but is not modified.
    _verify_and_autofix_scene_part_poses()
    # Apply evaluation-time offsets after nominal pose verification but before
    # any World/task wrapper snapshots the initial transforms.
    apply_xy_randomization(board_offset, part_offsets)
    # Optional contextual evaluation setup: place all earlier canonical parts
    # at their successful endpoints before capturing the reset snapshot.
    for placement in preplaced_success_parts or ():
        preplace_part_at_success_target(
            stage=omni.usd.get_context().get_stage(), placement=placement
        )
    # Snapshot scene-resident part xforms NOW, before physics starts.
    # restore_scene_part_xforms() uses this snapshot on each stop+play
    # so parts don't carry over their last-known PhysX-overwritten pose.
    _snapshot_scene_part_xforms()
    _apply_physx_determinism()
    return World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)


def _finalize_pick_place_setup(
    my_world: World,
    task_name: str = TASK_NAME,
    end_effector_initial_height: Optional[float] = None,
    events_dt: Optional[list] = None,
    total_t: Optional[float] = None,
    flag_robot_state_update: bool = False,
    enable_camera_viewports: bool = True,
    enable_camera_output: bool = True,
    base_translation_offset=None,
):
    """Reset the world, build per-arm PickPlaceControllers, cameras, viewports.

    Assumes a task named `task_name` has been added to `my_world`. Returns the
    same 9-tuple shape used throughout the codebase.
    """
    # ---- World reset: registers articulations & rigid bodies.
    my_world.reset()

# ---- Per-arm controllers on the shared /vega_1u articulation.
    task_params = my_world.get_task(task_name).get_params()
    L_robot_name = task_params["L_robot_name"]["value"]
    R_robot_name = task_params["R_robot_name"]["value"]
    my_L_arm = my_world.scene.get_object(L_robot_name)
    my_R_arm = my_world.scene.get_object(R_robot_name)

    # L owns_torso=False -> 7-DOF arm-only IK (Lift + torso_flip held by
    # the PD controller at their initial values). The 9-DOF version was
    # flipping torso_flip during cross-body transit (e.g. pick at x~-0.1 to
    # place at x~+0.25), swinging the arm "behind the back" before settling.
    # Re-enable the 9-DOF cspace if extended reach via torso lift/tilt is
    # needed and add a c-space bias on torso_flip to prevent the flip.
    # `end_effector_initial_height`, `events_dt`, `total_t` are accepted by
    # _finalize_pick_place_setup for backwards compatibility but are no
    # longer used by the EEPoseController (which is driven by waypoints
    # supplied by the caller, not a fixed state machine).
    L_controller = EEPoseController(
        name="controller_L",
        robot_articulation=my_L_arm,
        side="L",
        owns_torso=pc.OWNS_TORSO_L,
        owns_lift=getattr(pc, "OWNS_LIFT_L", pc.OWNS_TORSO_L),
        base_translation_offset=base_translation_offset,
    )
    R_controller = EEPoseController(
        name="controller_R",
        robot_articulation=my_R_arm,
        side="R",
        owns_torso=pc.OWNS_TORSO_R,
        owns_lift=getattr(pc, "OWNS_LIFT_R", pc.OWNS_TORSO_R),
        base_translation_offset=base_translation_offset,
    )

    # Both manipulator wrappers reference the same /vega_1u articulation, so a
    # single articulation_controller drives every joint.
    articulation_controller = my_L_arm.get_articulation_controller()
    init_joint_position = np.asarray(my_L_arm.get_joint_positions(),
                                     dtype=np.float64).copy()
    # If neither arm has Lift in its IK cspace (i.e. both are _armonly),
    # Lift is held by the PD controller at whatever value the USD initial
    # state had. Push it to param_config.LIFT_INIT_M so it (a) matches the
    # `Lift fixed value` in the *_armonly description yamls and (b) gives
    # the arm a sensible starting reach height. When either arm owns Lift
    # (_liftonly or full 9-DOF), Lift is part of that arm's cspace and the
    # IK sets it each step — no override needed.
    owns_lift_L = getattr(pc, "OWNS_LIFT_L", pc.OWNS_TORSO_L)
    owns_lift_R = getattr(pc, "OWNS_LIFT_R", pc.OWNS_TORSO_R)
    dof_names = list(my_L_arm.dof_names)
    if (not owns_lift_L) and (not owns_lift_R):
        if "Lift" in dof_names:
            init_joint_position[dof_names.index("Lift")] = float(
                getattr(pc, "LIFT_INIT_M", 0.0))

    # R-arm rest pose — see param_config.R_ARM_TUCKED. Tucked = j1=-90, rest 0.
    # Forward (USDA) = j1=-15, j2=-20, rest 0. Setting it here so the
    # snapshot R_arm_hold_q in run_pick_place.main() picks the chosen rest
    # pose and PDs R to it every step. The L descriptor yamls assume the
    # TUCKED layout for their R_arm_j* fixed values, so flipping this to
    # forward leaves Lula's R collision spheres slightly off — fine unless
    # R is actively doing IK.
    if getattr(pc, "R_ARM_TUCKED", True):
        R_REST = {"R_arm_j1": -1.57079633, "R_arm_j2": 0.0}
    else:
        R_REST = {"R_arm_j1": -0.26179939, "R_arm_j2": -0.34906585}
    for jname, val in R_REST.items():
        if jname in dof_names:
            init_joint_position[dof_names.index(jname)] = float(val)
    for jname in ("R_arm_j3", "R_arm_j4", "R_arm_j5", "R_arm_j6", "R_arm_j7"):
        if jname in dof_names:
            init_joint_position[dof_names.index(jname)] = 0.0

    my_L_arm.set_joint_positions(init_joint_position)

    # ---- Cameras (RGB + depth).
    # Cameras are authored in the robot USD at fixed prim paths. We do
    # NOT pass translation / orientation here — the USD pose stands. We
    # only (optionally) wrap them as Camera sensors so RGB/depth are
    # readable from Python, and/or open viewport tiles in Kit UI.
    HEAD_DEPTH_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/zed_depth_frame/headcam"
    L_WRIST_CAMERA_PATH    = f"{ROBOT_PRIM_PATH}/L_ee_link/gripper_link/L_wristcam"
    R_WRIST_CAMERA_PATH    = f"{ROBOT_PRIM_PATH}/R_ee_link/gripper_link/R_wristcam"

    head_depth_camera = None
    L_wrist_camera = None
    R_wrist_camera = None

    if enable_camera_output or enable_camera_viewports:
        # Verify the USD-authored camera prims actually exist. If a path
        # is missing, Camera(prim_path=...) would silently create a new
        # world-static Xform there, breaking the assumption that the cam
        # rides on the robot.
        _stage = omni.usd.get_context().get_stage()
        _missing = []
        for _label, _cam_path in (
            ("headcam",    HEAD_DEPTH_CAMERA_PATH),
            ("L_wristcam", L_WRIST_CAMERA_PATH),
            ("R_wristcam", R_WRIST_CAMERA_PATH),
        ):
            _prim = _stage.GetPrimAtPath(_cam_path) if _stage else None
            if not _prim or not _prim.IsValid():
                print(f"[setup] WARNING: {_label} not found at {_cam_path!r}. "
                      f"Skipping — check that the robot USD authors it.")
                _missing.append(_label)

        _set_camera_focal_length(
            _stage,
            HEAD_DEPTH_CAMERA_PATH,
            getattr(pc, "HEAD_DEPTH_CAMERA_FOCAL_LENGTH", 18.147562),
        )

    if enable_camera_output:
        # Wrap each USD camera as an Isaac Sim Camera sensor. Resolution
        # is the sensor buffer size (not authored on the USD prim).
        head_depth_camera = Camera(
            prim_path=HEAD_DEPTH_CAMERA_PATH,
            name="HeadDepthCam",
            resolution=(640, 480),
            frequency=30.0,
        )
        L_wrist_camera = Camera(
            prim_path=L_WRIST_CAMERA_PATH,
            name="LWristCam",
            resolution=(640, 480),
            frequency=30.0,
        )
        R_wrist_camera = Camera(
            prim_path=R_WRIST_CAMERA_PATH,
            name="RWristCam",
            resolution=(640, 480),
            frequency=30.0,
        )
        for cam in (head_depth_camera, L_wrist_camera, R_wrist_camera):
            cam.initialize()
            cam.add_distance_to_image_plane_to_frame()

    if enable_camera_viewports:
        # 3-tile viewport layout in Kit UI, stacked along the left edge.
        # Independent of sensor binding.
        create_viewport_for_camera(viewport_name="Head Depth View",  camera_prim_path=HEAD_DEPTH_CAMERA_PATH,
                                   width=240, height=200, position_x=50, position_y=50)
        create_viewport_for_camera(viewport_name="L Wrist View",     camera_prim_path=L_WRIST_CAMERA_PATH,
                                   width=240, height=200, position_x=50, position_y=250)
        create_viewport_for_camera(viewport_name="R Wrist View",     camera_prim_path=R_WRIST_CAMERA_PATH,
                                   width=240, height=200, position_x=50, position_y=450)
    # try:
    #     active_viewport = get_active_viewport_window()
    #     if active_viewport:
    #         omni.kit.commands.execute(
    #             "SetViewportCamera",
    #             camera_path="/OmniverseKit_Persp",
    #             viewport_api=active_viewport.viewport_api,
    #         )
    #         omni.kit.commands.execute(
    #             "ChangeProperty",
    #             prop_path=Sdf.Path("/OmniverseKit_Persp.focalLength"),
    #             value=50.0,
    #             prev=5.0,
    #         )
    #         persp_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/OmniverseKit_Persp")
    #         mat = UsdGeom.XformCache().GetLocalToWorldTransform(persp_prim)
    #         current_eye = np.array(mat.ExtractTranslation())
    #         new_eye = current_eye + np.array([-0.5, 0.0, 0.0])
    #         set_camera_view(eye=new_eye, target=np.array([0.0, 0.0, 1.0]),
    #                         camera_prim_path="/OmniverseKit_Persp")
    # except Exception as e:
    #     carb.log_warn(f"Could not set active viewport camera to perspective: {e}")

    my_controller = {"L": L_controller, "R": R_controller}
    my_robots = {"L": my_L_arm, "R": my_R_arm}
    reset_needed = False
    return (
        my_world,
        my_controller,
        my_robots,
        head_depth_camera,
        L_wrist_camera,
        R_wrist_camera,
        articulation_controller,
        task_params,
        reset_needed,
    )


def setup_pick_place_sim(
    L_object_prim_path: str,
    R_object_prim_path: str,
    L_target_position: np.ndarray,
    R_target_position: np.ndarray,
    end_effector_initial_height: Optional[float] = None,
    joint_opened_position: Optional[np.ndarray] = None,
    joint_closed_position: Optional[np.ndarray] = None,
    events_dt: Optional[list] = None,
    total_t: Optional[float] = None,
    flag_robot_state_update: bool = False,
    enable_camera_viewports: bool = True,
    enable_camera_output: bool = True,
    base_translation_offset=None,
    board_offset=None,
    part_offsets=None,
    preplaced_success_parts=None,
):
    """Build the 2-part bimanual pick-and-place world from the pre-built scene.

    ``board_offset`` and ``part_offsets`` are optional trial-level XY shifts;
    when provided they are applied before World snapshots the initial poses.
    ``preplaced_success_parts`` optionally describes all earlier canonical
    parts, which are placed at their successful endpoints before that snapshot.

    Returns
    -------
    my_world : isaacsim World
    my_controller : dict {"L": PickPlaceController, "R": PickPlaceController}
    my_robots : dict {"L": SingleManipulator, "R": SingleManipulator}
    head_depth_camera : Camera sensor wrapping the USD-authored headcam, or None
    L_wrist_camera, R_wrist_camera : Camera sensors wrapping the USD-authored
        wrist cams, or None when ``enable_camera_output`` is False
    articulation_controller : the shared ArticulationController for /vega_1u
    task_params : dict from PickPlaceTask_scene_bimanual.get_params()
    reset_needed : bool, always False on first call.
    """
    my_world = open_scene_and_world(
        board_offset=board_offset,
        part_offsets=part_offsets,
        preplaced_success_parts=preplaced_success_parts,
    )

    my_task = PickPlaceTask_scene_bimanual(
        name=TASK_NAME,
        L_object_prim_path=L_object_prim_path,
        R_object_prim_path=R_object_prim_path,
        L_target_position=L_target_position,
        R_target_position=R_target_position,
        joint_opened_position=joint_opened_position,
        joint_closed_position=joint_closed_position,
    )
    my_world.add_task(my_task)

    return _finalize_pick_place_setup(
        my_world,
        task_name=TASK_NAME,
        end_effector_initial_height=end_effector_initial_height,
        events_dt=events_dt,
        total_t=total_t,
        flag_robot_state_update=flag_robot_state_update,
        enable_camera_viewports=enable_camera_viewports,
        enable_camera_output=enable_camera_output,
        base_translation_offset=base_translation_offset,
    )
