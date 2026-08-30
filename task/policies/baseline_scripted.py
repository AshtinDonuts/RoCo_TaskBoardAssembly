"""Reference scripted policy for the IROS 2026 vega_1u assembly challenge.

Wraps the original `EEPathFollower`-driven pick-and-place: per part, build
a 9-phase EE-pose path from `PART_CONFIG`, drive Lula IK to follow it,
gate the snap_wait waypoint on `obs.snap_fired`.

Cartesian pacing (see `CARTESIAN_MAX_EE_SPEED_M_S`) walks the commanded
EE pose toward each terminal waypoint before each IK query, so motion is
physically slower without a separate planner. Close/open still fire only
at the original terminals.

This is the normal scripted baseline. Participants should not modify this
file; copy `template.py` instead.

Under fairness XY randomization, the harness normally supplies nominal
`PartTarget` waypoints, so this open-loop policy is expected to miss. The
development-only `--privileged-xy-randomization` flag shifts those waypoints
to generate expert rollouts. A competition policy must instead estimate the
part and board offsets from `Observation.rgb` / `Observation.depth`.
"""
from __future__ import annotations

import os.path
import sys

# Ensure `task/` is on sys.path so absolute imports of `param_config` and
# `controllers.*` work whether the policy is loaded as a top-level module
# or as `policies.baseline_scripted`.
_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402

import param_config as pc  # noqa: E402
from controllers.ee_pose_controller import (  # noqa: E402
    EEPathFollower,
    build_pick_place_phases,
)
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402
from policies._joint_rate_limiter import JointPositionRateLimiter  # noqa: E402


def make_l_path_for_part(part_name, target=None, snap_advance_when=None,
                         snap_timeout_steps=None, return_home_q=None,
                         safe_retract_pos=None, safe_retract_orn=None):
    """Build the 9-phase L path for one part using its PART_CONFIG entry.

    Pick/place world OBJECT positions are offset by `cfg["ee_offset"]` to
    get the EE-frame target positions. Orientations and per-part gripper
    open/close joint values are pulled from cfg too. Returns the (possibly
    `MAX_PHASES`-truncated) waypoint list, or `[]` if both pick_pos and
    place_pos are None.

    When `cfg["release_mode"] == "snap"`, `snap_advance_when` must be a
    callable returning True once the snap fires; the phase builder inserts
    a snap_wait waypoint between descend_place and open and gates the
    advance on it. `snap_timeout_steps` is the timeout fall-through.

    `return_home_q` (when set) prepends a `return_home` waypoint that
    drives the L arm joints directly to that c-space vector before
    starting hover_pick. The gripper is commanded to this part's
    `gripper_open` value during the return so it's at the right opening
    by the time the new pick begins.

    When `safe_retract_pos` is also set, a Cartesian `safe_retract`
    waypoint is inserted before `return_home`: raise the EE vertically
    (holding XY / orientation) clear of the table under existing
    Cartesian pacing, then begin the joint-space home motion.
    """
    # The harness PartTarget is nominal in competition mode and shifted only
    # under the explicit privileged development flag. In both cases consume
    # it verbatim; camera-aware policies should adjust a private copy first.
    cfg = (dict(target.extra) if target is not None and target.extra
           else pc.get_part_config(part_name))
    ee_off = np.asarray(cfg["ee_offset"], dtype=np.float64)
    orn_value = (target.ee_orientation if target is not None
                 and target.ee_orientation is not None
                 else cfg.get("ee_orientation"))
    orn = np.asarray(orn_value, dtype=np.float64)
    pick_pos = (target.pick_pos if target is not None else cfg.get("pick_pos"))
    place_pos = (target.place_pos if target is not None else cfg.get("place_pos"))
    pick_pos_ee = (
        None if pick_pos is None
        else np.asarray(pick_pos, dtype=np.float64) + ee_off
    )
    place_pos_ee = (
        None if place_pos is None
        else np.asarray(place_pos, dtype=np.float64) + ee_off
    )
    init_height = (cfg.get("init_height")
                   if cfg.get("init_height") is not None
                   else pc.INIT_HEIGHT)
    transit_steps = (cfg.get("transit_steps")
                     if cfg.get("transit_steps") is not None
                     else pc.TRANSIT_STEPS)
    final_height = (cfg.get("final_height")
                    if cfg.get("final_height") is not None
                    else getattr(pc, "FINAL_HEIGHT", None))
    full = build_pick_place_phases(
        pick_pos=pick_pos_ee,
        pick_orn=orn if pick_pos_ee is not None else None,
        place_pos=place_pos_ee,
        place_orn=orn if place_pos_ee is not None else None,
        init_height=init_height,
        final_height=final_height,
        include_close=pc.INCLUDE_CLOSE,
        include_open=pc.INCLUDE_OPEN,
        settle_close_steps=pc.SETTLE_CLOSE,
        settle_open_steps=pc.SETTLE_OPEN,
        settle_hover_place_steps=pc.SETTLE_HOVER_PLACE,
        settle_descend_place_steps=pc.SETTLE_DESCEND_PLACE,
        transit_steps=int(transit_steps),
        descend_pick_steps=pc.DESCEND_PICK_STEPS,
        descend_place_steps=pc.DESCEND_PLACE_STEPS,
        gripper_open_value=cfg.get("gripper_open"),
        gripper_close_value=cfg.get("gripper_close"),
        release_mode=(target.release_mode if target is not None
                      else cfg.get("release_mode", "open")),
        snap_advance_when=snap_advance_when,
        snap_timeout_steps=snap_timeout_steps,
        snap_search_n=int(((cfg.get("snap") or {}).get("search") or {})
                          .get("n", 0)),
        snap_search_extent_xy=tuple(((cfg.get("snap") or {}).get("search") or {})
                                    .get("extent_xy", (0.002, 0.002))),
        snap_search_dwell_steps=int(((cfg.get("snap") or {}).get("search") or {})
                                    .get("dwell_steps", 1)),
        return_home_q=return_home_q,
        return_home_gripper=cfg.get("gripper_open"),
        return_home_cspace_tol=getattr(pc, "RETURN_HOME_CSPACE_TOL", None),
        return_home_settle_steps=getattr(pc, "RETURN_HOME_SETTLE_STEPS", 20),
        return_home_max_velocity=getattr(
            pc, "RETURN_HOME_MAX_JOINT_VELOCITY_RAD_S", None
        ),
        return_home_max_acceleration=getattr(
            pc, "RETURN_HOME_MAX_JOINT_ACCELERATION_RAD_S2", None
        ),
        return_home_min_duration=getattr(
            pc, "RETURN_HOME_MIN_DURATION_S", None
        ),
        safe_retract_pos=safe_retract_pos,
        safe_retract_orn=safe_retract_orn,
        safe_retract_settle_steps=getattr(
            pc, "SAFE_RETRACT_SETTLE_STEPS", 0
        ),
    )
    if pc.MAX_PHASES is not None and pc.MAX_PHASES < len(full):
        return full[:int(pc.MAX_PHASES)]
    return full


def _cartesian_step_limits(env_info: EnvInfo):
    """Convert configured EE speeds to per-step caps using physics_dt."""
    physics_dt = float(getattr(env_info, "physics_dt", 1.0 / 200.0))
    if not np.isfinite(physics_dt) or physics_dt <= 0.0:
        physics_dt = 1.0 / 200.0
    max_speed = getattr(pc, "CARTESIAN_MAX_EE_SPEED_M_S", None)
    max_orn_speed = getattr(pc, "CARTESIAN_MAX_EE_ORN_SPEED_RAD_S", None)
    max_ee_step_m = (
        None if max_speed is None else float(max_speed) * physics_dt
    )
    max_ee_orn_step_rad = (
        None if max_orn_speed is None else float(max_orn_speed) * physics_dt
    )
    return max_ee_step_m, max_ee_orn_step_rad


class BaselinePolicy(Policy):
    """Scripted EE-path follower over each part's 9-phase pick-place plan.

    Requires `env_info.L_controller` (the harness sets it automatically).
    Other policies do not need this controller — they can produce joint
    targets directly.
    """

    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        L_controller = getattr(env_info, "L_controller", None)
        if L_controller is None:
            raise ValueError(
                "BaselinePolicy requires env_info.L_controller (an "
                "EEPoseController). The harness sets this by default; "
                "if you see this error, you may be running the policy "
                "outside the provided harness."
            )
        self._L_controller = L_controller
        max_ee_step_m, max_ee_orn_step_rad = _cartesian_step_limits(env_info)
        self._follower = EEPathFollower(
            L_controller,
            position_tolerance=getattr(pc, "POS_TOL", 0.005),
            orientation_tolerance=getattr(pc, "ORN_TOL", 0.05),
            default_timeout_steps=getattr(pc, "WAYPOINT_TIMEOUT_STEPS", None),
            max_ee_step_m=max_ee_step_m,
            max_ee_orn_step_rad=max_ee_orn_step_rad,
        )
        self._is_first_part = True
        self._max_joint_velocity = float(getattr(
            pc, "BASELINE_MAX_JOINT_VELOCITY_RAD_S", 0.5
        ))
        arm_indices = [env_info.dof_names.index(name)
                       for name in env_info.L_arm_joints]
        fallback_dt = float(getattr(
            pc, "BASELINE_CONTROL_DT_FALLBACK_S", 0.1
        ))
        self._control_dt_fallback = fallback_dt
        self._joint_rate_limiter = JointPositionRateLimiter(
            arm_indices,
            max_delta=self._max_joint_velocity * fallback_dt,
        )
        # _last_obs is read by the snap_advance_when closure, which is
        # invoked by EEPathFollower.step() to decide whether to advance
        # past the snap_wait waypoint. Updated every act() call.
        self._last_obs: Observation = None  # type: ignore[assignment]
        self._last_step_idx = None

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._last_obs = obs
        self._last_step_idx = int(obs.step_idx)
        self._joint_rate_limiter.reset(obs.joint_positions)

        snap_advance_when = None
        snap_timeout_steps = None
        if target.release_mode == "snap":
            snap_advance_when = self._read_snap_fired
            snap_cfg = target.extra.get("snap") if target.extra else None
            if snap_cfg is not None:
                snap_timeout_steps = snap_cfg.get("timeout_steps")

        return_home_q = (None if self._is_first_part
                         else self.env_info.L_arm_init_q)
        self._is_first_part = False

        # Optional Cartesian retreat before joint-space return_home
        # (ENABLE_SAFE_RETRACT). Even a smooth c-space path can sweep the
        # elbow/wrist through the table when started low.
        safe_retract_pos = None
        safe_retract_orn = None
        if (return_home_q is not None
                and bool(getattr(pc, "ENABLE_SAFE_RETRACT", False))):
            ee_pos, ee_orn = obs.ee_pose_L
            ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(-1)
            ee_orn = np.asarray(ee_orn, dtype=np.float64).reshape(-1)
            table_z = float(getattr(pc, "TABLE_Z", 1.0))
            clearance = float(getattr(pc, "SAFE_RETRACT_CLEARANCE_M", 0.35))
            safe_z = max(float(ee_pos[2]), table_z + clearance)
            safe_retract_pos = np.array(
                [ee_pos[0], ee_pos[1], safe_z], dtype=np.float64
            )
            safe_retract_orn = ee_orn

        path = make_l_path_for_part(
            target.name,
            target=target,
            snap_advance_when=snap_advance_when,
            snap_timeout_steps=snap_timeout_steps,
            return_home_q=return_home_q,
            safe_retract_pos=safe_retract_pos,
            safe_retract_orn=safe_retract_orn,
        )
        self._follower.reset()
        self._follower.set_path(path)

    def act(self, obs: Observation):
        step_idx = int(obs.step_idx)
        dt = None
        if self._last_step_idx is not None and step_idx > self._last_step_idx:
            dt = ((step_idx - self._last_step_idx)
                  * float(self.env_info.physics_dt))
        self._last_step_idx = step_idx
        self._last_obs = obs
        action = self._follower.step(dt=dt)
        effective_dt = self._control_dt_fallback if dt is None else dt
        return self._joint_rate_limiter.apply(
            action,
            max_delta=self._max_joint_velocity * effective_dt,
        )

    def is_done(self, obs: Observation) -> bool:
        self._last_obs = obs
        return self._follower.is_done()

    def _read_snap_fired(self) -> bool:
        return bool(self._last_obs is not None
                    and self._last_obs.snap_fired)

    # Diagnostic accessors used by the harness's stuck detector.
    @property
    def current_waypoint(self):
        return self._follower.current_waypoint()

    @property
    def current_index(self) -> int:
        return self._follower.current_index()
