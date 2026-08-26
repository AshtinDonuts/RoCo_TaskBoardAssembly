"""Privileged asset-centroid scripted policy (V1: gear_20teeth only).

V1 drives the **right** gripper (`env_info.R_controller`, `active_arms=("R",)`).
The policy reads live USD geometry at reset and plans a smooth top-down path.

Gripper close/open mode is configured by ``gripper.mode`` in
``config/asset_centroid_policy.json``:

- ``compliant`` (default): approach/release use ``parts.*.gripper_open_rad``
  (same idea as baseline ``part_grasp_open_rad``); close uses ``"close"`` →
  GripperCompliance slow-close toward 0 with stall hold (soft drives).
- ``aperture``: numeric ``parts.*.gripper_*_rad`` for both open and close.

Neither mode calls the Design-D geometric aperture resolver.
"""

from __future__ import annotations

import json
import os.path
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

try:  # Avoid bootstrapping Kit merely by importing this module offline.
    if "isaacsim" not in sys.modules:
        raise ModuleNotFoundError
    from isaacsim.core.utils.types import ArticulationAction
except ModuleNotFoundError:  # pragma: no cover - exercised only outside Isaac

    @dataclass
    class ArticulationAction:  # type: ignore[no-redef]
        joint_positions: list


from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402
from policies.asset_centroid_motion import (  # noqa: E402
    AssetCentroidConfig,
    ee_position_for_grasp_center,
    load_asset_centroid_config,
    local_aabb_midpoint,
    quat_angle,
    rotate_vector,
    sample_joint_segment,
    sample_pose_segment,
    top_down_yaw_candidates,
    top_down_yaw_quat,
    unwrap_revolute_delta,
)


SUPPORTED_PART = "gear_20teeth"
PICK_DROP_PARTS = frozenset(
    {"gear_20teeth", "gear_60teeth", "battery_size1", "battery_size5"}
)
SNAP_PARTS = frozenset({"rod_16mm", "bolt_8mm", "usb_a", "hdmi", "pin"})


@dataclass(frozen=True)
class _Phase:
    name: str
    kind: str
    pos: np.ndarray
    orn: np.ndarray
    gripper: float | str
    dwell_steps: int = 0


class AssetCentroidScriptedPolicy(Policy):
    """Smooth privileged-centroid pick/drop policy for gear_20teeth (right arm)."""

    # Harness default is left-only; V1 must command the right gripper.
    active_arms = ("R",)

    def __init__(
        self,
        env_info: EnvInfo,
        *,
        config: AssetCentroidConfig | None = None,
        config_path: str | None = None,
    ) -> None:
        super().__init__(env_info)
        self._cfg = config or load_asset_centroid_config(config_path)
        arm = self._cfg.active_arm
        controller = getattr(env_info, f"{arm}_controller", None)
        if controller is None:
            raise ValueError(
                f"AssetCentroidScriptedPolicy requires env_info.{arm}_controller "
                f"(from {self._cfg.path})"
            )
        self._controller = controller
        self.active_arms = (arm,)
        self._n_dof = len(env_info.dof_names)
        self._dt = float(env_info.physics_dt)
        self._phases: list[_Phase] = []
        self._phase_index = 0
        self._phase_ticks = 0
        self._segment = ()
        self._segment_q = ()
        self._segment_index = 0
        self._final_hold_steps = 0
        self._ik_failure_steps = 0
        self._done = True
        self._aborted = False
        self._last_action = self._noop()
        self._part_spec = None
        print(
            f"[asset_centroid] config={self._cfg.path} "
            f"arm={arm} "
            f"v_lin={self._cfg.motion.max_linear_speed_m_s:g} m/s "
            f"v_ang={np.degrees(self._cfg.motion.max_angular_speed_rad_s):g} deg/s "
            f"t_min={self._cfg.motion.minimum_move_s:g} s",
            flush=True,
        )

    def _noop(self) -> ArticulationAction:
        return ArticulationAction(joint_positions=[None] * self._n_dof)

    def _abort(self, message: str) -> None:
        if not self._aborted:
            print(f"[asset_centroid] ABORT: {message}", flush=True)
        self._aborted = True
        self._done = True
        self._last_action = self._noop()

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._phases = []
        self._phase_index = 0
        self._phase_ticks = 0
        self._segment = ()
        self._segment_q = ()
        self._segment_index = 0
        self._final_hold_steps = 0
        self._ik_failure_steps = 0
        self._done = True
        self._aborted = False
        self._last_action = self._noop()
        self._part_spec = None

        if target.release_mode == "snap":
            print(f"[asset_centroid] skip snap target {target.name!r}", flush=True)
            return
        if target.name != SUPPORTED_PART:
            print(
                f"[asset_centroid] skip unsupported V1 target {target.name!r}",
                flush=True,
            )
            return
        if target.place_pos is None:
            self._abort(f"{target.name}: missing scripted place_pos")
            return

        try:
            self._part_spec = self._cfg.part(target.name)
            centroid_world, asset_offset_world = self._live_asset_centroid(target.name)
            grasp_center = centroid_world + asset_offset_world
            print(
                f"[asset_centroid] localized {target.name}: centroid="
                f"{np.round(centroid_world, 6).tolist()} asset_offset_world="
                f"{np.round(asset_offset_world, 6).tolist()} grasp_center="
                f"{np.round(grasp_center, 6).tolist()} place_center="
                f"{np.round(target.place_pos, 6).tolist()}",
                flush=True,
            )
            pick_orn, place_orn, poses = self._choose_yaw_and_poses(
                grasp_center, np.asarray(target.place_pos, dtype=np.float64)
            )
        except Exception as exc:
            self._abort(f"{target.name}: localization/planning failed: {exc}")
            return

        timing = self._cfg.timing
        close_steps = max(1, int(np.ceil(timing.close_dwell_s / self._dt)))
        settle_steps = max(1, int(np.ceil(timing.settle_place_s / self._dt)))
        open_steps = max(1, int(np.ceil(timing.open_dwell_s / self._dt)))
        open_cmd, close_cmd = self._gripper_cmds()
        self._phases = [
            _Phase(
                "hover_pick",
                "move",
                poses["hover_pick"],
                pick_orn,
                open_cmd,
            ),
            _Phase(
                "descend_pick", "move", poses["pick"], pick_orn, open_cmd
            ),
            _Phase(
                "close",
                "dwell",
                poses["pick"],
                pick_orn,
                close_cmd,
                close_steps,
            ),
            _Phase(
                "lift_pick", "move", poses["lift_pick"], pick_orn, close_cmd
            ),
            _Phase(
                "hover_place",
                "move",
                poses["hover_place"],
                place_orn,
                close_cmd,
            ),
            _Phase(
                "descend_place", "move", poses["place"], place_orn, close_cmd
            ),
            _Phase(
                "settle_place",
                "dwell",
                poses["place"],
                place_orn,
                close_cmd,
                settle_steps,
            ),
            _Phase(
                "open",
                "dwell",
                poses["place"],
                place_orn,
                open_cmd,
                open_steps,
            ),
            _Phase("retract", "move", poses["retract"], place_orn, open_cmd),
        ]
        self._done = False
        print(
            f"[asset_centroid] {target.name}: centroid="
            f"{np.round(centroid_world, 5).tolist()} grasp_center="
            f"{np.round(grasp_center, 5).tolist()} "
            f"gripper.mode={self._cfg.gripper.mode} "
            f"open={open_cmd!r} close={close_cmd!r} "
            f"yaw-ready phases={len(self._phases)}",
            flush=True,
        )

    def _gripper_cmds(self) -> tuple[float | str, float | str]:
        """Return (open_cmd, close_cmd) for EEPoseController.forward."""
        if self._part_spec is None:
            raise RuntimeError("part spec not loaded")
        open_rad = float(self._part_spec.gripper_open_rad)
        if self._cfg.gripper.mode == "compliant":
            # Baseline-style approach aperture + soft stall close (→ 0 rad).
            return open_rad, "close"
        return open_rad, float(self._part_spec.gripper_close_rad)

    def _part_paths(self, name: str) -> tuple[str, Optional[str]]:
        path = os.path.join(_TASK_DIR, "part_init_poses.json")
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        entry = data.get(name)
        if not isinstance(entry, dict) or not entry.get("path"):
            raise KeyError(f"no part path metadata for {name!r}")
        return str(entry["path"]), entry.get("mesh_path")

    def _live_asset_centroid(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return live world AABB centroid and configured offset in world."""
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable")
        root_path, preferred_mesh_path = self._part_paths(name)
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"missing asset root {root_path}")

        if preferred_mesh_path:
            preferred = stage.GetPrimAtPath(preferred_mesh_path)
            if not preferred or not preferred.IsValid():
                raise RuntimeError(f"missing configured mesh {preferred_mesh_path}")
        meshes = [p for p in Usd.PrimRange(root) if p.GetTypeName() == "Mesh"]
        if not meshes:
            raise RuntimeError(f"no mesh descendants under {root_path}")

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        root_world = cache.GetLocalToWorldTransform(root)
        root_world_inv = root_world.GetInverse()
        points_root = []
        for mesh_prim in meshes:
            points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
            if not points:
                continue
            # Gf uses row-vector matrix composition: local-to-root is
            # mesh-to-world followed by world-to-root.
            mesh_to_root = cache.GetLocalToWorldTransform(mesh_prim) * root_world_inv
            for point in points:
                transformed = mesh_to_root.Transform(Gf.Vec3d(point))
                points_root.append(tuple(float(v) for v in transformed))
        centroid_local = local_aabb_midpoint(points_root)
        centroid_gf = root_world.Transform(Gf.Vec3d(*centroid_local.tolist()))
        centroid_world = np.asarray(tuple(centroid_gf), dtype=np.float64)

        offset = self._cfg.part(name).centroid_grasp_offset_asset
        offset_gf = root_world.TransformDir(Gf.Vec3d(*offset))
        offset_world = np.asarray(tuple(offset_gf), dtype=np.float64)
        return centroid_world, offset_world

    def _pose_set(
        self,
        grasp_center: np.ndarray,
        place_center: np.ndarray,
        pick_orn: np.ndarray,
        place_orn: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        if place_orn is None:
            place_orn = pick_orn
        if self._part_spec is None:
            raise RuntimeError("part spec not loaded")
        tcp = self._part_spec.tcp_to_grasp_tool
        clear = self._cfg.path_clearances
        frame = clear.tcp_offset_frame
        pick = ee_position_for_grasp_center(
            grasp_center, pick_orn, tcp, offset_frame=frame
        )
        place = ee_position_for_grasp_center(
            place_center, place_orn, tcp, offset_frame=frame
        )
        return {
            "pick": pick,
            "hover_pick": pick + np.array([0.0, 0.0, clear.hover_pick_m]),
            "lift_pick": pick + np.array([0.0, 0.0, clear.hover_pick_m]),
            "place": place,
            "hover_place": place + np.array([0.0, 0.0, clear.hover_place_m]),
            "retract": place + np.array([0.0, 0.0, clear.final_retract_m]),
        }

    def _yaw_candidates(self) -> tuple[np.ndarray, ...]:
        clear = self._cfg.path_clearances
        if clear.force_yaw_deg is not None:
            return (top_down_yaw_quat(clear.force_yaw_deg),)
        return top_down_yaw_candidates(clear.yaw_step_deg)

    def _prismatic_mask(self) -> np.ndarray:
        names = list(self._controller.cspace_joint_names())
        return np.asarray([name == "Lift" for name in names], dtype=bool)

    def _joint_step_cost(self, q_from: np.ndarray, q_to: np.ndarray) -> float:
        delta = unwrap_revolute_delta(
            q_from, q_to, prismatic_mask=self._prismatic_mask()
        )
        return float(np.dot(delta, delta))

    def _choose_yaw_and_poses(
        self,
        grasp_center: np.ndarray,
        place_center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Pick/place yaw from keyframe endpoint IK only.

        Dense Cartesian transfer is intentionally not required: like the
        baseline transit, runtime hover_place uses joint-space quintic lerp
        between lift_pick and hover_place solutions so Lula cannot flip
        wrist branches at midpoints.
        """
        seed0 = np.asarray(self._controller.current_cspace_q(), dtype=np.float64)
        best = None
        pick_keys = ("hover_pick", "pick", "lift_pick")
        # Do not require retract in preflight: yaw=0 place can be fine while
        # retract IK fails, which previously aborted the whole plan.
        place_keys = ("hover_place", "place")
        failure_counts = {key: 0 for key in pick_keys + place_keys}
        candidates = self._yaw_candidates()
        clear = self._cfg.path_clearances
        print(
            f"[asset_centroid] Test-path: tcp_offset_frame={clear.tcp_offset_frame!r} "
            f"force_yaw_deg={clear.force_yaw_deg} "
            f"n_yaw={len(candidates)} "
            f"tcp={list(self._part_spec.tcp_to_grasp_tool) if self._part_spec else None}",
            flush=True,
        )
        for pick_rank, pick_orn in enumerate(candidates):
            poses = self._pose_set(grasp_center, place_center, pick_orn)
            seed = seed0.copy()
            pick_score = pick_rank * 1e-6
            feasible = True
            for key in pick_keys:
                q, ok = self._controller.solve_q(poses[key], pick_orn, seed=seed)
                if not ok or q is None:
                    failure_counts[key] += 1
                    feasible = False
                    break
                q = np.asarray(q, dtype=np.float64).reshape(-1)
                pick_score += self._joint_step_cost(seed, q)
                seed = q
            if not feasible:
                continue
            lift_seed = seed.copy()
            for place_rank, place_orn in enumerate(candidates):
                pair_poses = self._pose_set(
                    grasp_center, place_center, pick_orn, place_orn
                )
                seed = lift_seed.copy()
                place_score = place_rank * 1e-6
                place_ok = True
                for key in place_keys:
                    q, ok = self._controller.solve_q(
                        pair_poses[key], place_orn, seed=seed
                    )
                    if not ok or q is None:
                        failure_counts[key] += 1
                        place_ok = False
                        break
                    q = np.asarray(q, dtype=np.float64).reshape(-1)
                    place_score += self._joint_step_cost(seed, q)
                    seed = q
                if not place_ok:
                    continue
                score = pick_score + place_score
                if best is None or score < best[0]:
                    best = (
                        score,
                        pick_orn.copy(),
                        place_orn.copy(),
                        pair_poses,
                    )
        if best is None and clear.force_yaw_deg is not None:
            fallback = top_down_yaw_candidates(clear.yaw_step_deg)
            print(
                f"[asset_centroid] forced yaw {clear.force_yaw_deg:g} deg infeasible "
                f"(failures={failure_counts}); falling back to {len(fallback)} yaws",
                flush=True,
            )
            candidates = fallback
            failure_counts = {key: 0 for key in pick_keys + place_keys}
            for pick_rank, pick_orn in enumerate(candidates):
                poses = self._pose_set(grasp_center, place_center, pick_orn)
                seed = seed0.copy()
                pick_score = pick_rank * 1e-6
                feasible = True
                for key in pick_keys:
                    q, ok = self._controller.solve_q(poses[key], pick_orn, seed=seed)
                    if not ok or q is None:
                        failure_counts[key] += 1
                        feasible = False
                        break
                    q = np.asarray(q, dtype=np.float64).reshape(-1)
                    pick_score += self._joint_step_cost(seed, q)
                    seed = q
                if not feasible:
                    continue
                lift_seed = seed.copy()
                for place_rank, place_orn in enumerate(candidates):
                    pair_poses = self._pose_set(
                        grasp_center, place_center, pick_orn, place_orn
                    )
                    seed = lift_seed.copy()
                    place_score = place_rank * 1e-6
                    place_ok = True
                    for key in place_keys:
                        q, ok = self._controller.solve_q(
                            pair_poses[key], place_orn, seed=seed
                        )
                        if not ok or q is None:
                            failure_counts[key] += 1
                            place_ok = False
                            break
                        q = np.asarray(q, dtype=np.float64).reshape(-1)
                        place_score += self._joint_step_cost(seed, q)
                        seed = q
                    if not place_ok:
                        continue
                    score = pick_score + place_score
                    if best is None or score < best[0]:
                        best = (
                            score,
                            pick_orn.copy(),
                            place_orn.copy(),
                            pair_poses,
                        )
        if best is None:
            preferred_poses = self._pose_set(
                grasp_center, place_center, candidates[0]
            )
            pose_text = ", ".join(
                f"{name}={np.round(preferred_poses[name], 5).tolist()}"
                for name in pick_keys
            )
            raise RuntimeError(
                "no top-down yaw candidate has feasible endpoint IK; "
                f"failures={failure_counts}; preferred poses: {pose_text}"
            )
        pick_yaw = self._nominal_yaw_degrees(best[1])
        place_yaw = self._nominal_yaw_degrees(best[2])
        poses = best[3]
        xy_err_pick = float(
            np.linalg.norm(poses["hover_pick"][:2] - grasp_center[:2])
        )
        print(
            "[asset_centroid] selected world-down TCP yaws "
            f"(pick={pick_yaw:.1f} deg, place={place_yaw:.1f} deg) "
            f"on {self._cfg.active_arm} arm; frame={self._cfg.path_clearances.tcp_offset_frame}; "
            f"hover_pick={np.round(poses['hover_pick'], 5).tolist()} "
            f"(|xy−grasp|={xy_err_pick * 1000:.1f} mm); "
            f"hover_place={np.round(poses['hover_place'], 5).tolist()}",
            flush=True,
        )
        return best[1], best[2], best[3]

    @staticmethod
    def _nominal_yaw_degrees(orn: np.ndarray) -> float:
        tool_x_world = rotate_vector(orn, [1.0, 0.0, 0.0])
        return float(np.degrees(np.arctan2(tool_x_world[1], tool_x_world[0])))

    def _actual_ee_pose(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Right EE world pose from the physical prim (obs has no ee_pose_R)."""
        ee = getattr(self._controller, "end_effector", None)
        if ee is not None:
            pos, orn = ee.get_world_pose()
            return (
                np.asarray(pos, dtype=np.float64).reshape(3),
                np.asarray(orn, dtype=np.float64).reshape(4),
            )
        # Offline unit tests inject a last commanded / seed pose on the
        # controller; never fall back to obs.ee_pose_L (left arm).
        seed = getattr(self._controller, "seed_ee_pose", None)
        if seed is not None:
            pos, orn = seed
            return (
                np.asarray(pos, dtype=np.float64).reshape(3),
                np.asarray(orn, dtype=np.float64).reshape(4),
            )
        raise RuntimeError("right EE pose unavailable (no end_effector)")

    def _start_segment(self, obs: Observation, phase: _Phase) -> None:
        start_pos, start_orn = self._actual_ee_pose(obs)
        limits = self._cfg.motion
        self._segment = sample_pose_segment(
            start_pos,
            start_orn,
            phase.pos,
            phase.orn,
            dt=self._dt,
            limits=limits,
        )
        self._segment_index = 0
        self._segment_q = ()
        # lift_pick → hover_place: joint-space transit (baseline lesson).
        if phase.name == "hover_place":
            q0 = np.asarray(
                self._controller.current_cspace_q(), dtype=np.float64
            )
            q1, ok = self._controller.solve_q(phase.pos, phase.orn, seed=q0)
            if not ok or q1 is None:
                self._abort(
                    "could not solve joint-space transfer endpoint at hover_place"
                )
                return
            q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
            self._segment_q = sample_joint_segment(
                q0,
                q1,
                dt=self._dt,
                linear_distance_m=float(np.linalg.norm(phase.pos - start_pos)),
                angular_distance_rad=quat_angle(start_orn, phase.orn),
                prismatic_mask=self._prismatic_mask(),
                limits=limits,
            )
            # Keep pose-segment length aligned for act() indexing.
            self._segment = tuple(
                (phase.pos, phase.orn) for _ in self._segment_q
            )
        self._final_hold_steps = 0
        mode = "joint" if self._segment_q else "cartesian"
        print(
            f"[asset_centroid] phase {phase.name}: "
            f"{len(self._segment)} {mode} steps, target="
            f"{np.round(phase.pos, 5).tolist()}",
            flush=True,
        )

    def _command(self, pos, orn, gripper) -> ArticulationAction:
        action = self._controller.forward(pos, orn, gripper)
        ik_ok = bool(getattr(getattr(self._controller, "ik", None), "ik_ok", True))
        if ik_ok:
            self._ik_failure_steps = 0
            self._last_action = action
        else:
            self._ik_failure_steps += 1
            max_fail = self._cfg.guards.max_ik_failure_steps
            if self._ik_failure_steps >= max_fail:
                self._abort(
                    f"IK failed for {max_fail} consecutive steps "
                    f"in phase {self._phases[self._phase_index].name!r}"
                )
        return self._last_action

    def _log_pose_error(self, obs: Observation, phase: _Phase) -> None:
        actual_pos, actual_orn = self._actual_ee_pose(obs)
        pos_err = float(np.linalg.norm(np.asarray(actual_pos) - phase.pos))
        orn_err = float(quat_angle(actual_orn, phase.orn))
        print(
            f"[asset_centroid] arrive {phase.name}: "
            f"cmd={np.round(phase.pos, 5).tolist()} "
            f"meas={np.round(actual_pos, 5).tolist()} "
            f"pos_err={pos_err * 1000:.1f} mm orn_err={np.degrees(orn_err):.1f} deg",
            flush=True,
        )

    def _advance_phase(self, obs: Optional[Observation] = None) -> None:
        completed = self._phases[self._phase_index].name
        if obs is not None:
            self._log_pose_error(obs, self._phases[self._phase_index])
        self._phase_index += 1
        self._phase_ticks = 0
        self._segment = ()
        self._segment_q = ()
        self._segment_index = 0
        self._final_hold_steps = 0
        if self._phase_index >= len(self._phases):
            self._done = True
            self._last_action = self._noop()
            print("[asset_centroid] trajectory complete", flush=True)
        else:
            print(f"[asset_centroid] phase {completed} complete", flush=True)

    def act(self, obs: Observation) -> ArticulationAction:
        if self._done:
            return self._noop()
        phase = self._phases[self._phase_index]
        if phase.kind == "dwell":
            action = self._command(phase.pos, phase.orn, phase.gripper)
            if self._done:
                return self._noop()
            if self._ik_failure_steps == 0:
                self._phase_ticks += 1
                if self._phase_ticks >= phase.dwell_steps:
                    self._advance_phase(obs)
            return action

        if not self._segment:
            self._start_segment(obs, phase)
            if self._done:
                return self._noop()
        if self._segment_index < len(self._segment):
            pos, orn = self._segment[self._segment_index]
            if self._segment_q:
                action = self._controller.forward_raw_q(
                    self._segment_q[self._segment_index], phase.gripper
                )
                self._last_action = action
                self._ik_failure_steps = 0
            else:
                action = self._command(pos, orn, phase.gripper)
            if self._done:
                return self._noop()
            if self._ik_failure_steps == 0:
                self._segment_index += 1
            return action

        action = self._command(phase.pos, phase.orn, phase.gripper)
        if self._done:
            return self._noop()
        actual_pos, actual_orn = self._actual_ee_pose(obs)
        guards = self._cfg.guards
        reached = (
            np.linalg.norm(np.asarray(actual_pos) - phase.pos) <= guards.pos_tol_m
            and quat_angle(actual_orn, phase.orn) <= guards.orn_tol_rad
        )
        if reached:
            self._advance_phase(obs)
        else:
            self._final_hold_steps += 1
            if self._final_hold_steps >= guards.max_final_hold_steps:
                self._abort(
                    f"phase {phase.name!r} did not converge; refusing to "
                    "continue into a grasp/release action"
                )
        return action

    def is_done(self, obs: Observation) -> bool:
        return self._done

    @property
    def current_waypoint(self):
        if self._done or self._phase_index >= len(self._phases):
            return None
        return self._phases[self._phase_index]

    @property
    def current_index(self) -> int:
        return self._phase_index
