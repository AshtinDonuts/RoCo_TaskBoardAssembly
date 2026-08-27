"""Privileged asset-centroid scripted policy for configured task parts.

V1 drives the **right** gripper (`env_info.R_controller`, `active_arms=("R",)`).
The policy reads live USD geometry at reset and plans a smooth top-down path.

Gripper close/open mode is configured by ``gripper.mode`` in
``config/asset_centroid_policy.json``:

- ``compliant`` (default): approach/release use ``parts.*.gripper_open_rad``;
  close uses GripperCompliance to slew toward ``parts.*.gripper_close_rad``
  and stall-hold earlier on contact. The configured aperture prevents a late
  stall from driving all the way toward 0.
- ``aperture``: numeric ``parts.*.gripper_*_rad`` for both open and close.

Explicit centroid specs use their JSON apertures. Parts without an explicit
spec use the Design-D aperture resolver from ``param_config``.
"""

from __future__ import annotations

import json
import os
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
    AssetMotionSpec,
    ee_position_for_grasp_center,
    load_asset_centroid_config,
    local_aabb_midpoint,
    quat_angle,
    quat_mul_wxyz,
    rotate_vector,
    sample_joint_segment,
    sample_pose_segment,
    top_down_yaw_candidates,
    top_down_yaw_quat,
    unwrap_revolute_delta,
)


@dataclass(frozen=True)
class _Phase:
    name: str
    kind: str
    pos: np.ndarray
    orn: np.ndarray
    gripper: float | str
    dwell_steps: int = 0
    orientation_cone_rad: float | None = None


class AssetCentroidScriptedPolicy(Policy):
    """Smooth privileged-centroid pick/drop policy for any configured part."""

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
        self._target: Optional[PartTarget] = None
        self._hover_pick_q: Optional[np.ndarray] = None
        self._hover_place_q: Optional[np.ndarray] = None
        self._place_q: Optional[np.ndarray] = None
        self._hold_q: Optional[np.ndarray] = None
        self._compliance_log_accum_s = 0.0
        self._approach_diagnostics: dict[str, object] = {}
        self._configure_compliance()
        print(
            f"[asset_centroid] config={self._cfg.path} "
            f"arm={arm} "
            f"v_lin={self._cfg.motion.max_linear_speed_m_s:g} m/s "
            f"v_ang={np.degrees(self._cfg.motion.max_angular_speed_rad_s):g} deg/s "
            f"t_min={self._cfg.motion.minimum_move_s:g} s",
            flush=True,
        )
        self._log_compliance_params(prefix="init")

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
        self._target = target
        self._hover_pick_q = None
        self._hover_place_q = None
        self._place_q = None
        self._hold_q = None
        self._compliance_log_accum_s = 0.0
        self._approach_diagnostics = {}

        if target.place_pos is None:
            self._abort(f"{target.name}: missing scripted place_pos")
            return

        try:
            self._part_spec = self._part_spec_for_target(target)
            self._configure_compliance()
            # Fresh compliance state so approach→close can latch HOLDING.
            comp = self._compliance()
            if comp is not None:
                meas_fn = getattr(self._controller, "_measured_gripper", None)
                q0 = None
                if callable(meas_fn):
                    try:
                        q0, _ = meas_fn()
                    except Exception:
                        q0 = None
                comp.reset(None if q0 is None else float(q0))
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
        relaxed = self._relaxed_approach_enabled()
        phases = [
            _Phase(
                "hover_pick_relaxed" if relaxed else "hover_pick",
                "move",
                poses["hover_pick"],
                pick_orn,
                open_cmd,
                orientation_cone_rad=(
                    self._cfg.approach_orientation.max_tilt_rad if relaxed else None
                ),
            ),
        ]
        if relaxed:
            # Relaxation is strictly approach-only: recover the chosen
            # top-down orientation at fixed hover XYZ before descending.
            phases.append(
                _Phase("align_hover_pick", "move", poses["hover_pick"], pick_orn, open_cmd)
            )
        phases.extend([
            _Phase("descend_pick", "move", poses["pick"], pick_orn, open_cmd),
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
            _Phase("hover_place", "move", poses["hover_place"], place_orn, close_cmd),
        ])
        # Null/near-null place descent (e.g. hover_place_height≈0) still used
        # to run a long joint segment and caused sideways EE wander; skip it.
        descend_delta = float(np.linalg.norm(poses["place"] - poses["hover_place"]))
        if descend_delta > 1e-4:
            phases.append(
                _Phase(
                    "descend_place",
                    "move",
                    poses["place"],
                    place_orn,
                    close_cmd,
                )
            )
        else:
            print(
                f"[asset_centroid] skip descend_place "
                f"(||place−hover_place||={descend_delta * 1000:.2f} mm)",
                flush=True,
            )
        phases.extend(
            [
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
        )
        self._phases = phases
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
        self._render_target_markers(
            part_name=target.name,
            centroid_world=centroid_world,
            grasp_center=grasp_center,
            poses=poses,
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

    def _part_spec_for_target(self, target: PartTarget) -> AssetMotionSpec:
        """Resolve a motion spec without duplicating every param-config entry.

        Explicit JSON entries remain available for measured, part-specific
        offsets. All other configured parts derive their tool offset and
        aperture from the resolved target that the runner provides.
        """
        try:
            return self._cfg.part(target.name)
        except KeyError:
            pass

        extra = target.extra or {}
        ee_offset = np.asarray(
            extra.get("ee_offset", (0.0, 0.016, 0.196)), dtype=np.float64
        ).reshape(-1)
        if ee_offset.shape != (3,):
            raise ValueError(f"{target.name}: ee_offset must be a length-3 vector")

        # param_config ee_offset is left-arm / world-frame tuned (Y ≈ +0.016):
        # baseline does EE = grasp + ee_offset. This policy uses tool-frame
        # EE = grasp - R(q)·tcp with top-down q = 180° about X.
        #
        # Matching L world offset would keep tcp.y = +ee_offset.y, but the R
        # fingertip in R_ee_link_gripper_link is at Y ≈ -0.0144 (mesh+URDF).
        # Keeping +Y here reproduces the pre-fix gear_20 lateral miss on R.
        ex, ey, ez = (float(ee_offset[0]), float(ee_offset[1]), float(ee_offset[2]))
        if self._cfg.active_arm == "R":
            tcp = (-ex, -abs(ey), ez)
        else:
            tcp = (-ex, ey, ez)
        open_rad = float(target.gripper_open)
        close_rad = float(target.gripper_close)
        try:
            from param_config import part_grasp_close_rad, part_grasp_open_rad

            open_rad = float(part_grasp_open_rad(target.name))
            close_rad = float(part_grasp_close_rad(target.name))
        except Exception as exc:
            # Custom parts can omit optional AABB geometry and still use
            # their resolved param_config aperture values.
            print(
                f"[asset_centroid] {target.name}: using param_config aperture "
                f"fallback ({exc})",
                flush=True,
            )
        if open_rad <= 0.0:
            raise ValueError(f"{target.name}: no positive gripper_open configured")
        if close_rad < 0.0:
            raise ValueError(f"{target.name}: gripper_close must be non-negative")
        print(
            f"[asset_centroid] {target.name}: no explicit centroid spec; "
            f"ee_offset={np.round(ee_offset, 5).tolist()} → "
            f"tcp_to_grasp_tool={list(tcp)} (arm={self._cfg.active_arm})",
            flush=True,
        )
        return AssetMotionSpec(
            centroid_grasp_offset_asset=(0.0, 0.0, 0.0),
            tcp_to_grasp_tool=tuple(float(v) for v in tcp),
            gripper_open_rad=open_rad,
            gripper_close_rad=close_rad,
        )

    def _compliance(self):
        return getattr(self._controller, "gripper_compliance", None)

    def _configure_compliance(self) -> None:
        """Apply policy-local speed and, once known, the part close bound."""
        g = self._compliance()
        if g is None:
            return
        g.cfg.close_speed_rad_s = float(self._cfg.gripper.close_speed_rad_s)
        if self._cfg.gripper.mode == "compliant" and self._part_spec is not None:
            g.cfg.close = float(self._part_spec.gripper_close_rad)

    def _log_compliance_params(self, *, prefix: str = "compliance") -> None:
        """Print GripperCompliance tunables + live state (call at ~1 Hz)."""
        g = self._compliance()
        phase_name = (
            self._phases[self._phase_index].name
            if self._phases and 0 <= self._phase_index < len(self._phases)
            else "-"
        )
        if g is None:
            print(
                f"[asset_centroid] {prefix}: no gripper_compliance on controller "
                f"(policy_phase={phase_name})",
                flush=True,
            )
            return
        cfg = g.cfg
        q_meas = qd_meas = None
        meas_fn = getattr(self._controller, "_measured_gripper", None)
        if callable(meas_fn):
            try:
                q_meas, qd_meas = meas_fn()
            except Exception:
                q_meas = qd_meas = None
        q_cmd = None if g.q_cmd is None else float(g.q_cmd)
        close_lag = (
            None if q_meas is None or q_cmd is None else float(q_meas) - q_cmd
        )
        q_meas_prev = getattr(g, "_q_meas_prev", None)
        meas_dq = (
            None
            if q_meas is None or q_meas_prev is None
            else float(q_meas) - float(q_meas_prev)
        )
        print(
            f"[asset_centroid] {prefix}: policy_phase={phase_name} "
            f"comp_phase={getattr(g.phase, 'value', g.phase)} "
            f"intent={g.intent!r} enabled={bool(g.enabled)} "
            f"q_cmd={None if q_cmd is None else f'{q_cmd:.4f}'} "
            f"q_hold={None if g._q_hold is None else f'{float(g._q_hold):.4f}'} "
            f"q_meas={None if q_meas is None else f'{float(q_meas):.4f}'} "
            f"qd_meas={None if qd_meas is None else f'{float(qd_meas):.4f}'} "
            f"close_lag={None if close_lag is None else f'{close_lag:.4f}'} "
            f"meas_dq={None if meas_dq is None else f'{meas_dq:.4f}'} "
            f"stall_ticks={int(g._stall_ticks)} "
            f"close_target={float(cfg.close):g} "
            f"close_speed={float(cfg.close_speed_rad_s):g} "
            f"open_speed={float(cfg.open_speed_rad_s):g} "
            f"stall_err={float(cfg.stall_err):g} "
            f"max_close_lag={float(cfg.max_close_lag):g} "
            f"stall_qd={float(cfg.stall_qd):g} "
            f"stall_dq={float(cfg.stall_dq):g} "
            f"stall_min_close={float(cfg.stall_min_close_rad):g} "
            f"hold_margin={float(cfg.hold_margin):g} "
            f"stall_hold_ticks={int(cfg.stall_hold_ticks)} "
            f"stall_progress={float(cfg.stall_progress):g}",
            flush=True,
        )

    def _maybe_log_compliance_1hz(self) -> None:
        self._compliance_log_accum_s += self._dt
        if self._compliance_log_accum_s < 1.0:
            return
        self._compliance_log_accum_s %= 1.0
        self._log_compliance_params()

    def _freeze_compliant_grasp_aperture(self) -> None:
        """After close dwell, hold the latched/measured grasp aperture.

        Prevents continued soft squeeze if HOLDING did not latch during dwell,
        while the configured part aperture remains a hard lower bound.
        """
        if self._cfg.gripper.mode != "compliant":
            return
        g = self._compliance()
        q_meas: Optional[float] = None
        meas_fn = getattr(self._controller, "_measured_gripper", None)
        if callable(meas_fn):
            try:
                q_meas, _ = meas_fn()
                q_meas = float(q_meas)
            except Exception:
                q_meas = None
        freeze: Optional[float] = None
        if g is not None:
            freeze = g.freeze(q_meas)
        elif q_meas is not None:
            freeze = q_meas
        if freeze is None:
            print(
                "[asset_centroid] compliant close ended but no aperture to freeze",
                flush=True,
            )
            return
        # Rewrite remaining phases that still request soft \"close\".
        new_phases: list[_Phase] = []
        replaced = 0
        for i, phase in enumerate(self._phases):
            if i >= self._phase_index and phase.gripper == "close":
                new_phases.append(
                    _Phase(
                        phase.name,
                        phase.kind,
                        phase.pos,
                        phase.orn,
                        float(freeze),
                        phase.dwell_steps,
                        phase.orientation_cone_rad,
                    )
                )
                replaced += 1
            else:
                new_phases.append(phase)
        self._phases = new_phases
        print(
            f"[asset_centroid] freeze compliant grasp at {freeze:.4f} rad "
            f"(rewrote {replaced} phase gripper cmds)",
            flush=True,
        )

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

        if self._part_spec is None:
            raise RuntimeError(f"{name}: part motion spec was not resolved")
        offset = self._part_spec.centroid_grasp_offset_asset
        offset_gf = root_world.TransformDir(Gf.Vec3d(*offset))
        offset_world = np.asarray(tuple(offset_gf), dtype=np.float64)
        return centroid_world, offset_world

    @staticmethod
    def _debug_markers_enabled() -> bool:
        # Default on so a GUI run shows targets without extra env setup.
        raw = os.environ.get("ROCO_ASSET_CENTROID_DEBUG_MARKERS", "1").strip().lower()
        return raw not in {"0", "false", "off", "no"}

    def _render_target_markers(
        self,
        *,
        part_name: str,
        centroid_world: np.ndarray,
        grasp_center: np.ndarray,
        poses: dict[str, np.ndarray],
    ) -> None:
        """Draw colored spheres for centroid / grasp / EE / baseline pick_pos.

        Legend (also printed):
          magenta — live AABB centroid (policy grasp source)
          yellow  — grasp_center (= centroid + asset offset)
          cyan    — hover_pick EE (joint/Cartesian approach endpoint)
          green   — PART_CONFIG pick_pos (baseline object point)
          orange  — baseline pick EE (= pick_pos + ee_offset, world add)
        """
        if not self._debug_markers_enabled():
            return
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
        except Exception as exc:
            print(f"[asset_centroid] debug markers skipped (no USD): {exc}", flush=True)
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        baseline_pick = None
        baseline_ee = None
        try:
            import param_config as pc

            cfg = pc.get_part_config(part_name)
            if cfg.get("pick_pos") is not None:
                baseline_pick = np.asarray(cfg["pick_pos"], dtype=np.float64).reshape(3)
                ee_off = np.asarray(cfg.get("ee_offset", (0.0, 0.0, 0.0)), dtype=np.float64)
                baseline_ee = baseline_pick + ee_off.reshape(3)
        except Exception:
            pass

        root = "/World/RoCoDebug/asset_centroid"
        if not stage.GetPrimAtPath(root).IsValid():
            UsdGeom.Xform.Define(stage, "/World/RoCoDebug")
            UsdGeom.Xform.Define(stage, root)

        markers = [
            ("centroid", centroid_world, 0.008, (1.0, 0.0, 1.0)),
            ("grasp_center", grasp_center, 0.007, (1.0, 1.0, 0.0)),
            ("hover_pick_ee", poses["hover_pick"], 0.010, (0.0, 1.0, 1.0)),
            ("pick_ee", poses["pick"], 0.008, (0.2, 0.6, 1.0)),
        ]
        if baseline_pick is not None:
            markers.append(("baseline_pick_pos", baseline_pick, 0.007, (0.0, 1.0, 0.0)))
        if baseline_ee is not None:
            markers.append(("baseline_pick_ee", baseline_ee, 0.008, (1.0, 0.5, 0.0)))

        def _set_sphere(path: str, xyz: np.ndarray, radius: float, rgb: tuple[float, float, float]) -> None:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                sphere = UsdGeom.Sphere.Define(stage, path)
            else:
                sphere = UsdGeom.Sphere(prim)
            sphere.CreateRadiusAttr(float(radius))
            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
            sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
            sphere.CreateDisplayOpacityAttr().Set([0.85])

        print(
            "[asset_centroid] debug markers (disable with "
            "ROCO_ASSET_CENTROID_DEBUG_MARKERS=0):",
            flush=True,
        )
        for name, xyz, radius, rgb in markers:
            path = f"{root}/{name}"
            _set_sphere(path, np.asarray(xyz, dtype=np.float64).reshape(3), radius, rgb)
            print(
                f"  {name:18s} rgb={rgb} pos={np.round(xyz, 5).tolist()}",
                flush=True,
            )
        print(
            "[asset_centroid] note: hover_pick uses joint-space approach — EE may "
            "not travel in a straight line toward the cyan sphere.",
            flush=True,
        )

    def _path_heights_m(self) -> tuple[float, float, float]:
        """Return (hover_pick, hover_place, retract) clearances in meters.

        When the active part sets ``use_param_config_height``, map PART_CONFIG
        ``init_height`` → hover_pick, ``hover_place_height`` → hover_place
        (falls back to ``init_height`` when unset), and ``final_height`` →
        retract. Otherwise use asset-centroid JSON path clearances.
        """
        clear = self._cfg.path_clearances
        hover_pick = float(clear.hover_pick_m)
        hover_place = float(clear.hover_place_m)
        retract = float(clear.final_retract_m)
        name = None if self._target is None else self._target.name
        if not name:
            return hover_pick, hover_place, retract
        try:
            import param_config as pc

            cfg = pc.get_part_config(name)
        except Exception:
            return hover_pick, hover_place, retract
        if not bool(cfg.get("use_param_config_height", False)):
            return hover_pick, hover_place, retract
        init_h = cfg.get("init_height")
        place_h = cfg.get("hover_place_height")
        final_h = cfg.get("final_height")
        if init_h is None:
            init_h = getattr(pc, "INIT_HEIGHT", hover_pick)
        if place_h is None:
            place_h = init_h
        if final_h is None:
            final_h = getattr(pc, "FINAL_HEIGHT", None)
        if final_h is None:
            final_h = init_h
        return float(init_h), float(place_h), float(final_h)

    def _place_z_offset_m(self) -> float:
        """World +Z on the place EE when use_param_config_height is set."""
        name = None if self._target is None else self._target.name
        if not name:
            return 0.0
        try:
            import param_config as pc

            cfg = pc.get_part_config(name)
        except Exception:
            return 0.0
        if not bool(cfg.get("use_param_config_height", False)):
            return 0.0
        return float(cfg.get("place_z_offset") or 0.0)

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
        hover_pick_m, hover_place_m, retract_m = self._path_heights_m()
        place_z_off = self._place_z_offset_m()
        pick = ee_position_for_grasp_center(
            grasp_center, pick_orn, tcp, offset_frame=frame
        )
        place = ee_position_for_grasp_center(
            place_center, place_orn, tcp, offset_frame=frame
        )
        place = place + np.array([0.0, 0.0, place_z_off], dtype=np.float64)
        return {
            "pick": pick,
            "hover_pick": pick + np.array([0.0, 0.0, hover_pick_m]),
            "lift_pick": pick + np.array([0.0, 0.0, hover_pick_m]),
            "place": place,
            "hover_place": place + np.array([0.0, 0.0, hover_place_m]),
            "retract": place + np.array([0.0, 0.0, retract_m]),
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

    def _relaxed_approach_enabled(self) -> bool:
        spec = self._cfg.approach_orientation
        part_name = None if self._target is None else self._target.name
        return (
            spec.enabled
            and spec.recover_at_hover
            and part_name in spec.enabled_parts
        )

    @staticmethod
    def _axis_angle_quat(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
        axis_arr = np.asarray(axis, dtype=np.float64)
        axis_arr /= np.linalg.norm(axis_arr)
        half = float(angle_rad) / 2.0
        return np.array([np.cos(half), *(axis_arr * np.sin(half))], dtype=np.float64)

    def _approach_tilt_candidates(self, top_down_orn: np.ndarray) -> tuple[np.ndarray, ...]:
        """Small tool-frame pitch/roll candidates; never add a Z/yaw turn."""
        spec = self._cfg.approach_orientation
        if not spec.enabled or spec.sample_tilt_rad <= 0.0:
            return (top_down_orn.copy(),)
        angle = spec.sample_tilt_rad
        return tuple(
            quat_mul_wxyz(top_down_orn, self._axis_angle_quat(axis, signed_angle))
            for axis, signed_angle in (
                ((1.0, 0.0, 0.0), angle),
                ((1.0, 0.0, 0.0), -angle),
                ((0.0, 1.0, 0.0), angle),
                ((0.0, 1.0, 0.0), -angle),
            )
        )

    def _preflight_relaxed_approach(
        self, seed: np.ndarray, hover_pos: np.ndarray, top_down_orn: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Probe the incoming hover path with sequential seeds and bounded tilts."""
        if not self._relaxed_approach_enabled():
            return seed, True
        start_pos, _ = self._actual_ee_pose(None)  # observation is not needed here
        candidates = self._approach_tilt_candidates(top_down_orn)
        chosen: list[float] = []
        # Two interior samples plus hover catch the common wrist-wrap failure
        # without turning endpoint preflight into an expensive dense trajectory.
        for waypoint_i, fraction in enumerate((1.0 / 3.0, 2.0 / 3.0, 1.0)):
            pos = start_pos + fraction * (hover_pos - start_pos)
            options = []
            for tilt_i, orn in enumerate(candidates):
                q, ok = self._controller.solve_q(pos, orn, seed=seed)
                if ok and q is not None:
                    q = np.asarray(q, dtype=np.float64).reshape(-1)
                    options.append((self._joint_step_cost(seed, q), tilt_i, q))
            if not options:
                self._approach_diagnostics = {
                    "first_failing_waypoint": waypoint_i,
                    "position": pos.copy(),
                    "chosen_tilt_index": chosen[-1] if chosen else None,
                }
                return seed, False
            _, tilt_i, seed = min(options, key=lambda item: item[0])
            chosen.append(int(tilt_i))
        self._approach_diagnostics = {
            "chosen_tilt_indices": tuple(chosen),
            "first_failing_waypoint": None,
        }
        return seed, True

    def _choose_yaw_and_poses(
        self,
        grasp_center: np.ndarray,
        place_center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Pick/place yaw from keyframe endpoint IK only.

        Dense Cartesian transfer is intentionally not required: runtime
        hover_pick(/_relaxed) and hover_place use joint-space quintic lerps
        between preflight endpoint solutions so Lula cannot flip wrist
        branches at midpoints. Short vertical descend_place stays Cartesian.
        """
        seed0 = np.asarray(self._controller.current_cspace_q(), dtype=np.float64)
        best = None
        best_approach_diagnostics = None
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
            seed, feasible = self._preflight_relaxed_approach(
                seed, poses["hover_pick"], pick_orn
            )
            if not feasible:
                failure_counts["hover_pick"] += 1
                continue
            hover_pick_q = None
            for key in pick_keys:
                q, ok = self._controller.solve_q(poses[key], pick_orn, seed=seed)
                if not ok or q is None:
                    failure_counts[key] += 1
                    feasible = False
                    break
                q = np.asarray(q, dtype=np.float64).reshape(-1)
                if key == "hover_pick":
                    hover_pick_q = q.copy()
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
                hover_place_q = None
                place_q = None
                for key in place_keys:
                    q, ok = self._controller.solve_q(
                        pair_poses[key], place_orn, seed=seed
                    )
                    if not ok or q is None:
                        failure_counts[key] += 1
                        place_ok = False
                        break
                    q = np.asarray(q, dtype=np.float64).reshape(-1)
                    if key == "hover_place":
                        hover_place_q = q.copy()
                    elif key == "place":
                        place_q = q.copy()
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
                        hover_pick_q,
                        hover_place_q,
                        place_q,
                    )
                    best_approach_diagnostics = dict(self._approach_diagnostics)
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
                seed, feasible = self._preflight_relaxed_approach(
                    seed, poses["hover_pick"], pick_orn
                )
                if not feasible:
                    failure_counts["hover_pick"] += 1
                    continue
                hover_pick_q = None
                for key in pick_keys:
                    q, ok = self._controller.solve_q(poses[key], pick_orn, seed=seed)
                    if not ok or q is None:
                        failure_counts[key] += 1
                        feasible = False
                        break
                    q = np.asarray(q, dtype=np.float64).reshape(-1)
                    if key == "hover_pick":
                        hover_pick_q = q.copy()
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
                    hover_place_q = None
                    place_q = None
                    for key in place_keys:
                        q, ok = self._controller.solve_q(
                            pair_poses[key], place_orn, seed=seed
                        )
                        if not ok or q is None:
                            failure_counts[key] += 1
                            place_ok = False
                            break
                        q = np.asarray(q, dtype=np.float64).reshape(-1)
                        if key == "hover_place":
                            hover_place_q = q.copy()
                        elif key == "place":
                            place_q = q.copy()
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
                            hover_pick_q,
                            hover_place_q,
                            place_q,
                        )
                        best_approach_diagnostics = dict(self._approach_diagnostics)
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
        if best_approach_diagnostics is not None:
            self._approach_diagnostics = best_approach_diagnostics
        self._hover_pick_q = best[4].copy()
        self._hover_place_q = best[5].copy()
        self._place_q = best[6].copy()
        hover_pick_m, hover_place_m, retract_m = self._path_heights_m()
        place_z_off = self._place_z_offset_m()
        xy_err_pick = float(
            np.linalg.norm(poses["hover_pick"][:2] - grasp_center[:2])
        )
        print(
            "[asset_centroid] path heights "
            f"hover_pick={hover_pick_m:.4f} hover_place={hover_place_m:.4f} "
            f"place_z_offset={place_z_off:.4f} retract={retract_m:.4f} | "
            f"place_ee_z={poses['place'][2]:.5f} "
            f"hover_place_z={poses['hover_place'][2]:.5f} "
            f"retract_z={poses['retract'][2]:.5f}",
            flush=True,
        )
        print(
            "[asset_centroid] selected world-down TCP yaws "
            f"(pick={pick_yaw:.1f} deg, place={place_yaw:.1f} deg) "
            f"on {self._cfg.active_arm} arm; frame={self._cfg.path_clearances.tcp_offset_frame}; "
            f"hover_pick={np.round(poses['hover_pick'], 5).tolist()} "
            f"(|xy−grasp|={xy_err_pick * 1000:.1f} mm); "
            f"hover_place={np.round(poses['hover_place'], 5).tolist()} "
            f"place={np.round(poses['place'], 5).tolist()}",
            flush=True,
        )
        if self._relaxed_approach_enabled():
            print(
                f"[asset_centroid] relaxed-approach diagnostics="
                f"{self._approach_diagnostics}",
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
        # Large transfers: joint-space quintic to a preflight endpoint so
        # per-step Lula cannot branch-flip mid-Cartesian (pin hover abort).
        # Short vertical descend_place stays Cartesian so XY does not arc.
        joint_target = None
        if phase.name in ("hover_pick", "hover_pick_relaxed"):
            joint_target = self._hover_pick_q
        elif phase.name == "hover_place":
            joint_target = self._hover_place_q
        if phase.name in ("hover_pick", "hover_pick_relaxed", "hover_place"):
            if joint_target is None:
                self._abort(f"missing preflight joint target for {phase.name}")
                return
            q0 = np.asarray(
                self._controller.current_cspace_q(), dtype=np.float64
            )
            q1 = np.asarray(joint_target, dtype=np.float64).reshape(-1).copy()
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

    def _command(self, pos, orn, gripper, *, orientation_cone_rad=None) -> ArticulationAction:
        """Command IK, passing a cone only for the relaxed pick approach."""
        try:
            action = self._controller.forward(
                pos, orn, gripper, orientation_cone_rad=orientation_cone_rad
            )
        except TypeError as exc:
            # Keep minimal offline controller doubles from older tests usable.
            if "orientation_cone_rad" not in str(exc):
                raise
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

    def _latch_hold_q(self) -> None:
        """Freeze arm joints for place settle/open (avoid per-tick re-IK jitter)."""
        if self._place_q is not None:
            self._hold_q = np.asarray(self._place_q, dtype=np.float64).reshape(-1).copy()
            return
        try:
            self._hold_q = np.asarray(
                self._controller.current_cspace_q(), dtype=np.float64
            ).reshape(-1).copy()
        except Exception:
            self._hold_q = None

    def _advance_phase(self, obs: Optional[Observation] = None) -> None:
        completed = self._phases[self._phase_index].name
        if obs is not None:
            self._log_pose_error(obs, self._phases[self._phase_index])
        if completed == "close":
            self._freeze_compliant_grasp_aperture()
        # Latch joints when we reach the release pose (after descend, or
        # after hover_place when descend was skipped).
        if completed in ("descend_place", "hover_place"):
            next_i = self._phase_index + 1
            next_name = (
                self._phases[next_i].name
                if next_i < len(self._phases)
                else None
            )
            if completed == "descend_place" or next_name == "settle_place":
                self._latch_hold_q()
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
        self._maybe_log_compliance_1hz()
        phase = self._phases[self._phase_index]
        if phase.kind == "dwell":
            if phase.name in ("settle_place", "open") and self._hold_q is not None:
                action = self._controller.forward_raw_q(
                    self._hold_q, phase.gripper
                )
                self._last_action = action
                self._ik_failure_steps = 0
            else:
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
                action = self._command(
                    pos, orn, phase.gripper,
                    orientation_cone_rad=phase.orientation_cone_rad,
                )
            if self._done:
                return self._noop()
            if self._ik_failure_steps == 0:
                self._segment_index += 1
            return action

        if self._segment_q:
            # Keep holding the exact endpoint selected for the joint-space
            # transfer. Re-running pose IK here can jump to a different
            # revolute wrap branch (observed on gear_60 hover_place), causing
            # a false 100-step IK abort after the transfer already succeeded.
            action = self._controller.forward_raw_q(
                self._segment_q[-1], phase.gripper
            )
            self._last_action = action
            self._ik_failure_steps = 0
        else:
            action = self._command(
                phase.pos, phase.orn, phase.gripper,
                orientation_cone_rad=phase.orientation_cone_rad,
            )
            if self._done:
                return self._noop()
        actual_pos, actual_orn = self._actual_ee_pose(obs)
        guards = self._cfg.guards
        orn_limit = (phase.orientation_cone_rad if phase.orientation_cone_rad is not None
                     else guards.orn_tol_rad)
        reached = (np.linalg.norm(np.asarray(actual_pos) - phase.pos) <= guards.pos_tol_m
                   and quat_angle(actual_orn, phase.orn) <= orn_limit)
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
