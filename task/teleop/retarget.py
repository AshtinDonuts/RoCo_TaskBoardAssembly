"""Relative Cartesian retargeting from an ALOHA leader to DexMate Vega 1U."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import transforms as T


@dataclass
class ProximityScaleConfig:
    """Laser-range band → scale in ``[scale_min, 1]``.

    Linear on ``[depth_inner_m, depth_outer_m]``: ``1`` at/above outer,
    ``scale_min`` at/below inner. Invalid depth fails open to ``1``.
    """

    enabled: bool = False
    depth_outer_m: float = 0.40
    depth_inner_m: float = 0.20
    scale_min: float = 0.20

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "ProximityScaleConfig":
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


# Back-compat alias for imports / older tests.
ProximitySlowdownConfig = ProximityScaleConfig


def proximity_band_scale(
    depth_m: Optional[float],
    cfg: Optional[ProximityScaleConfig] = None,
) -> float:
    """Return multiplier in ``[scale_min, 1]`` from approach range.

    If ``cfg`` is missing or ``enabled`` is False, returns ``1.0``.
    Invalid / unknown depth also fails open to ``1.0``.
    """
    c = cfg or ProximityScaleConfig()
    if not c.enabled:
        return 1.0
    if depth_m is None:
        return 1.0
    d = float(depth_m)
    if not np.isfinite(d) or d < 0.0:
        return 1.0
    outer = float(c.depth_outer_m)
    inner = float(c.depth_inner_m)
    s_min = float(np.clip(c.scale_min, 0.0, 1.0))
    if outer <= inner:
        return 1.0 if d >= outer else s_min
    if d >= outer:
        return 1.0
    if d <= inner:
        return s_min
    return float(s_min + (1.0 - s_min) * (d - inner) / (outer - inner))


# Back-compat name.
proximity_motion_scale = proximity_band_scale


@dataclass
class RetargetConfig:
    axes_perm: Tuple[int, int, int] = (0, 1, 2)
    axes_sign: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Optional 3x3: DexMate_delta = axes_map @ leader_delta.
    # When set, replaces axes_perm/axes_sign (used for head-camera view).
    axes_map: Optional[Tuple[Tuple[float, float, float], ...]] = None
    translation_gain: float = 1.0
    rotation_gain: float = 1.0
    # When True, hold dex_origin_quat forever between reanchors (leader /
    # keyboard wrist tilts ignored). Lula IK still receives the fixed quat
    # so joints compensate while translation tracks.
    fix_orientation: bool = False
    # When set, command this constant world/stage EE quat (wxyz) instead of
    # mapping leader wrist tilts. Translation and gripper still track.
    # Takes precedence over fix_orientation for the commanded orientation;
    # _rate_limit slews toward it at max_ang_vel (no hard snap).
    fixed_orientation_wxyz: Optional[Tuple[float, float, float, float]] = None
    # Claw-machine soft cone (rad) around fixed_orientation_wxyz / preferred
    # quat. When > 0, Lula tries preferred then in-cone tilts so XYZ motion
    # is not blocked when exact top-down is IK-infeasible. 0 / None = hard
    # orientation lock (legacy reject-all behavior).
    orientation_cone_rad: Optional[float] = None
    workspace_min: Tuple[float, float, float] = (-1.5, -1.5, 0.02)
    workspace_max: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    max_lin_vel: float = 0.35
    max_ang_vel: float = 1.2
    max_lin_acc: float = 2.0
    gripper_open_limit: float = 0.6649704
    gripper_close: float = 0.0
    # Master switch for slow-close + stall hold (controllers/gripper_compliance).
    # false → legacy immediate aperture commands (no slew / hold).
    gripper_compliance_enabled: bool = True
    # continuous (default): linear leader→aperture; binary: open/close intent.
    gripper_mode: str = "continuous"
    # Leader gripper_norm at/below which DexMate is fully closed (continuous)
    # or close intent (binary). Absorbs leader calibration slack.
    gripper_close_norm: float = 0.20
    gripper_hysteresis: float = 0.03
    # Slow-close + stall hold (see controllers/gripper_compliance.py).
    gripper_close_speed_rad_s: float = 0.4
    gripper_open_speed_rad_s: float = 0.8
    gripper_stall_qd: float = 0.02
    gripper_stall_err: float = 0.008
    gripper_stall_dq: float = 0.005
    gripper_stall_min_close_rad: float = 0.03
    gripper_hold_margin: float = 0.002
    # Cap how far q_cmd may lead closed vs q_meas (beats leader catch-up).
    gripper_max_close_lag: float = 0.01
    gripper_stall_hold_ticks: int = 12
    gripper_stall_progress: float = 0.002
    # Reject one-frame leader jumps larger than these (hold last good sample).
    # Rate limiting still applies afterward; this removes spurious spikes so
    # delta-gain does not permanently integrate a glitch. Enabled in the
    # Solo→Vega YAML; off by default so unit tests can take large steps.
    spike_filter_enabled: bool = False
    spike_max_lin_vel: float = 1.5  # m/s of leader EE translation
    spike_max_ang_vel: float = 8.0  # rad/s of leader EE rotation
    spike_max_gripper_vel: float = 5.0  # leader gripper_norm units/s
    stale_hold_s: float = 0.10
    stale_pause_s: float = 0.50
    # Throttle max_lin/ang_vel (+ accel); absolute map unchanged → catch-up.
    proximity_rate_limit: ProximityScaleConfig = field(
        default_factory=lambda: ProximityScaleConfig(enabled=False)
    )
    # Scale per-frame leader deltas into the absolute target (no catch-up).
    proximity_delta_gain: ProximityScaleConfig = field(
        default_factory=lambda: ProximityScaleConfig(enabled=False)
    )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "RetargetConfig":
        if not data:
            return cls()
        raw = dict(data)
        # Old YAML key → rate-limit mode.
        if "proximity_rate_limit" not in raw and "proximity_slowdown" in raw:
            raw["proximity_rate_limit"] = raw.pop("proximity_slowdown")
        else:
            raw.pop("proximity_slowdown", None)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "axes_perm" in kwargs:
            kwargs["axes_perm"] = tuple(int(v) for v in kwargs["axes_perm"])
        if "axes_sign" in kwargs:
            kwargs["axes_sign"] = tuple(float(v) for v in kwargs["axes_sign"])
        if "axes_map" in kwargs and kwargs["axes_map"] is not None:
            m = np.asarray(kwargs["axes_map"], dtype=np.float64).reshape(3, 3)
            kwargs["axes_map"] = tuple(tuple(float(x) for x in row) for row in m)
        if "workspace_min" in kwargs:
            kwargs["workspace_min"] = tuple(float(v) for v in kwargs["workspace_min"])
        if "workspace_max" in kwargs:
            kwargs["workspace_max"] = tuple(float(v) for v in kwargs["workspace_max"])
        if "fixed_orientation_wxyz" in kwargs:
            raw_q = kwargs["fixed_orientation_wxyz"]
            # null / false / [] / omit → unlocked (full 6DoF orientation).
            if raw_q is None or raw_q is False or raw_q == [] or raw_q == ():
                kwargs["fixed_orientation_wxyz"] = None
            else:
                q = tuple(float(v) for v in raw_q)
                if len(q) != 4:
                    raise ValueError(
                        "fixed_orientation_wxyz must be a length-4 wxyz quat "
                        f"(got len={len(q)})"
                    )
                kwargs["fixed_orientation_wxyz"] = q
        if "orientation_cone_rad" in kwargs:
            raw_c = kwargs["orientation_cone_rad"]
            if raw_c is None or raw_c is False:
                kwargs["orientation_cone_rad"] = None
            else:
                kwargs["orientation_cone_rad"] = float(raw_c)
        if "gripper_mode" in kwargs:
            kwargs["gripper_mode"] = str(kwargs["gripper_mode"]).lower()
        if "gripper_compliance_enabled" in kwargs:
            raw_e = kwargs["gripper_compliance_enabled"]
            if isinstance(raw_e, bool):
                kwargs["gripper_compliance_enabled"] = raw_e
            elif isinstance(raw_e, (int, float)):
                kwargs["gripper_compliance_enabled"] = bool(raw_e)
            else:
                s = str(raw_e).strip().lower()
                kwargs["gripper_compliance_enabled"] = s in (
                    "1", "true", "yes", "on",
                )
        for key in ("proximity_rate_limit", "proximity_delta_gain"):
            if key in kwargs:
                kwargs[key] = ProximityScaleConfig.from_dict(
                    kwargs[key] if isinstance(kwargs[key], dict) else None
                )
        return cls(**kwargs)

    def map_leader_vec(self, vec: Sequence[float]) -> np.ndarray:
        """Map a leader-base vector into DexMate world/stage axes."""
        if self.axes_map is not None:
            return T.apply_axes_matrix(vec, self.axes_map)
        return T.apply_axes_map(vec, self.axes_perm, self.axes_sign)

    def axes_matrix(self) -> np.ndarray:
        """3x3 map with DexMate_vec = axes_matrix @ leader_vec."""
        if self.axes_map is not None:
            return np.asarray(self.axes_map, dtype=np.float64).reshape(3, 3)
        # Build from signed permutation: out[i] = s[i] * v[p[i]]
        m = np.zeros((3, 3), dtype=np.float64)
        for i, (p, s) in enumerate(zip(self.axes_perm, self.axes_sign)):
            m[i, int(p)] = float(s)
        return m


@dataclass
class RetargetState:
    engaged: bool = False
    leader_origin_pos: Optional[np.ndarray] = None
    leader_origin_quat: Optional[np.ndarray] = None
    dex_origin_pos: Optional[np.ndarray] = None
    dex_origin_quat: Optional[np.ndarray] = None
    # Previous leader sample for incremental (delta-gain) integration.
    leader_prev_pos: Optional[np.ndarray] = None
    leader_prev_quat: Optional[np.ndarray] = None
    last_pos: Optional[np.ndarray] = None
    last_quat: Optional[np.ndarray] = None
    last_gripper: float = 0.0
    last_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_cmd: str = "none"
    # Last accepted leader sample for spike rejection (may lag a glitch).
    spike_pos: Optional[np.ndarray] = None
    spike_quat: Optional[np.ndarray] = None
    spike_gripper: Optional[float] = None


class CartesianRetargeter:
    def __init__(self, config: Optional[RetargetConfig] = None) -> None:
        self.cfg = config or RetargetConfig()
        self.state = RetargetState()

    def reset(self) -> None:
        self.state = RetargetState()

    def capture_origins(
        self,
        leader_pos: Sequence[float],
        leader_quat: Sequence[float],
        dex_pos: Sequence[float],
        dex_quat: Sequence[float],
    ) -> None:
        st = self.state
        st.engaged = True
        st.leader_origin_pos = T.as_vec(leader_pos, 3)
        st.leader_origin_quat = T.normalize_quat_wxyz(leader_quat)
        st.dex_origin_pos = T.as_vec(dex_pos, 3)
        st.dex_origin_quat = T.normalize_quat_wxyz(dex_quat)
        st.leader_prev_pos = st.leader_origin_pos.copy()
        st.leader_prev_quat = st.leader_origin_quat.copy()
        st.last_pos = st.dex_origin_pos.copy()
        st.last_quat = st.dex_origin_quat.copy()
        st.last_lin_vel[:] = 0.0
        st.spike_pos = st.leader_origin_pos.copy()
        st.spike_quat = st.leader_origin_quat.copy()
        st.spike_gripper = None

    def disengage(self) -> None:
        self.state.engaged = False
        self.state.leader_origin_pos = None
        self.state.leader_origin_quat = None
        self.state.dex_origin_pos = None
        self.state.dex_origin_quat = None
        self.state.leader_prev_pos = None
        self.state.leader_prev_quat = None
        self.state.spike_pos = None
        self.state.spike_quat = None
        self.state.spike_gripper = None

    def map_gripper(self, gripper_norm: float) -> float:
        cfg = self.cfg
        # When compliance module is off, always use legacy linear remap
        # (immediate aperture; no binary endpoint + slew/hold path).
        compliance_on = bool(getattr(cfg, "gripper_compliance_enabled", True))
        mode = str(getattr(cfg, "gripper_mode", "continuous")).lower()
        g = float(np.clip(gripper_norm, 0.0, 1.0))
        close_at = float(np.clip(cfg.gripper_close_norm, 0.0, 0.95))
        if compliance_on and mode == "binary":
            # Intent → endpoint only; slew/hold runs in GripperCompliance.
            open_at = float(
                np.clip(close_at + max(0.0, cfg.gripper_hysteresis), 0.0, 1.0)
            )
            prev = self.state.last_gripper
            prev_close = abs(prev - cfg.gripper_close) <= abs(
                prev - cfg.gripper_open_limit
            )
            if g <= close_at:
                intent_close = True
            elif g >= open_at:
                intent_close = False
            else:
                intent_close = prev_close
            target = cfg.gripper_close if intent_close else cfg.gripper_open_limit
            self.state.last_gripper = target
            return target

        # Linear remap: leader [close_norm, 1] → DexMate [closed, open].
        # Anything at/below close_norm is treated as fully closed so a
        # slightly short physical travel still seals the virtual fingers.
        if g <= close_at:
            g_eff = 0.0
        else:
            g_eff = (g - close_at) / (1.0 - close_at)
        span = cfg.gripper_open_limit - cfg.gripper_close
        target = cfg.gripper_close + g_eff * span
        prev = self.state.last_gripper
        if abs(target - prev) < cfg.gripper_hysteresis * span:
            return prev
        self.state.last_gripper = target
        return target

    def _locked_target_quat(self) -> Optional[np.ndarray]:
        """Normalized constant world EE quat, or None if unlocked."""
        q = self.cfg.fixed_orientation_wxyz
        if q is None:
            return None
        return T.normalize_quat_wxyz(q)

    def step(
        self,
        leader_pos: Sequence[float],
        leader_quat: Sequence[float],
        gripper_norm: float,
        dt: float,
        clutch: bool,
        deadman: bool,
        current_dex_pos: Sequence[float],
        current_dex_quat: Sequence[float],
        proximity_depth_m: Optional[float] = None,
        motion_scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        delta_scale = proximity_band_scale(
            proximity_depth_m, self.cfg.proximity_delta_gain
        )
        rate_scale = proximity_band_scale(
            proximity_depth_m, self.cfg.proximity_rate_limit
        )
        # Legacy override: explicit motion_scale replaces rate-limit scale.
        if motion_scale is not None:
            rate_scale = float(np.clip(motion_scale, 0.0, 1.0))

        info: Dict[str, Any] = {
            "held": False,
            "engaged": self.state.engaged,
            "delta_gain_scale": float(delta_scale),
            "rate_limit_scale": float(rate_scale),
            "motion_scale": float(rate_scale),  # back-compat alias
            "spike_rejected": False,
        }
        if not deadman:
            info["held"] = True
            info["reason"] = "deadman"
            self._sync_leader_prev(leader_pos, leader_quat)
            return self._hold(
                current_dex_pos, current_dex_quat, self.state.last_gripper, info
            )

        if not clutch:
            if self.state.engaged:
                self.disengage()
            info["held"] = True
            info["reason"] = "clutch_off"
            return self._hold(
                current_dex_pos, current_dex_quat, self.state.last_gripper, info
            )

        if clutch and not self.state.engaged:
            self.capture_origins(leader_pos, leader_quat, current_dex_pos, current_dex_quat)
            self.state.spike_gripper = float(np.clip(gripper_norm, 0.0, 1.0))
            grip = self.map_gripper(gripper_norm)
            info["engaged"] = True
            info["reason"] = "clutch_engage"
            return (
                self.state.last_pos.copy(),
                self.state.last_quat.copy(),
                grip,
                info,
            )

        leader_pos, leader_quat, gripper_norm, spike_info = self._spike_filter(
            leader_pos, leader_quat, gripper_norm, dt
        )
        info.update(spike_info)
        grip = self.map_gripper(gripper_norm)

        if self.cfg.proximity_delta_gain.enabled:
            pos, quat = self._incremental_target(
                leader_pos, leader_quat, delta_scale=float(delta_scale)
            )
        else:
            pos, quat = self._relative_target(leader_pos, leader_quat)
            self._sync_leader_prev(leader_pos, leader_quat)

        pos, quat = self._rate_limit(pos, quat, dt, motion_scale=float(rate_scale))
        pos = T.clamp_vec(pos, self.cfg.workspace_min, self.cfg.workspace_max)
        quat = T.normalize_quat_wxyz(quat)
        self.state.last_pos = pos
        self.state.last_quat = quat
        info["reason"] = "tracking"
        return pos, quat, grip, info

    def _spike_filter(
        self,
        leader_pos: Sequence[float],
        leader_quat: Sequence[float],
        gripper_norm: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """Drop one-frame leader jumps above configured velocity caps.

        Rejected samples are replaced with the last accepted leader pose /
        gripper so delta-gain does not integrate a glitch and absolute
        mapping does not snap to a bad FK sample.
        """
        cfg = self.cfg
        st = self.state
        pos = T.as_vec(leader_pos, 3)
        quat = T.normalize_quat_wxyz(leader_quat)
        grip = float(np.clip(gripper_norm, 0.0, 1.0))
        info: Dict[str, Any] = {
            "spike_rejected": False,
            "spike_lin": False,
            "spike_ang": False,
            "spike_gripper": False,
        }
        if not bool(getattr(cfg, "spike_filter_enabled", True)):
            st.spike_pos = pos.copy()
            st.spike_quat = quat.copy()
            st.spike_gripper = grip
            return pos, quat, grip, info

        dt = max(float(dt), 1e-4)
        if st.spike_pos is None or st.spike_quat is None:
            st.spike_pos = pos.copy()
            st.spike_quat = quat.copy()
            st.spike_gripper = grip
            return pos, quat, grip, info

        max_dp = float(cfg.spike_max_lin_vel) * dt
        max_da = float(cfg.spike_max_ang_vel) * dt
        max_dg = float(cfg.spike_max_gripper_vel) * dt

        dp = float(np.linalg.norm(pos - st.spike_pos))
        q_rel = T.quat_multiply_wxyz(quat, T.quat_conjugate_wxyz(st.spike_quat))
        da = float(np.linalg.norm(T.quat_wxyz_to_rotvec(q_rel)))
        prev_g = float(st.spike_gripper if st.spike_gripper is not None else grip)
        dg = abs(grip - prev_g)

        rejected = False
        out_pos, out_quat = pos, quat
        out_grip = grip
        if max_dp > 0.0 and dp > max_dp:
            info["spike_lin"] = True
            rejected = True
            out_pos = st.spike_pos.copy()
            out_quat = st.spike_quat.copy()
        if max_da > 0.0 and da > max_da:
            info["spike_ang"] = True
            rejected = True
            out_pos = st.spike_pos.copy()
            out_quat = st.spike_quat.copy()
        if max_dg > 0.0 and dg > max_dg:
            info["spike_gripper"] = True
            rejected = True
            out_grip = prev_g

        info["spike_rejected"] = rejected
        # Accept channels independently so a gripper glitch does not freeze EE.
        if not info["spike_lin"] and not info["spike_ang"]:
            st.spike_pos = pos.copy()
            st.spike_quat = quat.copy()
        if not info["spike_gripper"]:
            st.spike_gripper = grip
        return out_pos, out_quat, out_grip, info

    def _sync_leader_prev(
        self, leader_pos: Sequence[float], leader_quat: Sequence[float]
    ) -> None:
        self.state.leader_prev_pos = T.as_vec(leader_pos, 3)
        self.state.leader_prev_quat = T.normalize_quat_wxyz(leader_quat)

    def _hold(
        self,
        current_dex_pos: Sequence[float],
        current_dex_quat: Sequence[float],
        grip: float,
        info: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        if self.state.last_pos is None:
            self.state.last_pos = T.as_vec(current_dex_pos, 3)
            self.state.last_quat = T.normalize_quat_wxyz(current_dex_quat)
        self.state.last_lin_vel[:] = 0.0
        return self.state.last_pos.copy(), self.state.last_quat.copy(), grip, info

    def _relative_target(
        self,
        leader_pos: Sequence[float],
        leader_quat: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        st = self.state
        cfg = self.cfg
        dp = T.as_vec(leader_pos, 3) - st.leader_origin_pos
        dp_m = cfg.map_leader_vec(dp) * cfg.translation_gain
        pos = st.dex_origin_pos + dp_m

        locked = self._locked_target_quat()
        if locked is not None:
            return pos, locked

        if cfg.fix_orientation:
            return pos, T.normalize_quat_wxyz(st.dex_origin_quat)

        # Space-fixed relative orientation (headcam / world view):
        #   R_rel = R_leader @ R_leader0^T
        #   R_dex = (R_map @ R_rel @ R_map^T) @ R_dex0
        # Body-fixed (q0^{-1}*q) + axes_map on the rotvec is inconsistent:
        # axes_map is a space-frame map, so it must conjugate a space-fixed
        # relative rotation (left-multiply onto dex_origin).
        q_rel = T.quat_multiply_wxyz(
            T.normalize_quat_wxyz(leader_quat),
            T.quat_conjugate_wxyz(st.leader_origin_quat),
        )
        rotvec = T.quat_wxyz_to_rotvec(q_rel) * cfg.rotation_gain
        rotvec_m = cfg.map_leader_vec(rotvec)
        quat = T.quat_multiply_wxyz(
            T.rotvec_to_quat_wxyz(rotvec_m),
            st.dex_origin_quat,
        )
        return pos, quat

    def _incremental_target(
        self,
        leader_pos: Sequence[float],
        leader_quat: Sequence[float],
        *,
        delta_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate scaled per-frame leader deltas onto the last DexMate target.

        Path-dependent: changing ``delta_scale`` does not rewrite past motion
        (no snap, no catch-up to a high-gain absolute pose).
        """
        st = self.state
        cfg = self.cfg
        scale = float(np.clip(delta_scale, 0.0, 1.0))
        leader = T.as_vec(leader_pos, 3)
        q_leader = T.normalize_quat_wxyz(leader_quat)
        prev_l = st.leader_prev_pos
        prev_q = st.leader_prev_quat
        if prev_l is None or prev_q is None:
            self._sync_leader_prev(leader, q_leader)
            return st.last_pos.copy(), st.last_quat.copy()

        dL = leader - prev_l
        pos = st.last_pos + cfg.map_leader_vec(dL) * cfg.translation_gain * scale

        locked = self._locked_target_quat()
        if locked is not None:
            quat = locked
        elif cfg.fix_orientation:
            quat = T.normalize_quat_wxyz(st.last_quat)
        else:
            # Space-fixed leader rotation since previous sample, scaled.
            q_rel = T.quat_multiply_wxyz(q_leader, T.quat_conjugate_wxyz(prev_q))
            rotvec = T.quat_wxyz_to_rotvec(q_rel) * cfg.rotation_gain * scale
            rotvec_m = cfg.map_leader_vec(rotvec)
            quat = T.quat_multiply_wxyz(
                T.rotvec_to_quat_wxyz(rotvec_m),
                st.last_quat,
            )

        self._sync_leader_prev(leader, q_leader)
        return pos, quat

    def _rate_limit(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        dt: float,
        motion_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), 1e-4)
        scale = float(np.clip(motion_scale, 0.0, 1.0))
        prev_pos = self.state.last_pos
        prev_quat = self.state.last_quat
        max_dp = self.cfg.max_lin_vel * scale * dt
        desired_dp = pos - prev_pos
        desired_dist = float(np.linalg.norm(desired_dp))
        limited_dp = T.limit_delta(desired_dp, max_dp)
        if self.cfg.max_lin_acc > 0.0:
            max_dv = self.cfg.max_lin_acc * scale * dt
            desired_v = limited_dp / dt
            dv = desired_v - self.state.last_lin_vel
            dv = T.limit_delta(dv, max_dv)
            vel = self.state.last_lin_vel + dv
            limited_dp = vel * dt
            # Never integrate past the commanded target (accel limiting used to
            # coast/overshoot after the leader stopped).
            step = float(np.linalg.norm(limited_dp))
            if desired_dist <= T.EPS:
                limited_dp = np.zeros(3, dtype=np.float64)
                vel = np.zeros(3, dtype=np.float64)
            elif step > desired_dist and step > T.EPS:
                limited_dp = limited_dp * (desired_dist / step)
                vel = np.zeros(3, dtype=np.float64)
            self.state.last_lin_vel = vel
        else:
            self.state.last_lin_vel = limited_dp / dt
        pos_out = prev_pos + limited_dp

        # Constant world quat: slew toward it (do not hold prev_quat).
        # fix_orientation alone still freezes engage quat with no slew.
        locked = self._locked_target_quat()
        if locked is not None:
            quat = locked
        elif self.cfg.fix_orientation:
            return pos_out, T.normalize_quat_wxyz(prev_quat)

        max_angle = self.cfg.max_ang_vel * scale * dt
        q_rel = T.quat_multiply_wxyz(T.quat_conjugate_wxyz(prev_quat), quat)
        rotvec = T.quat_wxyz_to_rotvec(q_rel)
        angle = float(np.linalg.norm(rotvec))
        if angle > max_angle and angle > T.EPS:
            rotvec = rotvec * (max_angle / angle)
            quat_out = T.quat_multiply_wxyz(prev_quat, T.rotvec_to_quat_wxyz(rotvec))
        else:
            quat_out = quat
        return pos_out, quat_out
