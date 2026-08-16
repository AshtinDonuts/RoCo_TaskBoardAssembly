"""Relative Cartesian retargeting from an ALOHA leader to DexMate Vega 1U."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import transforms as T


@dataclass
class RetargetConfig:
    axes_perm: Tuple[int, int, int] = (0, 1, 2)
    axes_sign: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Optional 3x3: DexMate_delta = axes_map @ leader_delta.
    # When set, replaces axes_perm/axes_sign (used for head-camera view).
    axes_map: Optional[Tuple[Tuple[float, float, float], ...]] = None
    translation_gain: float = 1.0
    rotation_gain: float = 1.0
    workspace_min: Tuple[float, float, float] = (-1.5, -1.5, 0.02)
    workspace_max: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    max_lin_vel: float = 0.35
    max_ang_vel: float = 1.2
    max_lin_acc: float = 2.0
    gripper_open_limit: float = 0.6649704
    gripper_close: float = 0.0
    # Leader gripper_norm at/below which DexMate is fully closed. Absorbs
    # leader calibration slack so "almost closed" maps to aperture 0.
    gripper_close_norm: float = 0.20
    gripper_hysteresis: float = 0.03
    stale_hold_s: float = 0.10
    stale_pause_s: float = 0.50

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "RetargetConfig":
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
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
    last_pos: Optional[np.ndarray] = None
    last_quat: Optional[np.ndarray] = None
    last_gripper: float = 0.0
    last_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    last_cmd: str = "none"


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
        st.last_pos = st.dex_origin_pos.copy()
        st.last_quat = st.dex_origin_quat.copy()
        st.last_lin_vel[:] = 0.0

    def disengage(self) -> None:
        self.state.engaged = False
        self.state.leader_origin_pos = None
        self.state.leader_origin_quat = None
        self.state.dex_origin_pos = None
        self.state.dex_origin_quat = None

    def map_gripper(self, gripper_norm: float) -> float:
        cfg = self.cfg
        g = float(np.clip(gripper_norm, 0.0, 1.0))
        # Linear remap: leader [close_norm, 1] → DexMate [closed, open].
        # Anything at/below close_norm is treated as fully closed so a
        # slightly short physical travel still seals the virtual fingers.
        close_at = float(np.clip(cfg.gripper_close_norm, 0.0, 0.95))
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
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        info: Dict[str, Any] = {"held": False, "engaged": self.state.engaged}
        if not deadman:
            info["held"] = True
            info["reason"] = "deadman"
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

        grip = self.map_gripper(gripper_norm)
        if clutch and not self.state.engaged:
            self.capture_origins(leader_pos, leader_quat, current_dex_pos, current_dex_quat)
            info["engaged"] = True
            info["reason"] = "clutch_engage"
            return (
                self.state.last_pos.copy(),
                self.state.last_quat.copy(),
                grip,
                info,
            )

        pos, quat = self._relative_target(leader_pos, leader_quat)
        pos, quat = self._rate_limit(pos, quat, dt)
        pos = T.clamp_vec(pos, self.cfg.workspace_min, self.cfg.workspace_max)
        quat = T.normalize_quat_wxyz(quat)
        self.state.last_pos = pos
        self.state.last_quat = quat
        info["reason"] = "tracking"
        return pos, quat, grip, info

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

    def _rate_limit(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), 1e-4)
        prev_pos = self.state.last_pos
        prev_quat = self.state.last_quat
        max_dp = self.cfg.max_lin_vel * dt
        desired_dp = pos - prev_pos
        desired_dist = float(np.linalg.norm(desired_dp))
        limited_dp = T.limit_delta(desired_dp, max_dp)
        if self.cfg.max_lin_acc > 0.0:
            max_dv = self.cfg.max_lin_acc * dt
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

        max_angle = self.cfg.max_ang_vel * dt
        q_rel = T.quat_multiply_wxyz(T.quat_conjugate_wxyz(prev_quat), quat)
        rotvec = T.quat_wxyz_to_rotvec(q_rel)
        angle = float(np.linalg.norm(rotvec))
        if angle > max_angle and angle > T.EPS:
            rotvec = rotvec * (max_angle / angle)
            quat_out = T.quat_multiply_wxyz(prev_quat, T.rotvec_to_quat_wxyz(rotvec))
        else:
            quat_out = quat
        return pos_out, quat_out
