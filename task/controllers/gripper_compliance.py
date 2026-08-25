"""Binary / continuous gripper slew with stall hold-on-contact.

Position-only compliance: slowly drive toward open or closed, and when
fingers stall against an object while closing, freeze (slightly past)
the measured aperture so the PD drive stops pressing harder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np


class GripperPhase(str, Enum):
    OPEN = "open"
    CLOSING = "closing"
    HOLDING = "holding"
    OPENING = "opening"


def _as_bool(val: Any, default: bool = True) -> bool:
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(default)


@dataclass
class GripperComplianceConfig:
    """Tunables for slow-close + stall hold.

    Set ``enabled=False`` (YAML ``gripper_compliance_enabled: false``) to
    bypass the entire module: callers snap to open/close/target with no
    slew or stall hold.
    """

    enabled: bool = True
    mode: str = "binary"  # binary | continuous
    open_limit: float = 0.6649704
    close: float = 0.0
    close_norm: float = 0.20
    hysteresis: float = 0.03
    close_speed_rad_s: float = 0.05
    open_speed_rad_s: float = 0.25
    stall_qd: float = 0.02
    stall_err: float = 0.02
    stall_dq: float = 0.005
    stall_min_close_rad: float = 0.03
    hold_margin: float = 0.01

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "GripperComplianceConfig":
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}
        # Accept retarget YAML key names as aliases.
        alias = {
            "gripper_compliance_enabled": "enabled",
            "gripper_mode": "mode",
            "gripper_open_limit": "open_limit",
            "gripper_close": "close",
            "gripper_close_norm": "close_norm",
            "gripper_hysteresis": "hysteresis",
            "gripper_close_speed_rad_s": "close_speed_rad_s",
            "gripper_open_speed_rad_s": "open_speed_rad_s",
            "gripper_stall_qd": "stall_qd",
            "gripper_stall_err": "stall_err",
            "gripper_stall_dq": "stall_dq",
            "gripper_stall_min_close_rad": "stall_min_close_rad",
            "gripper_hold_margin": "hold_margin",
        }
        for key, val in data.items():
            field = alias.get(key, key)
            if field in known:
                kwargs[field] = val
        if "mode" in kwargs:
            kwargs["mode"] = str(kwargs["mode"]).lower()
        if "enabled" in kwargs:
            kwargs["enabled"] = _as_bool(kwargs["enabled"], True)
        return cls(**kwargs)

    @classmethod
    def from_retarget_cfg(cls, cfg: Any) -> "GripperComplianceConfig":
        """Build from a RetargetConfig-like object (attrs or dict)."""
        if cfg is None:
            return cls()
        if isinstance(cfg, dict):
            return cls.from_dict(cfg)
        return cls(
            enabled=_as_bool(getattr(cfg, "gripper_compliance_enabled", True), True),
            mode=str(getattr(cfg, "gripper_mode", "binary")).lower(),
            open_limit=float(getattr(cfg, "gripper_open_limit", 0.6649704)),
            close=float(getattr(cfg, "gripper_close", 0.0)),
            close_norm=float(getattr(cfg, "gripper_close_norm", 0.20)),
            hysteresis=float(getattr(cfg, "gripper_hysteresis", 0.03)),
            close_speed_rad_s=float(getattr(cfg, "gripper_close_speed_rad_s", 0.05)),
            open_speed_rad_s=float(getattr(cfg, "gripper_open_speed_rad_s", 0.25)),
            stall_qd=float(getattr(cfg, "gripper_stall_qd", 0.02)),
            stall_err=float(getattr(cfg, "gripper_stall_err", 0.02)),
            stall_dq=float(getattr(cfg, "gripper_stall_dq", 0.005)),
            stall_min_close_rad=float(
                getattr(cfg, "gripper_stall_min_close_rad", 0.03)
            ),
            hold_margin=float(getattr(cfg, "gripper_hold_margin", 0.01)),
        )


def binary_intent_from_norm(
    gripper_norm: float,
    *,
    close_norm: float,
    hysteresis: float,
    prev_intent: str,
) -> str:
    """Map leader gripper_norm to open/close with hysteresis band."""
    g = float(np.clip(gripper_norm, 0.0, 1.0))
    close_at = float(np.clip(close_norm, 0.0, 0.95))
    open_at = float(np.clip(close_at + max(0.0, hysteresis), 0.0, 1.0))
    if g <= close_at:
        return "close"
    if g >= open_at:
        return "open"
    return "open" if prev_intent not in ("open", "close") else prev_intent


def continuous_target_from_norm(
    gripper_norm: float,
    *,
    close: float,
    open_limit: float,
    close_norm: float,
) -> float:
    """Linear remap leader [close_norm, 1] → [close, open_limit]."""
    g = float(np.clip(gripper_norm, 0.0, 1.0))
    close_at = float(np.clip(close_norm, 0.0, 0.95))
    if g <= close_at:
        g_eff = 0.0
    else:
        g_eff = (g - close_at) / (1.0 - close_at)
    span = float(open_limit) - float(close)
    return float(close) + g_eff * span


class GripperCompliance:
    """Per-arm state machine: open / closing / holding / opening."""

    def __init__(self, cfg: Optional[GripperComplianceConfig] = None) -> None:
        self.cfg = cfg or GripperComplianceConfig()
        self.phase: GripperPhase = GripperPhase.OPEN
        self._intent: str = "open"
        self._q_cmd: Optional[float] = None
        self._q_hold: Optional[float] = None
        self._q_meas_prev: Optional[float] = None
        self._close_start_q: Optional[float] = None

    def reset(self, q_meas: Optional[float] = None) -> None:
        self.phase = GripperPhase.OPEN
        self._intent = "open"
        self._q_cmd = None if q_meas is None else float(q_meas)
        self._q_hold = None
        self._q_meas_prev = None if q_meas is None else float(q_meas)
        self._close_start_q = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", True))

    @property
    def intent(self) -> str:
        return self._intent

    @property
    def q_cmd(self) -> Optional[float]:
        return self._q_cmd

    def intent_from_norm(self, gripper_norm: float) -> str:
        return binary_intent_from_norm(
            gripper_norm,
            close_norm=self.cfg.close_norm,
            hysteresis=self.cfg.hysteresis,
            prev_intent=self._intent,
        )

    def _resolve_target(
        self,
        *,
        intent: Optional[str],
        target_rad: Optional[float],
        gripper_norm: Optional[float],
        q_cmd: float,
    ) -> tuple:
        """Return (intent, target_rad) from the caller's inputs."""
        cfg = self.cfg
        q_lo = float(min(cfg.close, cfg.open_limit))
        q_hi = float(max(cfg.close, cfg.open_limit))
        mode = str(cfg.mode).lower()

        if gripper_norm is not None and intent is None and target_rad is None:
            if mode == "continuous":
                target_rad = continuous_target_from_norm(
                    float(gripper_norm),
                    close=cfg.close,
                    open_limit=cfg.open_limit,
                    close_norm=cfg.close_norm,
                )
            else:
                intent = self.intent_from_norm(float(gripper_norm))

        if target_rad is not None:
            target = float(np.clip(target_rad, q_lo, q_hi))
            if target < q_cmd - 1e-9:
                intent = "close"
            elif target > q_cmd + 1e-9:
                intent = "open"
            else:
                intent = self._intent
            return intent, target

        if intent is None:
            intent = self._intent
        intent = str(intent).lower()
        if intent not in ("open", "close"):
            raise ValueError(f"intent must be 'open' or 'close'; got {intent!r}")
        target = cfg.close if intent == "close" else cfg.open_limit
        return intent, float(target)

    def update(
        self,
        *,
        q_meas: float,
        qd_meas: float,
        dt: float,
        intent: Optional[str] = None,
        target_rad: Optional[float] = None,
        gripper_norm: Optional[float] = None,
    ) -> float:
        """Advance one control tick; return commanded primary gripper aperture.

        Provide one of:
          - intent: \"open\" | \"close\"
          - target_rad: absolute aperture (continuous)
          - gripper_norm: leader norm; interpreted via cfg.mode

        When ``cfg.enabled`` is False, snaps immediately to the resolved
        target (no slew, no stall hold).
        """
        cfg = self.cfg
        q_meas = float(q_meas)
        qd_meas = float(qd_meas)
        dt = float(max(dt, 1e-6))
        q_lo = float(min(cfg.close, cfg.open_limit))
        q_hi = float(max(cfg.close, cfg.open_limit))

        if self._q_cmd is None:
            self._q_cmd = q_meas
        if self._q_meas_prev is None:
            self._q_meas_prev = q_meas

        intent, target = self._resolve_target(
            intent=intent,
            target_rad=target_rad,
            gripper_norm=gripper_norm,
            q_cmd=float(self._q_cmd),
        )

        if not self.enabled:
            # Master switch off: snap to target, clear hold state.
            self._intent = intent
            self._q_hold = None
            self._close_start_q = None
            self._q_cmd = float(target)
            self._q_meas_prev = q_meas
            self.phase = (
                GripperPhase.OPEN if intent == "open" else GripperPhase.CLOSING
            )
            if abs(self._q_cmd - cfg.open_limit) <= 1e-9:
                self.phase = GripperPhase.OPEN
            elif abs(self._q_cmd - cfg.close) <= 1e-9:
                self.phase = GripperPhase.CLOSING
            return float(np.clip(self._q_cmd, q_lo, q_hi))

        prev_intent = self._intent
        self._intent = intent

        if intent == "open":
            self._q_hold = None
            self._close_start_q = None
            if self.phase in (GripperPhase.HOLDING, GripperPhase.CLOSING, GripperPhase.OPEN):
                if abs(self._q_cmd - cfg.open_limit) > 1e-6:
                    self.phase = GripperPhase.OPENING
            self._q_cmd = self._slew(self._q_cmd, cfg.open_limit, cfg.open_speed_rad_s, dt)
            if abs(self._q_cmd - cfg.open_limit) <= 1e-6:
                self._q_cmd = cfg.open_limit
                self.phase = GripperPhase.OPEN
            else:
                self.phase = GripperPhase.OPENING
            self._q_meas_prev = q_meas
            return float(np.clip(self._q_cmd, q_lo, q_hi))

        # intent == close
        if prev_intent != "close":
            self._close_start_q = q_meas
            if self.phase != GripperPhase.HOLDING:
                self.phase = GripperPhase.CLOSING
        elif self.phase in (GripperPhase.OPEN, GripperPhase.OPENING):
            self._close_start_q = q_meas
            self.phase = GripperPhase.CLOSING

        if self.phase == GripperPhase.HOLDING and self._q_hold is not None:
            self._q_cmd = float(self._q_hold)
            self._q_meas_prev = q_meas
            return float(np.clip(self._q_cmd, q_lo, q_hi))

        # Slew toward close target (binary close or continuous target).
        close_target = float(target)
        self._q_cmd = self._slew(
            self._q_cmd, close_target, cfg.close_speed_rad_s, dt
        )
        self.phase = GripperPhase.CLOSING

        if self._stalled(q_meas, qd_meas, close_target):
            hold = float(np.clip(q_meas - cfg.hold_margin, q_lo, q_hi))
            self._q_hold = hold
            self._q_cmd = hold
            self.phase = GripperPhase.HOLDING

        self._q_meas_prev = q_meas
        return float(np.clip(self._q_cmd, q_lo, q_hi))

    def _slew(self, q: float, toward: float, speed: float, dt: float) -> float:
        delta = float(toward) - float(q)
        max_step = max(0.0, float(speed)) * dt
        if abs(delta) <= max_step:
            return float(toward)
        return float(q + np.sign(delta) * max_step)

    def _stalled(self, q_meas: float, qd_meas: float, close_target: float) -> bool:
        cfg = self.cfg
        if abs(qd_meas) >= cfg.stall_qd:
            return False
        q_cmd = float(self._q_cmd if self._q_cmd is not None else q_meas)
        start = float(self._close_start_q if self._close_start_q is not None else q_meas)
        # Larger aperture = more open; closing decreases q.
        closed_cmd = start - q_cmd
        closed_meas = start - float(q_meas)
        if closed_cmd < cfg.stall_min_close_rad and closed_meas < cfg.stall_min_close_rad:
            return False
        # Positive when command is more closed than the measured fingers.
        closing_err = float(q_meas) - q_cmd
        dq = abs(float(q_meas) - float(self._q_meas_prev))
        # Require lag AND near-zero motion so free-air tracking (cmd one
        # slew-step ahead) does not false-trigger hold.
        if closing_err < cfg.stall_err:
            return False
        if dq > cfg.stall_dq:
            return False
        # Ignore stall once we have already reached the close target with
        # negligible lag (empty grasp finished).
        if abs(q_cmd - float(close_target)) <= 1e-6 and closing_err < cfg.stall_err:
            return False
        return True

# Back-compat typing alias for call sites.
GripperCmd = Union[str, float, None]
