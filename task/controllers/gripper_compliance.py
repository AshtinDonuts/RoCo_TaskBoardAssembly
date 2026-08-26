"""Binary / continuous gripper slew with stall hold-on-contact.

Position-only compliance: slowly drive toward open or closed, and when
fingers stall against an object while closing, freeze near the measured
aperture so the PD drive stops pressing harder. Command lag is capped
(``max_close_lag``) so leader catch-up cannot keep squeezing through contact.
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
    mode: str = "continuous"  # continuous | binary
    open_limit: float = 0.6649704
    close: float = 0.0
    close_norm: float = 0.20
    hysteresis: float = 0.03
    close_speed_rad_s: float = 0.05
    open_speed_rad_s: float = 0.25
    stall_qd: float = 0.02
    stall_err: float = 0.008
    stall_dq: float = 0.005
    stall_min_close_rad: float = 0.03
    # Extra close past measured when latching hold (keep small).
    hold_margin: float = 0.002
    # Hard cap: command may not run more closed than q_meas by more than
    # this (blocks leader-0 "catch-up" through a contacted object).
    max_close_lag: float = 0.01
    # Consecutive lag-saturated, no-progress ticks before contact hold.
    # Keep high enough that slow soft-drive free-air creep does not latch.
    stall_hold_ticks: int = 12
    # Net measured close (rad) during a stall window that proves free-air
    # progress and resets the hold timer.
    stall_progress: float = 0.002

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
            "gripper_max_close_lag": "max_close_lag",
            "gripper_stall_hold_ticks": "stall_hold_ticks",
            "gripper_stall_progress": "stall_progress",
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
            mode=str(getattr(cfg, "gripper_mode", "continuous")).lower(),
            open_limit=float(getattr(cfg, "gripper_open_limit", 0.6649704)),
            close=float(getattr(cfg, "gripper_close", 0.0)),
            close_norm=float(getattr(cfg, "gripper_close_norm", 0.20)),
            hysteresis=float(getattr(cfg, "gripper_hysteresis", 0.03)),
            close_speed_rad_s=float(getattr(cfg, "gripper_close_speed_rad_s", 0.05)),
            open_speed_rad_s=float(getattr(cfg, "gripper_open_speed_rad_s", 0.25)),
            stall_qd=float(getattr(cfg, "gripper_stall_qd", 0.02)),
            stall_err=float(getattr(cfg, "gripper_stall_err", 0.008)),
            stall_dq=float(getattr(cfg, "gripper_stall_dq", 0.005)),
            stall_min_close_rad=float(
                getattr(cfg, "gripper_stall_min_close_rad", 0.03)
            ),
            hold_margin=float(getattr(cfg, "gripper_hold_margin", 0.002)),
            max_close_lag=float(getattr(cfg, "gripper_max_close_lag", 0.01)),
            stall_hold_ticks=int(getattr(cfg, "gripper_stall_hold_ticks", 12)),
            stall_progress=float(getattr(cfg, "gripper_stall_progress", 0.002)),
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
        self._stall_ticks: int = 0
        self._stall_q_ref: Optional[float] = None

    def reset(self, q_meas: Optional[float] = None) -> None:
        self.phase = GripperPhase.OPEN
        self._intent = "open"
        self._q_cmd = None if q_meas is None else float(q_meas)
        self._q_hold = None
        self._q_meas_prev = None if q_meas is None else float(q_meas)
        self._close_start_q = None
        self._stall_ticks = 0
        self._stall_q_ref = None

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
            self._stall_ticks = 0
            self._stall_q_ref = None
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
            self._stall_ticks = 0
            self._stall_q_ref = None
            open_target = float(target)
            if abs(self._q_cmd - open_target) > 1e-6:
                self.phase = GripperPhase.OPENING
            self._q_cmd = self._slew(self._q_cmd, open_target, cfg.open_speed_rad_s, dt)
            if abs(self._q_cmd - open_target) <= 1e-6:
                self._q_cmd = open_target
                self.phase = (
                    GripperPhase.OPEN
                    if abs(open_target - cfg.open_limit) <= 1e-6
                    else GripperPhase.OPENING
                )
            else:
                self.phase = GripperPhase.OPENING
            self._q_meas_prev = q_meas
            return float(np.clip(self._q_cmd, q_lo, q_hi))

        # intent == close — hold always wins over catch-up to a more-closed target.
        if self.phase == GripperPhase.HOLDING and self._q_hold is not None:
            # Only leave hold when the resolved target opens past the hold.
            if float(target) > float(self._q_hold) + max(cfg.hysteresis * 0.1, 1e-3):
                self._q_hold = None
                self._stall_ticks = 0
                self._stall_q_ref = None
                self.phase = GripperPhase.OPENING
                self._intent = "open"
                self._q_cmd = self._slew(
                    float(self._q_cmd), float(target), cfg.open_speed_rad_s, dt
                )
                self._q_meas_prev = q_meas
                return float(np.clip(self._q_cmd, q_lo, q_hi))
            self._q_cmd = float(self._q_hold)
            self._q_meas_prev = q_meas
            return float(np.clip(self._q_cmd, q_lo, q_hi))

        if prev_intent != "close":
            self._close_start_q = q_meas
            self._stall_ticks = 0
            self._stall_q_ref = None
            self.phase = GripperPhase.CLOSING
        elif self.phase in (GripperPhase.OPEN, GripperPhase.OPENING):
            self._close_start_q = q_meas
            self._stall_ticks = 0
            self._stall_q_ref = None
            self.phase = GripperPhase.CLOSING

        close_target = float(target)
        self._q_cmd = self._slew(
            self._q_cmd, close_target, cfg.close_speed_rad_s, dt
        )
        # Compliance priority: never command more closed than measured
        # beyond max_close_lag (stops leader-0 catch-up through contact).
        lag = max(0.0, float(cfg.max_close_lag))
        floor = float(q_meas) - lag
        if self._q_cmd < floor:
            self._q_cmd = floor
        self.phase = GripperPhase.CLOSING

        if self._stalled(q_meas, qd_meas, close_target):
            if self._stall_ticks == 0 or self._stall_q_ref is None:
                self._stall_q_ref = q_meas
            # Slow soft-drive free-air creep still reduces q_meas; that is
            # progress, not contact — reset the hold timer.
            progress = float(self._stall_q_ref) - float(q_meas)
            if progress > float(cfg.stall_progress):
                self._stall_q_ref = q_meas
                self._stall_ticks = 1
            else:
                self._stall_ticks += 1
        else:
            self._stall_ticks = 0
            self._stall_q_ref = None

        if self._stall_ticks >= max(1, int(cfg.stall_hold_ticks)):
            hold = float(np.clip(q_meas - cfg.hold_margin, q_lo, q_hi))
            self._q_hold = hold
            self._q_cmd = hold
            self.phase = GripperPhase.HOLDING
            self._stall_ticks = 0
            self._stall_q_ref = None

        self._q_meas_prev = q_meas
        return float(np.clip(self._q_cmd, q_lo, q_hi))

    def _slew(self, q: float, toward: float, speed: float, dt: float) -> float:
        delta = float(toward) - float(q)
        max_step = max(0.0, float(speed)) * dt
        if abs(delta) <= max_step:
            return float(toward)
        return float(q + np.sign(delta) * max_step)

    def _stalled(self, q_meas: float, qd_meas: float, close_target: float) -> bool:
        """True when lag is saturated and fingers are not making progress.

        Requires ``closing_err >= stall_err`` (command pressing against the
        lag cap / object). Slow free-air creep must not latch: callers also
        reset the hold timer on net ``stall_progress``. Near the close
        target we never hold (empty finish).
        """
        cfg = self.cfg
        if abs(qd_meas) >= cfg.stall_qd:
            return False
        q_cmd = float(self._q_cmd if self._q_cmd is not None else q_meas)
        start = float(self._close_start_q if self._close_start_q is not None else q_meas)
        closed_cmd = start - q_cmd
        closed_meas = start - float(q_meas)
        if closed_cmd < cfg.stall_min_close_rad and closed_meas < cfg.stall_min_close_rad:
            return False
        # Empty finish: already at/near commanded close — keep slewing, no hold.
        near = max(float(cfg.max_close_lag), float(cfg.stall_err))
        if float(q_meas) <= float(close_target) + near:
            return False
        want_more_close = float(close_target) < float(q_meas) - 1e-4
        if not want_more_close:
            return False
        closing_err = float(q_meas) - q_cmd
        # Must be pressing (lag saturated). Soft "want_more_close alone"
        # false-triggered mid free-air close with weak drives.
        if closing_err < cfg.stall_err:
            return False
        dq = abs(float(q_meas) - float(self._q_meas_prev))
        if dq > cfg.stall_dq:
            return False
        return True

# Back-compat typing alias for call sites.
GripperCmd = Union[str, float, None]
