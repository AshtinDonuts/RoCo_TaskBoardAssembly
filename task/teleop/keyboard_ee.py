"""Virtual Cartesian leader driven by keyboard hold/edge tokens."""
from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .protocol import COMMANDS
from . import transforms as T

# Keys not used by clutch / recording / part-done / estop.
EE_KEY_TO_CMD: Dict[str, str] = {
    "i": "ee+y",
    "k": "ee-y",
    "j": "ee-x",
    "l": "ee+x",
    "t": "ee+z",
    "g": "ee-z",
    "q": "ee+yaw",
    "a": "ee-yaw",
    "w": "ee+pitch",
    "d": "ee-pitch",
    "z": "ee+roll",
    "c": "ee-roll",
    "f": "grip_close",
    "v": "grip_open",
}

KEYBOARD_HELP = (
    "EEF translate (hold): i/k fwd/back  j/l left/right  t/g up/down\n"
    "EEF rotate (hold):    q/a yaw  w/d pitch  z/c roll\n"
    "Gripper:              f close  v open  (binary)\n"
    "Task: space=clutch  r=recenter  p=pause  u=resume  n=part_done  "
    "x=abort  e=estop  s=start\n"
    "Record: Right=save  Left=rerecord  Esc=stop\n"
    "Focus the Isaac window for carb keyboard teleop."
)

_LIN = {
    "ee+x": np.array([1.0, 0.0, 0.0]),
    "ee-x": np.array([-1.0, 0.0, 0.0]),
    "ee+y": np.array([0.0, 1.0, 0.0]),
    "ee-y": np.array([0.0, -1.0, 0.0]),
    "ee+z": np.array([0.0, 0.0, 1.0]),
    "ee-z": np.array([0.0, 0.0, -1.0]),
}
_ANG = {
    "ee+roll": np.array([1.0, 0.0, 0.0]),
    "ee-roll": np.array([-1.0, 0.0, 0.0]),
    "ee+pitch": np.array([0.0, 1.0, 0.0]),
    "ee-pitch": np.array([0.0, -1.0, 0.0]),
    "ee+yaw": np.array([0.0, 0.0, 1.0]),
    "ee-yaw": np.array([0.0, 0.0, -1.0]),
}


class KeyboardEE:
    """Integrates held-key velocities into a virtual leader EE pose."""

    def __init__(
        self,
        lin_vel_mps: float = 0.12,
        ang_vel_rps: float = 0.8,
        lin_step_m: float = 0.01,
        ang_step_rad: float = 0.05,
        workspace_min: Tuple[float, float, float] = (-0.35, -0.35, -0.15),
        workspace_max: Tuple[float, float, float] = (0.35, 0.35, 0.35),
    ) -> None:
        self.lin_vel_mps = float(lin_vel_mps)
        self.ang_vel_rps = float(ang_vel_rps)
        self.lin_step_m = float(lin_step_m)
        self.ang_step_rad = float(ang_step_rad)
        self.workspace_min = np.asarray(workspace_min, dtype=np.float64)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float64)
        self.pos = np.zeros(3, dtype=np.float64)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.gripper = 1.0
        self.clutch = True
        self.deadman = True
        self.pending: List[str] = []
        self._lock = threading.Lock()
        self.moved = False

    def apply(self, token: str) -> None:
        """One-shot nudge (TTY pulse / tests)."""
        with self._lock:
            if token in _LIN:
                self.pos = np.clip(
                    self.pos + self.lin_step_m * _LIN[token],
                    self.workspace_min,
                    self.workspace_max,
                )
                self.clutch = True
                self.moved = True
                return
            if token in _ANG:
                delta = T.rotvec_to_quat_wxyz(self.ang_step_rad * _ANG[token])
                self.quat = T.quat_multiply_wxyz(delta, self.quat)
                self.clutch = True
                self.moved = True
                return
            self._apply_edge_locked(token)

    def apply_holds(self, tokens: Iterable[str], dt: float) -> bool:
        """Integrate held motion tokens for ``dt`` seconds. Returns True if moved."""
        dt = max(float(dt), 0.0)
        moved = False
        with self._lock:
            lin = np.zeros(3, dtype=np.float64)
            ang = np.zeros(3, dtype=np.float64)
            for token in tokens:
                if token in _LIN:
                    lin = lin + _LIN[token]
                    moved = True
                elif token in _ANG:
                    ang = ang + _ANG[token]
                    moved = True
                elif token == "grip_close":
                    self.gripper = 0.0
                elif token == "grip_open":
                    self.gripper = 1.0
            if float(np.linalg.norm(lin)) > 1e-9:
                lin = lin / float(np.linalg.norm(lin))
                self.pos = np.clip(
                    self.pos + self.lin_vel_mps * dt * lin,
                    self.workspace_min,
                    self.workspace_max,
                )
                self.clutch = True
            if float(np.linalg.norm(ang)) > 1e-9:
                ang = ang / float(np.linalg.norm(ang))
                delta = T.rotvec_to_quat_wxyz(self.ang_vel_rps * dt * ang)
                self.quat = T.quat_multiply_wxyz(delta, self.quat)
                self.clutch = True
            if moved:
                self.moved = True
        return moved

    def apply_edge(self, token: str) -> None:
        with self._lock:
            self._apply_edge_locked(token)

    def _apply_edge_locked(self, token: str) -> None:
        if token == "grip_close":
            self.gripper = 0.0
            return
        if token == "grip_open":
            self.gripper = 1.0
            return
        if token == "clutch_toggle":
            self.clutch = not self.clutch
            self.pending.append("recenter" if self.clutch else "none")
            return
        if token == "estop":
            self.deadman = False
            self.clutch = False
            self.pending.append(token)
            return
        if token in ("start", "resume"):
            self.deadman = True
            self.pending.append(token)
            return
        if token in COMMANDS:
            self.pending.append(token)

    def _pop_cmd_locked(self) -> str:
        if not self.pending:
            return "none"
        cmd = self.pending.pop(0)
        return cmd if cmd in COMMANDS else "none"

    def pop_cmd(self) -> str:
        with self._lock:
            return self._pop_cmd_locked()

    def snapshot(self):
        with self._lock:
            cmd = self._pop_cmd_locked()
            return (
                self.pos.copy(),
                T.normalize_quat_wxyz(self.quat),
                float(self.gripper),
                bool(self.clutch),
                bool(self.deadman),
                cmd,
            )

    def take_sample(self) -> dict:
        with self._lock:
            cmd = self._pop_cmd_locked()
            return {
                "ee_pos": self.pos.copy().tolist(),
                "ee_quat_wxyz": T.normalize_quat_wxyz(self.quat).tolist(),
                "gripper_norm": float(self.gripper),
                "clutch": bool(self.clutch),
                "deadman": bool(self.deadman),
                "cmd": cmd,
                "joints": [0.0, -0.6, 0.5, 0.0, 1.2, 0.0],
                "timestamp_ns": None,
            }
