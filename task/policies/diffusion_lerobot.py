"""Deploy a trained LeRobot Diffusion Policy in the RoCo Isaac Sim harness.

The model runs in a separate LeRobot venv (`dp_server.py`) and this adapter
talks to it over a length-prefixed pickle pipe.

Action convention (HF dataset revision dc03b003...):
  14-D = left[xyz + intrinsic XYZ Euler + gripper] + right[...]
  Euler angles are unwrapped over time (may exceed ±π). Decode with
  ``Rotation.from_euler("XYZ", ...)`` — not rotation-vector conversion.

Control cadence:
  Dataset / training are 10 Hz. The harness runs physics at 200 Hz, while one
  rendered outer-loop iteration can advance multiple physics ticks. The
  adapter therefore schedules queries from ``obs.step_idx`` rather than by
  counting calls to ``act``. It consumes one LeRobot ``select_action`` step
  every 20 physics ticks and holds the absolute IK target between updates.
"""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys

import numpy as np

import param_config as pc  # noqa: E402  (kept for parity with other policies)
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Constants inlined so this file has no extra module dependencies. ---
GRIPPER_OPEN_LIMIT = 0.6649704       # L_gripper joint value at fully open
_IMG_H, _IMG_W = 240, 320            # RGB size the model was trained at
PHYSICS_HZ = 200
CONTROL_HZ = 10
CONTROL_PERIOD_TICKS = PHYSICS_HZ // CONTROL_HZ  # 20

# Published HF observation.state layout (44-D), documented explicitly.
STATE_SLICES = {
    "left_ee_pose": slice(0, 7),       # xyz + qwxyz
    "right_ee_pose": slice(7, 14),
    "left_joint_pos": slice(14, 21),
    "right_joint_pos": slice(21, 28),
    "left_joint_vel": slice(28, 35),
    "right_joint_vel": slice(35, 42),
    "left_gripper": slice(42, 43),
    "right_gripper": slice(43, 44),
}


def _resize_rgb(img):
    """HxWx3 uint8 (any size) -> (240,320,3) uint8. None -> zeros."""
    if img is None:
        return np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.shape[0] == _IMG_H and a.shape[1] == _IMG_W:
        return a.astype(np.uint8)
    try:
        import cv2
        return cv2.resize(a, (_IMG_W, _IMG_H),
                          interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        ys = max(1, a.shape[0] // _IMG_H)
        xs = max(1, a.shape[1] // _IMG_W)
        out = a[::ys, ::xs, :3]
        return out[:_IMG_H, :_IMG_W].astype(np.uint8)


def euler_xyz_to_quat_wxyz(rx, ry, rz):
    """Decode unwrapped intrinsic XYZ Euler radians -> quaternion wxyz.

    Values may exceed ±π (dataset applies ``np.unwrap``). SciPy's
    ``from_euler('XYZ', ...)`` accepts the unwrapped continuous target
    directly; do not wrap before conversion.
    """
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_euler("XYZ", [rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def left_action_to_ik_target(action_14: np.ndarray):
    """Map 14-D policy action -> (pos_xyz, quat_wxyz, gripper_joint).

    Only gripper ratios are clipped to [0, 1]. Position and unwrapped Euler
    orientation are preserved through model postprocessing as physical units.
    """
    a = np.asarray(action_14, dtype=np.float64).reshape(-1)
    if a.shape[0] != 14:
        raise ValueError(f"expected 14-D action, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"non-finite action: {a}")
    pos = a[:3].copy()
    quat = euler_xyz_to_quat_wxyz(a[3], a[4], a[5])
    grip = float(np.clip(a[6], 0.0, 1.0)) * GRIPPER_OPEN_LIMIT
    return pos, quat, grip


def camera_payload_from_obs(obs: Observation) -> dict:
    """Map harness RGB keys to the three HF camera streams (240x320)."""
    return {
        "head": _resize_rgb(obs.rgb.get("head")),
        "left": _resize_rgb(obs.rgb.get("L_wrist")),
        "right": _resize_rgb(obs.rgb.get("R_wrist")),
    }


class DiffusionLeRobotPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        # Right-arm controller is not part of the public EnvInfo contract; use
        # it only if the harness exposes it (see the RIGHT ARM note up top).
        self.R = getattr(env_info, "R_controller", None)
        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        # Derive control period from harness physics_dt when available.
        physics_dt = float(getattr(env_info, "physics_dt", 1.0 / PHYSICS_HZ))
        period = max(1, int(round((1.0 / CONTROL_HZ) / physics_dt)))
        self._control_period = period

        ckpt = os.environ.get("DP_CKPT")
        if not ckpt:
            raise ValueError("set DP_CKPT to the checkpoint pretrained_model dir")
        server_py = os.environ.get("DP_SERVER_PY")
        if not server_py:
            default_venv = os.path.join(
                os.path.dirname(_TASK_DIR), ".venv_lerobot", "bin", "python")
            server_py = default_venv if os.path.isfile(default_venv) else sys.executable
        # CLEAN env: the harness runs in Isaac's interpreter which exports
        # PYTHONPATH / LD_LIBRARY_PATH / CARB_* pointing at Isaac packages. If
        # the model's venv subprocess inherits those it loads Isaac's torch /
        # libs and crashes. Pass only what the server needs.
        keep = ("HOME", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES", "UV_PYTHON_INSTALL_DIR")
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin",
               "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        for k in keep:
            if k in os.environ:
                env[k] = os.environ[k]
        _log_path = os.environ.get(
            "DP_SERVER_LOG", os.path.join(_TASK_DIR, "dp_server.log"))
        self._err = open(_log_path, "w")
        self._proc = subprocess.Popen(
            [server_py, os.path.join(_TASK_DIR, "dp_server.py"), ckpt],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
            env=env, cwd=_TASK_DIR)
        print(f"[dp] spawned inference server (ckpt={ckpt})", flush=True)
        print(f"[dp] control period={self._control_period} ticks "
              f"({CONTROL_HZ} Hz @ physics_dt={physics_dt})", flush=True)
        # No handshake; give it a moment to load and check it's alive.
        import time
        time.sleep(2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"dp_server died on startup (exit {self._proc.returncode}); "
                f"see {_log_path}")
        self._last_action = None
        self._hold_target = None  # (pos, quat, grip)
        self._last_query_step = None

    # ---- length-prefixed pickle pipe ----
    def _send(self, obj):
        b = pickle.dumps(obj)
        self._proc.stdin.write(struct.pack(">I", len(b)) + b)
        self._proc.stdin.flush()

    def _recv(self):
        h = self._proc.stdout.read(4)
        if len(h) < 4:
            raise RuntimeError("dp_server closed")
        n = struct.unpack(">I", h)[0]
        buf = b""
        while len(buf) < n:
            buf += self._proc.stdout.read(n - len(buf))
        return pickle.loads(buf)

    # ---- Policy API ----
    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._send({"cmd": "reset"})
        self._recv()
        self._last_query_step = None
        self._hold_target = None
        self._last_action = None

    def _build_state(self, obs: Observation) -> np.ndarray:
        """Rebuild the 44-D training-time observation.state from `obs`."""
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)
        # L EE pose: prefer the controller (byte-for-byte what collection used);
        # fall back to the Observation field the public API guarantees.
        if self.L is not None:
            Lp, Lq = self.L.end_effector.get_world_pose()
        else:
            Lp, Lq = obs.ee_pose_L
        # R EE pose: not in the public EnvInfo -> identity unless R_controller given.
        if self.R is not None:
            Rp, Rq = self.R.end_effector.get_world_pose()
        else:
            Rp, Rq = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        ratio = lambda v: float(np.clip(v / GRIPPER_OPEN_LIMIT, 0, 1))
        return np.concatenate([
            np.asarray(Lp).reshape(-1)[:3], np.asarray(Lq).reshape(-1)[:4],
            np.asarray(Rp).reshape(-1)[:3], np.asarray(Rq).reshape(-1)[:4],
            q[self._Li], q[self._Ri], qd[self._Li], qd[self._Ri],
            [ratio(q[self._Lg])],
            [ratio(q[self._Rg]) if self._Rg is not None else 0.0],
        ]).astype(np.float32)

    def _query_policy(self, obs: Observation) -> np.ndarray:
        state = self._build_state(obs)
        cams = camera_payload_from_obs(obs)
        self._send({
            "state": state,
            "head": cams["head"],
            "left": cams["left"],
            "right": cams["right"],
        })
        a = np.asarray(self._recv()["action"], np.float64)
        self._last_action = a
        return a

    def act(self, obs: Observation):
        # Request one action per training-time interval. A rendered World.step
        # currently advances 20 physics ticks, so counting act() calls here
        # would accidentally reduce a 10 Hz policy to 0.5 Hz.
        step_idx = int(obs.step_idx)
        query_due = (
            self._hold_target is None
            or self._last_query_step is None
            or step_idx - self._last_query_step >= self._control_period
            or step_idx < self._last_query_step
        )
        if query_due:
            a = self._query_policy(obs)
            self._hold_target = left_action_to_ik_target(a)
            self._last_query_step = step_idx
        pos, quat, grip = self._hold_target
        return self.L.forward(pos, quat, grip)

    def is_done(self, obs: Observation) -> bool:
        return False  # harness advances on snap fire / per-part timeout

    def __del__(self):
        try:
            self._proc.terminate()
        except Exception:
            pass
