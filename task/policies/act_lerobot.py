"""Deploy a trained LeRobot ACT policy in the Isaac Sim harness.

Mirrors ``diffusion_lerobot.py``: ACT runs in a separate lerobot venv process
(``act_server.py``) and this adapter talks over a stdin/stdout pickle pipe.

Env:
  ACT_CKPT         path to the checkpoint pretrained_model dir (required)
  ACT_SERVER_PY    python for the model's venv (required)
  ACT_SERVER       optional path to act_server.py
  ACT_SERVER_LOG   optional stderr log path
  ACT_CUDA_VISIBLE_DEVICES  optional GPU id(s) for the sidecar
  ACT_N_ACTION_STEPS  optional override for closed-loop action horizon
  ACT_TEMPORAL_ENSEMBLE_COEFF  optional ACT temporal ensembling coeff
"""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import time

import numpy as np

from policy_api import EnvInfo, Observation, PartTarget, Policy

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRIPPER_OPEN_LIMIT = 0.6649704
_IMG_H, _IMG_W = 240, 320


def _resize_rgb(img):
    if img is None:
        return np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if a.size and float(np.nanmax(a)) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[0] == _IMG_H and a.shape[1] == _IMG_W:
        return a.astype(np.uint8, copy=False)
    try:
        import cv2

        return cv2.resize(a, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA).astype(
            np.uint8
        )
    except Exception:
        ys = max(1, a.shape[0] // _IMG_H)
        xs = max(1, a.shape[1] // _IMG_W)
        out = a[::ys, ::xs, :3]
        return out[:_IMG_H, :_IMG_W].astype(np.uint8)


def _rotvec_to_quat_wxyz(rx, ry, rz):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_rotvec([rx, ry, rz]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class ACTLeRobotPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self._proc = None
        self._err = None
        self.L = env_info.L_controller
        self.R = getattr(env_info, "R_controller", None)
        if self.L is None:
            raise ValueError("ACTLeRobotPolicy requires env_info.L_controller")

        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None

        ckpt = os.environ.get("ACT_CKPT")
        if not ckpt:
            raise ValueError("set ACT_CKPT to a LeRobot ACT checkpoint")
        server_py = os.environ.get("ACT_SERVER_PY")
        if not server_py:
            raise ValueError("set ACT_SERVER_PY to the LeRobot venv python")
        server_script = os.environ.get(
            "ACT_SERVER", os.path.join(_TASK_DIR, "act_server.py")
        )

        keep = (
            "HOME",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "UV_PYTHON_INSTALL_DIR",
            "HF_HOME",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "TOKENIZERS_PARALLELISM",
            "PYTORCH_CUDA_ALLOC_CONF",
            "ACT_N_ACTION_STEPS",
            "ACT_TEMPORAL_ENSEMBLE_COEFF",
        )
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
        for k in keep:
            if k in os.environ:
                env[k] = os.environ[k]
        # Prefer ACT-specific GPU visibility when set.
        if os.environ.get("ACT_CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = os.environ["ACT_CUDA_VISIBLE_DEVICES"]

        log_path = os.environ.get(
            "ACT_SERVER_LOG", os.path.join(_TASK_DIR, "act_server.log")
        )
        self._err = open(log_path, "w")
        # Avoid cwd=task/ — a polluted import path can break packaging.version
        # inside lerobot's safetensors load helper.
        self._proc = subprocess.Popen(
            [server_py, server_script, ckpt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            env=env,
            cwd=os.path.dirname(_TASK_DIR),
        )
        print(f"[act] spawned inference server (ckpt={ckpt})", flush=True)
        time.sleep(2)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"act_server died on startup (exit {self._proc.returncode}); "
                f"see {log_path}"
            )
        self._last_action = None

    def _send(self, obj):
        b = pickle.dumps(obj)
        self._proc.stdin.write(struct.pack(">I", len(b)) + b)
        self._proc.stdin.flush()

    def _recv(self):
        h = self._proc.stdout.read(4)
        if len(h) < 4:
            raise RuntimeError("act_server closed")
        n = struct.unpack(">I", h)[0]
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("act_server closed mid-message")
            buf += chunk
        return pickle.loads(buf)

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._send({"cmd": "reset"})
        self._recv()

    def _build_state(self, obs: Observation) -> np.ndarray:
        q = np.asarray(obs.joint_positions, np.float64)
        qd = np.asarray(obs.joint_velocities, np.float64)
        if self.L is not None:
            Lp, Lq = self.L.end_effector.get_world_pose()
        else:
            Lp, Lq = obs.ee_pose_L
        if self.R is not None:
            Rp, Rq = self.R.end_effector.get_world_pose()
        else:
            Rp, Rq = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        def ratio(v):
            return float(np.clip(v / GRIPPER_OPEN_LIMIT, 0, 1))

        return np.concatenate(
            [
                np.asarray(Lp).reshape(-1)[:3],
                np.asarray(Lq).reshape(-1)[:4],
                np.asarray(Rp).reshape(-1)[:3],
                np.asarray(Rq).reshape(-1)[:4],
                q[self._Li],
                q[self._Ri],
                qd[self._Li],
                qd[self._Ri],
                [ratio(q[self._Lg])],
                [ratio(q[self._Rg]) if self._Rg is not None else 0.0],
            ]
        ).astype(np.float32)

    def act(self, obs: Observation):
        state = self._build_state(obs)
        self._send(
            {
                "state": state,
                "head": _resize_rgb(obs.rgb.get("head")),
                "left": _resize_rgb(obs.rgb.get("L_wrist")),
                "right": _resize_rgb(obs.rgb.get("R_wrist")),
            }
        )
        a = np.asarray(self._recv()["action"], np.float64)
        self._last_action = a
        pos = a[:3]
        quat = _rotvec_to_quat_wxyz(a[3], a[4], a[5])
        grip = float(np.clip(a[6], 0, 1)) * GRIPPER_OPEN_LIMIT
        return self.L.forward(pos, quat, grip)

    def is_done(self, obs: Observation) -> bool:
        return False

    def __del__(self):
        try:
            if self._proc is not None:
                self._proc.terminate()
        except Exception:
            pass
        try:
            if self._err is not None:
                self._err.close()
        except Exception:
            pass
