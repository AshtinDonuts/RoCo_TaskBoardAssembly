"""ALOHA Solo leader teleoperation policy for the RoCo Task Board harness.

Reads 50 Hz leader samples over localhost JSON IPC, retargets relative
Cartesian motion onto the DexMate left arm through Lula IK, and streams
10 Hz LeRobot v3 frames to an isolated recorder sidecar.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_TASK_DIR)
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402
from teleop.leader_client import LeaderClient  # noqa: E402
from teleop.protocol import DEFAULT_HOST, DEFAULT_PORT, parse_endpoint  # noqa: E402
from teleop.recorder_client import RecorderClient  # noqa: E402
from teleop.retarget import CartesianRetargeter, RetargetConfig  # noqa: E402
from teleop.schema import (  # noqa: E402
    GRIPPER_OPEN_LIMIT,
    RECORD_HZ,
    encode_jpeg,
    gripper_ratio,
    pack_action,
    pack_state,
    resize_rgb,
)
from teleop import transforms as T  # noqa: E402

try:
    from isaacsim.core.utils.types import ArticulationAction
except Exception:  # pragma: no cover - template import in non-Isaac tests
    ArticulationAction = None  # type: ignore


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _ee_pose(controller, fallback) -> tuple:
    if controller is not None:
        try:
            pos, quat = controller.end_effector.get_world_pose()
            return np.asarray(pos, dtype=np.float64), T.normalize_quat_wxyz(quat)
        except Exception:
            pass
    pos, quat = fallback
    return np.asarray(pos, dtype=np.float64), T.normalize_quat_wxyz(quat)


class AlohaTeleopPolicy(Policy):
    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        if self.L is None:
            raise ValueError("AlohaTeleopPolicy requires env_info.L_controller")
        self.R = getattr(env_info, "R_controller", None)
        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None
        self._n_dof = len(dof)
        self._dt = float(env_info.physics_dt)
        self._record_stride = max(1, int(round((1.0 / RECORD_HZ) / self._dt)))

        cfg_path = Path(os.environ.get(
            "ALOHA_TELEOP_CONFIG",
            os.path.join(_REPO_ROOT, "config", "aloha_solo_to_vega_1u.yaml"),
        ))
        raw = _load_yaml(cfg_path) if cfg_path.exists() else {}
        self._cfg = raw
        self._retarget = CartesianRetargeter(RetargetConfig.from_dict(raw.get("retarget")))
        endpoint = os.environ.get("ALOHA_LEADER_ENDPOINT", raw.get("leader_endpoint", f"{DEFAULT_HOST}:{DEFAULT_PORT}"))
        host, port = parse_endpoint(str(endpoint))
        self._leader = LeaderClient(host=host, port=port)
        self._leader.start()

        self._paused = False
        self._estop = False
        self._abort = False
        self._part_done = False
        self._target: Optional[PartTarget] = None
        self._last_left_pos = None
        self._last_left_quat = None
        self._last_left_grip = 0.0
        self._right_pos = np.zeros(3, dtype=np.float64)
        self._right_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._ik_fail_streak = 0
        self._max_ik_fails = int(raw.get("max_ik_fail_streak", 15))
        self._stale_hold_s = float(raw.get("retarget", {}).get("stale_hold_s", 0.10))
        self._stale_pause_s = float(raw.get("retarget", {}).get("stale_pause_s", 0.50))
        self._last_status_s = 0.0
        self._frames_sent = 0
        self._drops = 0
        self._latencies_ms = []

        self._recorder = None
        server_py = os.environ.get("LEROBOT_SERVER_PY")
        if server_py:
            server_script = os.environ.get(
                "LEROBOT_SERVER_SCRIPT",
                os.path.join(_REPO_ROOT, "tools", "lerobot_recorder", "server.py"),
            )
            log_path = os.environ.get(
                "LEROBOT_SERVER_LOG",
                os.path.join(_TASK_DIR, "teleop_recorder.log"),
            )
            self._recorder = RecorderClient(server_py, server_script, log_path)
            init = self._recorder.send({
                "cmd": "init",
                "repo_id": os.environ.get("LEROBOT_REPO_ID", "local/roco_aloha_teleop"),
                "root": os.environ.get("LEROBOT_OUTPUT_ROOT", os.path.join(_REPO_ROOT, "runs")),
                "fps": RECORD_HZ,
                "run_id": os.environ.get("ALOHA_TELEOP_RUN_ID", time.strftime("%Y%m%d_%H%M%S")),
                "task": "Industrial Task Board Assembly",
                "robot_type": "vega_1u_gripper",
                "challenge_commit": os.environ.get("ROCO_COMMIT", ""),
                "config_path": str(cfg_path),
            })
            if not init or not init.get("ok"):
                raise RuntimeError(f"recorder init failed: {init}")

        print(
            f"[aloha_teleop] waiting for leader at {host}:{port}; "
            "close the leader gripper or send cmd=start",
            flush=True,
        )

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._target = target
        self._part_done = False
        self._paused = False
        left_pos, left_quat = _ee_pose(self.L, obs.ee_pose_L)
        right_pos, right_quat = _ee_pose(self.R, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
        self._last_left_pos = left_pos
        self._last_left_quat = left_quat
        self._last_left_grip = float(obs.L_gripper_position)
        self._right_pos = right_pos
        self._right_quat = right_quat
        self._retarget.disengage()
        self._ik_fail_streak = 0
        print(
            f"[aloha_teleop] part={target.name} release={target.release_mode} "
            f"pick={target.pick_pos} place={target.place_pos}",
            flush=True,
        )

    def act(self, obs: Observation):
        sample = self._leader.latest()
        cmd = self._leader.pop_cmd() if sample is not None else "none"
        self._apply_cmd(cmd)

        left_pos, left_quat = _ee_pose(self.L, obs.ee_pose_L)
        right_pos, right_quat = _ee_pose(self.R, (self._right_pos, self._right_quat))
        age = self._leader.age_s()
        hold = self._estop or self._paused or self._abort
        reason = "tracking"
        if sample is None or age > self._stale_pause_s:
            hold = True
            reason = "stale_pause" if sample is not None else "no_sample"
        elif age > self._stale_hold_s:
            hold = True
            reason = "stale_hold"
        if self._ik_fail_streak >= self._max_ik_fails:
            hold = True
            reason = "ik_fail"

        if hold or sample is None:
            pos = self._last_left_pos if self._last_left_pos is not None else left_pos
            quat = self._last_left_quat if self._last_left_quat is not None else left_quat
            grip = self._last_left_grip
        else:
            pos, quat, grip, info = self._retarget.step(
                leader_pos=sample["ee_pos"],
                leader_quat=sample["ee_quat_wxyz"],
                gripper_norm=sample["gripper_norm"],
                dt=self._dt,
                clutch=bool(sample["clutch"]) and not self._paused,
                deadman=bool(sample["deadman"]) and not self._estop,
                current_dex_pos=left_pos,
                current_dex_quat=left_quat,
            )
            reason = info.get("reason", reason)
            if sample.get("timestamp_ns"):
                latency_ms = max(0.0, (time.time_ns() - int(sample["timestamp_ns"])) / 1e6)
                self._latencies_ms.append(latency_ms)
                if len(self._latencies_ms) > 500:
                    self._latencies_ms = self._latencies_ms[-500:]

        self._last_left_pos = np.asarray(pos, dtype=np.float64)
        self._last_left_quat = T.normalize_quat_wxyz(quat)
        self._last_left_grip = float(grip)
        action = self.L.forward(self._last_left_pos, self._last_left_quat, float(self._last_left_grip))
        err = float(np.linalg.norm(self._last_left_pos - left_pos))
        if err > 0.12:
            self._ik_fail_streak += 1
        else:
            self._ik_fail_streak = 0

        if obs.step_idx % self._record_stride == 0:
            self._record(obs, left_pos, left_quat, right_pos, right_quat)
        self._maybe_log(obs, sample, reason, age)
        return action

    def is_done(self, obs: Observation) -> bool:
        if self._abort:
            return True
        if self._target is None:
            return False
        if self._target.release_mode == "snap":
            return False
        return bool(self._part_done)

    def _apply_cmd(self, cmd: str) -> None:
        if cmd in (None, "none"):
            return
        if cmd == "start":
            self._paused = False
            self._estop = False
        elif cmd == "pause":
            self._paused = True
        elif cmd == "resume":
            self._paused = False
        elif cmd == "recenter":
            self._retarget.disengage()
        elif cmd == "part_done":
            self._part_done = True
        elif cmd == "abort":
            self._abort = True
        elif cmd == "estop":
            self._estop = True
            self._paused = True
        elif cmd == "reset":
            self._retarget.reset()
            self._part_done = False
            self._paused = False
            self._estop = False

    def _record(self, obs: Observation, left_pos, left_quat, right_pos, right_quat) -> None:
        if self._recorder is None:
            return
        q = np.asarray(obs.joint_positions, dtype=np.float64)
        qd = np.asarray(obs.joint_velocities, dtype=np.float64)
        state = pack_state(
            left_pos,
            left_quat,
            right_pos,
            right_quat,
            q[self._Li],
            q[self._Ri],
            qd[self._Li],
            qd[self._Ri],
            gripper_ratio(q[self._Lg]),
            gripper_ratio(q[self._Rg] if self._Rg is not None else 0.0),
        )
        action = pack_action(
            self._last_left_pos,
            self._last_left_quat,
            gripper_ratio(self._last_left_grip),
            self._right_pos,
            self._right_quat,
            0.0,
        )
        images = {
            "head": resize_rgb(obs.rgb.get("head") if obs.rgb else None),
            "left_hand": resize_rgb(obs.rgb.get("L_wrist") if obs.rgb else None),
            "right_hand": resize_rgb(obs.rgb.get("R_wrist") if obs.rgb else None),
        }
        try:
            reply = self._recorder.send({
                "cmd": "frame",
                "step_idx": int(obs.step_idx),
                "timestamp_s": float(obs.step_idx * self._dt),
                "seq": int(self._frames_sent),
                "state": state.tolist(),
                "action": action.tolist(),
                "images": {k: encode_jpeg(v) for k, v in images.items()},
                "part_name": None if self._target is None else self._target.name,
                "release_mode": None if self._target is None else self._target.release_mode,
            })
            if reply and reply.get("ok"):
                self._frames_sent += 1
            else:
                self._drops += 1
        except Exception as exc:
            self._drops += 1
            print(f"[aloha_teleop] recorder frame drop: {exc}", flush=True)

    def _maybe_log(self, obs: Observation, sample, reason: str, age: float) -> None:
        now = time.time()
        if now - self._last_status_s < 1.0:
            return
        self._last_status_s = now
        p95 = 0.0
        if self._latencies_ms:
            p95 = float(np.percentile(self._latencies_ms, 95))
        print(
            f"[aloha_teleop] t={obs.step_idx * self._dt:6.2f}s part="
            f"{getattr(self._target, 'name', None)} reason={reason} "
            f"leader_hz={self._leader.hz:5.1f} age={age:.3f}s "
            f"p95_lat_ms={p95:.1f} frames={self._frames_sent} drops={self._drops} "
            f"connected={self._leader.connected}",
            flush=True,
        )

    def finalize(self, results_json: Optional[str] = None, success: Optional[bool] = None) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.send({
                "cmd": "end_episode",
                "results_json": results_json,
                "success": success,
                "stats": {
                    "frames": self._frames_sent,
                    "drops": self._drops,
                    "p95_latency_ms": (
                        float(np.percentile(self._latencies_ms, 95)) if self._latencies_ms else None
                    ),
                    "leader_hz": self._leader.hz,
                },
            })
        finally:
            self._recorder.close()
            self._leader.close()

    def __del__(self):
        try:
            if self._recorder is not None:
                self._recorder.close()
        except Exception:
            pass
        try:
            self._leader.close()
        except Exception:
            pass
