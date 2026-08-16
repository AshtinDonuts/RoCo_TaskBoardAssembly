"""ALOHA Solo leader teleoperation policy for the RoCo Task Board harness.

Reads 50 Hz leader samples over localhost JSON IPC, retargets relative
Cartesian motion onto the DexMate left arm through Lula IK, and streams
10 Hz LeRobot v3 frames to an isolated recorder sidecar.

Task success is logged only. The operator (or the episode timer) decides
whether an attempt is saved.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_TASK_DIR)
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402
from teleop.episode import EpisodeEvent, EpisodeSession  # noqa: E402
from teleop.keyboard_ee import KEYBOARD_HELP, KeyboardEE  # noqa: E402
from teleop.keyboard_input import KeyboardInput  # noqa: E402
from teleop.leader_client import LeaderClient  # noqa: E402
from teleop.protocol import DEFAULT_HOST, DEFAULT_PORT, parse_endpoint  # noqa: E402
from teleop.recorder_client import RecorderClient  # noqa: E402
from teleop.retarget import CartesianRetargeter, RetargetConfig  # noqa: E402
from teleop.schema import (  # noqa: E402
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

RECORDING_CMDS = {"save_episode", "rerecord_episode", "stop_recording"}


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return int(raw)


class AlohaTeleopPolicy(Policy):
    is_human_recording = True

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
        rec_cfg = raw.get("recording") or {}
        self._session = EpisodeSession(
            episode_time_s=_env_float(
                "ROCO_EPISODE_TIME_S", float(rec_cfg.get("episode_time_s", 600.0))
            ),
            warmup_time_s=_env_float(
                "ROCO_WARMUP_TIME_S", float(rec_cfg.get("warmup_time_s", 5.0))
            ),
            num_episodes=_env_int(
                "ROCO_NUM_EPISODES", int(rec_cfg.get("num_episodes", 1))
            ),
        )
        self._retarget = CartesianRetargeter(RetargetConfig.from_dict(raw.get("retarget")))
        self._keyboard_mode = os.environ.get("ALOHA_KEYBOARD_TELEOP", "").lower() in {
            "1", "true", "yes", "on",
        }
        self._kbd_ee: Optional[KeyboardEE] = None
        self._kbd_input: Optional[KeyboardInput] = None
        self._leader = None
        if self._keyboard_mode:
            kcfg = raw.get("keyboard") or {}
            self._kbd_ee = KeyboardEE(
                lin_vel_mps=float(kcfg.get("lin_vel_mps", 0.12)),
                ang_vel_rps=float(kcfg.get("ang_vel_rps", 0.8)),
            )
            self._kbd_input = KeyboardInput()
            backend = self._kbd_input.start()
            print(
                f"[aloha_teleop] keyboard teleop backend={backend}\n{KEYBOARD_HELP}",
                flush=True,
            )
            if backend == "none":
                print(
                    "[aloha_teleop] warning: no keyboard backend yet; "
                    "will retry carb.input after the sim is running",
                    flush=True,
                )
        else:
            endpoint = os.environ.get(
                "ALOHA_LEADER_ENDPOINT",
                raw.get("leader_endpoint", f"{DEFAULT_HOST}:{DEFAULT_PORT}"),
            )
            host, port = parse_endpoint(str(endpoint))
            self._leader = LeaderClient(host=host, port=port)
            self._leader.start()
            print(
                f"[aloha_teleop] waiting for leader at {host}:{port}; "
                "close the leader gripper or send cmd=start. "
                f"episode_time={self._session.episode_time_s:g}s "
                f"warmup={self._session.warmup_time_s:g}s "
                f"num_episodes={self._session.num_episodes}. "
                "Right=save  Left=rerecord  Esc=stop",
                flush=True,
            )
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
        self._latencies_ms: List[float] = []
        self._part_events: List[Dict[str, Any]] = []
        self._episode_seq = 0
        self._next_record_step: Optional[int] = None
        self._reset_requested = False
        self._closed = False
        self._logged_snap = False
        self._last_kbd_move_log_s = 0.0
        self._kbd_backend_retried = False
        self._last_ik_ok = True
        self._last_tracking_err_m = 0.0

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
        if not self._keyboard_mode:
            print(
                f"[aloha_teleop] episode_time={self._session.episode_time_s:g}s "
                f"warmup={self._session.warmup_time_s:g}s "
                f"num_episodes={self._session.num_episodes}",
                flush=True,
            )
        else:
            print(
                f"[aloha_teleop] episode_time={self._session.episode_time_s:g}s "
                f"warmup={self._session.warmup_time_s:g}s "
                f"num_episodes={self._session.num_episodes}. "
                "Hold i/k/... in the Isaac window after warmup.",
                flush=True,
            )

    @property
    def in_warmup(self) -> bool:
        return self._session.is_warmup

    def freeze_parts(self) -> bool:
        return self._abort or self._session.is_warmup or self._session.done

    def session_done(self) -> bool:
        return self._session.done

    def take_reset_request(self) -> bool:
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._target = target
        self._part_done = False
        self._paused = False
        self._logged_snap = False
        left_pos, left_quat = _ee_pose(self.L, obs.ee_pose_L)
        right_pos, right_quat = _ee_pose(self.R, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])))
        self._last_left_pos = left_pos
        self._last_left_quat = left_quat
        self._last_left_grip = float(obs.L_gripper_position)
        self._right_pos = right_pos
        self._right_quat = right_quat
        self._retarget.disengage()
        self._ik_fail_streak = 0
        lo = np.asarray(self._retarget.cfg.workspace_min, dtype=np.float64)
        hi = np.asarray(self._retarget.cfg.workspace_max, dtype=np.float64)
        in_workspace = bool(np.all(left_pos >= lo) and np.all(left_pos <= hi))
        self._part_events.append({
            "event": "part_start",
            "name": target.name,
            "release_mode": target.release_mode,
        })
        print(
            f"[aloha_teleop] part={target.name} release={target.release_mode} "
            f"pick={target.pick_pos} place={target.place_pos} "
            f"ee_start={[round(float(v), 4) for v in left_pos]} "
            f"workspace_ok={in_workspace}",
            flush=True,
        )
        if not in_workspace:
            print(
                f"[aloha_teleop] ERROR: initial EE is outside workspace "
                f"lo={lo.tolist()} hi={hi.tolist()}; refusing to clamp toward "
                "an unrelated target",
                flush=True,
            )

    def act(self, obs: Observation):
        now = time.monotonic()
        if self._session.phase == "idle" and not self._session.done:
            self._handle_episode_event(self._session.start(now), now)

        sample = self._read_leader_sample()
        cmd = "none" if sample is None else (sample.get("cmd") or "none")
        self._apply_cmd(cmd)
        self._handle_episode_event(self._session.step(cmd, now), now)
        self._note_snap(obs)

        left_pos, left_quat = _ee_pose(self.L, obs.ee_pose_L)
        right_pos, right_quat = _ee_pose(self.R, (self._right_pos, self._right_quat))
        age = 0.0 if self._keyboard_mode else self._leader.age_s()
        hold = (
            self._estop
            or self._paused
            or self._abort
            or self._session.done
        )
        # Keyboard teleop tracks during episode warmup; TCP leaders still hold.
        if (not self._keyboard_mode) and self._session.is_warmup:
            hold = True
        reason = "tracking"
        if self._session.is_warmup:
            reason = "warmup"
        elif self._abort:
            reason = "abort_hold"
        if sample is None or ((not self._keyboard_mode) and age > self._stale_pause_s):
            hold = True
            reason = "stale_pause" if sample is not None else "no_sample"
        elif (not self._keyboard_mode) and age > self._stale_hold_s:
            hold = True
            reason = "stale_hold"
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
        ik = getattr(self.L, "ik", None)
        ik_ok = bool(getattr(ik, "ik_ok", True))
        self._last_ik_ok = ik_ok
        self._last_tracking_err_m = float(
            np.linalg.norm(self._last_left_pos - left_pos)
        )
        if not ik_ok:
            self._ik_fail_streak += 1
            reason = "ik_reject"
        else:
            self._ik_fail_streak = 0

        if self._session.is_recording:
            if self._next_record_step is None:
                self._next_record_step = int(obs.step_idx)
            if int(obs.step_idx) >= self._next_record_step:
                self._record(obs, left_pos, left_quat, right_pos, right_quat)
                self._next_record_step = int(obs.step_idx) + self._record_stride
        self._maybe_log(obs, sample, reason, age, now)
        return action

    def _read_leader_sample(self) -> Optional[Dict[str, Any]]:
        if self._keyboard_mode and self._kbd_ee is not None and self._kbd_input is not None:
            self._maybe_upgrade_keyboard_backend()
            held = self._kbd_input.poll_held()
            moved = self._kbd_ee.apply_holds(held, self._dt)
            for edge in self._kbd_input.pop_edges():
                self._kbd_ee.apply_edge(edge)
            sample = self._kbd_ee.take_sample()
            if moved and (time.monotonic() - self._last_kbd_move_log_s) > 0.5:
                self._last_kbd_move_log_s = time.monotonic()
                print(
                    f"[aloha_teleop] kbd move held={sorted(held)} "
                    f"leader_pos={[round(v, 3) for v in sample['ee_pos']]} "
                    f"grip={sample['gripper_norm']:.0f} "
                    f"clutch={sample['clutch']} backend={self._kbd_input.backend}",
                    flush=True,
                )
            return sample
        if self._leader is None:
            return None
        sample = self._leader.latest()
        if sample is None:
            return None
        sample = dict(sample)
        sample["cmd"] = self._leader.pop_cmd()
        return sample

    def _maybe_upgrade_keyboard_backend(self) -> None:
        if self._kbd_input is None:
            return
        if self._kbd_input.backend == "carb":
            return
        # Kit may not expose the app window at Policy.__init__; retry once.
        if getattr(self, "_kbd_backend_retried", False):
            return
        self._kbd_backend_retried = True
        old = self._kbd_input.backend
        self._kbd_input.close()
        self._kbd_input = KeyboardInput()
        backend = self._kbd_input.start()
        print(
            f"[aloha_teleop] keyboard backend retry {old} -> {backend}",
            flush=True,
        )

    def is_done(self, obs: Observation) -> bool:
        if self._abort or self._session.is_warmup or self._session.done:
            return False
        if self._target is None:
            return False
        if self._target.release_mode == "snap":
            return False
        return bool(self._part_done)

    def _apply_cmd(self, cmd: str) -> None:
        if cmd in (None, "none") or cmd in RECORDING_CMDS:
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
            self._part_events.append({
                "event": "part_done",
                "name": None if self._target is None else self._target.name,
            })
        elif cmd == "abort":
            self._abort = True
            self._part_events.append({
                "event": "abort",
                "name": None if self._target is None else self._target.name,
            })
            print("[aloha_teleop] task abort: holding robot; recording continues", flush=True)
        elif cmd == "estop":
            self._estop = True
            self._paused = True
        elif cmd == "reset":
            self._retarget.reset()
            self._part_done = False
            self._paused = False
            self._estop = False

    def _note_snap(self, obs: Observation) -> None:
        if self._logged_snap or not getattr(obs, "snap_fired", False):
            return
        self._logged_snap = True
        self._part_events.append({
            "event": "snap_fired",
            "name": None if self._target is None else self._target.name,
        })

    def _handle_episode_event(self, ev: EpisodeEvent, now: float) -> None:
        if ev.is_noop:
            return
        print(
            f"[aloha_teleop] episode {ev.kind} reason={ev.reason} "
            f"attempt={ev.attempt_index} saved={ev.saved_episodes}",
            flush=True,
        )
        if ev.kind == "warmup_start":
            self._episode_seq = 0
            self._next_record_step = None
            self._abort = False
            self._part_done = False
            self._frames_sent = 0
            self._drops = 0
            return
        if ev.kind == "record_start":
            self._episode_seq = 0
            self._next_record_step = None
            if self._recorder is not None:
                reply = self._recorder.send({
                    "cmd": "begin_episode",
                    "episode_index": ev.episode_index,
                    "attempt_index": ev.attempt_index,
                })
                if not reply or not reply.get("ok"):
                    print(f"[aloha_teleop] begin_episode failed: {reply}", flush=True)
            return
        if ev.kind == "save":
            self._commit_episode(ev, now)
            return
        if ev.kind == "discard":
            self._discard_episode(ev, now)
            return
        if ev.kind == "session_end":
            self._session.mark_done()
            return

    def _commit_episode(self, ev: EpisodeEvent, now: float) -> None:
        duration_s = self._session.elapsed_episode_s(now)
        frames = 0
        if self._recorder is not None:
            reply = self._recorder.send({
                "cmd": "save_episode",
                "reason": ev.reason,
                "attempt_index": ev.attempt_index,
                "duration_s": duration_s,
                "task_log": self._task_log(),
                "results_json": os.environ.get("LEROBOT_RESULTS_JSON"),
                "stats": self._stats(),
            })
            if reply and reply.get("ok"):
                frames = int(reply.get("frames") or 0)
            else:
                print(f"[aloha_teleop] save_episode failed: {reply}", flush=True)
        self._session.complete_save(frames, ev.end_session, ev.reason)
        if self._session.needs_reset:
            self._part_events = []
            self._reset_requested = True
        print(
            f"[aloha_teleop] saved frames={frames} "
            f"episodes={self._session.saved_episodes}/{self._session.num_episodes} "
            f"duration={duration_s:.1f}s",
            flush=True,
        )

    def _discard_episode(self, ev: EpisodeEvent, now: float) -> None:
        if self._recorder is not None:
            reply = self._recorder.send({
                "cmd": "discard_episode",
                "reason": ev.reason,
                "attempt_index": ev.attempt_index,
                "duration_s": self._session.elapsed_episode_s(now),
                "task_log": self._task_log(),
            })
            if not reply or not reply.get("ok"):
                print(f"[aloha_teleop] discard_episode failed: {reply}", flush=True)
        self._session.complete_discard()
        if self._session.needs_reset:
            self._part_events = []
            self._reset_requested = True
        print("[aloha_teleop] discarded attempt; resetting for rerecord", flush=True)

    def _task_log(self) -> Dict[str, Any]:
        return {
            "aborted": bool(self._abort),
            "current_part": None if self._target is None else self._target.name,
            "events": list(self._part_events),
        }

    def _stats(self) -> Dict[str, Any]:
        return {
            "frames": self._frames_sent,
            "drops": self._drops,
            "p95_latency_ms": (
                float(np.percentile(self._latencies_ms, 95)) if self._latencies_ms else None
            ),
            "leader_hz": (
                None if self._leader is None else self._leader.hz
            ),
            "keyboard_mode": bool(self._keyboard_mode),
        }

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
        timestamp_s = float(self._episode_seq) / float(RECORD_HZ)
        try:
            reply = self._recorder.send({
                "cmd": "frame",
                "step_idx": int(obs.step_idx),
                "timestamp_s": timestamp_s,
                "seq": int(self._episode_seq),
                "state": state.tolist(),
                "action": action.tolist(),
                "images": {k: encode_jpeg(v) for k, v in images.items()},
                "part_name": None if self._target is None else self._target.name,
                "release_mode": None if self._target is None else self._target.release_mode,
            })
            if reply and reply.get("ok"):
                self._episode_seq += 1
                self._frames_sent += 1
            else:
                self._drops += 1
        except Exception as exc:
            self._drops += 1
            print(f"[aloha_teleop] recorder frame drop: {exc}", flush=True)

    def _maybe_log(self, obs: Observation, sample, reason: str, age: float, now: float) -> None:
        if now - self._last_status_s < 1.0:
            return
        self._last_status_s = now
        p95 = 0.0
        if self._latencies_ms:
            p95 = float(np.percentile(self._latencies_ms, 95))
        extra = ""
        if self._session.is_warmup:
            extra = f" warmup_left={self._session.remaining_warmup_s(now):.1f}s"
        elif self._session.is_recording:
            extra = f" rec_left={self._session.remaining_episode_s(now):.1f}s"
        print(
            f"[aloha_teleop] t={obs.step_idx * self._dt:6.2f}s part="
            f"{getattr(self._target, 'name', None)} reason={reason} "
            f"phase={self._session.phase}{extra} "
            f"ik_ok={self._last_ik_ok} "
            f"track_err_mm={self._last_tracking_err_m * 1000.0:.1f} "
            f"target={[round(float(v), 3) for v in self._last_left_pos]} "
            f"leader_hz={(0.0 if self._leader is None else self._leader.hz):5.1f} "
            f"age={age:.3f}s "
            f"p95_lat_ms={p95:.1f} frames={self._frames_sent} drops={self._drops} "
            f"connected="
            f"{True if self._keyboard_mode else (False if self._leader is None else self._leader.connected)}",
            flush=True,
        )

    def finalize(self, results_json: Optional[str] = None, success: Optional[bool] = None) -> None:
        del success  # task success must not decide whether data is kept
        if self._closed:
            return
        self._closed = True
        if self._recorder is not None:
            try:
                self._recorder.send({
                    "cmd": "finalize_session",
                    "reason": "session_end",
                    "results_json": results_json or os.environ.get("LEROBOT_RESULTS_JSON"),
                    "stats": self._stats(),
                    "task_log": self._task_log(),
                })
            except Exception as exc:
                print(f"[aloha_teleop] finalize_session failed: {exc}", flush=True)
            try:
                self._recorder.close()
            except Exception:
                pass
            self._recorder = None
        if self._kbd_input is not None:
            try:
                self._kbd_input.close()
            except Exception:
                pass
            self._kbd_input = None
        if self._leader is not None:
            try:
                self._leader.close()
            except Exception:
                pass
            self._leader = None

    def __del__(self):
        try:
            self.finalize()
        except Exception:
            pass
