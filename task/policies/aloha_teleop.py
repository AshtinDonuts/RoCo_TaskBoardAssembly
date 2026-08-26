"""ALOHA leader teleoperation policy for the RoCo Task Board harness.

Reads 50 Hz leader sample(s) over localhost JSON IPC, retargets relative
Cartesian motion onto the DexMate right arm (default) or both arms (dual)
through Lula IK, and streams 10 Hz LeRobot v3 frames to an isolated
recorder sidecar.

Task success is logged only. The operator (or the episode timer) decides
whether an attempt is saved.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_TASK_DIR)
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402
from teleop.control_arms import (  # noqa: E402
    merge_joint_position_actions,
    pack_teleop_action,
    prefer_cmd,
)
from teleop.episode import EpisodeEvent, EpisodeSession  # noqa: E402
from teleop.keyboard_ee import KEYBOARD_HELP, KeyboardEE  # noqa: E402
from teleop.keyboard_input import KeyboardInput  # noqa: E402
from teleop.leader_client import LeaderClient  # noqa: E402
from teleop.protocol import parse_endpoint  # noqa: E402
from teleop.recorder_client import RecorderClient  # noqa: E402
from teleop.retarget import CartesianRetargeter, RetargetConfig  # noqa: E402
from teleop.export_config import load_export_config  # noqa: E402
from teleop.grasp_aperture import (  # noqa: E402
    grasp_width_m as _grasp_width_m,
    resolve_grasp_close_rad,
)
from controllers.gripper_compliance import GripperComplianceConfig  # noqa: E402
import param_config as pc  # noqa: E402
from teleop.schema import (  # noqa: E402
    encode_jpeg,
    gripper_ratio,
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


def _start_leader(endpoint: str, label: str) -> LeaderClient:
    host, port = parse_endpoint(str(endpoint))
    client = LeaderClient(host=host, port=port)
    client.start()
    print(
        f"[aloha_teleop] waiting for {label} leader at {host}:{port}",
        flush=True,
    )
    return client


class AlohaTeleopPolicy(Policy):
    is_human_recording = True

    def __init__(self, env_info: EnvInfo) -> None:
        super().__init__(env_info)
        self.L = env_info.L_controller
        if self.L is None:
            raise ValueError("AlohaTeleopPolicy requires env_info.L_controller")
        self.R = getattr(env_info, "R_controller", None)
        if self.R is None:
            raise ValueError("AlohaTeleopPolicy requires env_info.R_controller")
        dof = list(env_info.dof_names)
        self._Li = [dof.index(j) for j in env_info.L_arm_joints]
        self._Ri = [dof.index(j) for j in env_info.R_arm_joints]
        self._Lg = dof.index(env_info.L_gripper_joint)
        self._Rg = dof.index("R_gripper_joint") if "R_gripper_joint" in dof else None
        self._n_dof = len(dof)
        self._dt = float(env_info.physics_dt)
        self._export = load_export_config()
        self._control_arms = str(self._export.control.arms)
        # Keyboard always drives the right arm only (left held by harness).
        self._keyboard_mode = os.environ.get("ALOHA_KEYBOARD_TELEOP", "").lower() in {
            "1", "true", "yes", "on",
        }
        if self._keyboard_mode:
            self.active_arms: Tuple[str, ...] = ("R",)
        else:
            self.active_arms = tuple(self._export.control.active_arms)
        # Sample + encode at the same fps. Wall clock (default) keeps mp4
        # realtime with the operator; sim clock gates on physics time.
        self._record_fps = float(self._export.fps)
        self._record_period_s = float(self._export.record_period_s)
        self._playback_clock = str(self._export.export.playback_clock)
        self._img_h = int(self._export.img_h)
        self._img_w = int(self._export.img_w)
        self._jpeg_quality = int(self._export.export.image.jpeg_quality)
        self._cameras = tuple(self._export.export.image.cameras)
        self._next_record_wall: Optional[float] = None
        self._next_record_sim: Optional[float] = None
        self._last_act_wall: Optional[float] = None

        cfg_path = Path(
            os.environ.get("ALOHA_TELEOP_CONFIG", str(self._export.paths.teleop_yaml))
        )
        raw = _load_yaml(cfg_path) if cfg_path.exists() else {}
        self._cfg = raw
        sess = self._export.session
        self._session = EpisodeSession(
            episode_time_s=_env_float("ROCO_EPISODE_TIME_S", float(sess.episode_time_s)),
            warmup_time_s=_env_float("ROCO_WARMUP_TIME_S", float(sess.warmup_time_s)),
            num_episodes=_env_int("ROCO_NUM_EPISODES", int(sess.num_episodes)),
        )
        retarget_cfg = RetargetConfig.from_dict(raw.get("retarget"))
        self._grasp_part = (
            os.environ.get("ROCO_TELEOP_PART")
            or os.environ.get("ALOHA_GRASP_PART")
            or ""
        ).strip() or None
        self._grasp_width_m: Optional[float] = None
        self._grasp_ui = None
        if self._grasp_part:
            self._apply_grasp_part_to_cfg(retarget_cfg, self._grasp_part)
        self._retarget_R = CartesianRetargeter(retarget_cfg)
        self._retarget_L: Optional[CartesianRetargeter] = None
        if self._control_arms == "dual" and not self._keyboard_mode:
            # Orientation lock is right-arm only even when YAML enables it.
            left_raw = dict(raw.get("retarget") or {})
            left_raw["fix_orientation"] = False
            left_raw["fixed_orientation_wxyz"] = None
            left_raw["orientation_cone_rad"] = None
            left_cfg = RetargetConfig.from_dict(left_raw)
            if self._grasp_part:
                left_cfg.gripper_close = float(retarget_cfg.gripper_close)
            self._retarget_L = CartesianRetargeter(left_cfg)
        # Primary retarget used by recenter / reset operator cmds.
        self._retarget = self._retarget_R
        locked_q = retarget_cfg.fixed_orientation_wxyz
        cone = retarget_cfg.orientation_cone_rad
        self._orientation_cone_rad_R = (
            float(cone) if cone is not None and float(cone) > 0.0 else None
        )
        if locked_q is not None:
            if self._orientation_cone_rad_R is not None:
                print(
                    "[aloha_teleop] right EE claw-machine: "
                    f"preferred_wxyz={list(locked_q)} "
                    f"orientation_cone_rad={self._orientation_cone_rad_R:g} "
                    "(XYZ+gripper; IK free-tilts in cone)",
                    flush=True,
                )
            else:
                print(
                    "[aloha_teleop] right EE fixed_orientation_wxyz="
                    f"{list(locked_q)} (world top-down hard lock; "
                    "track XYZ + gripper only)",
                    flush=True,
                )
        self._prox_rate = retarget_cfg.proximity_rate_limit
        self._prox_delta = retarget_cfg.proximity_delta_gain
        self._last_prox_rate_scale = 1.0
        self._last_prox_delta_scale = 1.0
        self._r_wrist_laser = getattr(env_info, "r_wrist_laser", None)
        if self._prox_rate.enabled:
            print(
                "[aloha_teleop] proximity_rate_limit: "
                f"outer={self._prox_rate.depth_outer_m:g} m → scale=1; "
                f"inner={self._prox_rate.depth_inner_m:g} m → "
                f"scale={self._prox_rate.scale_min:g} "
                "(R-wrist laser → max_lin/ang_vel; catch-up)",
                flush=True,
            )
        if self._prox_delta.enabled:
            print(
                "[aloha_teleop] proximity_delta_gain: "
                f"outer={self._prox_delta.depth_outer_m:g} m → scale=1; "
                f"inner={self._prox_delta.depth_inner_m:g} m → "
                f"scale={self._prox_delta.scale_min:g} "
                "(R-wrist laser → per-frame retarget deltas; no catch-up)",
                flush=True,
            )
        grip_cfg = GripperComplianceConfig.from_retarget_cfg(retarget_cfg)
        for ctrl, label in ((self.R, "R"), (self.L, "L")):
            if ctrl is None or not hasattr(ctrl, "gripper_compliance"):
                continue
            ctrl.gripper_compliance.cfg = grip_cfg
            ctrl._gripper_dt = self._dt
        # Soft PhysX drives (same as scripted baseline) are applied by the
        # harness via _apply_gripper_compliance (GRIPPER_DRIVE_*).
        print(
            "[aloha_teleop] soft gripper drives: harness "
            f"stiffness={pc.GRIPPER_DRIVE_STIFFNESS:g} "
            f"damping={pc.GRIPPER_DRIVE_DAMPING:g} "
            f"max_force={pc.GRIPPER_DRIVE_MAX_FORCE:g} "
            "(same path as scripted baseline)",
            flush=True,
        )
        if grip_cfg.enabled:
            part_note = (
                f" part={self._grasp_part!r}"
                if getattr(self, "_grasp_part", None)
                else ""
            )
            print(
                "[aloha_teleop] gripper command compliance: ON "
                f"mode={grip_cfg.mode} "
                f"close={grip_cfg.close:g} rad{part_note} "
                f"close_speed={grip_cfg.close_speed_rad_s:g} rad/s "
                f"open_speed={grip_cfg.open_speed_rad_s:g} rad/s "
                f"hold_margin={grip_cfg.hold_margin:g}",
                flush=True,
            )
        else:
            print(
                "[aloha_teleop] gripper command compliance: OFF "
                "(gripper_compliance_enabled=false; soft PhysX drives still on)",
                flush=True,
            )
        self._maybe_open_grasp_ui()
        if retarget_cfg.axes_map is not None:
            print(
                "[aloha_teleop] retarget frame=headcam_view (axes_map, "
                "space-fixed orientation); "
                "leader +X → into image, +Y/+Z → image left/up",
                flush=True,
            )
        else:
            print(
                f"[aloha_teleop] retarget axes_perm={retarget_cfg.axes_perm} "
                f"axes_sign={retarget_cfg.axes_sign}",
                flush=True,
            )
        self._kbd_ee: Optional[KeyboardEE] = None
        self._kbd_input: Optional[KeyboardInput] = None
        self._leader_R: Optional[LeaderClient] = None
        self._leader_L: Optional[LeaderClient] = None
        self._leader: Optional[LeaderClient] = None
        if self._keyboard_mode:
            kcfg = raw.get("keyboard") or {}
            self._kbd_ee = KeyboardEE(
                lin_vel_mps=float(kcfg.get("lin_vel_mps", 0.12)),
                ang_vel_rps=float(kcfg.get("ang_vel_rps", 0.8)),
            )
            self._kbd_input = KeyboardInput()
            backend = self._kbd_input.start()
            print(
                f"[aloha_teleop] keyboard teleop backend={backend} "
                f"control_arms={self._control_arms} (kbd drives right)\n"
                f"{KEYBOARD_HELP}",
                flush=True,
            )
            if self._control_arms == "dual":
                print(
                    "[aloha_teleop] keyboard+dual: right arm only; left held",
                    flush=True,
                )
            if backend == "none":
                print(
                    "[aloha_teleop] warning: no keyboard backend yet; "
                    "will retry carb.input after the sim is running",
                    flush=True,
                )
        else:
            # Prefer export JSON endpoints; yaml leader_endpoint is fallback for right.
            right_ep = self._export.leader_endpoint_for("right")
            if (
                self._control_arms == "right"
                and not os.environ.get("ALOHA_LEADER_ENDPOINT")
                and not os.environ.get("ALOHA_LEADER_ENDPOINT_RIGHT")
            ):
                yaml_ep = raw.get("leader_endpoint")
                if yaml_ep:
                    right_ep = str(yaml_ep)
            self._leader_R = _start_leader(right_ep, "right")
            self._leader = self._leader_R
            if self._control_arms == "dual":
                left_ep = self._export.leader_endpoint_for("left")
                self._leader_L = _start_leader(left_ep, "left")
            print(
                f"[aloha_teleop] control_arms={self._control_arms} "
                f"active={self.active_arms}; "
                "close a leader gripper or send cmd=start. "
                "Space pause/reanchor is WIP (not reliable yet). "
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
        self._last_right_pos = None
        self._last_right_quat = None
        self._last_right_grip = 0.0
        # Latched virtual poses / joints while paused for atomic reanchor.
        self._frozen_right: Optional[Dict[str, Any]] = None
        self._frozen_left: Optional[Dict[str, Any]] = None
        self._resume_hold_pending = False
        self._skip_record_once = False
        self._home_left_pos = np.zeros(3, dtype=np.float64)
        self._home_left_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._home_left_grip = 0.0
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
        self._reset_requested = False
        self._closed = False
        self._logged_snap = False
        self._last_kbd_move_log_s = 0.0
        self._kbd_backend_retried = False
        self._kbd_motion_active = False
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
            init_msg = {
                "cmd": "init",
                "root": os.environ.get(
                    "LEROBOT_OUTPUT_ROOT", str(self._export.paths.output_root)
                ),
                "run_id": os.environ.get(
                    "ALOHA_TELEOP_RUN_ID", time.strftime("%Y%m%d_%H%M%S")
                ),
                "challenge_commit": os.environ.get("ROCO_COMMIT", ""),
                "config_path": str(cfg_path),
            }
            init_msg.update(self._export.recorder_init_fields())
            init_msg["repo_id"] = os.environ.get(
                "LEROBOT_REPO_ID", init_msg["repo_id"]
            )
            init = self._recorder.send(init_msg)
            if not init or not init.get("ok"):
                raise RuntimeError(f"recorder init failed: {init}")
        suffix = ""
        if self._keyboard_mode:
            suffix = ". Hold i/k/... in the Isaac window after warmup."
        print(
            f"[aloha_teleop] export={self._export.source_path} "
            f"control_arms={self._control_arms} active={self.active_arms} "
            f"fps={self._record_fps:g} clock={self._playback_clock} "
            f"image={self._img_w}x{self._img_h} "
            f"episode_time={self._session.episode_time_s:g}s "
            f"warmup={self._session.warmup_time_s:g}s "
            f"num_episodes={self._session.num_episodes}{suffix}",
            flush=True,
        )

    @property
    def in_warmup(self) -> bool:
        return self._session.is_warmup

    def freeze_parts(self) -> bool:
        return (
            self._abort
            or self._paused
            or self._session.is_warmup
            or self._session.done
        )

    @property
    def pause_sim(self) -> bool:
        """When True the harness should render without stepping physics."""
        return bool(self._paused)

    def session_done(self) -> bool:
        return self._session.done

    def take_reset_request(self) -> bool:
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False

    def _apply_grasp_part_to_cfg(self, retarget_cfg: RetargetConfig, part: str) -> float:
        """Set ``retarget_cfg.gripper_close`` from geometric grasp_width_m."""
        fallback = float(pc.get_part_config(part).get("gripper_close", 0.0))
        close_rad = float(resolve_grasp_close_rad(part, fallback_rad=fallback))
        width = _grasp_width_m(part)
        retarget_cfg.gripper_close = close_rad
        self._grasp_part = part
        self._grasp_width_m = width
        if width is None:
            print(
                "[aloha_teleop] Design D grasp close: "
                f"part={part!r} grasp_width_m=n/a → gripper_close={close_rad:g} rad",
                flush=True,
            )
        else:
            print(
                "[aloha_teleop] Design D grasp close: "
                f"part={part!r} grasp_width_m={width:g} "
                f"({width * 1000.0:.2f} mm) → gripper_close={close_rad:g} rad",
                flush=True,
            )
        return close_rad

    def set_grasp_target_part(self, part: str) -> float:
        """Live Design D: set close endpoint from AABB ``grasp_width_m``."""
        part = str(part).strip()
        fallback = float(pc.get_part_config(part).get("gripper_close", 0.0))
        close_rad = float(resolve_grasp_close_rad(part, fallback_rad=fallback))
        width = _grasp_width_m(part)
        self._grasp_part = part
        self._grasp_width_m = width
        self._set_gripper_close_rad(close_rad)
        w_txt = "n/a" if width is None else f"{width * 1000.0:.2f} mm"
        print(
            "[aloha_teleop] grasp target ← "
            f"{part!r} width={w_txt} close={close_rad:g} rad",
            flush=True,
        )
        return close_rad

    def set_grasp_full_close(self) -> float:
        """Live Design D: command full close (0 rad)."""
        self._grasp_part = None
        self._grasp_width_m = 0.0
        self._set_gripper_close_rad(0.0)
        print("[aloha_teleop] grasp target ← full close (0 rad)", flush=True)
        return 0.0

    def _set_gripper_close_rad(self, close_rad: float) -> None:
        close_rad = float(close_rad)
        for ret in (self._retarget_R, self._retarget_L):
            if ret is None:
                continue
            ret.cfg.gripper_close = close_rad
        for ctrl in (self.R, self.L):
            if ctrl is None or not hasattr(ctrl, "gripper_compliance"):
                continue
            g = ctrl.gripper_compliance
            g.cfg.close = close_rad
            # Clear stall hold so the new close floor can be approached.
            g._q_hold = None
            if hasattr(g, "_stall_ticks"):
                g._stall_ticks = 0
            if hasattr(g, "phase"):
                from controllers.gripper_compliance import GripperPhase

                if g.phase == GripperPhase.HOLDING:
                    g.phase = GripperPhase.CLOSING

    def _grasp_status_text(self) -> str:
        part = self._grasp_part or "(none)"
        w = self._grasp_width_m
        close = float(self._retarget_R.cfg.gripper_close)
        if w is None:
            return f"part={part}\nclose={close:g} rad"
        return f"part={part}\nwidth={w * 1000.0:.2f} mm\nclose={close:g} rad"

    def _maybe_open_grasp_ui(self) -> None:
        if os.environ.get("ISAACSIM_HEADLESS", "").strip() in ("1", "true", "True"):
            return
        if os.environ.get("ROCO_GRASP_UI", "1").strip().lower() in (
            "0",
            "false",
            "off",
            "no",
        ):
            return
        try:
            from teleop.grasp_target_ui import GraspTargetPanel
            from teleop.grasp_aperture import list_grasp_parts as _list

            names = list(_list())
            if not names:
                names = list(pc.known_part_names())
            self._grasp_ui = GraspTargetPanel(
                names,
                on_select=self.set_grasp_target_part,
                on_full_close=self.set_grasp_full_close,
                get_status=self._grasp_status_text,
            )
        except Exception as exc:
            print(f"[aloha_teleop] grasp target UI not opened: {exc}", flush=True)

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._target = target
        self._part_done = False
        self._paused = False
        self._session.set_clock_paused(False, time.monotonic())
        self._resume_hold_pending = False
        self._skip_record_once = False
        self._frozen_right = None
        self._frozen_left = None
        self._logged_snap = False
        left_pos, left_quat = _ee_pose(self.L, obs.ee_pose_L)
        right_pos, right_quat = _ee_pose(
            self.R, (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        self._last_left_pos = left_pos
        self._last_left_quat = left_quat
        self._last_left_grip = float(obs.L_gripper_position)
        self._last_right_pos = right_pos
        self._last_right_quat = right_quat
        self._last_right_grip = float(getattr(obs, "R_gripper_position", 0.0) or 0.0)
        if self._Rg is not None:
            self._last_right_grip = float(obs.joint_positions[self._Rg])
        for ctrl, q0 in (
            (self.L, self._last_left_grip),
            (self.R, self._last_right_grip),
        ):
            if ctrl is not None and hasattr(ctrl, "gripper_compliance"):
                ctrl.gripper_compliance.reset(float(q0))
        self._home_left_pos = np.asarray(left_pos, dtype=np.float64).copy()
        self._home_left_quat = T.normalize_quat_wxyz(left_quat)
        self._home_left_grip = float(self._last_left_grip)
        self._retarget_R.disengage()
        if self._retarget_L is not None:
            self._retarget_L.disengage()
        self._ik_fail_streak = 0
        lo = np.asarray(self._retarget_R.cfg.workspace_min, dtype=np.float64)
        hi = np.asarray(self._retarget_R.cfg.workspace_max, dtype=np.float64)
        check_pos = right_pos if "R" in self.active_arms else left_pos
        in_workspace = bool(np.all(check_pos >= lo) and np.all(check_pos <= hi))
        self._part_events.append({
            "event": "part_start",
            "name": target.name,
            "release_mode": target.release_mode,
        })
        print(
            f"[aloha_teleop] part={target.name} release={target.release_mode} "
            f"pick={target.pick_pos} place={target.place_pos} "
            f"ee_R_start={[round(float(v), 4) for v in right_pos]} "
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
        if self._last_act_wall is None:
            wall_dt = self._dt
        else:
            wall_dt = max(1e-4, min(0.1, now - self._last_act_wall))
        self._last_act_wall = now
        if self._session.phase == "idle" and not self._session.done:
            self._handle_episode_event(self._session.start(now), now)

        sample_R, sample_L = self._read_leader_samples(wall_dt=wall_dt)
        cmd = prefer_cmd(
            None if sample_R is None else sample_R.get("cmd"),
            None if sample_L is None else sample_L.get("cmd"),
        )

        left_pos, left_quat = _ee_pose(self.L, obs.ee_pose_L)
        right_fallback = (
            (self._last_right_pos, self._last_right_quat)
            if self._last_right_pos is not None
            else (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        right_pos, right_quat = _ee_pose(self.R, right_fallback)

        self._apply_cmd(
            cmd,
            now,
            sample_R=sample_R,
            sample_L=sample_L,
            left_pos=left_pos,
            left_quat=left_quat,
            right_pos=right_pos,
            right_quat=right_quat,
        )
        self._handle_episode_event(self._session.step(cmd, now), now)
        self._note_snap(obs)

        drive_L = "L" in self.active_arms and self._retarget_L is not None
        drive_R = "R" in self.active_arms

        reason_R, age_R = self._step_arm(
            side="R",
            sample=sample_R,
            retarget=self._retarget_R,
            current_pos=right_pos,
            current_quat=right_quat,
            wall_dt=wall_dt,
            drive=drive_R,
            obs=obs,
        )
        reason_L = "held"
        age_L = 0.0
        if drive_L:
            reason_L, age_L = self._step_arm(
                side="L",
                sample=sample_L,
                retarget=self._retarget_L,
                current_pos=left_pos,
                current_quat=left_quat,
                wall_dt=wall_dt,
                drive=True,
                obs=obs,
            )
        elif self._last_left_pos is None:
            self._last_left_pos = np.asarray(left_pos, dtype=np.float64)
            self._last_left_quat = T.normalize_quat_wxyz(left_quat)

        reason = reason_R if drive_R else reason_L
        age = age_R if drive_R else age_L
        if drive_L and drive_R and reason_L != "tracking" and reason_R == "tracking":
            reason = reason_L

        if self._resume_hold_pending:
            action = self._emit_resume_hold_action()
            self._resume_hold_pending = False
            self._skip_record_once = True
            self._last_ik_ok = True
            self._last_tracking_err_m = 0.0
            sample_for_log = sample_R if sample_R is not None else sample_L
            self._maybe_log(obs, sample_for_log, "clutch_engage", age, now)
            return action

        actions = []
        ik_ok = True
        track_err = 0.0
        if drive_L:
            actions.append(
                self.L.forward(
                    self._last_left_pos,
                    self._last_left_quat,
                    float(self._last_left_grip),
                    apply_gripper_compliance=False,
                )
            )
            ik_L = getattr(self.L, "ik", None)
            ik_ok = ik_ok and bool(getattr(ik_L, "ik_ok", True))
            track_err = max(
                track_err,
                float(np.linalg.norm(self._last_left_pos - left_pos)),
            )
        if drive_R:
            actions.append(
                self.R.forward(
                    self._last_right_pos,
                    self._last_right_quat,
                    float(self._last_right_grip),
                    orientation_cone_rad=self._orientation_cone_rad_R,
                    apply_gripper_compliance=False,
                )
            )
            ik_R = getattr(self.R, "ik", None)
            ik_ok = ik_ok and bool(getattr(ik_R, "ik_ok", True))
            track_err = max(
                track_err,
                float(np.linalg.norm(self._last_right_pos - right_pos)),
            )
        action = merge_joint_position_actions(*actions, n_dof=self._n_dof)
        self._last_ik_ok = ik_ok
        self._last_tracking_err_m = track_err
        if not ik_ok:
            self._ik_fail_streak += 1
            reason = "ik_reject"
        else:
            self._ik_fail_streak = 0

        if (
            self._session.is_recording
            and not self._paused
            and not self._skip_record_once
            and self._should_record_frame(obs, now)
        ):
            self._record(obs, left_pos, left_quat, right_pos, right_quat)
        self._skip_record_once = False
        sample_for_log = sample_R if sample_R is not None else sample_L
        self._maybe_log(obs, sample_for_log, reason, age, now)
        return action

    def _global_hold(self) -> Tuple[bool, str]:
        if self._estop or self._paused or self._abort or self._session.done:
            if self._abort:
                return True, "abort_hold"
            if self._session.is_warmup:
                return True, "warmup"
            return True, "hold"
        if (not self._keyboard_mode) and self._session.is_warmup:
            return True, "warmup"
        return False, "tracking"

    def _step_arm(
        self,
        *,
        side: str,
        sample: Optional[Dict[str, Any]],
        retarget: CartesianRetargeter,
        current_pos,
        current_quat,
        wall_dt: float,
        drive: bool,
        obs: Observation,
    ) -> Tuple[str, float]:
        """Update last_* targets for one arm. Returns (reason, age_s)."""
        is_right = side == "R"
        if is_right:
            last_pos = self._last_right_pos
            last_quat = self._last_right_quat
            last_grip = self._last_right_grip
            ctrl = self.R
            g_idx = self._Rg
        else:
            last_pos = self._last_left_pos
            last_quat = self._last_left_quat
            last_grip = self._last_left_grip
            ctrl = self.L
            g_idx = self._Lg

        hold, reason = self._global_hold()
        age = 0.0
        if not self._keyboard_mode:
            leader = self._leader_R if is_right else self._leader_L
            age = float("inf") if leader is None else leader.age_s()
            if sample is None or age > self._stale_pause_s:
                hold = True
                reason = "stale_pause" if sample is not None else "no_sample"
            elif age > self._stale_hold_s:
                hold = True
                reason = "stale_hold"

        if not drive or hold or sample is None:
            pos = last_pos if last_pos is not None else current_pos
            quat = last_quat if last_quat is not None else current_quat
            grip = last_grip
        else:
            if (
                self._keyboard_mode
                and is_right
                and not self._kbd_motion_active
                and retarget.state.engaged
                and last_pos is not None
                and last_quat is not None
            ):
                retarget.capture_origins(
                    sample["ee_pos"],
                    sample["ee_quat_wxyz"],
                    last_pos,
                    last_quat,
                )
            # Reanchor from the commanded / frozen virtual target, not the
            # observed EE (which can lag IK and cause a resume jump).
            dex_pos = last_pos if last_pos is not None else current_pos
            dex_quat = last_quat if last_quat is not None else current_quat
            prox_depth = self._laser_range_m() if is_right else None
            pos, quat, _grip_hint, info = retarget.step(
                leader_pos=sample["ee_pos"],
                leader_quat=sample["ee_quat_wxyz"],
                gripper_norm=sample["gripper_norm"],
                dt=wall_dt,
                clutch=bool(sample["clutch"]) and not self._paused,
                deadman=bool(sample["deadman"]) and not self._estop,
                current_dex_pos=dex_pos,
                current_dex_quat=dex_quat,
                proximity_depth_m=prox_depth,
            )
            if g_idx is None:
                grip = float(_grip_hint)
            else:
                compliance = getattr(ctrl, "gripper_compliance", None)
                if compliance is None or not getattr(compliance, "enabled", True):
                    # Module off: use retarget map_gripper aperture directly.
                    grip = float(_grip_hint)
                else:
                    q_meas = float(obs.joint_positions[g_idx])
                    qd_meas = float(obs.joint_velocities[g_idx])
                    grip = float(
                        compliance.update(
                            q_meas=q_meas,
                            qd_meas=qd_meas,
                            dt=float(self._dt),
                            gripper_norm=float(sample["gripper_norm"]),
                        )
                    )
            if is_right:
                self._last_prox_rate_scale = float(
                    info.get("rate_limit_scale", 1.0)
                )
                self._last_prox_delta_scale = float(
                    info.get("delta_gain_scale", 1.0)
                )
            reason = info.get("reason", reason)
            if sample.get("timestamp_ns"):
                latency_ms = max(
                    0.0, (time.time_ns() - int(sample["timestamp_ns"])) / 1e6
                )
                self._latencies_ms.append(latency_ms)
                if len(self._latencies_ms) > 500:
                    self._latencies_ms = self._latencies_ms[-500:]

        pos = np.asarray(pos, dtype=np.float64)
        quat = T.normalize_quat_wxyz(quat)
        grip = float(grip)
        if is_right:
            self._last_right_pos = pos
            self._last_right_quat = quat
            self._last_right_grip = grip
        else:
            self._last_left_pos = pos
            self._last_left_quat = quat
            self._last_left_grip = grip
        return reason, age

    def _laser_range_m(self) -> Optional[float]:
        """R-wrist laser ``last_length``, or None if not ready (fail-open)."""
        laser = self._r_wrist_laser
        if laser is None:
            return None
        if not str(getattr(laser, "last_cast_source", "") or ""):
            return None
        return float(laser.last_length)

    def _advance_record_deadline(self, deadline: float, now: float) -> float:
        next_t = deadline + self._record_period_s
        if now >= next_t:
            missed = int((now - next_t) / self._record_period_s) + 1
            next_t += missed * self._record_period_s
        return next_t

    def _should_record_frame(self, obs: Observation, now: float) -> bool:
        if self._playback_clock == "sim":
            sim_t = float(obs.step_idx) * self._dt
            if self._next_record_sim is None:
                self._next_record_sim = sim_t
            if sim_t + 1e-12 < self._next_record_sim:
                return False
            self._next_record_sim = self._advance_record_deadline(
                self._next_record_sim, sim_t
            )
            return True
        if self._next_record_wall is None:
            self._next_record_wall = now
        if now + 1e-9 < self._next_record_wall:
            return False
        self._next_record_wall = self._advance_record_deadline(
            self._next_record_wall, now
        )
        return True

    def _read_leader_samples(
        self, wall_dt: Optional[float] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Return (sample_R, sample_L). Keyboard fills sample_R only."""
        if self._keyboard_mode and self._kbd_ee is not None and self._kbd_input is not None:
            self._maybe_upgrade_keyboard_backend()
            held = self._kbd_input.poll_held()
            self._kbd_motion_active = any(
                t.startswith("ee+") or t.startswith("ee-") for t in held
            )
            kbd_dt = float(self._dt if wall_dt is None else wall_dt)
            moved = self._kbd_ee.apply_holds(held, kbd_dt)
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
            return sample, None
        self._kbd_motion_active = False
        sample_R = None
        sample_L = None
        if self._leader_R is not None:
            sample_R = self._sample_from_leader(self._leader_R)
        if self._leader_L is not None:
            sample_L = self._sample_from_leader(self._leader_L)
        return sample_R, sample_L

    def _sample_from_leader(self, leader: LeaderClient) -> Optional[Dict[str, Any]]:
        """Prefer a queued cmd-event sample so pause/resume keep their pose."""
        event = leader.pop_cmd_event()
        if event is not None:
            return dict(event)
        raw = leader.latest()
        if raw is None:
            return None
        sample = dict(raw)
        sample["cmd"] = "none"
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

    def _disengage_retargeters(self) -> None:
        self._retarget_R.disengage()
        if self._retarget_L is not None:
            self._retarget_L.disengage()

    def _reset_retargeters(self) -> None:
        self._retarget_R.reset()
        if self._retarget_L is not None:
            self._retarget_L.reset()

    def _controller_cspace_q(self, controller) -> Optional[np.ndarray]:
        if controller is None:
            return None
        try:
            q = controller.current_cspace_q()
            return np.asarray(q, dtype=np.float64).reshape(-1).copy()
        except Exception:
            return None

    def _latch_frozen_arm(
        self,
        *,
        pos,
        quat,
        grip: float,
        controller,
    ) -> Dict[str, Any]:
        return {
            "pos": np.asarray(pos, dtype=np.float64).copy(),
            "quat": T.normalize_quat_wxyz(quat),
            "grip": float(grip),
            "q": self._controller_cspace_q(controller),
        }

    def _enter_pause(
        self,
        now: float,
        *,
        left_pos,
        left_quat,
        right_pos,
        right_quat,
    ) -> None:
        was = self._paused
        self._paused = True
        self._session.set_clock_paused(True, now)
        self._disengage_retargeters()
        self._resume_hold_pending = False
        # Latch the actual virtual EE so resume reanchors without yanking
        # toward a stale IK command or the physical leader's absolute pose.
        if "R" in self.active_arms:
            if right_pos is None:
                right_pos = (
                    self._last_right_pos
                    if self._last_right_pos is not None
                    else np.zeros(3)
                )
            if right_quat is None:
                right_quat = (
                    self._last_right_quat
                    if self._last_right_quat is not None
                    else np.array([1.0, 0.0, 0.0, 0.0])
                )
            grip = (
                float(self._last_right_grip)
                if self._last_right_grip is not None
                else 0.0
            )
            self._frozen_right = self._latch_frozen_arm(
                pos=right_pos,
                quat=right_quat,
                grip=grip,
                controller=self.R,
            )
            self._last_right_pos = self._frozen_right["pos"].copy()
            self._last_right_quat = self._frozen_right["quat"].copy()
            self._last_right_grip = float(self._frozen_right["grip"])
        if "L" in self.active_arms and self._retarget_L is not None:
            if left_pos is None:
                left_pos = (
                    self._last_left_pos
                    if self._last_left_pos is not None
                    else np.zeros(3)
                )
            if left_quat is None:
                left_quat = (
                    self._last_left_quat
                    if self._last_left_quat is not None
                    else np.array([1.0, 0.0, 0.0, 0.0])
                )
            grip = (
                float(self._last_left_grip)
                if self._last_left_grip is not None
                else 0.0
            )
            self._frozen_left = self._latch_frozen_arm(
                pos=left_pos,
                quat=left_quat,
                grip=grip,
                controller=self.L,
            )
            self._last_left_pos = self._frozen_left["pos"].copy()
            self._last_left_quat = self._frozen_left["quat"].copy()
            self._last_left_grip = float(self._frozen_left["grip"])
        if not was:
            self._next_record_wall = None
            self._next_record_sim = None
            print(
                "[aloha_teleop] PAUSE (WIP): physics and recording frozen. "
                "Reposition the leader, then Space or u to reanchor "
                "(reanchor path is not reliable yet).",
                flush=True,
            )

    def _resume_and_reanchor(
        self,
        now: float,
        *,
        sample_R: Optional[Dict[str, Any]],
        sample_L: Optional[Dict[str, Any]],
    ) -> None:
        """Pair frozen virtual pose with the Space-2 leader pose; no jump."""
        was = self._paused
        self._disengage_retargeters()

        if "R" in self.active_arms and sample_R is not None:
            frozen = self._frozen_right
            if frozen is None:
                frozen = self._latch_frozen_arm(
                    pos=self._last_right_pos
                    if self._last_right_pos is not None
                    else sample_R["ee_pos"],
                    quat=self._last_right_quat
                    if self._last_right_quat is not None
                    else sample_R["ee_quat_wxyz"],
                    grip=float(self._last_right_grip or 0.0),
                    controller=self.R,
                )
            self._retarget_R.capture_origins(
                sample_R["ee_pos"],
                sample_R["ee_quat_wxyz"],
                frozen["pos"],
                frozen["quat"],
            )
            self._retarget_R.state.last_gripper = float(frozen["grip"])
            self._last_right_pos = frozen["pos"].copy()
            self._last_right_quat = frozen["quat"].copy()
            self._last_right_grip = float(frozen["grip"])
            self._frozen_right = frozen

        if (
            "L" in self.active_arms
            and self._retarget_L is not None
            and sample_L is not None
        ):
            frozen = self._frozen_left
            if frozen is None:
                frozen = self._latch_frozen_arm(
                    pos=self._last_left_pos
                    if self._last_left_pos is not None
                    else sample_L["ee_pos"],
                    quat=self._last_left_quat
                    if self._last_left_quat is not None
                    else sample_L["ee_quat_wxyz"],
                    grip=float(self._last_left_grip or 0.0),
                    controller=self.L,
                )
            self._retarget_L.capture_origins(
                sample_L["ee_pos"],
                sample_L["ee_quat_wxyz"],
                frozen["pos"],
                frozen["quat"],
            )
            self._retarget_L.state.last_gripper = float(frozen["grip"])
            self._last_left_pos = frozen["pos"].copy()
            self._last_left_quat = frozen["quat"].copy()
            self._last_left_grip = float(frozen["grip"])
            self._frozen_left = frozen

        self._paused = False
        self._session.set_clock_paused(False, now)
        self._resume_hold_pending = True
        self._skip_record_once = True
        self._next_record_wall = None
        self._next_record_sim = None
        if was:
            print(
                "[aloha_teleop] RESUME (WIP): attempted origin recapture at the "
                "new leader pose; verify DexMate stays put before relying on this.",
                flush=True,
            )

    def _emit_resume_hold_action(self):
        """Raw joint hold so the first resumed physics step cannot yank."""
        actions = []
        if "L" in self.active_arms and self.L is not None:
            frozen = self._frozen_left
            q = None if frozen is None else frozen.get("q")
            grip = (
                float(self._last_left_grip)
                if self._last_left_grip is not None
                else 0.0
            )
            if q is not None and hasattr(self.L, "forward_raw_q"):
                actions.append(
                    self.L.forward_raw_q(
                        q, grip, apply_gripper_compliance=False
                    )
                )
            else:
                actions.append(
                    self.L.forward(
                        self._last_left_pos,
                        self._last_left_quat,
                        grip,
                        apply_gripper_compliance=False,
                    )
                )
        if "R" in self.active_arms and self.R is not None:
            frozen = self._frozen_right
            q = None if frozen is None else frozen.get("q")
            grip = (
                float(self._last_right_grip)
                if self._last_right_grip is not None
                else 0.0
            )
            if q is not None and hasattr(self.R, "forward_raw_q"):
                actions.append(
                    self.R.forward_raw_q(
                        q, grip, apply_gripper_compliance=False
                    )
                )
            else:
                actions.append(
                    self.R.forward(
                        self._last_right_pos,
                        self._last_right_quat,
                        grip,
                        orientation_cone_rad=self._orientation_cone_rad_R,
                        apply_gripper_compliance=False,
                    )
                )
        return merge_joint_position_actions(*actions, n_dof=self._n_dof)

    def _set_paused(self, paused: bool, now: float) -> None:
        """Legacy helper for start/reset/estop paths without latch poses."""
        if paused:
            right_pos = (
                self._last_right_pos
                if self._last_right_pos is not None
                else np.zeros(3)
            )
            right_quat = (
                self._last_right_quat
                if self._last_right_quat is not None
                else np.array([1.0, 0.0, 0.0, 0.0])
            )
            left_pos = (
                self._last_left_pos
                if self._last_left_pos is not None
                else np.zeros(3)
            )
            left_quat = (
                self._last_left_quat
                if self._last_left_quat is not None
                else np.array([1.0, 0.0, 0.0, 0.0])
            )
            self._enter_pause(
                now,
                left_pos=left_pos,
                left_quat=left_quat,
                right_pos=right_pos,
                right_quat=right_quat,
            )
            return
        self._paused = False
        self._session.set_clock_paused(False, now)
        self._disengage_retargeters()
        self._resume_hold_pending = False
        self._next_record_wall = None
        self._next_record_sim = None

    def _apply_cmd(
        self,
        cmd: str,
        now: float,
        *,
        sample_R: Optional[Dict[str, Any]] = None,
        sample_L: Optional[Dict[str, Any]] = None,
        left_pos=None,
        left_quat=None,
        right_pos=None,
        right_quat=None,
    ) -> None:
        if cmd in (None, "none") or cmd in RECORDING_CMDS:
            return
        if cmd == "start":
            self._estop = False
            if self._paused:
                self._resume_and_reanchor(
                    now, sample_R=sample_R, sample_L=sample_L
                )
            else:
                self._set_paused(False, now)
        elif cmd == "pause":
            self._enter_pause(
                now,
                left_pos=left_pos if left_pos is not None else self._last_left_pos,
                left_quat=left_quat
                if left_quat is not None
                else self._last_left_quat,
                right_pos=right_pos
                if right_pos is not None
                else self._last_right_pos,
                right_quat=right_quat
                if right_quat is not None
                else self._last_right_quat,
            )
        elif cmd == "resume":
            self._estop = False
            self._resume_and_reanchor(
                now, sample_R=sample_R, sample_L=sample_L
            )
        elif cmd == "recenter":
            self._disengage_retargeters()
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
            self._enter_pause(
                now,
                left_pos=left_pos if left_pos is not None else self._last_left_pos,
                left_quat=left_quat
                if left_quat is not None
                else self._last_left_quat,
                right_pos=right_pos
                if right_pos is not None
                else self._last_right_pos,
                right_quat=right_quat
                if right_quat is not None
                else self._last_right_quat,
            )
        elif cmd == "reset":
            self._reset_retargeters()
            self._part_done = False
            self._estop = False
            self._set_paused(False, now)

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
            self._next_record_wall = None
            self._next_record_sim = None
            self._abort = False
            self._part_done = False
            self._frames_sent = 0
            self._drops = 0
            return
        if ev.kind == "record_start":
            self._episode_seq = 0
            self._next_record_wall = None
            self._next_record_sim = None
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
            "control_arms": self._control_arms,
            "active_arms": list(self.active_arms),
        }

    def _stats(self) -> Dict[str, Any]:
        return {
            "frames": self._frames_sent,
            "drops": self._drops,
            "p95_latency_ms": (
                float(np.percentile(self._latencies_ms, 95)) if self._latencies_ms else None
            ),
            "leader_hz": (
                None if self._leader_R is None else self._leader_R.hz
            ),
            "leader_hz_left": (
                None if self._leader_L is None else self._leader_L.hz
            ),
            "control_arms": self._control_arms,
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
        action = pack_teleop_action(
            control_arms=self._control_arms if not self._keyboard_mode else "right",
            last_left_pos=self._last_left_pos,
            last_left_quat=self._last_left_quat,
            last_left_grip=float(self._last_left_grip),
            last_right_pos=self._last_right_pos,
            last_right_quat=self._last_right_quat,
            last_right_grip=float(self._last_right_grip),
            home_left_pos=self._home_left_pos,
            home_left_quat=self._home_left_quat,
            home_left_grip=float(self._home_left_grip),
        )
        rgb = obs.rgb or {}
        images = {}
        for cam in self._cameras:
            obs_key = self._export.obs_camera_key(cam)
            images[cam] = resize_rgb(
                rgb.get(obs_key),
                height=self._img_h,
                width=self._img_w,
            )
        timestamp_s = float(self._episode_seq) / float(self._record_fps)
        try:
            reply = self._recorder.send({
                "cmd": "frame",
                "step_idx": int(obs.step_idx),
                "timestamp_s": timestamp_s,
                "seq": int(self._episode_seq),
                "state": state.tolist(),
                "action": action.tolist(),
                "images": {
                    k: encode_jpeg(v, quality=self._jpeg_quality)
                    for k, v in images.items()
                },
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
        if self._paused:
            extra = " clutch=pause"
        if self._session.is_warmup:
            extra += f" warmup_left={self._session.remaining_warmup_s(now):.1f}s"
        elif self._session.is_recording:
            extra += f" rec_left={self._session.remaining_episode_s(now):.1f}s"
        target = self._last_right_pos if self._last_right_pos is not None else self._last_left_pos
        hz_r = 0.0 if self._leader_R is None else self._leader_R.hz
        hz_l = 0.0 if self._leader_L is None else self._leader_L.hz
        connected = True if self._keyboard_mode else (
            False if self._leader_R is None else self._leader_R.connected
        )
        clutch = None if sample is None else sample.get("clutch")
        print(
            f"[aloha_teleop] t={obs.step_idx * self._dt:6.2f}s part="
            f"{getattr(self._target, 'name', None)} reason={reason} "
            f"phase={self._session.phase}{extra} "
            f"arms={self._control_arms} "
            f"clutch={clutch} "
            f"ik_ok={self._last_ik_ok} "
            f"track_err_mm={self._last_tracking_err_m * 1000.0:.1f} "
            f"prox_delta={self._last_prox_delta_scale:.2f} "
            f"prox_rate={self._last_prox_rate_scale:.2f} "
            f"target={[round(float(v), 3) for v in target]} "
            f"leader_hz={hz_r:5.1f}"
            + (f"/{hz_l:5.1f}" if self._leader_L is not None else "")
            + f" age={age:.3f}s "
            f"p95_lat_ms={p95:.1f} frames={self._frames_sent} drops={self._drops} "
            f"connected={connected}",
            flush=True,
        )
        if reason == "clutch_off":
            print(
                "[aloha_teleop] DexMate held (pause/clutch off). Reposition the "
                "leader; Space or u reanchors without moving DexMate.",
                flush=True,
            )

    def finalize(self, results_json: Optional[str] = None, success: Optional[bool] = None) -> None:
        del success  # task success must not decide whether data is kept
        if self._closed:
            return
        self._closed = True
        if self._grasp_ui is not None:
            try:
                self._grasp_ui.destroy()
            except Exception:
                pass
            self._grasp_ui = None
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
        for leader in (self._leader_R, self._leader_L):
            if leader is not None:
                try:
                    leader.close()
                except Exception:
                    pass
        self._leader_R = None
        self._leader_L = None
        self._leader = None

    def __del__(self):
        try:
            self.finalize()
        except Exception:
            pass
