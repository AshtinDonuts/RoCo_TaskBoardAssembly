import argparse
from collections import deque
import json
import os
import traceback
from pathlib import Path
from textwrap import dedent

import numpy as np

import run_pick_place as rp
from isaacsim.core.utils.types import ArticulationAction
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from policy_api import EnvInfo, Observation, PartTarget


TASK_DESCRIPTION = (
    "assemble the parts onto the task board"
)
ROBOT_TYPE = "vega_1u"
GRIPPER_OPEN_LIMIT = 0.6649704
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect a TaskBoardAssembly scripted rollout into a local LeRobot v3 dataset "
            "using physics-callback sampling and episode-relative timestamps."
        ),
        epilog=dedent(
            """
            Runtime requirements:
              - Run with the Isaac Sim Python environment used by this repository.
              - Install lerobot 0.4.4 in that environment.
              - Keep numpy on a 1.x build compatible with Isaac Sim camera annotators.

            Example:
              OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_HEADLESS=1 \
              uv run --group collection python task/collect_lerobot_v3.py \
                --output-root artifacts/lerobot/source \
                --results-json artifacts/lerobot/results/seed-000.json \
                --random-seed 0 --sample-hz 10
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/lerobot_taskboard_episode"),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="taskboard/aware_35ec027_full_assembly",
    )
    parser.add_argument(
        "--policy",
        default="policies.baseline_scripted.BaselinePolicy",
    )
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=10.0,
        help=(
            "Target recording rate in Hz. Default is 10 Hz. "
            "Ignored when --sample-stride is set."
        ),
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help=(
            "Record one frame every N physics steps. Overrides --sample-hz. "
            "At 200 Hz physics, 200 means 1 Hz."
        ),
    )
    parser.add_argument(
        "--max-recorded-frames",
        type=int,
        default=0,
        help="Stop after recording this many frames. 0 disables the cap.",
    )
    parser.add_argument(
        "--max-parts",
        type=int,
        default=0,
        help="Stop after this many parts have become active. 0 means the full task order.",
    )
    parser.add_argument(
        "--max-sim-steps",
        type=int,
        default=0,
        help=(
            "Hard safety cap on physics steps. 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Final per-seed transaction/result JSON.",
    )
    parser.add_argument(
        "--save-summary",
        type=Path,
        default=None,
    )
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append to an existing compatible dataset (default: true).",
    )
    return parser.parse_args()


def _quat_wxyz_to_rotvec(quat):
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    xyz = q[1:]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = xyz / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    return (axis * angle).astype(np.float32)


def _pack_pose(pos, quat):
    pos_arr = np.asarray(pos, dtype=np.float32).reshape(3)
    quat_arr = np.asarray(quat, dtype=np.float32).reshape(4)
    return np.concatenate([pos_arr, quat_arr], axis=0)


def _pack_cartesian_action(pos, quat, gripper):
    pos_arr = np.asarray(pos, dtype=np.float32).reshape(3)
    rotvec = _quat_wxyz_to_rotvec(quat)
    grip = np.array([float(gripper)], dtype=np.float32)
    return np.concatenate([pos_arr, rotvec, grip], axis=0)


def _as_rgb(frame):
    if frame is None:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr.astype(np.uint8, copy=False)
    if arr.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH):
        return arr
    try:
        import cv2

        return cv2.resize(arr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.asarray(Image.fromarray(arr).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))


def _feature_spec():
    return {
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_ee_x", "left_ee_y", "left_ee_z",
                "left_ee_rx", "left_ee_ry", "left_ee_rz", "left_gripper",
                "right_ee_x", "right_ee_y", "right_ee_z",
                "right_ee_rx", "right_ee_ry", "right_ee_rz", "right_gripper",
            ],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (44,),
            "names": (
                [f"left_ee_{v}" for v in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                + [f"right_ee_{v}" for v in ("x", "y", "z", "qw", "qx", "qy", "qz")]
                + [f"left_joint_{i}" for i in range(7)]
                + [f"right_joint_{i}" for i in range(7)]
                + [f"left_joint_velocity_{i}" for i in range(7)]
                + [f"right_joint_velocity_{i}" for i in range(7)]
                + ["left_gripper", "right_gripper"]
            ),
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.left_hand": {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_hand": {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
    }


def _gripper_ratio(value):
    return np.float32(np.clip(float(value) / GRIPPER_OPEN_LIMIT, 0.0, 1.0))


def _pack_state(l_pos, l_quat, r_pos, r_quat, q, qd, li, ri, lg, rg):
    return np.concatenate([
        _pack_pose(l_pos, l_quat),
        _pack_pose(r_pos, r_quat),
        np.asarray(q, dtype=np.float32)[li],
        np.asarray(q, dtype=np.float32)[ri],
        np.asarray(qd, dtype=np.float32)[li],
        np.asarray(qd, dtype=np.float32)[ri],
        np.asarray([_gripper_ratio(q[lg]), _gripper_ratio(q[rg])], dtype=np.float32),
    ]).astype(np.float32)


def _part_segments(frame_parts, part_order):
    if not frame_parts:
        raise ValueError("cannot build part segments without recorded frames")
    segments = []
    begin = 0
    active = frame_parts[0]
    for idx, name in enumerate(frame_parts[1:], 1):
        if name != active:
            segments.append({"part": active, "begin": begin, "end": idx})
            begin, active = idx, name
    segments.append({"part": active, "begin": begin, "end": len(frame_parts)})
    actual = [segment["part"] for segment in segments]
    expected = list(part_order)
    if actual != expected:
        raise ValueError(f"recorded part order {actual!r} does not match runtime order {expected!r}")
    if any(segment["end"] <= segment["begin"] for segment in segments):
        raise ValueError("every part must have a nonempty recorded segment")
    return segments


def _waypoint_annotation(policy):
    """Return the active policy waypoint without changing the dataset schema."""
    try:
        waypoint = getattr(policy, "current_waypoint", None)
        if callable(waypoint):
            waypoint = waypoint()
        name = None if waypoint is None else getattr(waypoint, "name", None)
        index = getattr(policy, "current_index", None)
        if callable(index):
            index = index()
        return {
            "name": None if name is None else str(name),
            "waypoint_index": None if index is None else int(index),
        }
    except Exception:
        return {"name": None, "waypoint_index": None}


def _attach_phase_segments(segments, frame_waypoints):
    """Attach absolute, contiguous waypoint ranges to each part segment."""
    if not segments or len(frame_waypoints) != segments[-1]["end"]:
        raise ValueError("waypoint annotations must align one-to-one with recorded frames")
    complete = True
    for segment in segments:
        phases = []
        begin, end = int(segment["begin"]), int(segment["end"])
        phase_begin = begin
        active = dict(frame_waypoints[begin])
        for frame_idx in range(begin + 1, end):
            annotation = dict(frame_waypoints[frame_idx])
            if annotation != active:
                phases.append({**active, "begin": phase_begin, "end": frame_idx})
                phase_begin, active = frame_idx, annotation
        phases.append({**active, "begin": phase_begin, "end": end})
        segment["phases"] = phases
        if any(phase["name"] is None or phase["waypoint_index"] is None for phase in phases):
            complete = False
    return complete


def _resolve_gripper_cmd(command, cfg, current_value):
    if command is None:
        return float(current_value)
    if isinstance(command, str):
        cmd = command.lower()
        if cmd == "open":
            return float(cfg.get("gripper_open", current_value))
        if cmd == "close":
            return float(cfg.get("gripper_close", current_value))
    try:
        return float(command)
    except Exception:
        return float(current_value)


def _left_action_target(policy, obs, current_cfg):
    wp = getattr(policy, "current_waypoint", None)
    if callable(wp):
        wp = wp()
    ee_pos, ee_orn = obs.ee_pose_L
    if wp is None:
        return ee_pos, ee_orn, float(obs.L_gripper_position)
    if getattr(wp, "cspace_target", None) is not None:
        return ee_pos, ee_orn, _resolve_gripper_cmd(wp.gripper, current_cfg, obs.L_gripper_position)
    if getattr(wp, "joint_lerp_t", None) is not None or getattr(wp, "lock_pose", False):
        return ee_pos, ee_orn, _resolve_gripper_cmd(wp.gripper, current_cfg, obs.L_gripper_position)
    pos = wp.pos if getattr(wp, "pos", None) is not None else ee_pos
    orn = wp.orn if getattr(wp, "orn", None) is not None else ee_orn
    grip = _resolve_gripper_cmd(wp.gripper, current_cfg, obs.L_gripper_position)
    return pos, orn, grip


def _build_summary_path(args):
    return args.save_summary


def _resolve_sample_stride(args, physics_hz):
    if args.sample_stride is not None:
        if args.sample_stride <= 0:
            raise ValueError("--sample-stride must be > 0")
        stride = int(args.sample_stride)
    else:
        if args.sample_hz <= 0:
            raise ValueError("--sample-hz must be > 0 when --sample-stride is unset")
        stride = max(1, int(round(float(physics_hz) / float(args.sample_hz))))
    return stride


def _resolve_recording_config(args, physics_hz):
    if args.sample_stride is not None:
        stride = _resolve_sample_stride(args, physics_hz)
        fps = max(1, int(round(float(physics_hz) / float(stride))))
        return {
            "mode": "stride",
            "sample_stride": stride,
            "sample_fps": fps,
            "sample_period_s": float(stride) / float(physics_hz),
        }

    target_hz = float(args.sample_hz)
    return {
        "mode": "time",
        "sample_stride": None,
        "sample_fps": max(1, int(round(target_hz))),
        "sample_period_s": 1.0 / target_hz,
    }


def main():
    args = _parse_args()
    if args.max_recorded_frames < 0:
        raise ValueError("--max-recorded-frames must be >= 0")
    if args.max_parts < 0:
        raise ValueError("--max-parts must be >= 0")

    output_root = args.output_root.resolve()
    args.results_json = args.results_json.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)

    physics_hz = 200.0
    physics_dt = 1.0 / physics_hz
    recording_cfg = _resolve_recording_config(args, physics_hz)
    sample_stride = recording_cfg["sample_stride"]
    sample_fps = recording_cfg["sample_fps"]
    effective_max_parts = len(rp.pc.part_order) if args.max_parts == 0 else int(args.max_parts)
    effective_max_frames = None if args.max_recorded_frames == 0 else int(args.max_recorded_frames)
    expected_features = _feature_spec()
    if output_root.exists():
        if not args.append:
            raise FileExistsError(f"output root already exists: {output_root}")
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=output_root,
            streaming_encoding=True,
            encoder_queue_maxsize=120,
            vcodec="h264",
        )
        info = dataset.meta.info
        mismatches = []
        if int(info["fps"]) != sample_fps:
            mismatches.append(f"fps={info['fps']} (requested {sample_fps})")
        if info.get("robot_type") != ROBOT_TYPE:
            mismatches.append(f"robot_type={info.get('robot_type')!r}")
        for key, spec in expected_features.items():
            actual = info.get("features", {}).get(key)
            if actual is None:
                mismatches.append(f"missing feature {key}")
                continue
            if actual.get("dtype") != spec["dtype"] or tuple(actual.get("shape", ())) != tuple(spec["shape"]):
                mismatches.append(f"incompatible feature {key}: {actual}")
        extra = set(info.get("features", {})) - set(expected_features) - {
            "timestamp", "frame_index", "episode_index", "index", "task_index"
        }
        if extra:
            mismatches.append(f"unexpected features: {sorted(extra)}")
        if mismatches:
            raise ValueError("cannot append to incompatible dataset: " + "; ".join(mismatches))
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=output_root,
            fps=sample_fps,
            robot_type=ROBOT_TYPE,
            features=expected_features,
            use_videos=True,
            image_writer_threads=4,
            streaming_encoding=True,
            encoder_queue_maxsize=120,
            vcodec="h264",
        )
    episode_index = int(dataset.meta.total_episodes)
    pending_result_path = args.results_json.with_suffix(".pending.json")
    if pending_result_path.is_file():
        pending = json.loads(pending_result_path.read_text(encoding="utf-8"))
        pending_episode = int(pending.get("episode_index", -1))
        pending_seed = int(pending.get("seed", -1))
        if pending_seed != args.random_seed:
            raise ValueError(
                f"pending result seed {pending_seed} does not match requested seed {args.random_seed}"
            )
        if pending_episode < episode_index:
            args.results_json.parent.mkdir(parents=True, exist_ok=True)
            os.replace(pending_result_path, args.results_json)
            dataset.finalize()
            print(
                f"[collect] recovered finalized episode {pending_episode} "
                f"for seed {args.random_seed}",
                flush=True,
            )
            return
        if pending_episode == episode_index:
            pending_result_path.unlink()
            print(
                f"[collect] discarding uncommitted pending result for episode {pending_episode}",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"pending result episode {pending_episode} is ahead of dataset episode {episode_index}"
            )

    evaluated_parts = tuple(
        name for name in rp.pc.part_order
        if isinstance(name, str) and not name.startswith("<")
    )
    randomized_trial = rp.XYRandomization.sample(args.random_seed, evaluated_parts)
    print(
        f"[collect] randomization seed={args.random_seed} "
        f"episode_index={episode_index}",
        flush=True,
    )

    def _runtime_config(name):
        return randomized_trial.shifted_config(name, rp.pc.get_part_config(name))

    def _policy_config(name):
        return rp.resolve_policy_config(
            name, rp.pc.get_part_config(name), trial=randomized_trial, blind=False
        )

    camera_output_enabled = True
    enable_camera_viewports = bool(rp.pc.enable_camera_viewports and not rp._HEADLESS)
    render_each_step = True

    dummy_target = np.zeros(3, dtype=np.float64)
    (
        my_world,
        my_controller,
        my_robots,
        head_depth_camera,
        l_wrist_camera,
        r_wrist_camera,
        articulation_controller,
        _task_params,
        reset_needed,
    ) = rp.setup_pick_place_sim(
        L_object_prim_path=rp.pc.L_object_prim_path,
        R_object_prim_path=rp.pc.R_object_prim_path,
        L_target_position=dummy_target,
        R_target_position=dummy_target,
        joint_opened_position=np.array([rp.pc.PART_DEFAULTS["gripper_open"]]),
        joint_closed_position=np.array([rp.pc.PART_DEFAULTS["gripper_close"]]),
        enable_camera_viewports=enable_camera_viewports,
        enable_camera_output=camera_output_enabled,
        board_offset=randomized_trial.board_offset,
        part_offsets=randomized_trial.part_offsets,
    )
    rp.import_missing_parts(randomized_trial.part_offsets)

    l_controller = my_controller["L"]
    r_controller = my_controller["R"]
    l_robot = my_robots["L"]

    dof_names = list(l_robot.dof_names)
    r_arm_dof_indices = np.array([dof_names.index(j) for j in rp.R_ARM_JOINT_NAMES], dtype=np.int64)
    l_arm_dof_indices = np.array([dof_names.index(j) for j in sorted(j for j in dof_names if j.startswith("L_arm_j"))], dtype=np.int64)
    l_gripper_dof_index = dof_names.index("L_gripper_joint")
    r_gripper_dof_index = dof_names.index("R_gripper_joint")
    l_arm_joint_names = [j for j in dof_names if j.startswith("L_arm_j")]

    def _apply_init_joint_targets():
        targets = getattr(rp.pc, "INIT_JOINT_TARGETS", None)
        if not targets:
            return
        full_q = np.asarray(l_robot.get_joint_positions(), dtype=np.float64).copy()
        for jname, val in targets.items():
            if jname in dof_names:
                full_q[dof_names.index(jname)] = float(val)
        l_robot.set_joint_positions(full_q)
        l_robot.set_joint_velocities(np.zeros(len(dof_names)))

    _apply_init_joint_targets()
    r_arm_hold_q = np.asarray(l_robot.get_joint_positions())[r_arm_dof_indices].astype(np.float64)
    l_arm_init_q = np.asarray(l_controller.current_cspace_q(), dtype=np.float64).copy()

    env_info = EnvInfo(
        dof_names=dof_names,
        L_arm_joints=l_arm_joint_names,
        R_arm_joints=list(rp.R_ARM_JOINT_NAMES),
        L_gripper_joint="L_gripper_joint",
        L_arm_init_q=l_arm_init_q.copy(),
        physics_dt=physics_dt,
        enable_camera_output=True,
        L_controller=l_controller,
    )
    policy_class = rp._load_policy_class(args.policy)
    policy = policy_class(env_info)
    print(f"[collect] policy: {policy_class.__module__}.{policy_class.__name__}")

    stage = rp.omni.usd.get_context().get_stage()
    physx_iface = rp.omni.physx.get_physx_interface()
    parts_iter = iter(rp.pc.part_order)
    current_part = None
    current_snap_attacher = None
    current_snap_sub = None
    snap_fired_parts = set()
    part_step_count = 0
    run_complete = False
    part_activation_count = 0
    recorded_frames = 0
    policy_action_ready = False
    last_recorded_step = None
    next_record_time_s = 0.0
    record_time_origin_s = None
    pending_record_queue = deque()
    record_sub = None
    recorded_parts = []
    recorded_frame_parts = []
    recorded_frame_waypoints = []
    part_completion_reasons = []
    per_part_timeout_steps = int(
        getattr(policy, "PER_PART_TIMEOUT_STEPS",
                getattr(rp.pc, "PER_PART_TIMEOUT_STEPS", 3000))
    )
    warmup_steps = int(getattr(rp.pc, "WARMUP_STEPS", 0))
    effective_max_sim_steps = (
        None if args.max_sim_steps == 0 else int(args.max_sim_steps)
    )

    def _clear_snap_state():
        nonlocal current_snap_attacher, current_snap_sub
        current_snap_sub = None
        current_snap_attacher = None

    def _actual_ee_pose(controller):
        pos, orn = controller.end_effector.get_world_pose()
        return np.asarray(pos, dtype=np.float64), np.asarray(orn, dtype=np.float64)

    def _camera_payload():
        rgb = {"head": None, "L_wrist": None, "R_wrist": None}
        for key, cam in (("head", head_depth_camera), ("L_wrist", l_wrist_camera), ("R_wrist", r_wrist_camera)):
            if cam is None:
                continue
            try:
                rgba = cam.get_rgba()
                if rgba is not None and rgba.size > 0:
                    rgb[key] = np.asarray(rgba[..., :3], dtype=np.uint8)
            except Exception:
                pass
        return rgb

    def _build_observation(step_idx_override=None):
        full_q = np.asarray(l_robot.get_joint_positions(), dtype=np.float64)
        try:
            full_qd = np.asarray(l_robot.get_joint_velocities(), dtype=np.float64)
        except Exception:
            full_qd = np.zeros_like(full_q)

        ee_pos, ee_orn = _actual_ee_pose(l_controller)
        rgb = _camera_payload()

        snap_fired = bool(current_snap_attacher is not None and current_snap_attacher.attached)
        return Observation(
            step_idx=(int(my_world.current_time_step_index) if step_idx_override is None else int(step_idx_override)),
            joint_positions=full_q,
            joint_velocities=full_qd,
            L_gripper_position=float(full_q[l_gripper_dof_index]),
            ee_pose_L=(ee_pos, ee_orn),
            rgb=rgb,
            depth={"head": None, "L_wrist": None, "R_wrist": None},
            intrinsics={"head": None, "L_wrist": None, "R_wrist": None},
            snap_fired=snap_fired,
            target_part=current_part if isinstance(current_part, str) else None,
        )

    def _build_part_target(name):
        cfg = _policy_config(name)
        snap = cfg.get("snap") or {}
        def _arr(v):
            return None if v is None else np.asarray(v, dtype=np.float64).copy()
        return PartTarget(
            name=name,
            release_mode=cfg.get("release_mode", "open"),
            pick_pos=_arr(cfg.get("pick_pos")),
            spawn_orn=_arr(cfg.get("spawn_orn")),
            place_pos=_arr(cfg.get("place_pos")),
            grade_pos=_arr(cfg.get("grade_pos")),
            snap_target_pos=_arr(snap.get("target_pos")),
            snap_target_rot=_arr(snap.get("target_rot")),
            snap_pos_tol=_arr(snap.get("pos_tol_axes")),
            snap_rot_tol_deg=(None if snap.get("rot_tol_deg") is None else float(snap["rot_tol_deg"])),
            gripper_open=float(cfg.get("gripper_open", 0.0)),
            gripper_close=float(cfg.get("gripper_close", 0.0)),
            ee_orientation=_arr(cfg.get("ee_orientation")),
            extra=dict(cfg),
        )

    def _start_next_part():
        nonlocal current_part, current_snap_attacher, current_snap_sub
        nonlocal part_step_count, run_complete, part_activation_count, policy_action_ready

        if current_part is not None and current_snap_attacher is not None and current_snap_attacher.attached:
            snap_fired_parts.add(current_part)
        _clear_snap_state()
        policy_action_ready = False

        if part_activation_count >= effective_max_parts:
            current_part = None
            run_complete = True
            return None

        try:
            current_part = next(parts_iter)
        except StopIteration:
            current_part = None
            run_complete = True
            return None

        part_activation_count += 1
        recorded_parts.append(current_part)
        cfg = _runtime_config(current_part)
        if cfg.get("release_mode", "open") == "snap":
            snap_cfg = cfg.get("snap")
            current_snap_attacher = rp.build_snap_attacher(stage, current_part, snap_cfg)
            attacher = current_snap_attacher
            current_snap_sub = physx_iface.subscribe_physics_step_events(lambda dt, a=attacher: a.update())

        obs = _build_observation()
        target = _build_part_target(current_part)
        policy.reset(obs, target)
        part_step_count = 0
        print(f"[collect] now working on the part: {current_part}", flush=True)
        return current_part

    def _restart_iteration():
        nonlocal parts_iter, current_part, last_recorded_step, next_record_time_s
        nonlocal policy_action_ready, record_time_origin_s
        _clear_snap_state()
        snap_fired_parts.clear()
        pending_record_queue.clear()
        last_recorded_step = None
        next_record_time_s = 0.0
        record_time_origin_s = None
        policy_action_ready = False
        if stage is not None:
            for name in rp.pc.PART_CONFIG.keys():
                joint_path = f"/World/_snap_joint_{name}"
                if rp.is_prim_path_valid(joint_path):
                    stage.RemovePrim(joint_path)
        rp.restore_scene_part_xforms()
        parts_iter = iter(rp.pc.part_order)
        current_part = None
        _start_next_part()

    def _queue_record_frame(step_idx):
        nonlocal last_recorded_step, next_record_time_s, record_time_origin_s
        sim_time_s = step_idx * physics_dt
        if last_recorded_step is not None and step_idx < last_recorded_step:
            last_recorded_step = None
            next_record_time_s = 0.0
            record_time_origin_s = None

        if run_complete or current_part is None or not policy_action_ready:
            return
        if warmup_steps > 0 and step_idx < warmup_steps:
            return

        if recording_cfg["mode"] == "stride":
            if last_recorded_step is not None and step_idx - last_recorded_step < sample_stride:
                return
        else:
            if sim_time_s + 1e-9 < next_record_time_s:
                return

        if record_time_origin_s is None:
            record_time_origin_s = sim_time_s

        obs = _build_observation(step_idx_override=step_idx)
        l_pos, l_orn = obs.ee_pose_L
        r_pos, r_orn = _actual_ee_pose(r_controller)
        full_q = np.asarray(obs.joint_positions, dtype=np.float32)
        full_qd = np.asarray(obs.joint_velocities, dtype=np.float32)
        current_cfg = _policy_config(current_part) if current_part is not None else {}
        l_target_pos, l_target_orn, l_grip_cmd = _left_action_target(policy, obs, current_cfg)
        r_grip = float(full_q[r_gripper_dof_index])

        left_action = _pack_cartesian_action(
            l_target_pos, l_target_orn, _gripper_ratio(l_grip_cmd)
        )
        right_action = _pack_cartesian_action(
            r_pos, r_orn, _gripper_ratio(r_grip)
        )
        frame = {
            "action": np.concatenate([left_action, right_action]).astype(np.float32),
            "observation.state": _pack_state(
                l_pos, l_orn, r_pos, r_orn,
                full_q, full_qd,
                l_arm_dof_indices, r_arm_dof_indices,
                l_gripper_dof_index, r_gripper_dof_index,
            ),
            "observation.images.head": _as_rgb(obs.rgb.get("head")),
            "observation.images.left_hand": _as_rgb(obs.rgb.get("L_wrist")),
            "observation.images.right_hand": _as_rgb(obs.rgb.get("R_wrist")),
            "task": TASK_DESCRIPTION,
        }
        pending_record_queue.append({
            "frame": frame,
            "part": current_part,
            "waypoint": _waypoint_annotation(policy),
            "sim_time_s": float(sim_time_s - record_time_origin_s),
            "step_idx": int(step_idx),
        })
        last_recorded_step = step_idx
        if recording_cfg["mode"] == "time":
            while next_record_time_s <= sim_time_s + 1e-9:
                next_record_time_s += recording_cfg["sample_period_s"]

    def _flush_record_queue():
        nonlocal recorded_frames
        while pending_record_queue:
            if effective_max_frames is not None and recorded_frames >= effective_max_frames:
                pending_record_queue.clear()
                break
            item = pending_record_queue.popleft()
            dataset.add_frame(item["frame"])
            if dataset.episode_buffer is not None and dataset.episode_buffer["timestamp"]:
                dataset.episode_buffer["timestamp"][-1] = item["sim_time_s"]
            recorded_frame_parts.append(item["part"])
            recorded_frame_waypoints.append(item["waypoint"])
            recorded_frames += 1
            frame_cap_label = ("unbounded" if effective_max_frames is None else str(effective_max_frames))
            print(f"[collect] recorded frame {recorded_frames}/{frame_cap_label} at sim step {item['step_idx']}")

    def _on_physics_step(dt):
        del dt
        _queue_record_frame(int(my_world.current_time_step_index))

    summary = {
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "requested_sample_hz": args.sample_hz,
        "recording_mode": recording_cfg["mode"],
        "sample_stride": sample_stride,
        "sample_fps": sample_fps,
        "sample_period_s": recording_cfg["sample_period_s"],
        "max_recorded_frames": args.max_recorded_frames,
        "effective_max_recorded_frames": effective_max_frames,
        "max_parts": args.max_parts,
        "effective_max_parts": effective_max_parts,
        "max_sim_steps": args.max_sim_steps,
        "effective_max_sim_steps": effective_max_sim_steps,
        "episode_index": episode_index,
        "seed": args.random_seed,
        "xy_randomization": randomized_trial.as_dict(),
        "joint_dof_per_arm": 7,
        "action_encoding": "absolute_cartesian_target_xyz_rotvec_normalized_gripper",
        "notes": [
            "right arm is held at its initial pose by the current runner",
            "frames are selected from physics-step callbacks instead of the outer control loop",
            "stored timestamps are episode-relative physics times written explicitly into the dataset",
            "sample-hz mode uses simulation-time scheduling",
            "vega_1u exposes 7 arm joints per side, not 6",
        ],
    }
    max_sim_steps_label = "unbounded" if effective_max_sim_steps is None else str(effective_max_sim_steps)
    print(f"[collect] effective max sim steps: {max_sim_steps_label}")

    try:
        record_sub = physx_iface.subscribe_physics_step_events(_on_physics_step)
        _restart_iteration()
        while True:
            my_world.step(render=render_each_step)
            _flush_record_queue()
            if run_complete:
                break
            if not my_world.is_playing():
                if my_world.is_stopped():
                    reset_needed = True
                continue
            if reset_needed:
                my_world.reset()
                reset_needed = False
                _apply_init_joint_targets()
                l_controller.reset()
                r_controller.reset()
                _restart_iteration()
            if my_world.current_time_step_index == 0:
                _apply_init_joint_targets()
                l_controller.reset()
                r_controller.reset()
                _restart_iteration()

            if warmup_steps > 0 and my_world.current_time_step_index < warmup_steps:
                _apply_init_joint_targets()
                continue

            if effective_max_sim_steps is not None and int(my_world.current_time_step_index) >= effective_max_sim_steps:
                raise RuntimeError(
                    "collector exceeded max sim steps before finishing the requested episode"
                )

            if current_part is None:
                continue

            obs = _build_observation()
            if current_snap_attacher is not None and current_snap_attacher.attached:
                snap_fired_parts.add(current_part)

            cfg = _runtime_config(current_part)
            is_snap_done = (
                cfg.get("release_mode") == "snap"
                and current_snap_attacher is not None
                and current_snap_attacher.attached
            )
            is_timeout = part_step_count >= per_part_timeout_steps
            if policy.is_done(obs) or is_snap_done or is_timeout:
                completion_reason = (
                    "snap_done" if is_snap_done else
                    "timeout" if is_timeout else
                    "policy_done"
                )
                if current_part is not None:
                    part_completion_reasons.append({
                        "part": current_part,
                        "reason": completion_reason,
                        "step_count": part_step_count,
                    })
                _start_next_part()
                continue

            l_action = policy.act(obs)
            policy_action_ready = True
            if effective_max_frames is not None and recorded_frames >= effective_max_frames:
                run_complete = True
                break

            r_action_positions = [None] * len(dof_names)
            for j_idx, val in zip(r_arm_dof_indices, r_arm_hold_q.tolist()):
                r_action_positions[j_idx] = float(val)
            r_action = ArticulationAction(joint_positions=r_action_positions)
            merged = rp.merge_bimanual_actions(l_action, r_action, dof_names)
            articulation_controller.apply_action(merged)
            part_step_count += 1

        if recorded_frames == 0:
            raise RuntimeError("no frames were recorded")
        _flush_record_queue()
        if effective_max_parts != len(evaluated_parts):
            raise RuntimeError("only full nine-part episodes may be committed to the batch dataset")

        segments = _part_segments(recorded_frame_parts, evaluated_parts)
        waypoint_annotations_complete = _attach_phase_segments(
            segments, recorded_frame_waypoints
        )
        reasons = {row["part"]: row for row in part_completion_reasons}
        grade = rp._grade_task(
            stage,
            snap_fired_parts,
            results_json_path=None,
            metadata={
                "completion_reason": "complete",
                "completed_parts": len(recorded_parts),
                "total_task_steps": int(sum(row["step_count"] for row in part_completion_reasons)),
                "sim_time_s": float(my_world.current_time_step_index * physics_dt),
                "xy_randomization": randomized_trial.as_dict(),
                "blind_to_xy_randomization": False,
            },
            config_resolver=_runtime_config,
        )
        outcomes = {row["name"]: row for row in grade["per_part"]}
        for segment in segments:
            name = segment["part"]
            segment["name"] = segment.pop("part")
            segment["pass"] = bool(outcomes[name]["pass"])
            segment["completion_reason"] = reasons[name]["reason"]
            segment["task_steps"] = int(reasons[name]["step_count"])

        result = {
            "episode_index": episode_index,
            "seed": args.random_seed,
            "xy_randomization": randomized_trial.as_dict(),
            "completion_reason": "complete",
            "recorded_frames": recorded_frames,
            "recorded_parts": recorded_parts,
            "left_arm_home_q": l_arm_init_q.tolist(),
            "waypoint_annotations_complete": waypoint_annotations_complete,
            "segments": segments,
            **grade,
            "collector": dict(summary),
        }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        pending_tmp = pending_result_path.with_suffix(".tmp")
        pending_tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(pending_tmp, pending_result_path)

        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
        os.replace(pending_result_path, args.results_json)

        summary.update(result)
        summary_path = _build_summary_path(args)
        if summary_path is not None:
            summary_path = summary_path.resolve()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except Exception:
        print("[collect] unhandled exception:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        del record_sub
        try:
            rp.simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
