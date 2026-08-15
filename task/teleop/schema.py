"""Pinned LeRobot v3 observation/action contract for RoCo Task Board teleop."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from . import transforms as T

IMG_H = 240
IMG_W = 320
STATE_DIM = 44
ACTION_DIM = 14
RECORD_HZ = 10.0
GRIPPER_OPEN_LIMIT = 0.6649704

STATE_NAMES = (
    [f"left_ee_{n}" for n in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"right_ee_{n}" for n in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"left_joint_pos_{i}" for i in range(7)]
    + [f"right_joint_pos_{i}" for i in range(7)]
    + [f"left_joint_vel_{i}" for i in range(7)]
    + [f"right_joint_vel_{i}" for i in range(7)]
    + ["left_gripper_ratio", "right_gripper_ratio"]
)

ACTION_NAMES = (
    [f"left_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
    + [f"right_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
)

IMAGE_KEYS = ("head", "left_hand", "right_hand")


def gripper_ratio(joint_value: float, open_limit: float = GRIPPER_OPEN_LIMIT) -> float:
    if open_limit <= 0.0:
        return 0.0
    return float(np.clip(float(joint_value) / open_limit, 0.0, 1.0))


def pack_state(
    left_ee_pos: Sequence[float],
    left_ee_quat: Sequence[float],
    right_ee_pos: Sequence[float],
    right_ee_quat: Sequence[float],
    left_q: Sequence[float],
    right_q: Sequence[float],
    left_qd: Sequence[float],
    right_qd: Sequence[float],
    left_grip_ratio: float,
    right_grip_ratio: float,
) -> np.ndarray:
    return np.concatenate(
        [
            T.as_vec(left_ee_pos, 3),
            T.normalize_quat_wxyz(left_ee_quat),
            T.as_vec(right_ee_pos, 3),
            T.normalize_quat_wxyz(right_ee_quat),
            T.as_vec(left_q, 7),
            T.as_vec(right_q, 7),
            T.as_vec(left_qd, 7),
            T.as_vec(right_qd, 7),
            [float(np.clip(left_grip_ratio, 0.0, 1.0))],
            [float(np.clip(right_grip_ratio, 0.0, 1.0))],
        ]
    ).astype(np.float32)


def pack_action(
    left_pos: Sequence[float],
    left_quat: Sequence[float],
    left_grip_ratio: float,
    right_pos: Sequence[float],
    right_quat: Sequence[float],
    right_grip_ratio: float,
) -> np.ndarray:
    return np.concatenate(
        [
            T.as_vec(left_pos, 3),
            T.quat_wxyz_to_rotvec(left_quat),
            [float(np.clip(left_grip_ratio, 0.0, 1.0))],
            T.as_vec(right_pos, 3),
            T.quat_wxyz_to_rotvec(right_quat),
            [float(np.clip(right_grip_ratio, 0.0, 1.0))],
        ]
    ).astype(np.float32)


def resize_rgb(img: Optional[np.ndarray], height: int = IMG_H, width: int = IMG_W) -> np.ndarray:
    if img is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = arr[..., :3]
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr.astype(np.uint8, copy=False)
    try:
        import cv2

        return cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except Exception:
        ys = max(1, arr.shape[0] // height)
        xs = max(1, arr.shape[1] // width)
        out = arr[::ys, ::xs, :3]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        h = min(height, out.shape[0])
        w = min(width, out.shape[1])
        canvas[:h, :w] = out[:h, :w]
        return canvas


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    arr = np.asarray(img, dtype=np.uint8)
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", arr[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    from io import BytesIO

    from PIL import Image

    bio = BytesIO()
    Image.fromarray(arr).save(bio, format="JPEG", quality=int(quality))
    return bio.getvalue()


def decode_jpeg(payload: bytes) -> np.ndarray:
    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            return arr[:, :, ::-1].astype(np.uint8)
    except Exception:
        pass
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(payload)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def validate_frame(
    *,
    step_idx: int,
    timestamp_s: float,
    state: Sequence[float],
    action: Sequence[float],
    images: Dict[str, np.ndarray],
    prev_timestamp_s: Optional[float] = None,
    prev_seq: Optional[int] = None,
    seq: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if state_arr.size != STATE_DIM:
        raise ValueError(f"state must be {STATE_DIM}-D, got {state_arr.size}")
    if action_arr.size != ACTION_DIM:
        raise ValueError(f"action must be {ACTION_DIM}-D, got {action_arr.size}")
    if not np.all(np.isfinite(state_arr)):
        raise ValueError("state contains non-finite values")
    if not np.all(np.isfinite(action_arr)):
        raise ValueError("action contains non-finite values")
    if not (0.0 <= float(action_arr[6]) <= 1.0 and 0.0 <= float(action_arr[13]) <= 1.0):
        raise ValueError("action gripper ratios must be in [0, 1]")
    if prev_timestamp_s is not None and timestamp_s + 1e-6 < prev_timestamp_s:
        raise ValueError(f"timestamp went backwards: {prev_timestamp_s} -> {timestamp_s}")
    if prev_seq is not None and seq is not None and int(seq) < int(prev_seq):
        raise ValueError(f"sequence went backwards: {prev_seq} -> {seq}")
    checked = {}
    for key in IMAGE_KEYS:
        if key not in images:
            raise ValueError(f"missing image {key}")
        frame = np.asarray(images[key])
        if frame.shape != (IMG_H, IMG_W, 3):
            raise ValueError(f"{key} image shape {frame.shape} != {(IMG_H, IMG_W, 3)}")
        if frame.dtype != np.uint8:
            raise ValueError(f"{key} image dtype {frame.dtype} != uint8")
        checked[key] = frame
    if int(step_idx) < 0:
        raise ValueError("step_idx must be >= 0")
    return state_arr, action_arr, checked


def feature_spec() -> Dict[str, Any]:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(ACTION_NAMES),
        },
        "observation.images.head": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.left_hand": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_hand": {
            "dtype": "video",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "part_name": {"dtype": "string", "shape": (1,)},
        "release_mode": {"dtype": "string", "shape": (1,)},
        "sim_step": {"dtype": "int64", "shape": (1,)},
    }
