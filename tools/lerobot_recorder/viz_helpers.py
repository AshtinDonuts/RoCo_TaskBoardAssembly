"""Pure helpers for Aloha teleop dataset visualization (no LeRobot import)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Mirrors task/teleop/schema.py and writer._feature_spec (kept local so the
# lerobot conda env does not need the Isaac/teleop package tree).
STATE_DIM = 44
ACTION_DIM = 14
STATE_NAMES: tuple[str, ...] = tuple(
    [f"left_ee_{n}" for n in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"right_ee_{n}" for n in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"left_joint_pos_{i}" for i in range(7)]
    + [f"right_joint_pos_{i}" for i in range(7)]
    + [f"left_joint_vel_{i}" for i in range(7)]
    + [f"right_joint_vel_{i}" for i in range(7)]
    + ["left_gripper_ratio", "right_gripper_ratio"]
)
ACTION_NAMES: tuple[str, ...] = tuple(
    [f"left_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
    + [f"right_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
)
IMAGE_KEYS: tuple[str, ...] = ("head", "left_hand", "right_hand")
CAMERA_FEATURE_KEYS: tuple[str, ...] = tuple(f"observation.images.{k}" for k in IMAGE_KEYS)

DEFAULT_DATASET_FOLDER = "local_roco_aloha_teleop"
DEFAULT_REPO_ID = "local/roco_aloha_teleop"


@dataclass(frozen=True)
class ScalarGroup:
    """Named contiguous (or index) slice of a vector feature for Rerun panels."""

    entity: str
    indices: tuple[int, ...]
    names: tuple[str, ...]


# RoCo-friendly Rerun panels (avoids one 44-line plot).
STATE_GROUPS: tuple[ScalarGroup, ...] = (
    ScalarGroup("state/left_ee_xyz", (0, 1, 2), ("left_ee_x", "left_ee_y", "left_ee_z")),
    ScalarGroup(
        "state/left_ee_quat",
        (3, 4, 5, 6),
        ("left_ee_qw", "left_ee_qx", "left_ee_qy", "left_ee_qz"),
    ),
    ScalarGroup("state/right_ee_xyz", (7, 8, 9), ("right_ee_x", "right_ee_y", "right_ee_z")),
    ScalarGroup(
        "state/right_ee_quat",
        (10, 11, 12, 13),
        ("right_ee_qw", "right_ee_qx", "right_ee_qy", "right_ee_qz"),
    ),
    ScalarGroup(
        "state/left_joint_pos",
        tuple(range(14, 21)),
        tuple(f"left_joint_pos_{i}" for i in range(7)),
    ),
    ScalarGroup(
        "state/right_joint_pos",
        tuple(range(21, 28)),
        tuple(f"right_joint_pos_{i}" for i in range(7)),
    ),
    ScalarGroup(
        "state/left_joint_vel",
        tuple(range(28, 35)),
        tuple(f"left_joint_vel_{i}" for i in range(7)),
    ),
    ScalarGroup(
        "state/right_joint_vel",
        tuple(range(35, 42)),
        tuple(f"right_joint_vel_{i}" for i in range(7)),
    ),
    ScalarGroup(
        "state/grippers",
        (42, 43),
        ("left_gripper_ratio", "right_gripper_ratio"),
    ),
)

ACTION_GROUPS: tuple[ScalarGroup, ...] = (
    ScalarGroup(
        "action/left",
        tuple(range(0, 7)),
        tuple(f"left_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")),
    ),
    ScalarGroup(
        "action/right",
        tuple(range(7, 14)),
        tuple(f"right_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")),
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def datasets_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / "runs" / "datasets"


def resolve_dataset_root(
    dataset: Path | None = None,
    *,
    run_id: str | None = None,
    latest: bool = False,
    dataset_folder: str = DEFAULT_DATASET_FOLDER,
    runs_root: Path | None = None,
) -> Path:
    """Resolve a local LeRobot v3 dataset directory under runs/datasets/."""
    if sum(bool(x) for x in (dataset is not None, run_id is not None, latest)) != 1:
        raise ValueError("Provide exactly one of: dataset path, --run-id, or --latest")

    base = runs_root if runs_root is not None else datasets_root()

    if dataset is not None:
        path = dataset.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"dataset path does not exist: {path}")
        return path

    folder = base / dataset_folder
    if not folder.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {folder}")

    if run_id is not None:
        path = folder / run_id
        if not path.is_dir():
            raise FileNotFoundError(f"run not found: {path}")
        return path

    candidates = sorted(
        (p for p in folder.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no runs under {folder}")
    return candidates[0]


def load_repo_id(root: Path, default: str = DEFAULT_REPO_ID) -> str:
    meta_path = root / "run_meta.json"
    if not meta_path.is_file():
        return default
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return str(meta.get("repo_id") or default)


def load_info_features(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing meta/info.json under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"meta/info.json missing features dict: {info_path}")
    return features


def _as_shape_tuple(shape: Any) -> tuple[int, ...]:
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"feature shape must be list/tuple, got {shape!r}")
    return tuple(int(x) for x in shape)


def _as_names(names: Any) -> list[str] | None:
    if names is None:
        return None
    if not isinstance(names, list):
        raise ValueError(f"feature names must be a list, got {type(names).__name__}")
    return [str(n) for n in names]


def validate_teleop_contract(features: dict[str, Any]) -> None:
    """Raise ValueError if features do not match the Aloha teleop export contract."""
    state = features.get("observation.state")
    action = features.get("action")
    if not isinstance(state, dict) or not isinstance(action, dict):
        raise ValueError("features must include observation.state and action")

    state_shape = _as_shape_tuple(state.get("shape"))
    action_shape = _as_shape_tuple(action.get("shape"))
    if state_shape != (STATE_DIM,):
        raise ValueError(f"observation.state shape must be ({STATE_DIM},), got {state_shape}")
    if action_shape != (ACTION_DIM,):
        raise ValueError(f"action shape must be ({ACTION_DIM},), got {action_shape}")

    state_names = _as_names(state.get("names"))
    action_names = _as_names(action.get("names"))
    if state_names != list(STATE_NAMES):
        raise ValueError("observation.state names do not match teleop STATE_NAMES contract")
    if action_names != list(ACTION_NAMES):
        raise ValueError("action names do not match teleop ACTION_NAMES contract")

    for key in CAMERA_FEATURE_KEYS:
        cam = features.get(key)
        if not isinstance(cam, dict):
            raise ValueError(f"missing camera feature {key}")
        dtype = cam.get("dtype")
        if dtype not in ("video", "image"):
            raise ValueError(f"{key} dtype must be video or image, got {dtype!r}")


def assert_dataset_ready(root: Path) -> None:
    """Fail fast on quarantine / incomplete trees."""
    if "quarantine" in root.parts:
        raise ValueError(f"refusing to visualize quarantine path: {root}")
    info = root / "meta" / "info.json"
    data = root / "data"
    if not info.is_file():
        raise FileNotFoundError(f"incomplete dataset (no meta/info.json): {root}")
    if not data.is_dir():
        raise FileNotFoundError(f"incomplete dataset (no data/): {root}")


def slice_vector(values: Sequence[float], indices: Iterable[int]) -> list[float]:
    return [float(values[i]) for i in indices]


def episode_summary_lines(root: Path, episode_index: int) -> list[str]:
    """Return short text lines from episodes.jsonl for the given episode (if any)."""
    path = root / "episodes.jsonl"
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if int(row.get("episode_index", -1)) != episode_index:
            continue
        task = row.get("task") or {}
        events = task.get("events") or []
        event_bits = [
            f"{e.get('event')}={e.get('name')}" for e in events if isinstance(e, dict)
        ]
        return [
            f"disposition={row.get('disposition')} reason={row.get('reason')}",
            f"frames={row.get('frames')} duration_s={row.get('duration_s')}",
            f"current_part={task.get('current_part')} aborted={task.get('aborted')}",
            ("events: " + ", ".join(event_bits)) if event_bits else "events: (none)",
        ]
    return []
