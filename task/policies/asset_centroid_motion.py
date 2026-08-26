"""Pure geometry, config loading, and time-parameterisation for asset-centroid."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


TOP_DOWN_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "asset_centroid_policy.json"


@dataclass(frozen=True)
class MotionLimits:
    max_linear_speed_m_s: float
    max_angular_speed_rad_s: float
    minimum_move_s: float


@dataclass(frozen=True)
class PathClearances:
    hover_pick_m: float
    hover_place_m: float
    final_retract_m: float
    yaw_step_deg: float
    # None = search with yaw_step_deg; otherwise lock both pick/place to this yaw.
    force_yaw_deg: float | None
    # "world": baseline-style EE = grasp + offset (no yaw rotation of lateral).
    # "tool": EE = grasp - R(orn)·offset (lateral swings with yaw).
    tcp_offset_frame: str


@dataclass(frozen=True)
class TimingSpec:
    close_dwell_s: float
    settle_place_s: float
    open_dwell_s: float


@dataclass(frozen=True)
class GuardSpec:
    pos_tol_m: float
    orn_tol_rad: float
    max_final_hold_steps: int
    max_ik_failure_steps: int


@dataclass(frozen=True)
class AssetMotionSpec:
    """Hand-tuned grasp data; deliberately independent of AABB width config."""

    centroid_grasp_offset_asset: tuple[float, float, float]
    tcp_to_grasp_tool: tuple[float, float, float]
    gripper_open_rad: float
    gripper_close_rad: float


@dataclass(frozen=True)
class GripperSpec:
    """How close/open commands are issued to EEPoseController.

    - ``compliant``: approach/release use ``gripper_open_rad``; close is
      string ``"close"`` → GripperCompliance slow-close toward 0 with stall
      (soft PhysX drives yield). Matches baseline approach aperture + soft
      close endpoint.
    - ``aperture``: numeric ``gripper_*_rad`` from the part block.
    """

    mode: str  # "compliant" | "aperture"


@dataclass(frozen=True)
class AssetCentroidConfig:
    path: Path
    active_arm: str
    motion: MotionLimits
    path_clearances: PathClearances
    timing: TimingSpec
    guards: GuardSpec
    gripper: GripperSpec
    parts: dict[str, AssetMotionSpec]

    def part(self, name: str) -> AssetMotionSpec:
        try:
            return self.parts[name]
        except KeyError as exc:
            raise KeyError(
                f"no asset-centroid part config for {name!r} in {self.path}"
            ) from exc


def default_config_path() -> Path:
    override = os.environ.get("ROCO_ASSET_CENTROID_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_CONFIG_PATH


def _as_xyz(value, *, field: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{field} must be a length-3 list")
    return float(arr[0]), float(arr[1]), float(arr[2])


def load_asset_centroid_config(path: Path | str | None = None) -> AssetCentroidConfig:
    cfg_path = Path(path).expanduser().resolve() if path else default_config_path()
    with open(cfg_path, encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"asset-centroid config must be an object: {cfg_path}")

    motion_raw = raw.get("motion") or {}
    path_raw = raw.get("path") or {}
    timing_raw = raw.get("timing") or {}
    guards_raw = raw.get("guards") or {}
    gripper_raw = raw.get("gripper") or {}
    parts_raw = raw.get("parts") or {}

    motion = MotionLimits(
        max_linear_speed_m_s=float(motion_raw.get("max_linear_speed_m_s", 0.05)),
        max_angular_speed_rad_s=np.deg2rad(
            float(motion_raw.get("max_angular_speed_deg_s", 20.0))
        ),
        minimum_move_s=float(motion_raw.get("minimum_move_s", 0.5)),
    )
    path_clearances = PathClearances(
        hover_pick_m=float(path_raw.get("hover_pick_m", 0.05)),
        hover_place_m=float(path_raw.get("hover_place_m", 0.05)),
        final_retract_m=float(path_raw.get("final_retract_m", 0.10)),
        yaw_step_deg=float(path_raw.get("yaw_step_deg", 15.0)),
        force_yaw_deg=(
            None
            if path_raw.get("force_yaw_deg", None) is None
            else float(path_raw["force_yaw_deg"])
        ),
        tcp_offset_frame=str(path_raw.get("tcp_offset_frame", "tool")).lower(),
    )
    if path_clearances.tcp_offset_frame not in ("tool", "world"):
        raise ValueError(
            "path.tcp_offset_frame must be 'tool' or 'world', got "
            f"{path_clearances.tcp_offset_frame!r}"
        )
    timing = TimingSpec(
        close_dwell_s=float(timing_raw.get("close_dwell_s", 1.5)),
        settle_place_s=float(timing_raw.get("settle_place_s", 0.5)),
        open_dwell_s=float(timing_raw.get("open_dwell_s", 0.5)),
    )
    guards = GuardSpec(
        pos_tol_m=float(guards_raw.get("pos_tol_m", 0.004)),
        orn_tol_rad=float(guards_raw.get("orn_tol_rad", 0.05)),
        max_final_hold_steps=int(guards_raw.get("max_final_hold_steps", 500)),
        max_ik_failure_steps=int(guards_raw.get("max_ik_failure_steps", 100)),
    )
    gripper_mode = str(gripper_raw.get("mode", "compliant")).lower()
    if gripper_mode not in ("compliant", "aperture"):
        raise ValueError(
            "gripper.mode must be 'compliant' or 'aperture', got "
            f"{gripper_mode!r}"
        )
    gripper = GripperSpec(mode=gripper_mode)
    parts: dict[str, AssetMotionSpec] = {}
    for name, entry in parts_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"parts.{name} must be an object")
        parts[str(name)] = AssetMotionSpec(
            centroid_grasp_offset_asset=_as_xyz(
                entry.get("centroid_grasp_offset_asset_m", [0.0, 0.0, 0.0]),
                field=f"parts.{name}.centroid_grasp_offset_asset_m",
            ),
            tcp_to_grasp_tool=_as_xyz(
                entry.get("tcp_to_grasp_tool_m", [0.0, 0.0, 0.0]),
                field=f"parts.{name}.tcp_to_grasp_tool_m",
            ),
            gripper_open_rad=float(entry["gripper_open_rad"]),
            gripper_close_rad=float(entry["gripper_close_rad"]),
        )

    active_arm = str(raw.get("active_arm", "R")).upper()
    if active_arm not in ("L", "R"):
        raise ValueError(f"active_arm must be 'L' or 'R', got {active_arm!r}")

    return AssetCentroidConfig(
        path=cfg_path,
        active_arm=active_arm,
        motion=motion,
        path_clearances=path_clearances,
        timing=timing,
        guards=guards,
        gripper=gripper,
        parts=parts,
    )


@lru_cache(maxsize=4)
def cached_asset_centroid_config(path: str | None = None) -> AssetCentroidConfig:
    return load_asset_centroid_config(path)


def clear_asset_centroid_config_cache() -> None:
    cached_asset_centroid_config.cache_clear()


# Back-compat alias used by unit tests / older imports.
GEAR_20TEETH_SPEC = cached_asset_centroid_config().part("gear_20teeth")


def normalize_quat_wxyz(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    q = q / norm
    return -q if q[0] < 0.0 else q


def quat_mul_wxyz(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quat_wxyz(q1)
    w2, x2, y2, z2 = normalize_quat_wxyz(q2)
    return normalize_quat_wxyz(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_angle(q1: Sequence[float], q2: Sequence[float]) -> float:
    a = normalize_quat_wxyz(q1)
    b = normalize_quat_wxyz(q2)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def rotate_vector(quat: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(quat)
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rot @ np.asarray(vector, dtype=np.float64).reshape(3)


def top_down_yaw_quat(yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(yaw_deg))
    yaw = np.array(
        [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)],
        dtype=np.float64,
    )
    return quat_mul_wxyz(yaw, TOP_DOWN_WXYZ)


def top_down_yaw_candidates(step_degrees: float = 15.0) -> tuple[np.ndarray, ...]:
    """Top-down quaternions ordered 0, +step, -step, ... around world Z."""
    if step_degrees <= 0.0 or 360.0 % step_degrees > 1e-9:
        raise ValueError("step_degrees must be a positive divisor of 360")
    n = int(round(180.0 / step_degrees))
    angles_deg = [0.0]
    for i in range(1, n + 1):
        angles_deg.append(i * step_degrees)
        if i < n:
            angles_deg.append(-i * step_degrees)
    return tuple(top_down_yaw_quat(a) for a in angles_deg)


def local_aabb_midpoint(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("mesh points must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(pts)):
        raise ValueError("mesh points must be finite")
    return 0.5 * (np.min(pts, axis=0) + np.max(pts, axis=0))


def transform_point(
    point: Sequence[float],
    translation: Sequence[float],
    orientation_wxyz: Sequence[float],
) -> np.ndarray:
    return np.asarray(translation, dtype=np.float64).reshape(3) + rotate_vector(
        orientation_wxyz, point
    )


def ee_position_for_grasp_center(
    grasp_center_world: Sequence[float],
    ee_orientation_wxyz: Sequence[float],
    tcp_to_grasp_tool: Sequence[float],
    *,
    offset_frame: str = "tool",
) -> np.ndarray:
    """EE origin that places the grasp center at a world point.

    offset_frame:
      - ``tool``: EE = grasp − R(orn)·tcp  (lateral swings with yaw)
      - ``world``: EE = grasp + tcp  (baseline-style world ``ee_offset``)
    """
    grasp = np.asarray(grasp_center_world, dtype=np.float64).reshape(3)
    tcp = np.asarray(tcp_to_grasp_tool, dtype=np.float64).reshape(3)
    frame = str(offset_frame).lower()
    if frame == "world":
        return grasp + tcp
    if frame == "tool":
        return grasp - rotate_vector(ee_orientation_wxyz, tcp)
    raise ValueError(f"offset_frame must be 'tool' or 'world', got {offset_frame!r}")


def quintic_smoothstep(u: float | np.ndarray) -> float | np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def motion_duration(
    linear_distance_m: float,
    angular_distance_rad: float,
    *,
    max_linear_speed_m_s: float = 0.05,
    max_angular_speed_rad_s: float = np.deg2rad(20.0),
    minimum_s: float = 0.5,
) -> float:
    """Duration whose quintic peak speed respects both configured caps."""
    if max_linear_speed_m_s <= 0.0 or max_angular_speed_rad_s <= 0.0:
        raise ValueError("speed limits must be positive")
    # max d/du of 10u^3-15u^4+6u^5 is 1.875 at u=0.5.
    return float(
        max(
            minimum_s,
            1.875 * max(0.0, float(linear_distance_m)) / max_linear_speed_m_s,
            1.875 * max(0.0, float(angular_distance_rad)) / max_angular_speed_rad_s,
        )
    )


def duration_for_limits(
    linear_distance_m: float,
    angular_distance_rad: float,
    limits: MotionLimits,
) -> float:
    return motion_duration(
        linear_distance_m,
        angular_distance_rad,
        max_linear_speed_m_s=limits.max_linear_speed_m_s,
        max_angular_speed_rad_s=limits.max_angular_speed_rad_s,
        minimum_s=limits.minimum_move_s,
    )


def slerp_wxyz(q0: Sequence[float], q1: Sequence[float], u: float) -> np.ndarray:
    a = normalize_quat_wxyz(q0)
    b = normalize_quat_wxyz(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quat_wxyz(a + float(u) * (b - a))
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    return normalize_quat_wxyz(
        np.sin((1.0 - float(u)) * theta) / sin_theta * a
        + np.sin(float(u) * theta) / sin_theta * b
    )


def sample_pose_segment(
    start_pos: Sequence[float],
    start_orn: Sequence[float],
    end_pos: Sequence[float],
    end_orn: Sequence[float],
    *,
    dt: float,
    limits: MotionLimits | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return deterministic samples after the start pose through the endpoint."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    p0 = np.asarray(start_pos, dtype=np.float64).reshape(3)
    p1 = np.asarray(end_pos, dtype=np.float64).reshape(3)
    q0 = normalize_quat_wxyz(start_orn)
    q1 = normalize_quat_wxyz(end_orn)
    if limits is None:
        duration = motion_duration(float(np.linalg.norm(p1 - p0)), quat_angle(q0, q1))
    else:
        duration = duration_for_limits(
            float(np.linalg.norm(p1 - p0)), quat_angle(q0, q1), limits
        )
    count = max(1, int(np.ceil(duration / dt)))
    samples = []
    for i in range(1, count + 1):
        s = float(quintic_smoothstep(i / count))
        samples.append(((1.0 - s) * p0 + s * p1, slerp_wxyz(q0, q1, s)))
    return tuple(samples)


def unwrap_revolute_delta(
    start_q: Sequence[float],
    end_q: Sequence[float],
    *,
    prismatic_mask: Sequence[bool] | None = None,
) -> np.ndarray:
    """Shortest-path joint delta; leave prismatic joints unwrapped."""
    q0 = np.asarray(start_q, dtype=np.float64).reshape(-1)
    q1 = np.asarray(end_q, dtype=np.float64).reshape(-1)
    if q0.shape != q1.shape:
        raise ValueError("start_q and end_q must have the same shape")
    delta = q1 - q0
    if prismatic_mask is None:
        mask = np.zeros(q0.shape[0], dtype=bool)
    else:
        mask = np.asarray(prismatic_mask, dtype=bool).reshape(-1)
        if mask.shape != q0.shape:
            raise ValueError("prismatic_mask must match joint vector shape")
    revolute = ~mask
    delta[revolute] = np.arctan2(np.sin(delta[revolute]), np.cos(delta[revolute]))
    return delta


def sample_joint_segment(
    start_q: Sequence[float],
    end_q: Sequence[float],
    *,
    dt: float,
    linear_distance_m: float,
    angular_distance_rad: float,
    prismatic_mask: Sequence[bool] | None = None,
    limits: MotionLimits | None = None,
) -> tuple[np.ndarray, ...]:
    """Quintic joint-space samples from start (exclusive) through end."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    q0 = np.asarray(start_q, dtype=np.float64).reshape(-1)
    delta = unwrap_revolute_delta(q0, end_q, prismatic_mask=prismatic_mask)
    if limits is None:
        duration = motion_duration(linear_distance_m, angular_distance_rad)
    else:
        duration = duration_for_limits(linear_distance_m, angular_distance_rad, limits)
    count = max(1, int(np.ceil(duration / dt)))
    samples = []
    for i in range(1, count + 1):
        s = float(quintic_smoothstep(i / count))
        samples.append(q0 + s * delta)
    return tuple(samples)
