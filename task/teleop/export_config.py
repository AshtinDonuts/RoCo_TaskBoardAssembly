"""Load and validate teleop / LeRobot data-export settings from JSON."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .schema import ACTION_DIM, IMAGE_KEYS, RECORD_HZ, STATE_DIM

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_CONFIG = REPO_ROOT / "config" / "teleop_export.json"
ALLOWED_CLOCKS = frozenset({"wall", "sim"})
CAMERA_TO_OBS = {
    "head": "head",
    "left_hand": "L_wrist",
    "right_hand": "R_wrist",
}


@dataclass(frozen=True)
class ImageExportConfig:
    height: int = 240
    width: int = 320
    jpeg_quality: int = 90
    cameras: Tuple[str, ...] = IMAGE_KEYS


@dataclass(frozen=True)
class VideoExportConfig:
    vcodec: str = "av1"
    pix_fmt: str = "yuv420p"


@dataclass(frozen=True)
class DatasetExportConfig:
    repo_id: str = "local/roco_aloha_teleop"
    robot_type: str = "vega_1u_gripper"
    task: str = "Industrial Task Board Assembly"
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM


@dataclass(frozen=True)
class ExportSection:
    fps: float = RECORD_HZ
    playback_clock: str = "wall"
    image: ImageExportConfig = field(default_factory=ImageExportConfig)
    video: VideoExportConfig = field(default_factory=VideoExportConfig)
    dataset: DatasetExportConfig = field(default_factory=DatasetExportConfig)


@dataclass(frozen=True)
class SessionExportConfig:
    episode_time_s: float = 600.0
    warmup_time_s: float = 5.0
    num_episodes: int = 1


@dataclass(frozen=True)
class PathsExportConfig:
    output_root: Path = field(default_factory=lambda: REPO_ROOT / "runs")
    teleop_yaml: Path = field(
        default_factory=lambda: REPO_ROOT / "config" / "aloha_solo_to_vega_1u.yaml"
    )


@dataclass(frozen=True)
class TeleopExportConfig:
    schema_version: int
    export: ExportSection
    session: SessionExportConfig
    paths: PathsExportConfig
    source_path: Path

    @property
    def fps(self) -> float:
        return float(self.export.fps)

    @property
    def record_period_s(self) -> float:
        return 1.0 / float(self.export.fps)

    @property
    def img_h(self) -> int:
        return int(self.export.image.height)

    @property
    def img_w(self) -> int:
        return int(self.export.image.width)

    def obs_camera_key(self, camera: str) -> str:
        if camera not in CAMERA_TO_OBS:
            raise KeyError(f"unknown export camera {camera!r}")
        return CAMERA_TO_OBS[camera]

    def recorder_init_fields(self) -> Dict[str, Any]:
        return {
            "fps": float(self.export.fps),
            "repo_id": self.export.dataset.repo_id,
            "robot_type": self.export.dataset.robot_type,
            "task": self.export.dataset.task,
            "state_dim": int(self.export.dataset.state_dim),
            "action_dim": int(self.export.dataset.action_dim),
            "image_height": int(self.export.image.height),
            "image_width": int(self.export.image.width),
            "cameras": list(self.export.image.cameras),
            "jpeg_quality": int(self.export.image.jpeg_quality),
            "vcodec": self.export.video.vcodec,
            "pix_fmt": self.export.video.pix_fmt,
            "playback_clock": self.export.playback_clock,
            "export_config_path": str(self.source_path),
        }


def default_export_config_path() -> Path:
    raw = os.environ.get("ALOHA_EXPORT_CONFIG")
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_EXPORT_CONFIG


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _resolve_path(raw: Any, *, base: Path, default: Path) -> Path:
    if raw in (None, ""):
        path = default
    else:
        path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    else:
        path = path.resolve()
    return path


def _parse_cameras(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return IMAGE_KEYS
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("export.image.cameras must be a list of strings")
    cameras = tuple(str(c) for c in raw)
    if not cameras:
        raise ValueError("export.image.cameras must be non-empty")
    unknown = [c for c in cameras if c not in CAMERA_TO_OBS]
    if unknown:
        raise ValueError(
            f"export.image.cameras unknown entries {unknown}; "
            f"allowed={sorted(CAMERA_TO_OBS)}"
        )
    required = set(IMAGE_KEYS)
    missing = required.difference(cameras)
    if missing:
        raise ValueError(
            f"export.image.cameras must include challenge cameras {sorted(required)}; "
            f"missing={sorted(missing)}"
        )
    return cameras


def validate_export_dict(raw: Mapping[str, Any], *, source_path: Path) -> TeleopExportConfig:
    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version {schema_version}")

    export_raw = _as_mapping(raw.get("export"), "export")
    image_raw = _as_mapping(export_raw.get("image"), "export.image")
    video_raw = _as_mapping(export_raw.get("video"), "export.video")
    dataset_raw = _as_mapping(export_raw.get("dataset"), "export.dataset")
    session_raw = _as_mapping(raw.get("session"), "session")
    paths_raw = _as_mapping(raw.get("paths"), "paths")

    fps = float(export_raw.get("fps", RECORD_HZ))
    if not (fps > 0.0):
        raise ValueError(f"export.fps must be > 0, got {fps}")

    playback_clock = str(export_raw.get("playback_clock", "wall")).strip().lower()
    if playback_clock not in ALLOWED_CLOCKS:
        raise ValueError(
            f"export.playback_clock must be one of {sorted(ALLOWED_CLOCKS)}, "
            f"got {playback_clock!r}"
        )

    height = int(image_raw.get("height", 240))
    width = int(image_raw.get("width", 320))
    if height <= 0 or width <= 0:
        raise ValueError(f"export.image height/width must be > 0, got {height}x{width}")
    jpeg_quality = int(image_raw.get("jpeg_quality", 90))
    if not (1 <= jpeg_quality <= 100):
        raise ValueError(f"export.image.jpeg_quality must be in 1..100, got {jpeg_quality}")
    cameras = _parse_cameras(image_raw.get("cameras"))

    state_dim = int(dataset_raw.get("state_dim", STATE_DIM))
    action_dim = int(dataset_raw.get("action_dim", ACTION_DIM))
    if state_dim != STATE_DIM:
        raise ValueError(f"export.dataset.state_dim must be {STATE_DIM}, got {state_dim}")
    if action_dim != ACTION_DIM:
        raise ValueError(f"export.dataset.action_dim must be {ACTION_DIM}, got {action_dim}")

    episode_time_s = float(session_raw.get("episode_time_s", 600.0))
    warmup_time_s = float(session_raw.get("warmup_time_s", 5.0))
    num_episodes = int(session_raw.get("num_episodes", 1))
    if episode_time_s <= 0.0:
        raise ValueError("session.episode_time_s must be > 0")
    if warmup_time_s < 0.0:
        raise ValueError("session.warmup_time_s must be >= 0")
    if num_episodes <= 0:
        raise ValueError("session.num_episodes must be > 0")

    # Relative paths in the default config are repo-root relative.
    output_root = _resolve_path(
        paths_raw.get("output_root"),
        base=REPO_ROOT,
        default=REPO_ROOT / "runs",
    )
    teleop_yaml = _resolve_path(
        paths_raw.get("teleop_yaml"),
        base=REPO_ROOT,
        default=REPO_ROOT / "config" / "aloha_solo_to_vega_1u.yaml",
    )

    return TeleopExportConfig(
        schema_version=schema_version,
        export=ExportSection(
            fps=fps,
            playback_clock=playback_clock,
            image=ImageExportConfig(
                height=height,
                width=width,
                jpeg_quality=jpeg_quality,
                cameras=cameras,
            ),
            video=VideoExportConfig(
                vcodec=str(video_raw.get("vcodec", "av1")),
                pix_fmt=str(video_raw.get("pix_fmt", "yuv420p")),
            ),
            dataset=DatasetExportConfig(
                repo_id=str(dataset_raw.get("repo_id", "local/roco_aloha_teleop")),
                robot_type=str(dataset_raw.get("robot_type", "vega_1u_gripper")),
                task=str(dataset_raw.get("task", "Industrial Task Board Assembly")),
                state_dim=state_dim,
                action_dim=action_dim,
            ),
        ),
        session=SessionExportConfig(
            episode_time_s=episode_time_s,
            warmup_time_s=warmup_time_s,
            num_episodes=num_episodes,
        ),
        paths=PathsExportConfig(output_root=output_root, teleop_yaml=teleop_yaml),
        source_path=source_path.resolve(),
    )


def load_export_config(path: Optional[os.PathLike | str] = None) -> TeleopExportConfig:
    cfg_path = Path(path).expanduser() if path else default_export_config_path()
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"teleop export config not found: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("export config root must be a JSON object")
    return validate_export_dict(raw, source_path=cfg_path)
