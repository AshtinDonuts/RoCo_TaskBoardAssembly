#!/usr/bin/env python3
"""LeRobot v3 episode writer used by the recorder sidecar.

Task success never decides whether an episode is committed. Quarantine is
reserved for sessions that saved zero episodes (interrupt before any save,
or a corrupt writer).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

IMG_H, IMG_W = 240, 320
STATE_DIM, ACTION_DIM = 44, 14
IMAGE_KEYS = ("head", "left_hand", "right_hand")


def _decode_jpeg(payload: bytes):
    import numpy as np

    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            return arr[:, :, ::-1]
    except Exception:
        pass
    from io import BytesIO

    from PIL import Image

    return np.asarray(Image.open(BytesIO(payload)).convert("RGB"))


def _validate_frame(msg: Dict[str, Any], prev_ts, prev_seq):
    import numpy as np

    state = np.asarray(msg["state"], dtype=np.float32).reshape(-1)
    action = np.asarray(msg["action"], dtype=np.float32).reshape(-1)
    if state.size != STATE_DIM:
        raise ValueError(f"state dim {state.size}")
    if action.size != ACTION_DIM:
        raise ValueError(f"action dim {action.size}")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError("non-finite state/action")
    if not (0.0 <= float(action[6]) <= 1.0 and 0.0 <= float(action[13]) <= 1.0):
        raise ValueError("gripper ratio out of range")
    ts = float(msg["timestamp_s"])
    seq = int(msg["seq"])
    if prev_ts is not None and ts + 1e-6 < prev_ts:
        raise ValueError("timestamp went backwards")
    if prev_seq is not None and seq < prev_seq:
        raise ValueError("seq went backwards")
    images = {}
    for key in IMAGE_KEYS:
        raw = msg["images"][key]
        if isinstance(raw, (bytes, bytearray)):
            frame = _decode_jpeg(bytes(raw))
        else:
            frame = np.asarray(raw, dtype=np.uint8)
        if frame.shape != (IMG_H, IMG_W, 3):
            raise ValueError(f"{key} shape {frame.shape}")
        images[key] = frame
    return state, action, images, ts, seq


def _feature_spec():
    state_names = (
        [f"left_ee_{n}" for n in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        + [f"right_ee_{n}" for n in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        + [f"left_joint_pos_{i}" for i in range(7)]
        + [f"right_joint_pos_{i}" for i in range(7)]
        + [f"left_joint_vel_{i}" for i in range(7)]
        + [f"right_joint_vel_{i}" for i in range(7)]
        + ["left_gripper_ratio", "right_gripper_ratio"]
    )
    action_names = (
        [f"left_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
        + [f"right_{n}" for n in ("x", "y", "z", "rx", "ry", "rz", "gripper")]
    )
    return {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": state_names},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": action_names},
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
    }


def _create_dataset(root: Path, repo_id: str, fps: float, robot_type: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    kwargs = dict(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=_feature_spec(),
        use_videos=True,
        image_writer_threads=4,
    )
    try:
        return LeRobotDataset.create(root=root, **kwargs)
    except TypeError:
        os.environ["HF_LEROBOT_HOME"] = str(root.parent)
        return LeRobotDataset.create(**kwargs)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


class EpisodeWriter:
    def __init__(self) -> None:
        self.dataset = None
        self.staging = None
        self.final_root = None
        self.quarantine = None
        self.meta: Dict[str, Any] = {}
        self.prev_ts = None
        self.prev_seq = None
        self.n_ok = 0
        self.n_drop = 0
        self.n_ok_total = 0
        self.n_saved = 0
        self.n_discarded = 0
        self.episode_active = False
        self.task = "Industrial Task Board Assembly"
        self._log_path: Optional[Path] = None
        self._finalized = False

    def init(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        root = Path(msg.get("root") or "runs").expanduser().resolve()
        run_id = msg.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
        repo_id = msg.get("repo_id") or "local/roco_aloha_teleop"
        self.staging = root / "staging" / run_id
        self.final_root = root / "datasets" / repo_id.replace("/", "_") / run_id
        self.quarantine = root / "quarantine" / run_id
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.dataset = _create_dataset(
                self.staging,
                repo_id=repo_id,
                fps=int(msg.get("fps", 10)),
                robot_type=msg.get("robot_type") or "vega_1u_gripper",
            )
        except FileExistsError:
            shutil.rmtree(self.staging, ignore_errors=True)
            self.dataset = _create_dataset(
                self.staging,
                repo_id=repo_id,
                fps=int(msg.get("fps", 10)),
                robot_type=msg.get("robot_type") or "vega_1u_gripper",
            )
        self.task = msg.get("task") or self.task
        self.meta = {
            "run_id": run_id,
            "repo_id": repo_id,
            "challenge_commit": msg.get("challenge_commit"),
            "config_path": msg.get("config_path"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "schema": {
                "observation.state": STATE_DIM,
                "action": ACTION_DIM,
                "images": f"{IMG_H}x{IMG_W}x3",
                "rotation": "rotvec",
            },
        }
        (self.staging / "run_meta.json").write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )
        self._log_path = self.staging / "episodes.jsonl"
        sys.stderr.write(f"[recorder] staging {self.staging}\n")
        sys.stderr.flush()
        return {"ok": True, "staging": str(self.staging)}

    def begin(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("init was not called")
        if self.episode_active and self.n_ok > 0:
            self.discard({"reason": "implicit_begin"})
        self.prev_ts = None
        self.prev_seq = None
        self.n_ok = 0
        self.n_drop = 0
        self.episode_active = True
        sys.stderr.write(
            f"[recorder] begin_episode index={msg.get('episode_index')} "
            f"attempt={msg.get('attempt_index')}\n"
        )
        sys.stderr.flush()
        return {
            "ok": True,
            "episode_index": int(msg.get("episode_index") or self.n_saved),
            "attempt_index": msg.get("attempt_index"),
        }

    def add_frame(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("init was not called")
        if not self.episode_active:
            return {"ok": False, "error": "no active episode"}
        try:
            state, action, images, ts, seq = _validate_frame(msg, self.prev_ts, self.prev_seq)
        except Exception as exc:
            self.n_drop += 1
            return {"ok": False, "error": str(exc)}
        frame = {
            "observation.state": state,
            "action": action,
            "observation.images.head": images["head"],
            "observation.images.left_hand": images["left_hand"],
            "observation.images.right_hand": images["right_hand"],
            "task": self.task,
        }
        self.dataset.add_frame(frame)
        self.prev_ts = ts
        self.prev_seq = seq
        self.n_ok += 1
        self.n_ok_total += 1
        return {"ok": True, "n": self.n_ok}

    def save(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset is None:
            return {"ok": False, "error": "no dataset"}
        if not self.episode_active:
            return {"ok": False, "error": "no active episode"}
        frames = self.n_ok
        stats = dict(msg.get("stats") or {})
        stats.update({
            "frames_ok": self.n_ok,
            "frames_drop": self.n_drop,
            "frames_ok_total": self.n_ok_total,
        })
        record = {
            "episode_index": self.n_saved if frames > 0 else None,
            "attempt_index": msg.get("attempt_index"),
            "disposition": "saved" if frames > 0 else "empty_skipped",
            "reason": msg.get("reason"),
            "frames": frames,
            "drops": self.n_drop,
            "duration_s": msg.get("duration_s"),
            "task": msg.get("task_log") or {},
            "stats": stats,
        }
        results_json = msg.get("results_json")
        if results_json and Path(results_json).exists() and self.staging is not None:
            dest = self.staging / f"results_attempt_{msg.get('attempt_index', self.n_saved)}.json"
            shutil.copy2(results_json, dest)
            record["results_json"] = dest.name
        if frames > 0:
            self.dataset.save_episode()
            self.n_saved += 1
            record["episode_index"] = self.n_saved - 1
        else:
            self._clear_buffer()
        self._append_log(record)
        self.episode_active = False
        self.n_ok = 0
        self.prev_ts = None
        self.prev_seq = None
        sys.stderr.write(
            f"[recorder] save reason={record['reason']} frames={frames} "
            f"saved={self.n_saved}\n"
        )
        sys.stderr.flush()
        return {
            "ok": True,
            "status": record["disposition"],
            "frames": frames,
            "n_saved": self.n_saved,
        }

    def discard(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset is None:
            return {"ok": False, "error": "no dataset"}
        frames = self.n_ok
        self._clear_buffer()
        record = {
            "episode_index": None,
            "attempt_index": msg.get("attempt_index"),
            "disposition": "discarded",
            "reason": msg.get("reason", "rerecord_episode"),
            "frames": frames,
            "drops": self.n_drop,
            "duration_s": msg.get("duration_s"),
            "task": msg.get("task_log") or {},
        }
        self._append_log(record)
        self.n_discarded += 1
        self.episode_active = False
        self.n_ok = 0
        self.prev_ts = None
        self.prev_seq = None
        sys.stderr.write(
            f"[recorder] discard reason={record['reason']} frames={frames}\n"
        )
        sys.stderr.flush()
        return {"ok": True, "status": "discarded", "frames": frames}

    def finalize_session(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._finalized:
            dest = self.final_root if self.n_saved > 0 else self.quarantine
            return {
                "ok": True,
                "status": "committed" if self.n_saved > 0 else "quarantined",
                "path": None if dest is None else str(dest),
                "n_saved": self.n_saved,
            }
        if self.episode_active:
            self.discard({
                "reason": msg.get("reason", "interrupted"),
                "attempt_index": msg.get("attempt_index"),
                "duration_s": msg.get("duration_s"),
                "task_log": msg.get("task_log"),
            })
        stats = msg.get("stats") or {}
        stats.update({
            "episodes_saved": self.n_saved,
            "episodes_discarded": self.n_discarded,
            "frames_ok_total": self.n_ok_total,
        })
        if self.staging is not None:
            results_json = msg.get("results_json")
            if results_json and Path(results_json).exists():
                shutil.copy2(results_json, self.staging / "results.json")
            (self.staging / "stats.json").write_text(
                json.dumps(stats, indent=2), encoding="utf-8"
            )
        if self.dataset is not None and hasattr(self.dataset, "finalize"):
            try:
                self.dataset.finalize()
            except Exception as exc:
                sys.stderr.write(f"[recorder] finalize warning: {exc}\n")
                sys.stderr.flush()
        dest = self.final_root if self.n_saved > 0 else self.quarantine
        if dest is None or self.staging is None:
            return {"ok": False, "error": "not initialized"}
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(self.staging), str(dest))
        self._finalized = True
        status = "committed" if self.n_saved > 0 else "quarantined"
        sys.stderr.write(
            f"[recorder] {status} -> {dest} episodes={self.n_saved} "
            f"frames={self.n_ok_total}\n"
        )
        sys.stderr.flush()
        return {
            "ok": True,
            "status": status,
            "path": str(dest),
            "n_saved": self.n_saved,
            "n_discarded": self.n_discarded,
        }

    def end(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible: save current buffer then finalize the session."""
        if self.episode_active:
            saved = self.save(msg)
            if not saved.get("ok"):
                return saved
        return self.finalize_session(msg)

    def _clear_buffer(self) -> None:
        if self.dataset is None:
            return
        clearer = getattr(self.dataset, "clear_episode_buffer", None)
        if clearer is None:
            return
        try:
            clearer()
        except TypeError:
            clearer(delete_images=True)
        except Exception as exc:
            sys.stderr.write(f"[recorder] clear_episode_buffer warning: {exc}\n")
            sys.stderr.flush()

    def _append_log(self, record: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        _append_jsonl(self._log_path, record)
