#!/usr/bin/env python3
"""LeRobot v3 recorder sidecar. Runs in the Python 3.12 conda env.

Length-prefixed pickle protocol on stdin/stdout. Never inherit Isaac's
PYTHONPATH. All numpy arrays from Isaac are converted to lists/bytes first.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

HEADER = struct.Struct(">I")
IMG_H, IMG_W = 240, 320
STATE_DIM, ACTION_DIM = 44, 14
IMAGE_KEYS = ("head", "left_hand", "right_hand")

_in = sys.stdin.buffer
_out = sys.stdout.buffer


def _read() -> Optional[Dict[str, Any]]:
    header = _in.read(HEADER.size)
    if len(header) < HEADER.size:
        return None
    (n,) = HEADER.unpack(header)
    buf = b""
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write(obj: Dict[str, Any]) -> None:
    payload = pickle.dumps(obj, protocol=4)
    _out.write(HEADER.pack(len(payload)) + payload)
    _out.flush()


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


def _success_from_results(path: Optional[str]) -> Optional[bool]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    parts = data.get("parts") or data.get("results") or data
    if isinstance(parts, dict) and "parts" in parts:
        parts = parts["parts"]
    if isinstance(parts, dict):
        flags = []
        for item in parts.values():
            if isinstance(item, dict) and "pass" in item:
                flags.append(bool(item["pass"]))
            elif isinstance(item, bool):
                flags.append(item)
        if flags:
            return all(flags) and len(flags) >= 9
    if isinstance(parts, list):
        flags = []
        for item in parts:
            if isinstance(item, dict):
                flags.append(bool(item.get("pass", item.get("success", False))))
        if flags:
            return all(flags) and len(flags) >= 9
    passed = data.get("passed") or data.get("n_pass") or data.get("score")
    total = data.get("total") or data.get("n_total")
    if passed is not None and total is not None:
        return int(passed) == int(total) and int(total) >= 9
    return None


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
        self.task = "Industrial Task Board Assembly"

    def init(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        root = Path(msg.get("root") or "runs").expanduser().resolve()
        run_id = msg.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
        repo_id = msg.get("repo_id") or "local/roco_aloha_teleop"
        self.staging = root / "staging" / run_id
        self.final_root = root / "datasets" / repo_id.replace("/", "_")
        self.quarantine = root / "quarantine" / run_id
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.staging.parent.mkdir(parents=True, exist_ok=True)
        if self.staging.exists():
            shutil.rmtree(self.staging)
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
        sys.stderr.write(f"[recorder] staging {self.staging}\n")
        sys.stderr.flush()
        return {"ok": True, "staging": str(self.staging)}

    def add_frame(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("init was not called")
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
        return {"ok": True, "n": self.n_ok}

    def end(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset is None:
            return {"ok": False, "error": "no dataset"}
        stats = msg.get("stats") or {}
        stats.update({"frames_ok": self.n_ok, "frames_drop": self.n_drop})
        results_json = msg.get("results_json")
        success = msg.get("success")
        if success is None:
            success = _success_from_results(results_json)
        if results_json and Path(results_json).exists():
            shutil.copy2(results_json, self.staging / "results.json")
        (self.staging / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        if self.n_ok > 0:
            self.dataset.save_episode()
        if hasattr(self.dataset, "finalize"):
            try:
                self.dataset.finalize()
            except Exception as exc:
                sys.stderr.write(f"[recorder] finalize warning: {exc}\n")
                sys.stderr.flush()
        dest = self.final_root if success else self.quarantine
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(self.staging), str(dest))
        status = "committed" if success else "quarantined"
        sys.stderr.write(f"[recorder] {status} -> {dest} frames={self.n_ok}\n")
        sys.stderr.flush()
        return {"ok": True, "status": status, "path": str(dest), "success": success}


def main() -> None:
    writer = EpisodeWriter()
    sys.stderr.write("[recorder] ready\n")
    sys.stderr.flush()
    while True:
        msg = _read()
        if msg is None:
            break
        cmd = msg.get("cmd")
        try:
            if cmd == "init":
                _write(writer.init(msg))
            elif cmd == "frame":
                _write(writer.add_frame(msg))
            elif cmd == "end_episode":
                _write(writer.end(msg))
            elif cmd == "shutdown":
                _write({"ok": True})
                break
            else:
                _write({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _write({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
