#!/usr/bin/env python3
"""Validate the pinned RoCo Industrial Assembly LeRobot dataset before training.

Pins:
  repo:     rocochallenge2025/rocochallenge2026_Industrial_Assembly
  revision: dc03b003f94d184b2b20465ed986456ee1bf2a3c
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ID = "rocochallenge2025/rocochallenge2026_Industrial_Assembly"
REVISION = "dc03b003f94d184b2b20465ed986456ee1bf2a3c"
EXPECTED_FPS = 10
EXPECTED_EPISODES = 200
EXPECTED_FRAMES = 121_454
STATE_DIM = 44
ACTION_DIM = 14
CAMERAS = (
    "observation.images.head",
    "observation.images.left_hand",
    "observation.images.right_hand",
)
IMG_HW = (240, 320)


def _finite_stats(stats: dict, key: str) -> None:
    for field in ("min", "max", "mean", "std"):
        if field not in stats[key]:
            raise AssertionError(f"missing stats[{key}][{field}]")
        arr = stats[key][field]
        flat = arr.reshape(-1).tolist() if hasattr(arr, "reshape") else list(arr)
        if not all(math.isfinite(float(x)) for x in flat):
            raise AssertionError(f"non-finite stats for {key}.{field}")
        if field == "std" and any(float(x) == 0.0 for x in flat):
            raise AssertionError(f"zero-std dim in {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root override")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"[validate] loading {REPO_ID}@{REVISION}")
    # Prefer pyav: torchcodec often fails without matching system FFmpeg libs.
    ds = LeRobotDataset(
        REPO_ID,
        root=args.root,
        revision=REVISION,
        download_videos=True,
        video_backend="pyav",
    )
    meta = ds.meta

    assert meta.fps == EXPECTED_FPS, meta.fps
    assert meta.total_episodes == EXPECTED_EPISODES, meta.total_episodes
    assert meta.total_frames == EXPECTED_FRAMES, meta.total_frames
    assert ds.num_episodes == EXPECTED_EPISODES
    assert ds.num_frames == EXPECTED_FRAMES

    feats = meta.features
    assert feats["observation.state"]["shape"] == (STATE_DIM,), feats["observation.state"]
    assert feats["action"]["shape"] == (ACTION_DIM,), feats["action"]
    for cam in CAMERAS:
        assert cam in feats, f"missing camera {cam}"
        shape = tuple(feats[cam]["shape"])
        assert shape[:2] == IMG_HW or shape[-3:-1] == IMG_HW, (cam, shape)

    assert meta.stats is not None
    _finite_stats(meta.stats, "observation.state")
    _finite_stats(meta.stats, "action")

    sample = ds[args.sample_index]
    state = sample["observation.state"]
    action = sample["action"]
    assert tuple(state.shape) == (STATE_DIM,), state.shape
    assert tuple(action.shape) == (ACTION_DIM,), action.shape
    assert state.isfinite().all().item()
    assert action.isfinite().all().item()

    for cam in CAMERAS:
        img = sample[cam]
        assert img.ndim == 3 and img.shape[0] == 3, (cam, img.shape)
        assert img.shape[1:] == IMG_HW, (cam, img.shape)

    # Spot-check timing on a few episode boundaries.
    fps = float(meta.fps)
    for ep in (0, 1, min(5, EXPECTED_EPISODES - 1)):
        ep_meta = meta.episodes[ep]
        length = int(ep_meta["length"])
        assert 500 <= length <= 800, (ep, length)
        # timestamps should advance at ~1/fps within an episode
        start = int(ep_meta["dataset_from_index"])
        t0 = float(ds[start]["timestamp"])
        t1 = float(ds[start + 1]["timestamp"])
        dt = t1 - t0
        assert abs(dt - 1.0 / fps) < 1e-2, (ep, dt)

    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "fps": meta.fps,
        "num_episodes": ds.num_episodes,
        "num_frames": ds.num_frames,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "cameras": list(CAMERAS),
        "image_hw": list(IMG_HW),
        "ok": True,
    }
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[validate] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[validate] FAILED: {exc}", file=sys.stderr)
        raise
