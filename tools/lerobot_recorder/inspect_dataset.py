#!/usr/bin/env python3
"""Inspect a recorded LeRobot Task Board episode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        try:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError as exc:
        raise SystemExit(
            "Run this with the lerobot conda env: conda run -n lerobot python ..."
        ) from exc

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    root = args.dataset
    print(f"dataset: {root}")
    repo_id = "local/roco_aloha_teleop"
    for path in [root / "run_meta.json"]:
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            print(f"\n== {path.name} ==")
            print(json.dumps(meta, indent=2))
            repo_id = meta.get("repo_id") or repo_id
    stats_path = root / "stats.json"
    if stats_path.exists():
        print("\n== stats.json ==")
        print(stats_path.read_text(encoding="utf-8"))

    try:
        md = LeRobotDatasetMetadata(repo_id=repo_id, root=str(root))
        print("LeRobotDatasetMetadata frames/info:", getattr(md, "info", md))
    except Exception as exc:
        print(f"metadata helper: {exc}")

    ds = None
    last = None
    for kwargs in (
        {"repo_id": repo_id, "root": root},
        {"repo_id": "local/" + root.name.replace("_", "/"), "root": root},
    ):
        try:
            ds = LeRobotDataset(**kwargs)
            break
        except Exception as exc:
            last = exc
    if ds is None:
        raise SystemExit(f"could not open dataset: {last}")

    print(f"frames={ds.num_frames} episodes={ds.num_episodes} fps={getattr(ds, 'fps', None)}")
    sample = ds[0]
    state = np.asarray(sample["observation.state"])
    action = np.asarray(sample["action"])
    print(f"state shape={state.shape} action shape={action.shape}")
    for key in (
        "observation.images.head",
        "observation.images.left_hand",
        "observation.images.right_hand",
    ):
        if key in sample:
            print(f"{key} shape={np.asarray(sample[key]).shape}")

    if args.plot:
        import matplotlib.pyplot as plt

        n = min(ds.num_frames, 2000)
        ee = np.stack([np.asarray(ds[i]["observation.state"])[:3] for i in range(n)])
        act = np.stack([np.asarray(ds[i]["action"])[:3] for i in range(n)])
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(ee)
        ax[0].set_ylabel("left EE xyz")
        ax[1].plot(act)
        ax[1].set_ylabel("left action xyz")
        ax[1].set_xlabel("frame")
        out = root / "inspect_ee.png"
        fig.tight_layout()
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
