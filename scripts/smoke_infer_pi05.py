#!/usr/bin/env python3
"""Load a full/LoRA pi0.5 checkpoint and emit one finite 14-D RoCo action."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
sys.path.insert(0, str(TASK_DIR))

from lerobot.policies import make_pre_post_processors
from lerobot.utils.constants import ACTION
from pi05_server import load_pi05_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path, help="LeRobot pretrained_model directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    policy, policy_cfg, kind = load_pi05_policy(args.ckpt, args.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(args.ckpt),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()

    def image() -> torch.Tensor:
        return torch.rand(3, 240, 320, dtype=torch.float32)

    observation = {
        "observation.state": torch.zeros(44, dtype=torch.float32),
        "observation.images.head": image(),
        "observation.images.left_hand": image(),
        "observation.images.right_hand": image(),
        "task": "assemble parts onto the task board",
    }
    with torch.inference_mode():
        batch = preprocessor(observation)
        action = postprocessor(policy.select_action(batch))
    if isinstance(action, dict):
        action = action[ACTION]
    values = action.squeeze(0).float().cpu().numpy().reshape(-1)
    assert values.shape == (14,), values.shape
    assert np.isfinite(values).all(), values

    report = {
        "checkpoint": str(args.ckpt),
        "checkpoint_kind": kind,
        "device": args.device,
        "state_dim": 44,
        "action_shape": list(values.shape),
        "action": values.tolist(),
        "ok": True,
    }
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
