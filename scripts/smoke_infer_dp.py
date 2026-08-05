#!/usr/bin/env python3
"""Smoke-test: load a Diffusion checkpoint + processors and emit a finite 14-D action."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=Path, help="pretrained_model directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    ckpt = args.ckpt
    if not ckpt.is_dir():
        raise FileNotFoundError(ckpt)

    policy = DiffusionPolicy.from_pretrained(str(ckpt))
    policy.eval().to(args.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.reset()

    state = torch.zeros(1, 44, dtype=torch.float32)
    def img():
        return torch.rand(1, 3, 240, 320, dtype=torch.float32)
    obs = {
        "observation.state": state,
        "observation.images.head": img(),
        "observation.images.left_hand": img(),
        "observation.images.right_hand": img(),
    }
    with torch.no_grad():
        obs = preprocessor(obs)
        action = policy.select_action(obs)
        action = postprocessor(action)
    a = action.squeeze(0).float().cpu().numpy()
    assert a.shape == (14,), a.shape
    assert np.isfinite(a).all(), a

    report = {
        "ckpt": str(ckpt),
        "device": args.device,
        "action_shape": list(a.shape),
        "action": a.tolist(),
        "n_obs_steps": policy.config.n_obs_steps,
        "horizon": policy.config.horizon,
        "n_action_steps": policy.config.n_action_steps,
        "ok": True,
    }
    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[smoke_infer] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke_infer] FAILED: {exc}", file=sys.stderr)
        raise
