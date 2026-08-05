"""Pi0.5 inference server for the LeRobot environment.

Runs outside Isaac Sim's Python environment. The Isaac-side policy talks to
this process through a length-prefixed pickle protocol.
"""
from __future__ import annotations

import os
import pickle
import struct
import sys
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION
from pi05_checkpoint_utils import checkpoint_kind


def load_pi05_policy(checkpoint: str, device: str):
    """Load a standalone pi0.5 policy or a PEFT adapter and its recorded base."""
    checkpoint = str(checkpoint)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    kind = checkpoint_kind(checkpoint)
    if kind == "lora":
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(checkpoint)
        base_path = peft_cfg.base_model_name_or_path
        if not base_path:
            raise ValueError("PEFT checkpoint does not record base_model_name_or_path")
        base_policy = PI05Policy.from_pretrained(base_path, config=policy_cfg)
        policy = PeftModel.from_pretrained(base_policy, checkpoint)
    else:
        policy = PI05Policy.from_pretrained(checkpoint, config=policy_cfg)
    policy.eval().to(device)
    return policy, policy_cfg, kind


def _read(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    size = struct.unpack(">I", header)[0]
    buf = b""
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write(stream, obj):
    payload = pickle.dumps(obj)
    stream.write(struct.pack(">I", len(payload)) + payload)
    stream.flush()


def _img(arr):
    # HxWx3 uint8/float -> 3xHxW float32 in [0, 1].
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype == np.uint8:
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return t.float().div(255.0)
    t = torch.from_numpy(np.ascontiguousarray(a[..., :3])).permute(2, 0, 1)
    return t.float().clamp(0.0, 1.0)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python pi05_server.py /path/to/checkpoint/pretrained_model")

    checkpoint = sys.argv[1]
    device = os.environ.get("PI05_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    task = os.environ.get("PI05_TASK", "assemble parts onto the task board")
    policy, policy_cfg, kind = load_pi05_policy(checkpoint, device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        pretrained_revision=getattr(policy_cfg, "pretrained_revision", None),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    sys.stderr.write(f"[pi05_server] loaded {kind} checkpoint {checkpoint} on {device}\n")
    sys.stderr.flush()
    in_stream = sys.stdin.buffer
    out_stream = sys.stdout.buffer

    while True:
        msg = _read(in_stream)
        if msg is None:
            break
        if msg.get("cmd") == "reset":
            policy.reset()
            _write(out_stream, {"ok": True})
            continue

        obs = {
            "observation.state": torch.as_tensor(msg["state"], dtype=torch.float32),
            "observation.images.head": _img(msg["head"]),
            "observation.images.left_hand": _img(msg["left"]),
            "observation.images.right_hand": _img(msg["right"]),
            "task": msg.get("task", task),
        }

        with torch.inference_mode():
            batch = preprocessor(obs)
            action = policy.select_action(batch)
            action = postprocessor(action)
        if isinstance(action, dict):
            action = action[ACTION]
        action_np = action.squeeze(0).float().cpu().numpy().reshape(-1)
        if action_np.shape != (14,):
            raise RuntimeError(f"expected 14-D pi0.5 action, got shape {action_np.shape}")
        if not np.isfinite(action_np).all():
            raise RuntimeError("pi0.5 action contains non-finite values")
        _write(out_stream, {"action": action_np.tolist()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
