"""ACT inference server (runs in the lerobot venv).

Length-prefixed pickle protocol over stdin/stdout — same contract as
``dp_server.py`` so the Isaac harness can query ACT without importing
lerobot in-process.

Message in : {"cmd":"reset"} OR
             {"state": (44,) f32, "head"/"left"/"right": (240,320,3) uint8}
Message out: {"ok":True} OR {"action": (14,) f32}

Usage: python act_server.py <checkpoint_pretrained_model_dir>
"""
import os
import pickle
import struct
import sys
import warnings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
warnings.filterwarnings("ignore")

# Ensure packaging.version is bound before safetensors/lerobot import
# (some launch cwds leave `packaging` without its submodule attribute).
import packaging.version as _packaging_version
import packaging

packaging.version = _packaging_version

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = sys.argv[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

policy = ACTPolicy.from_pretrained(CKPT)
policy.eval().to(DEV)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": DEV}},
)
sys.stderr.write(f"[act_server] loaded {CKPT} (+processors) on {DEV}\n")
sys.stderr.flush()

_in = sys.stdin.buffer
_out = sys.stdout.buffer


def _read():
    h = _in.read(4)
    if len(h) < 4:
        return None
    n = struct.unpack(">I", h)[0]
    buf = b""
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write(obj):
    b = pickle.dumps(obj)
    _out.write(struct.pack(">I", len(b)) + b)
    _out.flush()


def _img(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    return t.unsqueeze(0).float().div(255.0)


while True:
    msg = _read()
    if msg is None:
        break
    if msg.get("cmd") == "reset":
        policy.reset()
        _write({"ok": True})
        continue
    obs = {
        "observation.state": torch.from_numpy(
            np.asarray(msg["state"], np.float32)
        ).unsqueeze(0),
        "observation.images.head": _img(msg["head"]),
        "observation.images.left_hand": _img(msg["left"]),
        "observation.images.right_hand": _img(msg["right"]),
    }
    with torch.no_grad():
        obs = preprocessor(obs)
        a = policy.select_action(obs)
        a = postprocessor(a)
    _write({"action": a.squeeze(0).float().cpu().numpy().tolist()})
