"""CPU-only tests for the pi0.5 Isaac adapter contract."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
sys.path.insert(0, str(TASK_DIR))
sys.modules.setdefault(
    "policy_api",
    SimpleNamespace(EnvInfo=object, Observation=object, PartTarget=object, Policy=object),
)

from pi05_checkpoint_utils import checkpoint_kind
from policies.pi05_lerobot import (
    _IMG_H,
    _IMG_W,
    CONTROL_PERIOD_TICKS,
    GRIPPER_OPEN_LIMIT,
    STATE_SLICES,
    Pi05LeRobotPolicy,
    _resize_rgb,
    camera_payload_from_obs,
    euler_xyz_to_quat_wxyz,
    left_action_to_ik_target,
    query_due,
)


def test_state_layout_and_values():
    assert list(STATE_SLICES.values()) == [
        slice(0, 7),
        slice(7, 14),
        slice(14, 21),
        slice(21, 28),
        slice(28, 35),
        slice(35, 42),
        slice(42, 43),
        slice(43, 44),
    ]
    policy = Pi05LeRobotPolicy.__new__(Pi05LeRobotPolicy)
    policy.L = SimpleNamespace(
        end_effector=SimpleNamespace(
            get_world_pose=lambda: (np.array([1, 2, 3]), np.array([1, 0, 0, 0]))
        )
    )
    policy.R = SimpleNamespace(
        end_effector=SimpleNamespace(
            get_world_pose=lambda: (np.array([4, 5, 6]), np.array([1, 0, 0, 0]))
        )
    )
    policy._Li = list(range(7))
    policy._Ri = list(range(7, 14))
    policy._Lg = 14
    policy._Rg = 15
    q = np.arange(16, dtype=float)
    q[14] = GRIPPER_OPEN_LIMIT
    q[15] = GRIPPER_OPEN_LIMIT / 2
    obs = SimpleNamespace(joint_positions=q, joint_velocities=np.arange(16, dtype=float) + 20)
    state = policy._build_state(obs)
    assert state.shape == (44,)
    np.testing.assert_allclose(state[:14], [1, 2, 3, 1, 0, 0, 0, 4, 5, 6, 1, 0, 0, 0])
    np.testing.assert_allclose(state[42:], [1.0, 0.5])


def test_resize_and_camera_mapping():
    assert _resize_rgb(None).shape == (_IMG_H, _IMG_W, 3)
    obs = SimpleNamespace(
        rgb={
            "head": np.zeros((480, 640, 3), np.uint8),
            "L_wrist": np.ones((480, 640, 3), np.uint8),
            "R_wrist": np.full((480, 640, 3), 2, np.uint8),
        }
    )
    cameras = camera_payload_from_obs(obs)
    assert set(cameras) == {"head", "left", "right"}
    assert all(image.shape == (_IMG_H, _IMG_W, 3) for image in cameras.values())


def test_unwrapped_intrinsic_euler_and_gripper_clip():
    quat = euler_xyz_to_quat_wxyz(7.0, -4.0, 2 * math.pi + 0.2)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)
    action = np.zeros(14)
    action[:7] = [0.1, 0.2, 0.3, 7.0, -4.0, 3.5, 2.0]
    pos, quat, grip = left_action_to_ik_target(action)
    np.testing.assert_allclose(pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)
    assert grip == pytest.approx(GRIPPER_OPEN_LIMIT)


def test_action_guards_and_legacy_rotvec():
    with pytest.raises(ValueError, match="14-D"):
        left_action_to_ik_target(np.zeros(7))
    bad = np.zeros(14)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        left_action_to_ik_target(bad)
    action = np.zeros(14)
    action[3:6] = [0.4, 0.2, -0.1]
    euler_quat = left_action_to_ik_target(action, "euler_xyz")[1]
    rotvec_quat = left_action_to_ik_target(action, "rotvec")[1]
    assert not np.allclose(euler_quat, rotvec_quat)


def test_10hz_query_and_reset_behavior():
    assert CONTROL_PERIOD_TICKS == 20
    assert query_due(60, None, False, 20)
    assert not query_due(61, 60, True, 20)
    assert query_due(80, 60, True, 20)
    assert query_due(0, 80, True, 20)
    # Adapter reset clears both values, making the next observation query immediately.
    assert query_due(100, None, False, 20)


def test_full_and_lora_checkpoint_detection(tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    (full / "model.safetensors").touch()
    assert checkpoint_kind(full) == "full"

    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert checkpoint_kind(lora) == "lora"

    with pytest.raises(FileNotFoundError):
        checkpoint_kind(tmp_path / "missing")
