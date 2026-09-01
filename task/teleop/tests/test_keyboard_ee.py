import numpy as np

from teleop.keyboard_ee import EE_KEY_TO_CMD, KeyboardEE
from teleop.protocol import CHAR_TO_CMD, KeyDecoder
from teleop.retarget import CartesianRetargeter, RetargetConfig


def test_ee_keys_do_not_collide_with_task_keys():
    overlap = set(EE_KEY_TO_CMD) & set(CHAR_TO_CMD)
    assert not overlap


def test_key_decoder_extra_chars_motion():
    dec = KeyDecoder(extra_chars=EE_KEY_TO_CMD)
    assert dec.feed(b"i", now=1.0) == ["ee-y"]
    assert dec.feed(b"l", now=1.0) == ["ee-x"]
    assert dec.feed(b"f", now=1.0) == ["grip_close"]
    assert dec.feed(b"v", now=1.0) == ["grip_open"]
    assert dec.feed(b"n", now=1.0) == ["part_done"]
    assert dec.feed(b"\x1b[C", now=1.0) == ["save_episode"]


def test_keyboard_ee_hold_integrates_and_binary_gripper():
    ee = KeyboardEE(lin_vel_mps=0.10, ang_vel_rps=1.0)
    assert ee.apply_holds(["ee+y"], dt=0.1) is True
    assert abs(ee.pos[1] - 0.01) < 1e-9
    ee.apply_holds(["grip_close"], dt=0.1)
    assert ee.gripper == 0.0
    ee.apply_holds(["grip_open"], dt=0.1)
    assert ee.gripper == 1.0
    before = ee.quat.copy()
    ee.apply_holds(["ee+yaw"], dt=0.1)
    assert float(np.linalg.norm(ee.quat - before)) > 0.0
    sample = ee.take_sample()
    assert sample["clutch"] is True
    assert sample["deadman"] is True
    assert sample["cmd"] == "none"
    assert sample["gripper_norm"] == 1.0
    assert abs(sample["ee_pos"][1] - 0.01) < 1e-9


def test_keyboard_hold_moves_retarget_target():
    ee = KeyboardEE(lin_vel_mps=0.20)
    ret = CartesianRetargeter(RetargetConfig(translation_gain=1.0, max_lin_acc=0.0))
    dex0 = np.array([0.3, 0.0, 0.2])
    quat0 = np.array([1.0, 0.0, 0.0, 0.0])
    # Engage clutch at origin.
    pos, quat, grip, info = ret.step(
        leader_pos=ee.pos,
        leader_quat=ee.quat,
        gripper_norm=ee.gripper,
        dt=0.005,
        clutch=True,
        deadman=True,
        current_dex_pos=dex0,
        current_dex_quat=quat0,
    )
    assert info["reason"] == "clutch_engage"
    ee.apply_holds(["ee+y"], dt=0.05)  # +1 cm leader Y
    pos2, _, _, info2 = ret.step(
        leader_pos=ee.pos,
        leader_quat=ee.quat,
        gripper_norm=ee.gripper,
        dt=0.005,
        clutch=True,
        deadman=True,
        current_dex_pos=dex0,
        current_dex_quat=quat0,
    )
    assert info2["reason"] == "tracking"
    # Rate limit caps one physics step; still must move positive Y.
    assert pos2[1] > pos[1]
    for _ in range(20):
        pos2, _, _, _ = ret.step(
            leader_pos=ee.pos,
            leader_quat=ee.quat,
            gripper_norm=ee.gripper,
            dt=0.005,
            clutch=True,
            deadman=True,
            current_dex_pos=dex0,
            current_dex_quat=quat0,
        )
    assert pos2[1] >= 0.009
