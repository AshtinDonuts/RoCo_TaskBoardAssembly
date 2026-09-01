from __future__ import annotations

import numpy as np
import pytest

from teleop.schema import (
    ACTION_DIM,
    STATE_DIM,
    pack_action,
    pack_state,
    resize_rgb,
    validate_frame,
)


def test_pack_dimensions():
    state = pack_state(
        [0, 0, 0], [1, 0, 0, 0],
        [0, 0, 0], [1, 0, 0, 0],
        [0] * 7, [0] * 7,
        [0] * 7, [0] * 7,
        0.1, 0.2,
    )
    action = pack_action(
        [0.1, 0.2, 0.3], [1, 0, 0, 0], 0.4,
        [0, 0, 0], [1, 0, 0, 0], 0.0,
    )
    assert state.shape == (STATE_DIM,)
    assert action.shape == (ACTION_DIM,)


def test_validate_frame_accepts_good_row():
    state = np.zeros(STATE_DIM, dtype=np.float32)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    images = {
        "head": np.zeros((240, 320, 3), dtype=np.uint8),
        "left_hand": np.zeros((240, 320, 3), dtype=np.uint8),
        "right_hand": np.zeros((240, 320, 3), dtype=np.uint8),
    }
    validate_frame(
        step_idx=0,
        timestamp_s=0.0,
        state=state,
        action=action,
        images=images,
    )


def test_validate_frame_rejects_bad_action_dim():
    with pytest.raises(ValueError):
        validate_frame(
            step_idx=0,
            timestamp_s=0.0,
            state=np.zeros(STATE_DIM),
            action=np.zeros(3),
            images={
                "head": np.zeros((240, 320, 3), np.uint8),
                "left_hand": np.zeros((240, 320, 3), np.uint8),
                "right_hand": np.zeros((240, 320, 3), np.uint8),
            },
        )


def test_resize_none_is_black():
    img = resize_rgb(None)
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8
