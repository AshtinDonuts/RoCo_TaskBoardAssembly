from teleop.keyboard_ee import KeyboardEE
from teleop.protocol import (
    clutch_mode_after_cmd,
    clutch_mode_engaged,
    clutch_transition_cmd,
    cycle_clutch_mode,
)


def test_space_cycles_track_pause():
    assert cycle_clutch_mode("track") == "pause"
    assert cycle_clutch_mode("pause") == "track"
    assert cycle_clutch_mode("freeze") == "pause"
    assert clutch_mode_engaged("track") is True
    assert clutch_mode_engaged("freeze") is False
    assert clutch_mode_engaged("pause") is False


def test_clutch_transition_cmds():
    assert clutch_transition_cmd("track", "pause") == "pause"
    assert clutch_transition_cmd("pause", "track") == "resume"
    assert clutch_transition_cmd("freeze", "track") == "recenter"
    assert clutch_transition_cmd("track", "freeze") == "none"
    assert clutch_mode_after_cmd("pause", "track") == "pause"
    assert clutch_mode_after_cmd("resume", "pause") == "track"
    assert clutch_mode_after_cmd("clutch_toggle", "track") == "pause"
    assert clutch_mode_after_cmd("clutch_toggle", "pause") == "track"
    assert clutch_mode_after_cmd("clutch_toggle", "freeze") == "pause"


def test_keyboard_ee_space_cycles_pause_and_emits_wire_cmds():
    ee = KeyboardEE()
    assert ee.clutch_mode == "track"
    assert ee.clutch is True

    ee.apply_edge("clutch_toggle")
    sample = ee.take_sample()
    assert ee.clutch_mode == "pause"
    assert sample["clutch"] is False
    assert sample["cmd"] == "pause"

    ee.apply_edge("clutch_toggle")
    sample = ee.take_sample()
    assert ee.clutch_mode == "track"
    assert sample["clutch"] is True
    assert sample["cmd"] == "resume"


def test_keyboard_p_and_u_jump_to_pause_and_track():
    ee = KeyboardEE()
    ee.apply_edge("pause")
    sample = ee.take_sample()
    assert ee.clutch_mode == "pause"
    assert sample["clutch"] is False
    assert sample["cmd"] == "pause"

    ee.apply_edge("resume")
    sample = ee.take_sample()
    assert ee.clutch_mode == "track"
    assert sample["clutch"] is True
    assert sample["cmd"] == "resume"


def test_keyboard_motion_during_pause_does_not_auto_resume():
    ee = KeyboardEE()
    ee.apply_edge("pause")
    assert ee.take_sample()["cmd"] == "pause"
    moved = ee.apply_holds(["ee+x"], 0.05)
    assert moved is True
    sample = ee.take_sample()
    assert ee.clutch_mode == "pause"
    assert sample["clutch"] is False
    assert sample["cmd"] == "none"
