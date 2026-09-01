from teleop.protocol import (
    COMMANDS,
    KeyDecoder,
    make_leader_sample,
    map_operator_token,
    validate_leader_sample,
)


def test_recording_commands_are_on_the_wire():
    for cmd in ("save_episode", "rerecord_episode", "stop_recording"):
        assert cmd in COMMANDS
        sample = make_leader_sample(
            seq=0,
            joints=[0] * 6,
            ee_pos=[0, 0, 0],
            ee_quat_wxyz=[1, 0, 0, 0],
            gripper_norm=0.0,
            clutch=False,
            deadman=True,
            cmd=cmd,
        )
        assert validate_leader_sample(sample)["cmd"] == cmd


def test_map_operator_token_aliases_and_letters():
    assert map_operator_token("n") == "part_done"
    assert map_operator_token("save") == "save_episode"
    assert map_operator_token("save_episode") == "save_episode"
    assert map_operator_token("rerecord") == "rerecord_episode"
    assert map_operator_token("stop") == "stop_recording"
    assert map_operator_token("s") == "start"
    assert map_operator_token(" ") == "clutch_toggle"
    assert map_operator_token("nope") is None


def test_key_decoder_letters_and_arrows():
    dec = KeyDecoder(esc_timeout_s=0.05)
    assert dec.feed(b"n", now=1.0) == ["part_done"]
    assert dec.feed(b"\x1b[C", now=1.0) == ["save_episode"]
    assert dec.feed(b"\x1b[D", now=1.0) == ["rerecord_episode"]
    assert dec.feed(b"\x1bOC", now=1.0) == ["save_episode"]
    assert dec.feed(b"\x1bOD", now=1.0) == ["rerecord_episode"]


def test_key_decoder_esc_timeout_is_stop():
    dec = KeyDecoder(esc_timeout_s=0.05)
    assert dec.feed(b"\x1b", now=10.0) == []
    assert dec.poll_timeout(now=10.02) == []
    assert dec.poll_timeout(now=10.06) == ["stop_recording"]
    assert dec.poll_timeout(now=10.20) == []


def test_key_decoder_csi_modified_arrow():
    dec = KeyDecoder()
    assert dec.feed(b"\x1b[1;5C", now=1.0) == ["save_episode"]
