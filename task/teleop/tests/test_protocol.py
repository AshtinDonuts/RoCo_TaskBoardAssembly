from __future__ import annotations

import socket
import threading
import time

from teleop.protocol import (
    PROTOCOL_VERSION,
    bind_server,
    connect_with_retry,
    make_leader_sample,
    recv_message,
    send_message,
    validate_leader_sample,
)


def test_sample_roundtrip():
    sample = make_leader_sample(
        seq=3,
        joints=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ee_pos=[0.01, 0.02, 0.03],
        ee_quat_wxyz=[1, 0, 0, 0],
        gripper_norm=0.4,
        clutch=True,
        deadman=True,
        cmd="recenter",
    )
    assert sample["version"] == PROTOCOL_VERSION
    again = validate_leader_sample(sample)
    assert again["cmd"] == "recenter"


def test_socket_roundtrip():
    server = bind_server("127.0.0.1", 0)
    server.settimeout(2.0)
    host, port = server.getsockname()[:2]
    received = {}

    def accept():
        conn, _ = server.accept()
        received["msg"] = recv_message(conn)
        conn.close()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    client = connect_with_retry(host, port, timeout_s=2.0)
    send_message(
        client,
        make_leader_sample(
            seq=1,
            joints=[0] * 6,
            ee_pos=[0, 0, 0],
            ee_quat_wxyz=[1, 0, 0, 0],
            gripper_norm=0.0,
            clutch=False,
            deadman=True,
        ),
    )
    thread.join(timeout=2.0)
    client.close()
    server.close()
    assert received["msg"]["seq"] == 1


def test_gripper_clamp():
    sample = make_leader_sample(
        seq=0,
        joints=[0] * 6,
        ee_pos=[0, 0, 0],
        ee_quat_wxyz=[1, 0, 0, 0],
        gripper_norm=1.04,
        clutch=False,
        deadman=True,
    )
    assert sample["gripper_norm"] == 1.0
