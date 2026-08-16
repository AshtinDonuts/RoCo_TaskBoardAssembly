"""LeaderClient command-event queue (no TCP)."""
from __future__ import annotations

import time

from teleop.leader_client import LeaderClient


def test_cmd_event_queue_preserves_pose_at_edge():
    client = LeaderClient(cmd_queue_max=8)
    now = time.time()
    with client._lock:
        client._ingest_sample_locked(
            {
                "cmd": "pause",
                "ee_pos": [0.0, 0.0, 0.0],
                "ee_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "clutch": False,
                "gripper_norm": 1.0,
            },
            now,
        )
        client._ingest_sample_locked(
            {
                "cmd": "none",
                "ee_pos": [0.1, 0.0, 0.0],
                "ee_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "clutch": False,
                "gripper_norm": 1.0,
            },
            now + 0.02,
        )
        client._ingest_sample_locked(
            {
                "cmd": "resume",
                "ee_pos": [0.2, 0.1, 0.0],
                "ee_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "clutch": True,
                "gripper_norm": 0.5,
            },
            now + 0.04,
        )
        client._ingest_sample_locked(
            {
                "cmd": "none",
                "ee_pos": [0.3, 0.0, 0.0],
                "ee_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "clutch": True,
                "gripper_norm": 0.5,
            },
            now + 0.06,
        )

    pause_ev = client.pop_cmd_event()
    assert pause_ev is not None
    assert pause_ev["cmd"] == "pause"
    assert pause_ev["ee_pos"] == [0.0, 0.0, 0.0]
    assert pause_ev["clutch"] is False

    resume_ev = client.pop_cmd_event()
    assert resume_ev is not None
    assert resume_ev["cmd"] == "resume"
    assert resume_ev["ee_pos"] == [0.2, 0.1, 0.0]
    assert resume_ev["clutch"] is True

    assert client.pop_cmd_event() is None
    assert client.pop_cmd() == "none"

    latest = client.latest()
    assert latest is not None
    assert latest["ee_pos"] == [0.3, 0.0, 0.0]
    assert latest["cmd"] == "none"
