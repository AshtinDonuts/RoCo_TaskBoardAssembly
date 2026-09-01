from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "lerobot_recorder"))

from writer import ACTION_DIM, IMG_H, IMG_W, STATE_DIM, EpisodeWriter  # noqa: E402


class FakeDataset:
    def __init__(self) -> None:
        self.frames = []
        self.saved = []
        self.cleared = 0
        self.finalized = False

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self, episode_data=None, parallel_encoding=True):
        self.saved.append(list(self.frames))
        self.frames = []

    def clear_episode_buffer(self, delete_images=True):
        self.cleared += 1
        self.frames = []

    def finalize(self):
        self.finalized = True


def _frame(seq: int = 0) -> dict:
    images = {
        key: np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        for key in ("head", "left_hand", "right_hand")
    }
    return {
        "state": np.zeros(STATE_DIM, dtype=np.float32),
        "action": np.zeros(ACTION_DIM, dtype=np.float32),
        "images": images,
        "timestamp_s": float(seq) * 0.1,
        "seq": seq,
    }


def _writer(tmp_path: Path) -> EpisodeWriter:
    writer = EpisodeWriter()
    writer.staging = tmp_path / "staging"
    writer.staging.mkdir()
    writer.final_root = tmp_path / "datasets" / "run"
    writer.quarantine = tmp_path / "quarantine" / "run"
    writer.dataset = FakeDataset()
    writer._log_path = writer.staging / "episodes.jsonl"
    return writer


def test_failed_task_episode_is_still_committed(tmp_path):
    writer = _writer(tmp_path)
    writer.begin({"episode_index": 0, "attempt_index": 1})
    assert writer.add_frame(_frame(0))["ok"]
    saved = writer.save({
        "reason": "save_episode",
        "attempt_index": 1,
        "duration_s": 1.2,
        "task_log": {"aborted": True, "events": [{"event": "abort"}]},
    })
    assert saved["status"] == "saved"
    assert saved["n_saved"] == 1
    out = writer.finalize_session({"reason": "session_end"})
    assert out["status"] == "committed"
    assert writer.final_root.exists()
    assert not writer.quarantine.exists()
    log = [
        json.loads(line)
        for line in (writer.final_root / "episodes.jsonl").read_text().splitlines()
        if line
    ]
    assert log[0]["disposition"] == "saved"
    assert log[0]["task"]["aborted"] is True


def test_rerecord_then_save_keeps_one_episode(tmp_path):
    writer = _writer(tmp_path)
    writer.begin({"attempt_index": 1})
    writer.add_frame(_frame(0))
    discarded = writer.discard({"reason": "rerecord_episode", "attempt_index": 1})
    assert discarded["status"] == "discarded"
    assert writer.dataset.cleared == 1
    writer.begin({"attempt_index": 2})
    writer.add_frame(_frame(0))
    writer.add_frame(_frame(1))
    writer.save({"reason": "timeout", "attempt_index": 2})
    out = writer.finalize_session({})
    assert out["n_saved"] == 1
    assert out["n_discarded"] == 1
    assert len(writer.dataset.saved) == 1
    assert len(writer.dataset.saved[0]) == 2


def test_interrupt_before_save_goes_to_quarantine(tmp_path):
    writer = _writer(tmp_path)
    writer.begin({"attempt_index": 1})
    writer.add_frame(_frame(0))
    out = writer.finalize_session({"reason": "interrupted"})
    assert out["status"] == "quarantined"
    assert out["n_saved"] == 0
    assert writer.quarantine.exists()
    assert writer.dataset.cleared >= 1


def test_frames_rejected_before_begin(tmp_path):
    writer = _writer(tmp_path)
    reply = writer.add_frame(_frame(0))
    assert reply["ok"] is False
    assert "no active episode" in reply["error"]
