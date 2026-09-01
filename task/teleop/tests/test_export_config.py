"""Tests for teleop export JSON loading / validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from teleop.export_config import (
    DEFAULT_EXPORT_CONFIG,
    load_export_config,
    validate_export_dict,
)


def test_load_default_export_config():
    cfg = load_export_config(DEFAULT_EXPORT_CONFIG)
    assert cfg.schema_version == 1
    assert cfg.fps == 10.0
    assert cfg.export.playback_clock == "wall"
    assert cfg.img_h == 240 and cfg.img_w == 320
    assert cfg.export.image.cameras == ("head", "left_hand", "right_hand")
    assert cfg.export.dataset.state_dim == 44
    assert cfg.export.dataset.action_dim == 14
    assert cfg.session.episode_time_s == 60.0
    assert cfg.paths.teleop_yaml.name == "aloha_solo_to_vega_1u.yaml"
    assert cfg.control_arms == "right"
    assert cfg.control.active_arms == ("R",)
    assert cfg.control.gravity_compensation is False
    assert cfg.leader_endpoint_for("right") == "127.0.0.1:19850"
    fields = cfg.recorder_init_fields()
    assert fields["fps"] == 10.0
    assert fields["image_height"] == 240
    assert fields["cameras"] == ["head", "left_hand", "right_hand"]
    assert fields["control_arms"] == "right"


def test_control_arms_defaults_when_omitted(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw.pop("control", None)
    cfg = validate_export_dict(raw, source_path=tmp_path / "no_control.json")
    assert cfg.control_arms == "right"
    assert cfg.control.active_arms == ("R",)
    assert cfg.control.gravity_compensation is False


def test_gravity_compensation_opt_in(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["control"]["gravity_compensation"] = True
    cfg = validate_export_dict(raw, source_path=tmp_path / "gcomp.json")
    assert cfg.control.gravity_compensation is True


def test_reject_bad_gravity_compensation(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["control"]["gravity_compensation"] = "sometimes"
    with pytest.raises(ValueError, match="gravity_compensation"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_control_arms_dual(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["control"]["arms"] = "dual"
    cfg = validate_export_dict(raw, source_path=tmp_path / "dual.json")
    assert cfg.control_arms == "dual"
    assert cfg.control.active_arms == ("L", "R")
    assert cfg.leader_endpoint_for("left") == "127.0.0.1:19850"
    assert cfg.leader_endpoint_for("right") == "127.0.0.1:19851"


def test_reject_bad_control_arms(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["control"]["arms"] = "left"
    with pytest.raises(ValueError, match="control.arms"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_reject_dual_same_endpoints(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["control"]["arms"] = "dual"
    raw["control"]["leader_endpoints"] = {
        "left": "127.0.0.1:19850",
        "right": "127.0.0.1:19850",
    }
    with pytest.raises(ValueError, match="must differ"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_reject_bad_fps(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["export"]["fps"] = 0
    with pytest.raises(ValueError, match="fps"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_reject_bad_image_size(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["export"]["image"]["height"] = 0
    with pytest.raises(ValueError, match="height/width"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_reject_incomplete_cameras(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["export"]["image"]["cameras"] = ["head"]
    with pytest.raises(ValueError, match="must include challenge cameras"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_reject_unknown_camera(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["export"]["image"]["cameras"] = ["head", "left_hand", "right_hand", "ego"]
    with pytest.raises(ValueError, match="unknown"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_reject_wrong_contract_dims(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["export"]["dataset"]["state_dim"] = 32
    with pytest.raises(ValueError, match="state_dim"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_reject_bad_playback_clock(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["export"]["playback_clock"] = "render"
    with pytest.raises(ValueError, match="playback_clock"):
        validate_export_dict(raw, source_path=tmp_path / "bad.json")


def test_load_custom_file(tmp_path: Path):
    raw = json.loads(DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8"))
    raw["export"]["fps"] = 20
    raw["session"]["warmup_time_s"] = 2
    path = tmp_path / "export.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_export_config(path)
    assert cfg.fps == 20.0
    assert cfg.session.warmup_time_s == 2.0
    assert cfg.record_period_s == pytest.approx(0.05)
