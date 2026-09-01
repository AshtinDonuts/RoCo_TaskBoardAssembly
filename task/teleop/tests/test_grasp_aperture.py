"""Tests for geometric Design D grasp_width_m → close rad."""
from __future__ import annotations

import json

import numpy as np
import pytest

from teleop.grasp_aperture import (
    clear_aabb_cache,
    close_rad_to_aperture_m,
    grasp_width_m,
    grasp_width_to_close_rad,
    list_grasp_parts,
    part_grasp_close_rad_from_width,
    resolve_grasp_close_rad,
)


def setup_function():
    clear_aabb_cache()


def test_list_parts_includes_gears():
    names = list_grasp_parts()
    assert "gear_20teeth" in names
    assert "battery_size1" in names


def test_rod_width_maps_to_measured_aperture_with_clearance():
    w = grasp_width_m("rod_16mm")
    assert w is not None
    np.testing.assert_allclose(w, 0.016, atol=5e-5)
    q = grasp_width_to_close_rad(w)
    np.testing.assert_allclose(q, 0.075966, atol=1e-5)
    np.testing.assert_allclose(close_rad_to_aperture_m(q), w + 0.0005, atol=1e-6)


def test_part_resolution_matches_converter():
    q_resolved = part_grasp_close_rad_from_width("gear_20teeth")
    w = grasp_width_m("gear_20teeth")
    q_conv = grasp_width_to_close_rad(w)
    np.testing.assert_allclose(q_resolved, q_conv, atol=1e-5)


def test_all_precomputed_close_values_match_converter():
    from teleop.grasp_aperture import load_aabb_extents

    data = load_aabb_extents()
    for name, part in data["parts"].items():
        q_conv = grasp_width_to_close_rad(part["grasp_width_m"])
        np.testing.assert_allclose(
            part["grasp_close_rad"], q_conv, atol=1e-5, err_msg=name
        )


def test_runtime_resolution_uses_calibration_not_precomputed(tmp_path):
    path = tmp_path / "extents.json"
    path.write_text(
        json.dumps(
            {
                "parts": {
                    "widget": {
                        "grasp_width_m": 0.1,
                        "grasp_close_rad": 0.9,
                    }
                },
                "aperture_calibration": {
                    "q_closed_rad": 0.0,
                    "gap_at_q_closed_m": 0.0,
                    "q_open_rad": 1.0,
                    "gap_at_q_open_m": 1.0,
                    "margin_m": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    q = part_grasp_close_rad_from_width("widget", path=str(path))
    np.testing.assert_allclose(q, 0.1)


def test_piecewise_mapping_and_clearance(tmp_path):
    path = tmp_path / "piecewise.json"
    path.write_text(
        json.dumps(
            {
                "parts": {},
                "aperture_calibration": {
                    "model": "piecewise_linear_measured",
                    "clearance_m": 0.001,
                    "samples": [
                        {"q_rad": 0.0, "gap_m": 0.0},
                        {"q_rad": 0.2, "gap_m": 0.01},
                        {"q_rad": 0.5, "gap_m": 0.04},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    q = grasp_width_to_close_rad(0.019, path=str(path))
    np.testing.assert_allclose(q, 0.3)
    np.testing.assert_allclose(close_rad_to_aperture_m(q, path=str(path)), 0.02)


def test_resolve_fallback():
    q = resolve_grasp_close_rad("not_a_real_part_xyz", fallback_rad=0.123)
    np.testing.assert_allclose(q, 0.123)


def test_param_config_helper_uses_geometric_resolution():
    import param_config as pc

    q = pc.part_grasp_close_rad("battery_size1")
    expected = grasp_width_to_close_rad(grasp_width_m("battery_size1"))
    np.testing.assert_allclose(q, expected)


def test_scripted_open_target_clears_every_known_object():
    import param_config as pc

    for name in list_grasp_parts():
        q_open = pc.part_grasp_open_rad(name)
        q_close = pc.part_grasp_close_rad(name)
        assert q_open > q_close, name
        cfg = pc.get_part_config(name)
        if bool(cfg.get("use_param_config_aperture", False)):
            np.testing.assert_allclose(q_open, float(cfg["gripper_open"]))
            np.testing.assert_allclose(q_close, float(cfg["gripper_close"]))
            continue
        explicit = pc.PART_CONFIG.get(name, {}).get("gripper_open")
        if explicit is not None:
            # Hand-tuned PART_CONFIG open is honored (clamped above close).
            assert q_open >= float(explicit) - 1e-9 or q_open >= q_close + 1e-3 - 1e-9, name
            continue
        assert close_rad_to_aperture_m(q_open) >= (
            grasp_width_m(name) + pc.GRASP_OPEN_CLEARANCE_M - 1e-6
        ), name


def test_bolt_uses_param_config_aperture_when_flag_set():
    import param_config as pc

    cfg = pc.get_part_config("bolt_8mm")
    assert cfg["use_param_config_aperture"] is True
    q_open = pc.part_grasp_open_rad("bolt_8mm")
    q_close = pc.part_grasp_close_rad("bolt_8mm")
    np.testing.assert_allclose(q_open, float(cfg["gripper_open"]))
    np.testing.assert_allclose(q_close, float(cfg["gripper_close"]))
    # Design D alone would be ~0.062; flag must bypass that.
    assert q_close < 0.01
def test_param_config_does_not_hide_malformed_geometry(tmp_path, monkeypatch):
    import param_config as pc

    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    monkeypatch.setenv("ROCO_PART_AABB_JSON", str(path))
    clear_aabb_cache()

    with pytest.raises(json.JSONDecodeError):
        pc.part_grasp_close_rad("rod_16mm")
