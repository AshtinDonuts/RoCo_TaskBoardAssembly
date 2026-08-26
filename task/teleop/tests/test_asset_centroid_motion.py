"""Unit tests for the dependency-light asset-centroid motion helpers."""

from __future__ import annotations

import numpy as np

from policies.asset_centroid_motion import (
    GEAR_20TEETH_SPEC,
    clear_asset_centroid_config_cache,
    ee_position_for_grasp_center,
    load_asset_centroid_config,
    local_aabb_midpoint,
    motion_duration,
    quat_angle,
    quintic_smoothstep,
    rotate_vector,
    sample_joint_segment,
    sample_pose_segment,
    top_down_yaw_candidates,
    transform_point,
    unwrap_revolute_delta,
)


def test_local_aabb_midpoint_not_vertex_mean():
    points = np.array([[-2.0, -1.0, 3.0], [4.0, 5.0, 9.0], [4.0, 5.0, 9.0]])
    np.testing.assert_allclose(local_aabb_midpoint(points), [1.0, 2.0, 6.0])


def test_centroid_transforms_with_live_pose():
    angle = np.pi / 2.0
    q_z90 = [np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)]
    result = transform_point([1.0, 0.0, 0.0], [10.0, 20.0, 30.0], q_z90)
    np.testing.assert_allclose(result, [10.0, 21.0, 30.0], atol=1e-9)


def test_all_yaw_candidates_are_exactly_world_down():
    candidates = top_down_yaw_candidates(15.0)
    assert len(candidates) == 24
    for quat in candidates:
        np.testing.assert_allclose(
            rotate_vector(quat, [0.0, 0.0, 1.0]),
            [0.0, 0.0, -1.0],
            atol=1e-9,
        )


def test_manual_gear_apertures_and_tcp_transform():
    assert GEAR_20TEETH_SPEC.gripper_open_rad == 0.12
    assert GEAR_20TEETH_SPEC.gripper_close_rad == 0.065
    ee_tool = ee_position_for_grasp_center(
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0, 0.0],
        GEAR_20TEETH_SPEC.tcp_to_grasp_tool,
        offset_frame="tool",
    )
    ee_world = ee_position_for_grasp_center(
        [1.0, 2.0, 3.0],
        [0.0, 1.0, 0.0, 0.0],
        GEAR_20TEETH_SPEC.tcp_to_grasp_tool,
        offset_frame="world",
    )
    np.testing.assert_allclose(ee_tool, [1.0, 2.016, 3.197], atol=1e-9)
    np.testing.assert_allclose(ee_world, [1.0, 2.016, 3.197], atol=1e-9)


def test_quintic_has_zero_endpoint_velocity():
    eps = 1e-6
    assert quintic_smoothstep(0.0) == 0.0
    assert quintic_smoothstep(1.0) == 1.0
    assert quintic_smoothstep(eps) / eps < 1e-8
    assert (1.0 - quintic_smoothstep(1.0 - eps)) / eps < 1e-8


def test_duration_caps_quintic_peak_linear_speed():
    distance = 0.20
    duration = motion_duration(distance, 0.0, max_linear_speed_m_s=0.05)
    np.testing.assert_allclose(duration, 7.5, atol=1e-12)
    assert 1.875 * distance / duration <= 0.05 + 1e-12


def test_pose_sampling_is_deterministic_and_reaches_endpoint():
    kwargs = dict(
        start_pos=[0.0, 0.0, 0.0],
        start_orn=[1.0, 0.0, 0.0, 0.0],
        end_pos=[0.01, 0.02, 0.03],
        end_orn=[0.0, 1.0, 0.0, 0.0],
        dt=0.01,
    )
    a = sample_pose_segment(**kwargs)
    b = sample_pose_segment(**kwargs)
    assert len(a) == len(b) > 1
    for (pa, qa), (pb, qb) in zip(a, b):
        np.testing.assert_allclose(pa, pb)
        np.testing.assert_allclose(qa, qb)
    np.testing.assert_allclose(a[-1][0], kwargs["end_pos"], atol=1e-12)
    assert quat_angle(a[-1][1], kwargs["end_orn"]) < 1e-9


def test_joint_segment_unwraps_shortest_revolute_path():
    start = np.array([0.0, 3.0])
    end = np.array([0.1, -3.0])  # raw delta ~-6; shortest ~+0.283 via wrap
    delta = unwrap_revolute_delta(start, end, prismatic_mask=[True, False])
    np.testing.assert_allclose(delta[0], 0.1, atol=1e-12)
    assert abs(delta[1]) < np.pi
    samples = sample_joint_segment(
        start,
        end,
        dt=0.1,
        linear_distance_m=0.05,
        angular_distance_rad=0.2,
        prismatic_mask=[True, False],
    )
    assert len(samples) >= 1
    np.testing.assert_allclose(samples[-1], start + delta, atol=1e-12)


def test_json_config_exposes_speed_and_gear_grasp():
    clear_asset_centroid_config_cache()
    cfg = load_asset_centroid_config()
    assert cfg.active_arm == "R"
    assert cfg.path_clearances.tcp_offset_frame == "world"
    assert cfg.path_clearances.force_yaw_deg == 0.0
    assert cfg.motion.max_linear_speed_m_s > 0.0
    np.testing.assert_allclose(
        np.degrees(cfg.motion.max_angular_speed_rad_s), 20.0
    )
    gear = cfg.part("gear_20teeth")
    assert gear.gripper_open_rad == 0.12
    assert gear.gripper_close_rad == 0.065
    assert GEAR_20TEETH_SPEC.gripper_close_rad == gear.gripper_close_rad

