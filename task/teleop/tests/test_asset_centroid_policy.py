"""Offline state-machine tests for the privileged scripted policy."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from controllers.gripper_compliance import (
    GripperCompliance,
    GripperComplianceConfig,
    GripperPhase,
)
from policies.asset_centroid_scripted import AssetCentroidScriptedPolicy
from policies.asset_centroid_motion import quat_angle, quat_mul_wxyz
from policy_api import Observation, PartTarget


class _Controller:
    def __init__(
        self, *, ik_ok=True, solve_ok=True, solve_delta=0.0, fail_forward=None
    ):
        self.ik = SimpleNamespace(ik_ok=ik_ok)
        self.solve_ok = solve_ok
        self.solve_delta = solve_delta
        self.fail_forward = fail_forward
        self.forward_targets = []
        self.seed_ee_pose = (
            np.array([0.0, 0.0, 1.3]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        )

    def current_cspace_q(self):
        return np.zeros(7)

    def cspace_joint_names(self):
        return [f"R_arm_j{i}" for i in range(1, 8)]

    def solve_q(self, pos, orn, seed=None):
        return (
            np.asarray(seed, dtype=np.float64) + self.solve_delta,
            self.solve_ok,
        )

    def forward(self, pos, orn, gripper):
        pos = np.asarray(pos, dtype=np.float64).copy()
        self.forward_targets.append(pos)
        if self.fail_forward is not None:
            self.ik.ik_ok = not bool(self.fail_forward(pos))
        return SimpleNamespace(joint_positions=[0.0] * 8)

    def forward_raw_q(self, q_cspace, gripper):
        return SimpleNamespace(joint_positions=[0.0] * 8)


def _policy(controller=None, *, config=None):
    controller = controller or _Controller()
    env = SimpleNamespace(
        R_controller=controller,
        L_controller=None,
        dof_names=[f"q{i}" for i in range(8)],
        physics_dt=0.005,
    )
    return AssetCentroidScriptedPolicy(env, config=config)


def _obs():
    return Observation(
        step_idx=0,
        joint_positions=np.zeros(8),
        joint_velocities=np.zeros(8),
        L_gripper_position=0.12,
        ee_pose_L=(
            np.array([0.0, 0.0, 1.3]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ),
        rgb={"head": None, "L_wrist": None, "R_wrist": None},
        depth={"head": None, "L_wrist": None, "R_wrist": None},
        intrinsics={"head": None, "L_wrist": None, "R_wrist": None},
    )


def test_param_config_fallback_flips_r_tcp_y():
    """Non-JSON parts inherit L-tuned ee_offset; R tool TCP must negate Y."""
    policy = _policy()
    target = PartTarget(
        "pin",
        "open",
        place_pos=np.array([0.1, 0.0, 1.0]),
        gripper_open=0.2,
        gripper_close=0.0,
        extra={"ee_offset": np.array([0.0, 0.016, 0.2])},
    )
    spec = policy._part_spec_for_target(target)
    np.testing.assert_allclose(spec.tcp_to_grasp_tool, [0.0, -0.016, 0.2], atol=1e-9)


def _configured_target(name: str) -> PartTarget:
    import param_config as pc

    cfg = pc.get_part_config(name)
    return PartTarget(
        name,
        cfg["release_mode"],
        pick_pos=np.asarray(cfg["pick_pos"], dtype=np.float64),
        place_pos=np.asarray(cfg["place_pos"], dtype=np.float64),
        gripper_open=float(cfg["gripper_open"]),
        gripper_close=float(cfg["gripper_close"]),
        extra=dict(cfg),
    )


def test_pin_fallback_spec_uses_r_tcp_and_geometry_apertures():
    """Pin must not silently inherit the L-arm lateral TCP convention."""
    import param_config as pc

    policy = _policy()
    target = _configured_target("pin")
    spec = policy._part_spec_for_target(target)

    np.testing.assert_allclose(spec.tcp_to_grasp_tool, [0.0, -0.017, 0.185])
    assert spec.gripper_open_rad == pytest.approx(pc.part_grasp_open_rad("pin"))
    assert spec.gripper_close_rad == pytest.approx(pc.part_grasp_close_rad("pin"))
    assert spec.gripper_open_rad > spec.gripper_close_rad > 0.0


def test_pin_centroid_builds_finite_approach_with_configured_clearance(monkeypatch):
    """Separate bad centroid/TCP construction from an IK continuity failure."""
    policy = _policy()
    centroid = np.array([-0.03105, -0.01476, 1.04437])
    monkeypatch.setattr(
        policy, "_live_asset_centroid", lambda _: (centroid.copy(), np.zeros(3))
    )

    policy.reset(_obs(), _configured_target("pin"))

    assert not policy.is_done(_obs())
    phases = {phase.name: phase for phase in policy._phases}
    pick = phases["descend_pick"].pos
    hover = phases["hover_pick"].pos
    assert np.all(np.isfinite(pick))
    np.testing.assert_allclose(hover[:2], pick[:2], atol=1e-12)
    assert hover[2] - pick[2] == pytest.approx(policy._path_heights_m()[0])


def test_relaxed_hover_precedes_strict_alignment_and_disabled_keeps_legacy_sequence(
    monkeypatch,
):
    def centroid(_):
        return np.array([0.0, 0.0, 1.0]), np.zeros(3)

    relaxed = _policy()
    monkeypatch.setattr(relaxed, "_live_asset_centroid", centroid)
    relaxed.reset(_obs(), PartTarget("gear_60teeth", "open", place_pos=np.array([0.1, 0.0, 1.0])))
    phases = relaxed._phases
    names = [phase.name for phase in phases]
    assert names[:3] == ["hover_pick_relaxed", "align_hover_pick", "descend_pick"]
    assert phases[0].orientation_cone_rad == pytest.approx(
        relaxed._cfg.approach_orientation.max_tilt_rad
    )
    assert phases[1].orientation_cone_rad is None
    assert phases[2].orientation_cone_rad is None

    strict = _policy(config=relaxed._cfg)
    monkeypatch.setattr(strict, "_live_asset_centroid", centroid)
    strict.reset(_obs(), PartTarget("gear_20teeth", "open", place_pos=np.array([0.1, 0.0, 1.0])))
    assert [phase.name for phase in strict._phases][:2] == ["hover_pick", "descend_pick"]


def test_tilt_candidates_cover_signed_tool_axes_without_free_yaw():
    policy = _policy()
    base = np.array([0.0, 1.0, 0.0, 0.0])
    candidates = policy._approach_tilt_candidates(base)
    assert len(candidates) == 4
    for candidate in candidates:
        assert quat_angle(candidate, base) == pytest.approx(
            policy._cfg.approach_orientation.sample_tilt_rad
        )
        # A roll/pitch perturbation is composed on the right (tool frame),
        # so it contains no independent world-Z/yaw rotation.
        relative = quat_mul_wxyz(np.array([0.0, -1.0, 0.0, 0.0]), candidate)
        assert abs(relative[3]) < 1e-12


def test_pin_hover_pick_joint_approach_bypasses_dense_cartesian_ik(monkeypatch):
    """Joint hover_pick must not call pose IK each sample (pin mid-path abort)."""
    controller = _Controller(fail_forward=lambda pos: True)
    policy = _policy(controller)
    centroid = np.array([-0.03105, -0.01476, 1.04437])
    monkeypatch.setattr(
        policy, "_live_asset_centroid", lambda _: (centroid.copy(), np.zeros(3))
    )
    obs = _obs()
    # Seed EE at hover so arrival check can pass after the joint segment.
    hover_pos = None
    policy.reset(obs, _configured_target("pin"))
    assert not policy.is_done(obs)
    assert policy._hover_pick_q is not None
    hover_phase = next(
        p for p in policy._phases if p.name in ("hover_pick", "hover_pick_relaxed")
    )
    hover_pos = hover_phase.pos.copy()
    controller.seed_ee_pose = (hover_pos, hover_phase.orn.copy())

    for _ in range(len(policy._segment_q) + 5 if policy._segment_q else 200):
        policy.act(obs)
        if policy._phase_index > 0 or policy.is_done(obs):
            break
        if not policy._segment:
            # First act starts the segment.
            continue

    assert controller.forward_targets == [], "joint approach must not call forward()/IK"
    assert not policy._aborted
    assert policy._phases[policy._phase_index].name != hover_phase.name or (
        policy._segment_index >= len(policy._segment)
    )


def test_policy_requires_right_controller_and_active_arms():
    with pytest.raises(ValueError, match="R_controller"):
        AssetCentroidScriptedPolicy(
            SimpleNamespace(
                R_controller=None,
                dof_names=["q0"],
                physics_dt=0.005,
            )
        )
    policy = _policy()
    assert policy.active_arms == ("R",)


@pytest.mark.parametrize(
    ("name", "release_mode"),
    [
        ("gear_60teeth", "open"),
        ("battery_size1", "open"),
        ("rod_16mm", "open"),
        ("pin", "open"),
    ],
)
def test_configured_targets_are_planned_not_skipped(monkeypatch, name, release_mode):
    policy = _policy()
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda _: (np.array([0.0, 0.0, 1.0]), np.zeros(3)),
    )
    obs = _obs()
    policy.reset(
        obs,
        PartTarget(name, release_mode, place_pos=np.array([0.1, 0.0, 1.0])),
    )
    assert not policy.is_done(obs)
    assert policy._part_spec is not None


def test_missing_geometry_aborts_without_motion(monkeypatch):
    policy = _policy()
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda name: (_ for _ in ()).throw(RuntimeError("missing mesh")),
    )
    obs = _obs()
    policy.reset(
        obs,
        PartTarget("gear_20teeth", "open", place_pos=np.array([0.0, 0.0, 1.0])),
    )
    assert policy.is_done(obs)
    assert all(v is None for v in policy.act(obs).joint_positions)


def test_param_config_height_overrides_json_clearances():
    import param_config as pc

    policy = _policy()
    policy._target = PartTarget(
        "bolt_8mm", "open", place_pos=np.array([0.1, 0.0, 1.0])
    )
    cfg = pc.get_part_config("bolt_8mm")
    assert cfg.get("use_param_config_height") is True
    init_h = float(cfg["init_height"])
    place_h = float(cfg["hover_place_height"])
    final_h = float(cfg["final_height"])
    hover_pick, hover_place, retract = policy._path_heights_m()
    np.testing.assert_allclose(hover_pick, init_h)
    np.testing.assert_allclose(hover_place, place_h)
    np.testing.assert_allclose(retract, final_h)
    np.testing.assert_allclose(policy._place_z_offset_m(), float(cfg["place_z_offset"]))
    # Must differ from JSON path clearances when PART_CONFIG values differ.
    json_path = policy._cfg.path_clearances
    assert (hover_pick, hover_place, retract) != (
        float(json_path.hover_pick_m),
        float(json_path.hover_place_m),
        float(json_path.final_retract_m),
    )


def test_hover_place_height_falls_back_to_init_height(monkeypatch):
    import param_config as pc

    policy = _policy()
    policy._target = PartTarget(
        "bolt_8mm", "open", place_pos=np.array([0.1, 0.0, 1.0])
    )
    base = pc.get_part_config("bolt_8mm")
    patched = dict(base)
    patched["hover_place_height"] = None
    monkeypatch.setattr(pc, "get_part_config", lambda name: patched)
    hover_pick, hover_place, retract = policy._path_heights_m()
    np.testing.assert_allclose(hover_pick, float(base["init_height"]))
    np.testing.assert_allclose(hover_place, float(base["init_height"]))
    np.testing.assert_allclose(retract, float(base["final_height"]))


def test_json_path_heights_used_when_flag_false():
    policy = _policy()
    policy._target = PartTarget(
        "gear_20teeth", "open", place_pos=np.array([0.1, 0.0, 1.0])
    )
    hover_pick, hover_place, retract = policy._path_heights_m()
    np.testing.assert_allclose(hover_pick, policy._cfg.path_clearances.hover_pick_m)
    np.testing.assert_allclose(hover_place, policy._cfg.path_clearances.hover_place_m)
    np.testing.assert_allclose(retract, policy._cfg.path_clearances.final_retract_m)


def test_endpoint_motion_larger_than_runtime_jump_gate_still_plans(monkeypatch):
    policy = _policy(_Controller(solve_delta=2.0))
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda name: (np.array([0.0, 0.0, 1.0]), np.zeros(3)),
    )
    obs = _obs()
    policy.reset(
        obs,
        PartTarget("gear_20teeth", "open", place_pos=np.array([0.1, 0.0, 1.0])),
    )
    assert not policy.is_done(obs)
    # Default gripper.mode=compliant: numeric open (baseline-style) + soft close.
    assert policy._cfg.gripper.mode == "compliant"
    by_name = {p.name: p.gripper for p in policy._phases}
    assert by_name["close"] == "close"
    assert by_name["hover_pick"] == pytest.approx(0.12)
    assert by_name["open"] == pytest.approx(0.12)


def test_close_dwell_freezes_remaining_grasp_at_measured_aperture(monkeypatch):
    controller = _Controller(solve_delta=2.0)
    controller.gripper_compliance = GripperCompliance(
        GripperComplianceConfig(hold_margin=0.002)
    )
    controller._measured_gripper = lambda: (0.09, 0.0)
    policy = _policy(controller)
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda name: (np.array([0.0, 0.0, 1.0]), np.zeros(3)),
    )
    obs = _obs()
    policy.reset(
        obs,
        PartTarget("gear_20teeth", "open", place_pos=np.array([0.1, 0.0, 1.0])),
    )
    assert controller.gripper_compliance.cfg.close_speed_rad_s == pytest.approx(0.40)
    assert controller.gripper_compliance.cfg.close == pytest.approx(0.075)
    policy._phase_index = next(
        i for i, phase in enumerate(policy._phases) if phase.name == "close"
    )
    controller.gripper_compliance._q_cmd = 0.08

    policy._freeze_compliant_grasp_aperture()

    assert controller.gripper_compliance.phase == GripperPhase.HOLDING
    assert controller.gripper_compliance.q_cmd == pytest.approx(0.088)
    by_name = {phase.name: phase.gripper for phase in policy._phases}
    for name in ("close", "lift_pick", "hover_place", "descend_place", "settle_place"):
        assert by_name[name] == pytest.approx(0.088)
    assert by_name["open"] == pytest.approx(0.12)


def test_descend_place_uses_cartesian_segment(monkeypatch):
    """Short place descent should be Cartesian (task-space Z), not joint arc."""
    policy = _policy(_Controller(solve_delta=0.01))
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda name: (np.array([0.0, 0.0, 1.0]), np.zeros(3)),
    )
    obs = _obs()
    policy.reset(
        obs,
        PartTarget("gear_20teeth", "open", place_pos=np.array([0.1, 0.0, 1.0])),
    )
    assert not policy.is_done(obs)
    phase_i = next(
        i for i, phase in enumerate(policy._phases) if phase.name == "descend_place"
    )
    policy._phase_index = phase_i
    policy._segment = ()
    policy._segment_q = ()
    policy._start_segment(obs, policy._phases[phase_i])
    assert len(policy._segment) > 0
    assert policy._segment_q == ()


def test_hover_pick_uses_joint_space_segment(monkeypatch):
    """Approach to hover_pick must quintic-lerp joints (preflight q), not dense IK."""
    policy = _policy(_Controller(solve_delta=0.01))
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda name: (np.array([-0.031, -0.015, 1.04]), np.zeros(3)),
    )
    obs = _obs()
    policy.reset(
        obs,
        PartTarget("pin", "open", place_pos=np.array([0.0324, 0.00616, 1.065])),
    )
    assert not policy.is_done(obs)
    assert policy._hover_pick_q is not None
    phase_i = next(
        i
        for i, phase in enumerate(policy._phases)
        if phase.name in ("hover_pick", "hover_pick_relaxed")
    )
    policy._phase_index = phase_i
    policy._segment = ()
    policy._segment_q = ()
    policy._start_segment(obs, policy._phases[phase_i])
    assert len(policy._segment_q) > 0
    np.testing.assert_allclose(policy._segment_q[-1], policy._hover_pick_q)


def test_bolt_skips_null_descend_and_latches_hold_q(monkeypatch):
    """When hover≈place, skip descend_place; settle/open freeze joints."""
    import param_config as pc

    policy = _policy(_Controller(solve_delta=0.01))
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda name: (np.array([-0.17, 0.0, 1.04]), np.zeros(3)),
    )
    base = dict(pc.get_part_config("bolt_8mm"))
    base["hover_place_height"] = 0.0
    monkeypatch.setattr(pc, "get_part_config", lambda name: base)
    obs = _obs()
    policy.reset(
        obs,
        PartTarget("bolt_8mm", "open", place_pos=np.array([-0.00174, 0.13135, 1.06])),
    )
    assert not policy.is_done(obs)
    names = [p.name for p in policy._phases]
    assert "descend_place" not in names
    assert "settle_place" in names
    # Simulate finishing hover_place → settle; hold_q must latch.
    policy._phase_index = names.index("hover_place")
    policy._advance_phase(obs)
    assert policy._hold_q is not None
    policy._phase_index = names.index("settle_place")
    action = policy.act(obs)
    assert action is not None


def test_bolt_transport_hover_clears_release_height():
    """hover_place must sit above place so transit does not scrape the table."""
    import param_config as pc

    policy = _policy()
    policy._target = PartTarget(
        "bolt_8mm", "open", place_pos=np.array([-0.00174, 0.13135, 1.06])
    )
    policy._part_spec = policy._part_spec_for_target(policy._target)
    cfg = pc.get_part_config("bolt_8mm")
    assert float(cfg["hover_place_height"]) > 0.0
    poses = policy._pose_set(
        np.array([-0.17, 0.0, 1.04]),
        np.array([-0.00174, 0.13135, 1.06]),
        np.array([0.0, 1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        poses["hover_place"][2] - poses["place"][2],
        float(cfg["hover_place_height"]),
    )


def test_place_z_offset_lowers_release_pose(monkeypatch):
    import param_config as pc

    policy = _policy()
    policy._target = PartTarget(
        "bolt_8mm", "open", place_pos=np.array([-0.00174, 0.13135, 1.06])
    )
    policy._part_spec = policy._part_spec_for_target(policy._target)
    grasp = np.array([-0.17, 0.0, 1.04])
    place_c = np.array([-0.00174, 0.13135, 1.06])
    orn = np.array([0.0, 1.0, 0.0, 0.0])
    off = policy._place_z_offset_m()
    assert off < 0.0
    poses = policy._pose_set(grasp, place_c, orn)
    base = dict(pc.get_part_config("bolt_8mm"))
    base["place_z_offset"] = 0.0
    monkeypatch.setattr(pc, "get_part_config", lambda name: base)
    poses0 = policy._pose_set(grasp, place_c, orn)
    np.testing.assert_allclose(poses["place"][2], poses0["place"][2] + off)
    np.testing.assert_allclose(
        poses["hover_place"][2] - poses["place"][2],
        float(base["hover_place_height"]),
    )
    np.testing.assert_allclose(
        poses["retract"][2] - poses["place"][2],
        float(base["final_height"]),
    )


def test_repeated_runtime_ik_failure_aborts_at_guard(monkeypatch):
    """Cartesian phases still honor the consecutive IK-failure guard."""
    controller = _Controller(ik_ok=False, solve_ok=True)
    policy = _policy(controller)
    monkeypatch.setattr(
        policy,
        "_live_asset_centroid",
        lambda name: (np.array([0.0, 0.0, 1.0]), np.zeros(3)),
    )
    obs = _obs()
    policy.reset(
        obs,
        PartTarget("gear_20teeth", "open", place_pos=np.array([0.1, 0.0, 1.0])),
    )
    assert not policy.is_done(obs)
    # hover_pick is joint-space now; exercise the guard on descend_pick.
    phase_i = next(
        i for i, phase in enumerate(policy._phases) if phase.name == "descend_pick"
    )
    policy._phase_index = phase_i
    policy._segment = ()
    policy._segment_q = ()
    for _ in range(99):
        policy.act(obs)
        assert not policy.is_done(obs)
    action = policy.act(obs)
    assert policy.is_done(obs)
    assert all(v is None for v in action.joint_positions)
