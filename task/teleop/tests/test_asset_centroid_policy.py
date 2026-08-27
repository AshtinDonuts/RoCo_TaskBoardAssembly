"""Offline state-machine tests for the privileged scripted policy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from controllers.gripper_compliance import (
    GripperCompliance,
    GripperComplianceConfig,
    GripperPhase,
)
from policies.asset_centroid_scripted import AssetCentroidScriptedPolicy
from policy_api import Observation, PartTarget


class _Controller:
    def __init__(self, *, ik_ok=True, solve_ok=True, solve_delta=0.0):
        self.ik = SimpleNamespace(ik_ok=ik_ok)
        self.solve_ok = solve_ok
        self.solve_delta = solve_delta
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
        return SimpleNamespace(joint_positions=[0.0] * 8)

    def forward_raw_q(self, q_cspace, gripper):
        return SimpleNamespace(joint_positions=[0.0] * 8)


def _policy(controller=None):
    controller = controller or _Controller()
    env = SimpleNamespace(
        R_controller=controller,
        L_controller=None,
        dof_names=[f"q{i}" for i in range(8)],
        physics_dt=0.005,
    )
    return AssetCentroidScriptedPolicy(env)


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
        "gear_60teeth",
        "open",
        place_pos=np.array([0.1, 0.0, 1.0]),
        gripper_open=0.2,
        gripper_close=0.0,
        extra={"ee_offset": np.array([0.0, 0.016, 0.2])},
    )
    spec = policy._part_spec_for_target(target)
    np.testing.assert_allclose(spec.tcp_to_grasp_tool, [0.0, -0.016, 0.2], atol=1e-9)


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


def test_no_feasible_yaw_aborts_without_motion(monkeypatch):
    policy = _policy(_Controller(solve_ok=False))
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
    assert policy.is_done(obs)
    assert all(v is None for v in policy.act(obs).joint_positions)


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


def test_repeated_runtime_ik_failure_aborts_at_guard(monkeypatch):
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
    for _ in range(99):
        policy.act(obs)
        assert not policy.is_done(obs)
    action = policy.act(obs)
    assert policy.is_done(obs)
    assert all(v is None for v in action.joint_positions)
