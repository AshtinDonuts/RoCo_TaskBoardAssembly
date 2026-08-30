"""Unit tests for Cartesian EE-pose pacing in EEPathFollower."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_follower_module():
    """Load the follower with tiny Isaac/Lula stubs for unit testing."""
    module_names = (
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.api",
        "isaacsim.core.api.controllers",
        "isaacsim.core.api.controllers.base_controller",
        "isaacsim.core.utils",
        "isaacsim.core.utils.types",
        "task.controllers.lula_ik_controller",
    )
    previous = {name: sys.modules.get(name) for name in module_names}

    class BaseController:
        def __init__(self, name):
            self.name = name

        def reset(self):
            pass

    class ArticulationAction:
        def __init__(self, joint_positions=None):
            self.joint_positions = joint_positions

    stubs = {name: types.ModuleType(name) for name in module_names}
    stubs["isaacsim.core.api.controllers.base_controller"].BaseController = (
        BaseController
    )
    stubs["isaacsim.core.utils.types"].ArticulationAction = ArticulationAction
    stubs["task.controllers.lula_ik_controller"].LulaIKController = object
    sys.modules.update(stubs)

    module_name = "task.controllers._ee_pose_controller_cart_test"
    path = Path(__file__).parents[1] / "controllers" / "ee_pose_controller.py"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


FOLLOWER_MODULE = _load_follower_module()
EEPathFollower = FOLLOWER_MODULE.EEPathFollower
Waypoint = FOLLOWER_MODULE.Waypoint
_quat_slerp = FOLLOWER_MODULE._quat_slerp
_quat_angle = FOLLOWER_MODULE._quat_angle


class _EndEffector:
    def __init__(self, position, orn=None):
        self.position = np.asarray(position, dtype=np.float64)
        self.orn = np.asarray(
            orn if orn is not None else [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )

    def get_world_pose(self):
        return self.position.copy(), self.orn.copy()


class _Controller:
    def __init__(self, position, orn=None):
        self._n_dof = 3
        self.end_effector = _EndEffector(position, orn)
        self.last_forward = None

    def reset(self):
        pass

    def forward(self, target_position, target_orientation, gripper_cmd):
        self.last_forward = (
            np.asarray(target_position, dtype=np.float64).copy(),
            (None if target_orientation is None
             else np.asarray(target_orientation, dtype=np.float64).copy()),
            gripper_cmd,
        )
        return types.SimpleNamespace(
            joint_positions=[0.0, 0.0, gripper_cmd],
            gripper=gripper_cmd,
        )


class QuatSlerpTests(unittest.TestCase):
    def test_endpoints_and_short_arc(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(_quat_slerp(q0, q1, 0.0), q0)
        np.testing.assert_allclose(_quat_slerp(q0, q1, 1.0), q1)
        mid = _quat_slerp(q0, q1, 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(mid)), 1.0, places=6)
        # q0→q1 is a 180° rotation; halfway is 90°.
        self.assertAlmostEqual(_quat_angle(q0, mid), np.pi / 2, places=5)
        self.assertAlmostEqual(_quat_angle(mid, q1), np.pi / 2, places=5)


class CartesianPacingTests(unittest.TestCase):
    def test_ik_target_steps_toward_terminal(self):
        controller = _Controller([0.0, 0.0, 0.0])
        follower = EEPathFollower(
            controller,
            position_tolerance=0.001,
            max_ee_step_m=0.05,
            max_ee_orn_step_rad=0.1,
        )
        terminal = np.array([0.2, 0.0, 0.0])
        follower.set_path([
            Waypoint(terminal, np.array([1.0, 0.0, 0.0, 0.0]), "open", 0, "hover"),
        ])

        follower.step()
        ik_pos, _, grip = controller.last_forward
        np.testing.assert_allclose(ik_pos, [0.05, 0.0, 0.0], atol=1e-9)
        self.assertIsNone(grip)
        self.assertEqual(follower.current_index(), 0)

        follower.step()
        ik_pos, _, grip = controller.last_forward
        np.testing.assert_allclose(ik_pos, [0.10, 0.0, 0.0], atol=1e-9)
        self.assertIsNone(grip)

    def test_gripper_only_after_command_reaches_terminal(self):
        controller = _Controller([0.0, 0.0, 0.0])
        follower = EEPathFollower(
            controller,
            position_tolerance=0.001,
            max_ee_step_m=0.1,
        )
        terminal = np.array([0.25, 0.0, 0.0])
        follower.set_path([
            Waypoint(terminal, None, "open", 0, "hover"),
        ])

        for _ in range(2):
            follower.step()
            self.assertIsNone(controller.last_forward[2])

        follower.step()  # command arrives at 0.25
        self.assertEqual(controller.last_forward[2], "open")
        np.testing.assert_allclose(
            controller.last_forward[0], terminal, atol=1e-9
        )
        # Actual EE still at origin — must not advance yet.
        self.assertEqual(follower.current_index(), 0)

    def test_advance_requires_actual_ee_at_terminal(self):
        controller = _Controller([0.0, 0.0, 0.0])
        follower = EEPathFollower(
            controller,
            position_tolerance=0.001,
            max_ee_step_m=0.1,
        )
        terminal = np.array([0.1, 0.0, 0.0])
        follower.set_path([
            Waypoint(terminal, None, None, 0, "hover"),
            Waypoint(np.array([0.1, 0.0, -0.05]), None, None, 0, "descend"),
        ])

        follower.step()  # cmd reaches terminal
        self.assertEqual(follower.current_index(), 0)

        controller.end_effector.position[:] = terminal
        follower.step()
        self.assertEqual(follower.current_index(), 1)

    def test_close_lock_pose_issues_gripper_immediately(self):
        controller = _Controller([0.0, 0.0, 0.0])
        follower = EEPathFollower(
            controller,
            position_tolerance=0.001,
            max_ee_step_m=0.01,
        )
        follower.set_path([
            Waypoint(np.zeros(3), None, "close", 1, "close", True),
        ])
        action_grip = follower.step().gripper
        self.assertEqual(action_grip, "close")
        np.testing.assert_allclose(
            controller.last_forward[0], [0.0, 0.0, 0.0], atol=1e-9
        )

    def test_disabled_pacing_snaps_ik_target(self):
        controller = _Controller([0.0, 0.0, 0.0])
        follower = EEPathFollower(controller, position_tolerance=0.001)
        terminal = np.array([0.3, 0.0, 0.0])
        follower.set_path([
            Waypoint(terminal, None, "open", 0, "hover"),
        ])
        follower.step()
        np.testing.assert_allclose(
            controller.last_forward[0], terminal, atol=1e-9
        )
        self.assertEqual(controller.last_forward[2], "open")


if __name__ == "__main__":
    unittest.main()
