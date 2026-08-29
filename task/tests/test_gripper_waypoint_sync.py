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

    module_name = "task.controllers._ee_pose_controller_sync_test"
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


class _EndEffector:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float64)

    def get_world_pose(self):
        return self.position.copy(), np.array([1.0, 0.0, 0.0, 0.0])


class _Controller:
    def __init__(self, position):
        self._n_dof = 3
        self.end_effector = _EndEffector(position)

    def reset(self):
        pass

    def forward(self, target_position, target_orientation, gripper_cmd):
        return types.SimpleNamespace(
            joint_positions=[0.0, 0.0, gripper_cmd],
            gripper=gripper_cmd,
        )


class GripperWaypointSyncTests(unittest.TestCase):
    def test_close_waits_for_actual_pick_pose_then_dwells(self):
        controller = _Controller([1.0, 0.0, 0.0])
        follower = EEPathFollower(controller, position_tolerance=0.001)
        follower.set_path([
            Waypoint(np.zeros(3), None, None, 0, "descend_pick"),
            Waypoint(np.zeros(3), None, "close", 2, "close", True),
        ])

        for _ in range(5):
            action = follower.step()
            self.assertIsNone(action.gripper)
            self.assertEqual(follower.current_index(), 0)

        controller.end_effector.position[:] = 0.0
        action = follower.step()
        self.assertEqual(action.gripper, "close")
        self.assertEqual(follower.current_index(), 1)

        for _ in range(2):
            self.assertEqual(follower.step().gripper, "close")
            self.assertFalse(follower.is_done())
        follower.step()
        self.assertTrue(follower.is_done())

    def test_open_waits_for_actual_place_pose(self):
        controller = _Controller([1.0, 0.0, 0.0])
        follower = EEPathFollower(controller, position_tolerance=0.001)
        follower.set_path([
            Waypoint(np.zeros(3), None, None, 0, "descend_place"),
            Waypoint(np.zeros(3), None, "open", 0, "open", True),
        ])

        for _ in range(5):
            self.assertIsNone(follower.step().gripper)

        controller.end_effector.position[:] = 0.0
        self.assertEqual(follower.step().gripper, "open")

    def test_snap_mode_open_waits_for_snap_confirmation(self):
        snapped = False

        def snap_fired():
            return snapped

        controller = _Controller([0.0, 0.0, 0.0])
        follower = EEPathFollower(controller, position_tolerance=0.001)
        follower.set_path([
            Waypoint(
                np.zeros(3), None, None, 0, "descend_place",
                False, None, snap_fired,
            ),
            Waypoint(np.zeros(3), None, "open", 0, "open", True),
        ])

        for _ in range(5):
            self.assertIsNone(follower.step().gripper)
            self.assertEqual(follower.current_index(), 0)

        snapped = True
        self.assertEqual(follower.step().gripper, "open")


if __name__ == "__main__":
    unittest.main()
