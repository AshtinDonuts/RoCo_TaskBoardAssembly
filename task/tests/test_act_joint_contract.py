import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_policy_module():
    policy_api = types.ModuleType("policy_api")
    policy_api.EnvInfo = policy_api.Observation = policy_api.PartTarget = object

    class Policy:
        def __init__(self, env_info):
            self.env_info = env_info

    policy_api.Policy = Policy
    previous = sys.modules.get("policy_api")
    sys.modules["policy_api"] = policy_api
    try:
        path = Path(__file__).resolve().parents[1] / "policies/act_lerobot.py"
        spec = importlib.util.spec_from_file_location("act_joint_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("policy_api", None)
        else:
            sys.modules["policy_api"] = previous


act = _load_policy_module()


class ACTJointContractTest(unittest.TestCase):
    def _policy(self):
        policy = act.ACTLeRobotPolicy.__new__(act.ACTLeRobotPolicy)
        policy._schema = "joint"
        policy._state_dim = act.JOINT_STATE_DIM
        policy._action_dim = act.JOINT_ACTION_DIM
        policy._Li = list(range(2, 9))
        policy._Lg = 18
        policy.env_info = SimpleNamespace(dof_names=[f"dof_{i}" for i in range(20)])
        return policy

    def test_state_is_left_joint_only(self):
        policy = self._policy()
        q = np.arange(20, dtype=np.float64)
        qd = q + 100
        state = policy._build_state(SimpleNamespace(joint_positions=q, joint_velocities=qd))
        np.testing.assert_array_equal(state, [
            2, 3, 4, 5, 6, 7, 8,
            102, 103, 104, 105, 106, 107, 108,
            18,
        ])

    def test_joint_action_owns_only_left_arm_and_gripper(self):
        class ArticulationAction:
            def __init__(self, joint_positions):
                self.joint_positions = joint_positions

        modules = {
            "omni": types.ModuleType("omni"),
            "omni.isaac": types.ModuleType("omni.isaac"),
            "omni.isaac.core": types.ModuleType("omni.isaac.core"),
            "omni.isaac.core.utils": types.ModuleType("omni.isaac.core.utils"),
            "omni.isaac.core.utils.types": types.ModuleType("omni.isaac.core.utils.types"),
        }
        modules["omni.isaac.core.utils.types"].ArticulationAction = ArticulationAction
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            positions = self._policy()._joint_action(np.arange(8)).joint_positions
        finally:
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
        self.assertEqual([positions[i] for i in range(2, 9)], list(map(float, range(7))))
        self.assertEqual(positions[18], 7.0)
        self.assertTrue(all(value is None for i, value in enumerate(positions) if i not in {*range(2, 9), 18}))

    def test_invalid_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite shape"):
            self._policy()._joint_action(np.zeros(7))
        bad = np.zeros(8)
        bad[3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite shape"):
            self._policy()._joint_action(bad)

    def test_right_camera_payload_is_black_and_actions_are_held(self):
        policy = self._policy()
        policy._last_action = None
        policy._next_infer_step = 0
        sent = []
        policy._send = sent.append
        policy._recv = lambda: {"action": list(np.arange(8, dtype=float))}
        policy._joint_action = lambda values: tuple(np.asarray(values).tolist())
        rgb = {
            "head": np.full((240, 320, 3), 1, np.uint8),
            "L_wrist": np.full((240, 320, 3), 2, np.uint8),
            "R_wrist": np.full((240, 320, 3), 255, np.uint8),
        }
        q = np.arange(20, dtype=np.float64)
        obs = SimpleNamespace(step_idx=100, joint_positions=q, joint_velocities=q, rgb=rgb)
        first = policy.act(obs)
        self.assertEqual(len(sent), 1)
        self.assertFalse(sent[0]["right"].any())
        obs.step_idx = 119
        held = policy.act(obs)
        self.assertEqual(first, held)
        self.assertEqual(len(sent), 1)
        obs.step_idx = 120
        policy.act(obs)
        self.assertEqual(len(sent), 2)


if __name__ == "__main__":
    unittest.main()
