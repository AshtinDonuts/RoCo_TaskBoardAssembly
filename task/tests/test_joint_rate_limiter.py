import unittest
from types import SimpleNamespace

import numpy as np

from task.policies._joint_rate_limiter import JointPositionRateLimiter


class JointPositionRateLimiterTests(unittest.TestCase):
    def test_limits_selected_joints_in_both_directions(self):
        limiter = JointPositionRateLimiter([1, 3], max_delta=0.005)
        limiter.reset([10.0, 0.0, 20.0, 1.0, 30.0])

        first = SimpleNamespace(
            joint_positions=[10.0, 0.2, 20.0, 0.8, 30.0],
            joint_velocities=[1, 2, 3, 4, 5],
        )
        self.assertIs(limiter.apply(first), first)
        np.testing.assert_allclose(
            first.joint_positions, [10.0, 0.005, 20.0, 0.995, 30.0]
        )
        self.assertEqual(first.joint_velocities, [1, 2, 3, 4, 5])

        second = SimpleNamespace(
            joint_positions=[11.0, 0.2, 21.0, 0.8, 31.0]
        )
        limiter.apply(second)
        np.testing.assert_allclose(
            second.joint_positions, [11.0, 0.01, 21.0, 0.99, 31.0]
        )

    def test_passes_through_unselected_and_none_targets(self):
        limiter = JointPositionRateLimiter([0, 2], max_delta=0.005)
        limiter.reset([0.5, 1.5, 2.5, 3.5])
        action = SimpleNamespace(
            joint_positions=[None, -9.0, 3.0, 0.42]
        )

        limiter.apply(action)

        self.assertIsNone(action.joint_positions[0])
        self.assertEqual(action.joint_positions[1], -9.0)
        self.assertAlmostEqual(action.joint_positions[2], 2.505)
        self.assertEqual(action.joint_positions[3], 0.42)

    def test_reset_uses_current_observed_positions(self):
        limiter = JointPositionRateLimiter([1], max_delta=0.005)
        limiter.reset([0.0, 1.0])
        first = SimpleNamespace(joint_positions=[None, 2.0])
        limiter.apply(first)
        self.assertAlmostEqual(first.joint_positions[1], 1.005)

        limiter.reset([0.0, -1.0])
        after_reset = SimpleNamespace(joint_positions=[None, 2.0])
        limiter.apply(after_reset)
        self.assertAlmostEqual(after_reset.joint_positions[1], -0.995)

    def test_action_without_positions_passes_through(self):
        limiter = JointPositionRateLimiter([0], max_delta=0.005)
        limiter.reset([0.0])
        action = SimpleNamespace(joint_positions=None, marker="unchanged")
        self.assertIs(limiter.apply(action), action)
        self.assertEqual(action.marker, "unchanged")

    def test_per_call_delta_override(self):
        limiter = JointPositionRateLimiter([0], max_delta=0.5)
        limiter.reset([0.0])
        action = SimpleNamespace(joint_positions=[1.0])
        limiter.apply(action, max_delta=0.02)
        self.assertAlmostEqual(action.joint_positions[0], 0.02)


if __name__ == "__main__":
    unittest.main()
