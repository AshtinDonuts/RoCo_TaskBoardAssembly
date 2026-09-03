import unittest

from task.param_config import FULL_PART_ORDER
from task.eval_context import previous_part, successful_target_spec


class EvalContextTests(unittest.TestCase):
    ORDER = FULL_PART_ORDER

    def test_canonical_order_has_all_nine_subtasks(self):
        self.assertEqual(len(self.ORDER), 9)
        self.assertEqual(self.ORDER[0], "gear_60teeth")
        self.assertEqual(self.ORDER[-1], "battery_size5")

    def test_previous_part_uses_immediate_ordered_predecessor(self):
        self.assertIsNone(previous_part(self.ORDER[0], self.ORDER))
        self.assertEqual(previous_part("hdmi", self.ORDER), "usb_a")
        self.assertEqual(previous_part("battery_size1", self.ORDER), "pin")

    def test_previous_part_rejects_unknown_part(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            previous_part("unknown", self.ORDER)

    def test_snap_success_uses_final_connect_pose(self):
        spec = successful_target_spec({
            "release_mode": "snap",
            "snap": {
                "target_pos": (1.0, 2.0, 3.0),
                "target_rot": (1.0, 0.0, 0.0, 0.0),
                "connect_pos": (4.0, 5.0, 6.0),
                "connect_rot": (0.0, 1.0, 0.0, 0.0),
                "parent_body_path": "/World/board",
            },
        })
        self.assertEqual(spec["position"], (4.0, 5.0, 6.0))
        self.assertEqual(spec["rotation"], (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(spec["measure"], "mesh_pose")
        self.assertTrue(spec["fixed_joint"])

    def test_open_success_uses_settled_grade_target(self):
        spec = successful_target_spec({
            "release_mode": "open",
            "place_pos": (1.0, 2.0, 3.0),
            "grade_pos": (4.0, 5.0, 6.0),
            "grade_use_aabb": True,
        })
        self.assertEqual(spec["position"], (4.0, 5.0, 6.0))
        self.assertEqual(spec["measure"], "aabb_midpoint")
        self.assertFalse(spec["fixed_joint"])


if __name__ == "__main__":
    unittest.main()
