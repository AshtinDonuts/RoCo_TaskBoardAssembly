import copy
import unittest

import numpy as np

from task.eval_randomization import (
    SUPPORT_COUPLED_PARTS,
    XY_MAX_M,
    XY_MIN_M,
    XY_LIMIT_M,
    XYRandomization,
    resolve_policy_config,
)


class XYRandomizationTests(unittest.TestCase):
    PARTS = (
        "gear_60teeth", "gear_20teeth", "rod_16mm", "bolt_8mm", "usb_a"
    )

    def test_sampling_is_deterministic_and_bounded(self):
        a = XYRandomization.sample(7, self.PARTS)
        b = XYRandomization.sample(7, reversed(self.PARTS))
        np.testing.assert_array_equal(a.board_offset, b.board_offset)
        for name in self.PARTS:
            np.testing.assert_array_equal(a.offset_for(name), b.offset_for(name))
            self.assertTrue(np.all(np.abs(a.offset_for(name)[:2]) <= XY_LIMIT_M))
            self.assertEqual(float(a.offset_for(name)[2]), 0.0)

    def test_support_parts_share_board_offset(self):
        trial = XYRandomization.sample(11, self.PARTS)
        for name in SUPPORT_COUPLED_PARTS:
            np.testing.assert_array_equal(trial.offset_for(name), trial.board_offset)
        self.assertFalse(
            np.array_equal(trial.offset_for("gear_20teeth"), trial.board_offset)
        )
        self.assertFalse(
            np.array_equal(trial.offset_for("usb_a"), trial.board_offset)
        )

    def test_config_shift_is_deep_and_does_not_mutate_input(self):
        config = {
            "pick_pos": np.array([1.0, 2.0, 3.0]),
            "place_pos": np.array([4.0, 5.0, 6.0]),
            "grade_pos": np.array([7.0, 8.0, 9.0]),
            "snap": {
                "target_pos": (10.0, 11.0, 12.0),
                "connect_pos": (13.0, 14.0, 15.0),
                "pos_tol_axes": (0.1, 0.2, 0.3),
            },
            "ee_offset": np.array([0.0, 0.0, 0.2]),
        }
        original = copy.deepcopy(config)
        trial = XYRandomization(
            seed=0,
            board_offset=np.array([0.1, 0.2, 0.0]),
            part_offsets={"ordinary": np.array([0.3, 0.4, 0.0])},
        )
        shifted = trial.shifted_config("ordinary", config)
        np.testing.assert_array_equal(shifted["pick_pos"], [1.3, 2.4, 3.0])
        np.testing.assert_array_equal(shifted["place_pos"], [4.1, 5.2, 6.0])
        np.testing.assert_array_equal(shifted["grade_pos"], [7.1, 8.2, 9.0])
        self.assertEqual(shifted["snap"]["target_pos"], (10.1, 11.2, 12.0))
        self.assertEqual(shifted["snap"]["connect_pos"], (13.1, 14.2, 15.0))
        np.testing.assert_array_equal(config["pick_pos"], original["pick_pos"])
        np.testing.assert_array_equal(config["place_pos"], original["place_pos"])
        self.assertEqual(config["snap"], original["snap"])

    def test_resolve_policy_config_camera_only_by_default(self):
        config = {
            "pick_pos": np.array([1.0, 2.0, 3.0]),
            "place_pos": np.array([4.0, 5.0, 6.0]),
            "grade_pos": np.array([7.0, 8.0, 9.0]),
        }
        trial = XYRandomization(
            seed=0,
            board_offset=np.array([0.1, 0.2, 0.0]),
            part_offsets={"usb_a": np.array([0.3, 0.4, 0.0])},
        )
        camera_only = resolve_policy_config("usb_a", config, trial=trial)
        privileged = resolve_policy_config(
            "usb_a", config, trial=trial, privileged=True
        )
        scene = trial.shifted_config("usb_a", config)

        np.testing.assert_array_equal(privileged["pick_pos"], scene["pick_pos"])
        np.testing.assert_array_equal(privileged["place_pos"], scene["place_pos"])
        np.testing.assert_array_equal(camera_only["pick_pos"], config["pick_pos"])
        np.testing.assert_array_equal(camera_only["place_pos"], config["place_pos"])
        self.assertFalse(
            np.array_equal(camera_only["pick_pos"], scene["pick_pos"])
        )
        self.assertFalse(
            np.array_equal(camera_only["place_pos"], scene["place_pos"])
        )

    def test_result_metadata_declares_official_range(self):
        trial = XYRandomization.sample(3, self.PARTS)
        self.assertEqual(trial.as_dict()["xy_range_m"], [XY_MIN_M, XY_MAX_M])


if __name__ == "__main__":
    unittest.main()
