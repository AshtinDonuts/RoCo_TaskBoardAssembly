import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_collector_helpers():
    stubs = {}
    stubs["run_pick_place"] = types.ModuleType("run_pick_place")
    isaac_types = types.ModuleType("isaacsim.core.utils.types")
    isaac_types.ArticulationAction = object
    stubs["isaacsim"] = types.ModuleType("isaacsim")
    stubs["isaacsim.core"] = types.ModuleType("isaacsim.core")
    stubs["isaacsim.core.utils"] = types.ModuleType("isaacsim.core.utils")
    stubs["isaacsim.core.utils.types"] = isaac_types
    policy_api = types.ModuleType("policy_api")
    policy_api.EnvInfo = policy_api.Observation = policy_api.PartTarget = object
    stubs["policy_api"] = policy_api
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        path = Path(__file__).resolve().parents[1] / "collect_lerobot_v3.py"
        spec = importlib.util.spec_from_file_location("collector_contract_under_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


collector = _load_collector_helpers()


class LeRobotCollectionContractTest(unittest.TestCase):
    def test_feature_contract(self):
        features = collector._feature_spec()
        self.assertEqual(set(features), {
            "action", "observation.state", "observation.images.head",
            "observation.images.left_hand", "observation.images.right_hand",
        })
        self.assertEqual(features["action"]["shape"], (14,))
        self.assertEqual(features["observation.state"]["shape"], (44,))
        self.assertEqual(features["observation.images.head"]["shape"], (240, 320, 3))

    def test_state_order_and_gripper_normalization(self):
        q = np.arange(20, dtype=np.float32)
        qd = q + 100
        q[18] = collector.GRIPPER_OPEN_LIMIT
        q[19] = 0
        state = collector._pack_state(
            [1, 2, 3], [1, 0, 0, 0], [4, 5, 6], [1, 0, 0, 0],
            q, qd, np.arange(7), np.arange(7, 14), 18, 19,
        )
        self.assertEqual(state.shape, (44,))
        np.testing.assert_array_equal(state[:14], [1, 2, 3, 1, 0, 0, 0, 4, 5, 6, 1, 0, 0, 0])
        np.testing.assert_array_equal(state[14:28], np.arange(14))
        np.testing.assert_array_equal(state[28:42], np.arange(14) + 100)
        np.testing.assert_array_equal(state[42:], [1, 0])

    def test_action_rotation_and_sampling(self):
        action = collector._pack_cartesian_action([1, 2, 3], [1, 0, 0, 0], 0.5)
        np.testing.assert_array_equal(action, [1, 2, 3, 0, 0, 0, 0.5])
        cfg = collector._resolve_recording_config(
            SimpleNamespace(sample_stride=None, sample_hz=10.0), 200.0
        )
        self.assertEqual(cfg["sample_fps"], 10)
        self.assertAlmostEqual(cfg["sample_period_s"], 0.1)

    def test_frame_parts_form_exact_contiguous_segments(self):
        parts = ("a", "b", "c")
        segments = collector._part_segments(["a", "a", "b", "c", "c"], parts)
        self.assertEqual(segments, [
            {"part": "a", "begin": 0, "end": 2},
            {"part": "b", "begin": 2, "end": 3},
            {"part": "c", "begin": 3, "end": 5},
        ])
        with self.assertRaisesRegex(ValueError, "does not match"):
            collector._part_segments(["a", "c", "b"], parts)

    def test_waypoints_form_absolute_contiguous_phase_ranges(self):
        segments = [
            {"part": "a", "begin": 0, "end": 3},
            {"part": "b", "begin": 3, "end": 5},
        ]
        waypoints = [
            {"name": "hover_pick", "waypoint_index": 0},
            {"name": "hover_pick", "waypoint_index": 0},
            {"name": "close", "waypoint_index": 1},
            {"name": "return_home", "waypoint_index": 0},
            {"name": "hover_pick", "waypoint_index": 1},
        ]
        self.assertTrue(collector._attach_phase_segments(segments, waypoints))
        self.assertEqual(segments[0]["phases"], [
            {"name": "hover_pick", "waypoint_index": 0, "begin": 0, "end": 2},
            {"name": "close", "waypoint_index": 1, "begin": 2, "end": 3},
        ])
        self.assertEqual(segments[1]["phases"][0]["name"], "return_home")

    def test_missing_waypoint_marks_annotations_incomplete(self):
        segments = [{"part": "a", "begin": 0, "end": 1}]
        self.assertFalse(collector._attach_phase_segments(
            segments, [{"name": None, "waypoint_index": None}]
        ))

    def test_missing_image_uses_training_resolution(self):
        image = collector._as_rgb(None)
        self.assertEqual(image.shape, (240, 320, 3))
        self.assertEqual(image.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
