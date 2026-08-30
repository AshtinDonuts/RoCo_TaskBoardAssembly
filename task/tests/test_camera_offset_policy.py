"""Unit tests for camera-only XY offset estimation (no Isaac Sim)."""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_TASK_DIR = str(Path(__file__).resolve().parents[1])
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from policies.camera_offset.constants import (  # noqa: E402
    SUPPORT_COUPLED_PARTS,
    XY_MAX_M,
    XY_MIN_M,
)
from policies.camera_offset.estimator import OffsetEstimator  # noqa: E402
from policies.camera_offset.geometry import (  # noqa: E402
    clamp_xy,
    pixel_delta_to_world_xy,
    world_xy_to_pixel_delta,
)
from policies.camera_offset.matching import (  # noqa: E402
    ncc_search,
    phase_correlate,
    pick_best_candidate,
    subpixel_quadratic,
)
from policies.camera_offset.reference import (  # noqa: E402
    PartTemplate,
    make_bundle,
)
from policies.camera_offset.targets import (  # noqa: E402
    adjust_part_target,
    estimated_part_offset,
)
from policies.camera_offset_scripted import CameraOffsetScriptedPolicy  # noqa: E402
from policy_api import EnvInfo, Observation, PartTarget  # noqa: E402


def _blob(h, w, cy, cx, radius, value, bg=30):
    img = np.full((h, w), bg, dtype=np.float64)
    yy, xx = np.ogrid[:h, :w]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = value
    return img


def _rgb(gray):
    g = np.clip(gray, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _synthetic_bundle():
    h, w = 96, 96
    jac = np.array([[0.001, 0.0], [0.0, 0.001]], dtype=np.float64)
    board = _blob(h, w, 48, 48, 28, 180, bg=20)
    # Unique corner so translation is unambiguous.
    board[20:28, 20:28] = 255
    board[70:78, 60:72] = 90
    part_a = _blob(h, w, 40, 30, 6, 0, bg=0)
    part_b = _blob(h, w, 55, 70, 5, 0, bg=0)
    gray = np.where(part_a > 0, 40, board)
    gray = np.where(part_b > 0, 220, gray)
    rgb = _rgb(gray)
    depth = np.full((h, w), 0.40, dtype=np.float64)
    depth[part_a > 0] = 0.38
    depth[part_b > 0] = 0.37
    K = np.array([[400.0, 0.0, 48.0], [0.0, 400.0, 48.0], [0.0, 0.0, 1.0]])
    board_mask = (board > 50) & (part_a == 0) & (part_b == 0)
    half = 8
    parts = {}
    for name, (cy, cx) in (("usb_a", (40, 30)), ("hdmi", (55, 70)),
                           ("gear_60teeth", (48, 48))):
        y0, x0 = int(cy - half), int(cx - half)
        parts[name] = PartTemplate(
            name=name,
            rgb=rgb[y0:y0 + 2 * half, x0:x0 + 2 * half].copy(),
            depth=depth[y0:y0 + 2 * half, x0:x0 + 2 * half].copy(),
            mask=np.ones((2 * half, 2 * half), dtype=bool),
            search_center_uv=np.array([cx, cy], dtype=np.float64),
        )
    return make_bundle(
        rgb=rgb,
        depth=depth,
        intrinsics=K,
        board_mask=board_mask,
        jacobian_xy_per_px=jac,
        board_center_uv=np.array([48.0, 48.0]),
        parts=parts,
        buffer_frames=1,
    ), gray, depth


def _shift_image(arr, du, dv):
    return np.roll(np.roll(arr, int(dv), axis=0), int(du), axis=1)


class GeometryTests(unittest.TestCase):
    def test_pixel_world_roundtrip(self):
        jac = np.array([[0.002, 0.0], [0.0, -0.0015]])
        world = np.array([0.006, -0.003])
        pix = world_xy_to_pixel_delta(world, jac)
        back = pixel_delta_to_world_xy(pix, jac)
        np.testing.assert_allclose(back, world)

    def test_clamp_keeps_interior_and_clips_overshoot(self):
        np.testing.assert_allclose(clamp_xy([0.004, -0.007]), [0.004, -0.007])
        clipped = clamp_xy([XY_MAX_M + 2e-4, XY_MIN_M - 2e-4])
        np.testing.assert_allclose(clipped, [XY_MAX_M, XY_MIN_M])
        far = clamp_xy([0.05, -0.05])
        np.testing.assert_allclose(far, [XY_MAX_M, XY_MIN_M])


class MatchingTests(unittest.TestCase):
    def test_phase_correlate_recovers_integer_shift(self):
        base = _blob(64, 64, 20, 22, 8, 200)
        du, dv = 5, -3
        moved = _shift_image(base, du, dv)
        delta, score = phase_correlate(base, moved)
        self.assertGreater(score, 1.0)
        np.testing.assert_allclose(delta, [du, dv], atol=0.25)

    def test_ncc_recovers_template_shift(self):
        img = _blob(80, 80, 40, 40, 7, 210)
        tmpl = img[32:48, 32:48]
        shifted = _shift_image(img, 4, -2)
        result = ncc_search(
            shifted, tmpl, search_origin_uv=(40, 40), search_half=(12, 12)
        )
        self.assertTrue(result["valid"])
        self.assertGreater(result["score"], 0.9)
        np.testing.assert_allclose([result["du"], result["dv"]], [4, -2], atol=0.3)

    def test_subpixel_quadratic_peak(self):
        s = np.array([[1.0, 2.0, 1.5], [2.0, 4.0, 3.0], [1.0, 2.5, 1.0]])
        du, dv = subpixel_quadratic(s, (1, 1))
        self.assertTrue(np.isfinite(du) and np.isfinite(dv))

    def test_tie_break_is_lexicographic_after_score_depth_distance(self):
        # score, dist, y, x
        cands = [
            (0.9, 3.0, 10, 8),
            (0.9, 3.0, 10, 4),
            (0.9, 3.0, 11, 1),
        ]
        best = pick_best_candidate(cands, [0.1, 0.1, 0.1])
        self.assertEqual((best[3], best[4]), (10, 4))


class TargetAdjustTests(unittest.TestCase):
    def _target(self):
        return PartTarget(
            name="usb_a",
            release_mode="snap",
            pick_pos=np.array([1.0, 2.0, 3.0]),
            place_pos=np.array([4.0, 5.0, 6.0]),
            grade_pos=np.array([7.0, 8.0, 9.0]),
            snap_target_pos=np.array([10.0, 11.0, 12.0]),
            extra={
                "pick_pos": np.array([1.0, 2.0, 3.0]),
                "place_pos": np.array([4.0, 5.0, 6.0]),
                "grade_pos": np.array([7.0, 8.0, 9.0]),
                "snap": {
                    "target_pos": (10.0, 11.0, 12.0),
                    "connect_pos": [13.0, 14.0, 15.0],
                    "search": {"n": 5},
                },
                "ee_offset": np.array([0.0, 0.0, 0.2]),
            },
        )

    def test_adjust_does_not_mutate_and_shifts_snap_fields(self):
        target = self._target()
        original = copy.deepcopy(target)
        adjusted = adjust_part_target(
            target,
            board_offset=[0.01, -0.005],
            part_offset=[-0.002, 0.003],
        )
        np.testing.assert_allclose(adjusted.pick_pos, [0.998, 2.003, 3.0])
        np.testing.assert_allclose(adjusted.place_pos, [4.01, 4.995, 6.0])
        np.testing.assert_allclose(adjusted.grade_pos, [7.01, 7.995, 9.0])
        np.testing.assert_allclose(adjusted.snap_target_pos, [10.01, 10.995, 12.0])
        self.assertEqual(adjusted.extra["snap"]["target_pos"], (10.01, 10.995, 12.0))
        self.assertEqual(adjusted.extra["snap"]["connect_pos"], [13.01, 13.995, 15.0])
        np.testing.assert_array_equal(target.pick_pos, original.pick_pos)
        self.assertEqual(target.extra["snap"]["target_pos"], original.extra["snap"]["target_pos"])
        np.testing.assert_array_equal(adjusted.extra["ee_offset"], [0.0, 0.0, 0.2])
        self.assertEqual(adjusted.extra["snap"]["search"], {"n": 5})

    def test_support_coupled_parts_use_board_offset(self):
        board = np.array([0.004, -0.006])
        parts = {"usb_a": np.array([0.001, 0.002])}
        for name in SUPPORT_COUPLED_PARTS:
            off = estimated_part_offset(name, parts, board)
            np.testing.assert_allclose(off[:2], board)
        off = estimated_part_offset("usb_a", parts, board)
        np.testing.assert_allclose(off[:2], [0.001, 0.002])


class ReferenceBundleTests(unittest.TestCase):
    def test_save_load_roundtrip(self):
        bundle, _, _ = _synthetic_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            bundle.save(tmp)
            loaded = type(bundle).load(tmp)
        np.testing.assert_array_equal(loaded.rgb, bundle.rgb)
        self.assertEqual(loaded.content_hash, bundle.content_hash)
        self.assertIn("usb_a", loaded.parts)


class EstimatorTests(unittest.TestCase):
    def test_recovers_known_translation_and_is_deterministic(self):
        bundle, gray, depth = _synthetic_bundle()
        du, dv = 4, -3
        rgb_cur = _rgb(_shift_image(gray, du, dv))
        depth_cur = _shift_image(depth, du, dv)
        est = OffsetEstimator(bundle)
        est.add_frame(rgb_cur, depth_cur, bundle.intrinsics)
        a = est.estimate()
        est2 = OffsetEstimator(bundle)
        est2.add_frame(rgb_cur.copy(), depth_cur.copy(), bundle.intrinsics.copy())
        b = est2.estimate()
        expected = pixel_delta_to_world_xy([du, dv], bundle.jacobian_xy_per_px)
        np.testing.assert_allclose(a.board_xy, expected, atol=0.0015)
        np.testing.assert_array_equal(a.board_xy, b.board_xy)
        np.testing.assert_array_equal(a.part_xy["usb_a"], b.part_xy["usb_a"])
        np.testing.assert_allclose(a.part_xy["gear_60teeth"], a.board_xy)

    def test_missing_head_frame_fails_clearly(self):
        bundle, _, _ = _synthetic_bundle()
        est = OffsetEstimator(bundle)
        with self.assertRaises(RuntimeError):
            est.estimate()
        with self.assertRaises(RuntimeError):
            bundle.assert_observation_shape(None)


class PolicyWrapperTests(unittest.TestCase):
    def _env(self):
        return EnvInfo(
            dof_names=["j1", "j2", "grip"],
            L_arm_joints=["j1", "j2"],
            R_arm_joints=[],
            L_gripper_joint="grip",
            L_arm_init_q=np.zeros(2),
            physics_dt=0.005,
            enable_camera_output=True,
        )

    def _obs(self, rgb, depth, K):
        return Observation(
            step_idx=0,
            joint_positions=np.array([0.1, 0.2, 0.0]),
            joint_velocities=np.zeros(3),
            L_gripper_position=0.0,
            ee_pose_L=(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
            rgb={"head": rgb, "L_wrist": None, "R_wrist": None},
            depth={"head": depth, "L_wrist": None, "R_wrist": None},
            intrinsics={"head": K, "L_wrist": None, "R_wrist": None},
        )

    def test_missing_rgb_raises(self):
        bundle, gray, depth = _synthetic_bundle()
        fake = SimpleNamespace(reset_targets=[], done=False)

        class Baseline:
            def reset(self, obs, target):
                fake.reset_targets.append(target)

            def act(self, obs):
                return "act"

            def is_done(self, obs):
                return False

        policy = CameraOffsetScriptedPolicy(
            self._env(), bundle=bundle, baseline=Baseline()
        )
        obs = self._obs(None, depth, bundle.intrinsics)
        target = PartTarget(name="usb_a", release_mode="open")
        with self.assertRaises(RuntimeError):
            policy.reset(obs, target)

    def test_camera_disabled_raises(self):
        bundle, _, _ = _synthetic_bundle()
        env = self._env()
        env.enable_camera_output = False
        with self.assertRaises(RuntimeError):
            CameraOffsetScriptedPolicy(
                env, bundle=bundle, baseline=SimpleNamespace()
            )

    def test_delegates_adjusted_target_and_identical_frames_match(self):
        bundle, gray, depth = _synthetic_bundle()
        rgb = _rgb(_shift_image(gray, 3, 1))
        depth_s = _shift_image(depth, 3, 1)
        seen = []

        class Baseline:
            def reset(self, obs, target):
                seen.append(copy.deepcopy(target))

            def act(self, obs):
                return "go"

            def is_done(self, obs):
                return True

        env = self._env()
        target = PartTarget(
            name="usb_a",
            release_mode="open",
            pick_pos=np.array([0.1, 0.2, 1.0]),
            place_pos=np.array([0.3, 0.4, 1.0]),
        )
        p1 = CameraOffsetScriptedPolicy(env, bundle=bundle, baseline=Baseline())
        p2 = CameraOffsetScriptedPolicy(env, bundle=bundle, baseline=Baseline())
        obs = self._obs(rgb, depth_s, bundle.intrinsics)
        p1.reset(obs, target)
        a1 = p1.act(obs)
        p2.reset(obs, target)
        a2 = p2.act(obs)
        self.assertEqual(a1, a2)
        self.assertEqual(len(seen), 2)
        np.testing.assert_array_equal(seen[0].pick_pos, seen[1].pick_pos)
        np.testing.assert_array_equal(seen[0].place_pos, seen[1].place_pos)
        self.assertFalse(np.allclose(seen[0].place_pos, target.place_pos))
        np.testing.assert_array_equal(target.pick_pos, [0.1, 0.2, 1.0])


if __name__ == "__main__":
    unittest.main()
