"""Camera-aware wrapper around the scripted baseline.

Estimates the fairness XY board/part offsets from the head RGB-D stream
and a packaged nominal reference, then hands an adjusted private
``PartTarget`` copy to ``BaselinePolicy``. Grasp heights, orientations,
gripper values, snap search, IK, and grading are unchanged.

Run with:

    TASK_ENABLE_CAMERA_OUTPUT=1 uv run python task/run_pick_place.py \\
        --policy policies.camera_offset_scripted.CameraOffsetScriptedPolicy \\
        --random-seed N
"""
from __future__ import annotations

import os.path
import sys
from types import SimpleNamespace

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from policies.camera_offset.estimator import OffsetEstimator  # noqa: E402
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402
from policies.camera_offset.targets import (  # noqa: E402
    adjust_part_target,
    estimated_part_offset,
)
from policy_api import EnvInfo, Observation, PartTarget, Policy  # noqa: E402


def _hold_action(env_info: EnvInfo, obs: Observation):
    n = len(env_info.dof_names)
    joints = [None] * n
    q = obs.joint_positions
    for name in env_info.L_arm_joints:
        idx = env_info.dof_names.index(name)
        joints[idx] = float(q[idx])
    gidx = env_info.dof_names.index(env_info.L_gripper_joint)
    joints[gidx] = float(obs.L_gripper_position)
    try:
        from omni.isaac.core.utils.types import ArticulationAction
        return ArticulationAction(joint_positions=joints)
    except ImportError:
        return SimpleNamespace(joint_positions=joints)


class CameraOffsetScriptedPolicy(Policy):
    """Estimate XY offsets from the head camera, then run BaselinePolicy."""

    def __init__(
        self,
        env_info: EnvInfo,
        *,
        bundle: ReferenceBundle | None = None,
        estimator: OffsetEstimator | None = None,
        baseline: Policy | None = None,
        reference_dir=None,
    ) -> None:
        super().__init__(env_info)
        if bundle is None:
            path = reference_dir or ReferenceBundle.default_dir()
            bundle = ReferenceBundle.load(path)
        self.bundle = bundle
        self._estimator = estimator or OffsetEstimator(bundle)
        if baseline is None:
            from policies.baseline_scripted import BaselinePolicy
            baseline = BaselinePolicy(env_info)
        self._baseline = baseline
        self._estimate = None
        self._pending_target: PartTarget | None = None
        self._baseline_started = False
        self._missing_rgb_steps = 0
        # Head cameras often need a few sim steps before the first RGB arrives.
        self._missing_rgb_limit = 600
        if not getattr(env_info, "enable_camera_output", True):
            raise RuntimeError(
                "CameraOffsetScriptedPolicy requires camera output. "
                "Set TASK_ENABLE_CAMERA_OUTPUT=1 before evaluation."
            )

    def reset(self, obs: Observation, target: PartTarget) -> None:
        self._pending_target = target
        self._baseline_started = False
        if self._estimate is None:
            if self._ingest(obs) and self._estimator.ready():
                self._estimate = self._estimator.estimate()
        if self._estimate is not None:
            self._start_baseline(obs, target)

    def act(self, obs: Observation):
        if self._estimate is None:
            if not self._ingest(obs):
                return _hold_action(self.env_info, obs)
            if self._estimator.ready():
                self._estimate = self._estimator.estimate()
                if self._pending_target is None:
                    raise RuntimeError("offset estimate completed before Policy.reset")
                self._start_baseline(obs, self._pending_target)
            else:
                return _hold_action(self.env_info, obs)
        if not self._baseline_started:
            if self._pending_target is None:
                raise RuntimeError("Policy.act called before Policy.reset")
            self._start_baseline(obs, self._pending_target)
        return self._baseline.act(obs)

    def is_done(self, obs: Observation) -> bool:
        if self._estimate is None or not self._baseline_started:
            return False
        return self._baseline.is_done(obs)

    @property
    def current_waypoint(self):
        if hasattr(self._baseline, "current_waypoint"):
            return self._baseline.current_waypoint
        return None

    @property
    def current_index(self) -> int:
        if hasattr(self._baseline, "current_index"):
            return self._baseline.current_index
        return 0

    def _ingest(self, obs: Observation) -> bool:
        """Buffer a head frame. Returns False while the stream is still warming up."""
        rgb = None if obs.rgb is None else obs.rgb.get("head")
        depth = None if obs.depth is None else obs.depth.get("head")
        K = None if obs.intrinsics is None else obs.intrinsics.get("head")
        if rgb is None:
            self._missing_rgb_steps += 1
            if self._missing_rgb_steps >= self._missing_rgb_limit:
                raise RuntimeError(
                    "CameraOffsetScriptedPolicy requires Observation.rgb['head']. "
                    "Set TASK_ENABLE_CAMERA_OUTPUT=1 and wait for the head stream."
                )
            return False
        self._missing_rgb_steps = 0
        self._estimator.add_frame(rgb, depth, K)
        return True

    def _start_baseline(self, obs: Observation, target: PartTarget) -> None:
        if self._estimate is None:
            raise RuntimeError("cannot start baseline before offset estimate")
        part_off = estimated_part_offset(
            target.name, self._estimate.part_xy, self._estimate.board_xy
        )
        board_off = self._estimate.board_xy
        adjusted = adjust_part_target(target, board_off, part_off[:2])
        self._baseline.reset(obs, adjusted)
        self._baseline_started = True
