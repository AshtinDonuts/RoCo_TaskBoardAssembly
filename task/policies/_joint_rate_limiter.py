"""Small stateful rate limiter for scripted joint-position commands."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class JointPositionRateLimiter:
    """Limit selected joint targets without changing other action fields."""

    def __init__(self, joint_indices: Sequence[int], max_delta: float) -> None:
        self._joint_indices = tuple(int(i) for i in joint_indices)
        self._max_delta = float(max_delta)
        if not np.isfinite(self._max_delta) or self._max_delta <= 0.0:
            raise ValueError(f"max_delta must be finite and > 0, got {max_delta!r}")
        self._previous = None

    def reset(self, joint_positions) -> None:
        """Seed the next command from the currently observed articulation."""
        positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        self._previous = np.asarray(
            [positions[i] for i in self._joint_indices], dtype=np.float64
        )
        if not np.all(np.isfinite(self._previous)):
            raise ValueError("observed joint positions must be finite")

    def apply(self, action, max_delta=None):
        """Rate-limit selected entries of ``action.joint_positions`` in place.

        Entries outside ``joint_indices`` are untouched. A selected entry set
        to ``None`` is also untouched and does not change the stored command.
        ``max_delta`` optionally overrides the constructor limit for this call,
        allowing callers to convert a velocity bound using the measured control
        interval.
        """
        if action is None:
            return None
        joint_positions = getattr(action, "joint_positions", None)
        if joint_positions is None:
            return action
        if self._previous is None:
            raise RuntimeError("reset() must be called before apply()")

        limit = self._max_delta if max_delta is None else float(max_delta)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(f"max_delta must be finite and > 0, got {limit!r}")

        limited = list(joint_positions)
        for state_i, action_i in enumerate(self._joint_indices):
            desired = limited[action_i]
            if desired is None:
                continue
            desired = float(desired)
            if not np.isfinite(desired):
                continue
            previous = float(self._previous[state_i])
            step = float(np.clip(
                desired - previous, -limit, limit
            ))
            command = previous + step
            limited[action_i] = command
            self._previous[state_i] = command

        action.joint_positions = limited
        return action
