"""Evaluation-time XY randomization helpers.

This module intentionally has no simulator imports so the sampling and
configuration rules can be tested independently of Isaac Sim.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


XY_LIMIT_M = 0.01
SUPPORT_COUPLED_PARTS = frozenset({"gear_60teeth", "rod_16mm", "bolt_8mm"})


def _shift_xy(value, offset):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if arr.size < 2:
        raise ValueError(f"position must have at least two values, got {value!r}")
    arr[:2] += np.asarray(offset, dtype=np.float64)[:2]
    return arr


def _shift_xy_preserving_container(value, offset):
    shifted = _shift_xy(value, offset)
    if shifted is None:
        return None
    if isinstance(value, tuple):
        return tuple(float(x) for x in shifted)
    if isinstance(value, list):
        return [float(x) for x in shifted]
    return shifted


@dataclass(frozen=True)
class XYRandomization:
    """One immutable-in-use trial randomization record."""

    seed: int
    board_offset: np.ndarray
    part_offsets: Mapping[str, np.ndarray]

    @classmethod
    def sample(cls, seed: int, part_names: Iterable[str]) -> "XYRandomization":
        rng = np.random.default_rng(int(seed))
        board_xy = rng.uniform(-XY_LIMIT_M, XY_LIMIT_M, size=2)
        board_offset = np.array([board_xy[0], board_xy[1], 0.0], dtype=np.float64)

        offsets = {}
        # Sorting makes the mapping independent of incidental part-order
        # changes while keeping the seed-to-trial mapping reproducible.
        for name in sorted({str(n) for n in part_names}):
            if name in SUPPORT_COUPLED_PARTS:
                offsets[name] = board_offset.copy()
                continue
            xy = rng.uniform(-XY_LIMIT_M, XY_LIMIT_M, size=2)
            offsets[name] = np.array([xy[0], xy[1], 0.0], dtype=np.float64)
        return cls(seed=int(seed), board_offset=board_offset, part_offsets=offsets)

    def offset_for(self, part_name: str) -> np.ndarray:
        return np.asarray(
            self.part_offsets.get(part_name, np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).copy()

    def shifted_config(self, part_name: str, config: Mapping) -> dict:
        """Deep-copy and shift the position fields used by the evaluator."""
        out = copy.deepcopy(dict(config))
        part_offset = self.offset_for(part_name)
        board_offset = np.asarray(self.board_offset, dtype=np.float64)
        if out.get("pick_pos") is not None:
            out["pick_pos"] = _shift_xy(out["pick_pos"], part_offset)
        for key in ("place_pos", "grade_pos"):
            if out.get(key) is not None:
                out[key] = _shift_xy(out[key], board_offset)

        snap = out.get("snap")
        if isinstance(snap, Mapping):
            snap = dict(snap)
            for key in ("target_pos", "connect_pos"):
                if snap.get(key) is not None:
                    snap[key] = _shift_xy_preserving_container(
                        snap[key], board_offset
                    )
            out["snap"] = snap
        return out

    def as_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "board_offset": [float(x) for x in self.board_offset],
            "part_offsets": {
                name: [float(x) for x in offset]
                for name, offset in sorted(self.part_offsets.items())
            },
        }
