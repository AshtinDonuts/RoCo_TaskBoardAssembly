"""Private PartTarget adjustment: shift only randomized XY fields."""
from __future__ import annotations

import copy

import numpy as np

from .constants import SUPPORT_COUPLED_PARTS
from .geometry import as_xy


def _shift_xy_value(value, offset_xy):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if arr.size < 2:
        raise ValueError(f"position must have at least two values, got {value!r}")
    dxy = as_xy(offset_xy)
    arr[0] += dxy[0]
    arr[1] += dxy[1]
    if isinstance(value, tuple):
        return tuple(float(x) for x in arr)
    if isinstance(value, list):
        return [float(x) for x in arr]
    return arr


def support_coupled(part_name: str) -> bool:
    return str(part_name) in SUPPORT_COUPLED_PARTS


def estimated_part_offset(part_name: str, part_offsets: dict,
                          board_offset) -> np.ndarray:
    if support_coupled(part_name):
        xy = as_xy(board_offset)
        return np.array([xy[0], xy[1], 0.0], dtype=np.float64)
    offset = part_offsets.get(part_name)
    if offset is None:
        return np.zeros(3, dtype=np.float64)
    xy = as_xy(offset)
    return np.array([xy[0], xy[1], 0.0], dtype=np.float64)


def adjust_part_target(target, board_offset, part_offset):
    """Deep-copy ``target`` and add estimated XY offsets. Does not mutate input."""
    adjusted = copy.deepcopy(target)
    board_xy = as_xy(board_offset)
    part_xy = as_xy(part_offset)
    adjusted.pick_pos = _shift_xy_value(adjusted.pick_pos, part_xy)
    adjusted.place_pos = _shift_xy_value(adjusted.place_pos, board_xy)
    adjusted.grade_pos = _shift_xy_value(adjusted.grade_pos, board_xy)
    adjusted.snap_target_pos = _shift_xy_value(adjusted.snap_target_pos, board_xy)

    extra = dict(adjusted.extra) if adjusted.extra else {}
    if extra.get("pick_pos") is not None:
        extra["pick_pos"] = _shift_xy_value(extra.get("pick_pos"), part_xy)
    if extra.get("place_pos") is not None:
        extra["place_pos"] = _shift_xy_value(extra.get("place_pos"), board_xy)
    if extra.get("grade_pos") is not None:
        extra["grade_pos"] = _shift_xy_value(extra.get("grade_pos"), board_xy)
    snap = extra.get("snap")
    if isinstance(snap, dict):
        snap = dict(snap)
        snap["target_pos"] = _shift_xy_value(snap.get("target_pos"), board_xy)
        snap["connect_pos"] = _shift_xy_value(snap.get("connect_pos"), board_xy)
        extra["snap"] = snap
    adjusted.extra = extra
    return adjusted
