"""Geometric Design D: grasp_width_m → primary gripper close command (rad).

Reads ``config/part_local_aabb_extents.json`` (local-root AABB + derived
``grasp_width_m``) and maps jaw gap to joint angle via the file's measured
piecewise-linear ``aperture_calibration``. Legacy two-point linear
calibrations remain supported for test fixtures and older config files.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_DEFAULT_JSON = (
    Path(__file__).resolve().parents[2] / "config" / "part_local_aabb_extents.json"
)


def extents_json_path() -> Path:
    env = os.environ.get("ROCO_PART_AABB_JSON", "").strip()
    return Path(env) if env else _DEFAULT_JSON


@lru_cache(maxsize=4)
def load_aabb_extents(path: Optional[str] = None) -> Dict[str, Any]:
    p = Path(path) if path else extents_json_path()
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def clear_aabb_cache() -> None:
    load_aabb_extents.cache_clear()


def list_grasp_parts(path: Optional[str] = None) -> Tuple[str, ...]:
    data = load_aabb_extents(path)
    parts = data.get("parts") or {}
    return tuple(sorted(parts.keys()))


def grasp_width_m(name: str, path: Optional[str] = None) -> Optional[float]:
    data = load_aabb_extents(path)
    part = (data.get("parts") or {}).get(name)
    if not part:
        return None
    if "grasp_width_m" in part:
        return float(part["grasp_width_m"])
    return None


def _calibration(data: Dict[str, Any]) -> Dict[str, Any]:
    cal = dict(data.get("aperture_calibration") or {})
    return {
        "model": str(cal.get("model", "linear")),
        "q_open_rad": float(cal.get("q_open_rad", 0.6649704)),
        "gap_at_q_open_m": float(cal.get("gap_at_q_open_m", 0.141577671)),
        "q_closed_rad": float(cal.get("q_closed_rad", 0.0)),
        "gap_at_q_closed_m": float(cal.get("gap_at_q_closed_m", 0.0)),
        "clearance_m": float(cal.get("clearance_m", 0.0)),
        "margin_m": float(cal.get("margin_m", 0.0)),
        "samples": tuple(cal.get("samples") or ()),
    }


def _piecewise_arrays(cal: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    samples = cal.get("samples") or ()
    if not samples:
        return None
    if len(samples) < 2:
        raise ValueError("aperture_calibration.samples requires at least 2 points")
    q = np.asarray([float(s["q_rad"]) for s in samples], dtype=np.float64)
    gap = np.asarray([float(s["gap_m"]) for s in samples], dtype=np.float64)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(gap)):
        raise ValueError("aperture calibration samples must be finite")
    if np.any(np.diff(q) <= 0.0) or np.any(np.diff(gap) <= 0.0):
        raise ValueError("aperture calibration q_rad and gap_m must increase")
    return q, gap


def grasp_width_to_close_rad(
    width_m: float,
    *,
    path: Optional[str] = None,
    margin_m: Optional[float] = None,
    clearance_m: Optional[float] = None,
) -> float:
    """Map jaw gap (m) → primary gripper joint angle (rad).

    Uses measured piecewise interpolation when calibration samples are
    present, otherwise the legacy two-point linear mapping.
    """
    data = load_aabb_extents(path)
    cal = _calibration(data)
    q0 = cal["q_closed_rad"]
    q1 = cal["q_open_rad"]
    g0 = cal["gap_at_q_closed_m"]
    g1 = cal["gap_at_q_open_m"]
    margin = cal["margin_m"] if margin_m is None else float(margin_m)
    clearance = (
        cal["clearance_m"] if clearance_m is None else float(clearance_m)
    )
    w = max(0.0, float(width_m) + clearance - margin)
    piecewise = _piecewise_arrays(cal)
    if piecewise is not None:
        q_samples, gap_samples = piecewise
        return float(np.interp(w, gap_samples, q_samples))
    if g1 <= g0 + 1e-12:
        return float(q0)
    t = (w - g0) / (g1 - g0)
    q = q0 + t * (q1 - q0)
    return float(np.clip(q, min(q0, q1), max(q0, q1)))


def close_rad_to_aperture_m(q_rad: float, *, path: Optional[str] = None) -> float:
    """Map primary joint angle to the measured physical distal jaw gap."""
    cal = _calibration(load_aabb_extents(path))
    q = float(q_rad)
    piecewise = _piecewise_arrays(cal)
    if piecewise is not None:
        q_samples, gap_samples = piecewise
        return float(np.interp(q, q_samples, gap_samples))
    q0 = cal["q_closed_rad"]
    q1 = cal["q_open_rad"]
    g0 = cal["gap_at_q_closed_m"]
    g1 = cal["gap_at_q_open_m"]
    if q1 <= q0 + 1e-12:
        return float(g0)
    t = (q - q0) / (q1 - q0)
    gap = g0 + t * (g1 - g0)
    return float(np.clip(gap, min(g0, g1), max(g0, g1)))


def part_grasp_close_rad_from_width(
    name: str, *, path: Optional[str] = None
) -> Optional[float]:
    """Close rad for ``name`` from geometric ``grasp_width_m``, or None."""
    w = grasp_width_m(name, path=path)
    if w is None:
        return None
    # ``grasp_width_m`` + ``aperture_calibration`` are the runtime source of
    # truth.  ``grasp_close_rad`` in the JSON is a precomputed audit value;
    # using it here would make calibration tuning silently ineffective.
    return grasp_width_to_close_rad(w, path=path)


def resolve_grasp_close_rad(
    name: str,
    *,
    path: Optional[str] = None,
    fallback_rad: Optional[float] = None,
) -> float:
    """Geometric close rad, else ``fallback_rad``, else 0."""
    q = part_grasp_close_rad_from_width(name, path=path)
    if q is not None:
        return float(q)
    if fallback_rad is not None:
        return float(fallback_rad)
    return 0.0
