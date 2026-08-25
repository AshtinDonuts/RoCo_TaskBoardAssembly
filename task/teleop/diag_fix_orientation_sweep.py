"""Deterministic Cartesian retarget sweeps: fix_orientation vs free vs lock.

No Isaac required. Proves whether ``fix_orientation`` / ``fixed_orientation_wxyz``
restrict **commanded** XYZ (they should not) vs only freezing orientation.

Sweeps a synthetic leader through a planar XY grid (then a Z column) with
three retarget configs and prints commanded DexMate AABB + final quat.

    python3 task/teleop/diag_fix_orientation_sweep.py
    python3 -m teleop.diag_fix_orientation_sweep   # from task/
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:
    sys.path.insert(0, _TASK_DIR)

from teleop.retarget import CartesianRetargeter, RetargetConfig  # noqa: E402
from teleop import transforms as T  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


_TOP_DOWN = (0.0, 1.0, 0.0, 0.0)
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _load_yaml_retarget() -> Dict:
    path = os.path.join(
        os.path.dirname(_TASK_DIR), "config", "aloha_solo_to_vega_1u.yaml"
    )
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return dict(raw.get("retarget") or {})


def _make_cfg(
    base: Dict,
    *,
    fix_orientation: bool = False,
    fixed_orientation_wxyz: Optional[Tuple[float, ...]] = None,
) -> RetargetConfig:
    data = dict(base)
    data["fix_orientation"] = fix_orientation
    data["fixed_orientation_wxyz"] = fixed_orientation_wxyz
    # Disable proximity so the sweep is path-independent / absolute.
    data["proximity_delta_gain"] = {"enabled": False}
    data["proximity_rate_limit"] = {"enabled": False}
    # Fast enough that one step reaches the absolute map target.
    data["max_lin_vel"] = 10.0
    data["max_ang_vel"] = 10.0
    data["max_lin_acc"] = 0.0
    return RetargetConfig.from_dict(data)


def _sweep(
    name: str,
    cfg: RetargetConfig,
    dex_origin: np.ndarray,
    dex_quat0: np.ndarray,
    leader_xy: np.ndarray,
    leader_z: float = 0.0,
) -> Dict:
    r = CartesianRetargeter(cfg)
    r.capture_origins(
        [0.0, 0.0, 0.0],
        _IDENTITY,
        dex_origin,
        dex_quat0,
    )
    positions: List[np.ndarray] = []
    quats: List[np.ndarray] = []
    for xy in leader_xy:
        pos, quat, _, info = r.step(
            leader_pos=[float(xy[0]), float(xy[1]), leader_z],
            leader_quat=_IDENTITY,  # wrist never tilts; orientation lock is the var
            gripper_norm=0.5,
            dt=0.05,
            clutch=True,
            deadman=True,
            current_dex_pos=dex_origin,
            current_dex_quat=dex_quat0,
        )
        positions.append(np.asarray(pos, dtype=np.float64))
        quats.append(np.asarray(quat, dtype=np.float64))
        assert info["reason"] in ("tracking", "clutch_engage"), info

    P = np.stack(positions, axis=0)
    Q = np.stack(quats, axis=0)
    # Quat uniqueness (up to sign).
    q_ref = Q[0]
    q_spread = float(
        np.max(
            [
                np.linalg.norm(T.quat_wxyz_to_rotvec(
                    T.quat_multiply_wxyz(
                        T.quat_conjugate_wxyz(q_ref),
                        T.normalize_quat_wxyz(q),
                    )
                ))
                for q in Q
            ]
        )
    )
    return {
        "name": name,
        "n": len(P),
        "pos_min": P.min(axis=0),
        "pos_max": P.max(axis=0),
        "pos_span": P.max(axis=0) - P.min(axis=0),
        "quat_first": Q[0],
        "quat_last": Q[-1],
        "quat_angle_spread_rad": q_spread,
        "fix_orientation": cfg.fix_orientation,
        "fixed_orientation_wxyz": cfg.fixed_orientation_wxyz,
    }


def main() -> None:
    base = _load_yaml_retarget()
    # Home-like DexMate origin near board (stage meters).
    dex_origin = np.array([0.25, -0.15, 1.05], dtype=np.float64)
    # Non-top-down "engage" quat — what fix_orientation would freeze.
    engage_quat = T.normalize_quat_wxyz(
        T.rotvec_to_quat_wxyz(np.array([0.4, -0.6, 0.2]))
    )

    # Leader XY grid in meters (before translation_gain / axes_map).
    xs = np.linspace(-0.08, 0.08, 9)
    ys = np.linspace(-0.08, 0.08, 9)
    leader_xy = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    modes = [
        (
            "free_6dof",
            _make_cfg(base, fix_orientation=False, fixed_orientation_wxyz=None),
        ),
        (
            "fix_orientation",
            _make_cfg(base, fix_orientation=True, fixed_orientation_wxyz=None),
        ),
        (
            "fixed_top_down",
            _make_cfg(
                base,
                fix_orientation=False,
                fixed_orientation_wxyz=_TOP_DOWN,
            ),
        ),
    ]

    print("=== deterministic retarget XYZ sweeps ===")
    print(f"leader XY grid: {len(xs)}x{len(ys)} over ±0.08 m")
    print(f"dex_origin={dex_origin.tolist()}")
    print(f"engage_quat (simulated home)={np.round(engage_quat, 4).tolist()}")
    if base.get("axes_map") is not None:
        print("axes_map: from aloha_solo_to_vega_1u.yaml")
    print(f"translation_gain={base.get('translation_gain', 1.0)}")
    print()

    rows = []
    for name, cfg in modes:
        # Warm fixed_top_down onto lock so span isn't polluted by slew frames.
        if cfg.fixed_orientation_wxyz is not None:
            warm = CartesianRetargeter(cfg)
            warm.capture_origins(
                [0, 0, 0], _IDENTITY, dex_origin, engage_quat
            )
            pos, quat = dex_origin, engage_quat
            for _ in range(120):
                pos, quat, _, _ = warm.step(
                    leader_pos=[0, 0, 0],
                    leader_quat=_IDENTITY,
                    gripper_norm=0.5,
                    dt=0.05,
                    clutch=True,
                    deadman=True,
                    current_dex_pos=pos,
                    current_dex_quat=quat,
                )
            # Re-seed sweep from locked orientation at origin.
            r = _sweep(name, cfg, dex_origin, quat, leader_xy)
        else:
            r = _sweep(name, cfg, dex_origin, engage_quat, leader_xy)
        rows.append(r)
        print(f"[{r['name']}]")
        print(
            f"  pos_span xyz (m) = {np.round(r['pos_span'], 4).tolist()}  "
            f"AABB [{np.round(r['pos_min'], 4).tolist()} .. "
            f"{np.round(r['pos_max'], 4).tolist()}]"
        )
        print(
            f"  quat_angle_spread = {r['quat_angle_spread_rad']:.4f} rad  "
            f"last={np.round(r['quat_last'], 4).tolist()}"
        )

    free_span = rows[0]["pos_span"]
    fix_span = rows[1]["pos_span"]
    lock_span = rows[2]["pos_span"]
    print("\n=== verdict (retarget layer) ===")
    print(
        "If fix_orientation / fixed_top_down pos_span ≈ free_6dof, "
        "retarget is NOT clamping XYZ. Restricted board approach under "
        "fix_orientation is then Lula IK failing for the frozen engage "
        "quat (run find_reachable_r_arm_orn_compare.py)."
    )
    print(f"  free_6dof     span={np.round(free_span, 4).tolist()}")
    print(f"  fix_orientation span={np.round(fix_span, 4).tolist()}")
    print(f"  fixed_top_down  span={np.round(lock_span, 4).tolist()}")
    span_err = float(np.max(np.abs(fix_span - free_span)))
    print(f"  max |fix−free| span component = {span_err:.6f} m")
    if span_err < 1e-6:
        print("  → commanded XYZ workspace identical; orientation lock only.")
    else:
        print("  → unexpected XYZ difference — inspect workspace clamp / map.")


if __name__ == "__main__":
    main()
