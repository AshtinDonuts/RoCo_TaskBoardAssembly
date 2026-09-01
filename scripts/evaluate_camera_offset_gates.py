#!/usr/bin/env python3
"""Offline verification gates for CameraOffsetScriptedPolicy (no Isaac Sim).

Covers plan §6 items that can run without the simulator:

- nominal observation → near-zero offsets
- synthetic held-out translations (default 100) via Jacobian warps
- deliberate ±1 cm boundary corners on both axes
- identical-frame determinism
- optional real seed-*.npz / results JSON accuracy report

Usage:

  uv run python scripts/evaluate_camera_offset_gates.py \\
      --reference task/policies/camera_reference \\
      --nominal-observation artifacts/randomization-final-frames/reference/nominal-observation.npz \\
      --observations artifacts/randomization-final-frames/observations \\
      --results artifacts/randomization-final-frames/results \\
      --json-out artifacts/camera_offset_gates.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from policies.camera_offset.constants import (  # noqa: E402
    SUPPORT_COUPLED_PARTS,
    XY_MAX_M,
    XY_MIN_M,
)
from policies.camera_offset.estimator import OffsetEstimator  # noqa: E402
from policies.camera_offset.geometry import (  # noqa: E402
    clamp_xy,
    world_xy_to_pixel_delta,
)
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402
from policies.camera_offset.estimator import OffsetEstimate  # noqa: E402


CONNECTOR_PARTS = ("usb_a", "hdmi", "pin")


def _stats(vals_m):
    a = np.asarray(vals_m, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mae_mm": float(np.mean(a) * 1e3),
        "p95_mm": float(np.percentile(a, 95) * 1e3),
        "worst_mm": float(np.max(a) * 1e3),
    }


def _shift_rgbd(rgb, depth, du, dv):
    du_i, dv_i = int(round(du)), int(round(dv))
    return (
        np.roll(np.roll(rgb, dv_i, axis=0), du_i, axis=1),
        np.roll(np.roll(depth, dv_i, axis=0), du_i, axis=1),
    )


def _estimate(bundle, rgb, depth, K, n_frames=1):
    est = OffsetEstimator(bundle)
    n = int(n_frames)
    for _ in range(max(1, n)):
        est.add_frame(rgb, depth, K)
    return est.estimate()


def _estimate_board_only(bundle, rgb, depth, K):
    """Fast path for rigid synthetic warps: board registration only."""
    est = OffsetEstimator(bundle)
    est.add_frame(rgb, depth, K)
    board_xy, board_c, board_src = est._estimate_board(rgb, depth)
    parts = {name: board_xy.copy() for name in bundle.parts}
    conf = {name: float(board_c) for name in bundle.parts}
    return OffsetEstimate(
        board_xy=clamp_xy(board_xy),
        part_xy=parts,
        board_confidence=float(board_c),
        part_confidence=conf,
        diagnostics={"board_source": board_src, "mode": "board_only"},
    )


def _load_gt(path: Path) -> dict:
    payload = json.loads(path.read_text())
    meta = payload.get("xy_randomization") or payload.get("metadata", {}).get(
        "xy_randomization"
    )
    if not meta:
        raise ValueError(f"no xy_randomization in {path}")
    return {
        "board": np.asarray(meta["board_offset"], dtype=np.float64)[:2],
        "parts": {
            k: np.asarray(v, dtype=np.float64)[:2]
            for k, v in meta["part_offsets"].items()
        },
    }


def evaluate_nominal(bundle, rgb, depth, K, tol_mm=1.5):
    out = _estimate(bundle, rgb, depth, K)
    board_err = float(np.linalg.norm(out.board_xy))
    part_errs = {
        name: float(np.linalg.norm(out.part_offset(name)[:2]))
        for name in bundle.parts
    }
    worst = max([board_err, *part_errs.values()])
    return {
        "pass": worst * 1e3 <= tol_mm,
        "tol_mm": tol_mm,
        "board_err_mm": board_err * 1e3,
        "part_err_mm": {k: v * 1e3 for k, v in part_errs.items()},
        "worst_mm": worst * 1e3,
    }


def evaluate_determinism(bundle, rgb, depth, K):
    a = _estimate(bundle, rgb, depth, K)
    b = _estimate(bundle, rgb.copy(), depth.copy(), K.copy())
    same_board = np.array_equal(a.board_xy, b.board_xy)
    same_parts = all(
        np.array_equal(a.part_xy[n], b.part_xy[n]) for n in a.part_xy
    )
    return {
        "pass": bool(same_board and same_parts),
        "board_equal": bool(same_board),
        "parts_equal": bool(same_parts),
    }


def evaluate_synthetic(bundle, rgb, depth, K, n_seeds=100, seed=0):
    rng = np.random.RandomState(int(seed))
    board_err = []
    part_err = []
    connector_err = []
    rows = []
    for i in range(int(n_seeds)):
        board_xy = rng.uniform(XY_MIN_M, XY_MAX_M, size=2)
        # Global image warp uses the board Jacobian (synthetic gate).
        du_dv = world_xy_to_pixel_delta(board_xy, bundle.jacobian_xy_per_px)
        cur_rgb, cur_depth = _shift_rgbd(rgb, depth, du_dv[0], du_dv[1])
        out = _estimate_board_only(bundle, cur_rgb, cur_depth, K)
        be = float(np.linalg.norm(out.board_xy - board_xy))
        board_err.append(be)
        row = {"i": i, "board_err_m": be, "gt_board": board_xy.tolist()}
        for name in bundle.parts:
            # Rigid warp: every part shares the board translation.
            gt = board_xy
            pe = float(np.linalg.norm(out.part_offset(name)[:2] - gt))
            part_err.append(pe)
            row[f"{name}_err_m"] = pe
            if name in CONNECTOR_PARTS:
                connector_err.append(pe)
        rows.append(row)
    return {
        "n_seeds": int(n_seeds),
        "board": _stats(board_err),
        "parts": _stats(part_err),
        "connectors": _stats(connector_err),
        "pass_board_mae_mm_lt_1": bool(_stats(board_err).get("mae_mm", 9) < 1.0),
        "pass_connector_mae_mm_lt_1": bool(
            _stats(connector_err).get("mae_mm", 9) < 1.0
        ),
        "rows_sample": rows[:5],
    }


def evaluate_corners(bundle, rgb, depth, K):
    corners = [
        (XY_MIN_M, XY_MIN_M),
        (XY_MIN_M, XY_MAX_M),
        (XY_MAX_M, XY_MIN_M),
        (XY_MAX_M, XY_MAX_M),
        (XY_MIN_M, 0.0),
        (XY_MAX_M, 0.0),
        (0.0, XY_MIN_M),
        (0.0, XY_MAX_M),
    ]
    board_err = []
    connector_err = []
    details = []
    for xy in corners:
        board_xy = np.asarray(xy, dtype=np.float64)
        du_dv = world_xy_to_pixel_delta(board_xy, bundle.jacobian_xy_per_px)
        cur_rgb, cur_depth = _shift_rgbd(rgb, depth, du_dv[0], du_dv[1])
        out = _estimate_board_only(bundle, cur_rgb, cur_depth, K)
        be = float(np.linalg.norm(out.board_xy - board_xy))
        board_err.append(be)
        crow = {"gt": board_xy.tolist(), "board_err_mm": be * 1e3}
        for name in CONNECTOR_PARTS:
            if name not in bundle.parts:
                continue
            pe = float(np.linalg.norm(out.part_offset(name)[:2] - board_xy))
            connector_err.append(pe)
            crow[f"{name}_err_mm"] = pe * 1e3
        details.append(crow)
    return {
        "board": _stats(board_err),
        "connectors": _stats(connector_err),
        "pass_worst_board_mm_lt_2": bool(_stats(board_err).get("worst_mm", 9) < 2.0),
        "details": details,
    }


def evaluate_real_seeds(bundle, obs_dir: Path, res_dir: Path):
    board_err = []
    part_err = []
    connector_err = []
    conf_fail = 0
    rows = []
    for npz_path in sorted(obs_dir.glob("*.npz")):
        json_path = res_dir / f"{npz_path.stem}.json"
        if not json_path.is_file():
            continue
        snap = np.load(npz_path)
        out = _estimate(
            bundle, snap["head_rgb"], snap["head_depth"], snap["head_intrinsics"],
            n_frames=1,
        )
        gt = _load_gt(json_path)
        be = float(np.linalg.norm(out.board_xy - gt["board"]))
        board_err.append(be)
        row = {"file": npz_path.name, "board_err_mm": be * 1e3}
        if out.board_confidence < 0.15:
            conf_fail += 1
        for name, xy in gt["parts"].items():
            pe = float(np.linalg.norm(out.part_offset(name)[:2] - xy))
            part_err.append(pe)
            row[f"{name}_err_mm"] = pe * 1e3
            if name in CONNECTOR_PARTS:
                connector_err.append(pe)
            if out.part_confidence.get(name, 1.0) < 0.15:
                conf_fail += 1
        rows.append(row)
    return {
        "board": _stats(board_err),
        "parts": _stats(part_err),
        "connectors": _stats(connector_err),
        "confidence_failures": conf_fail,
        "n_files": len(rows),
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--nominal-observation", required=True)
    parser.add_argument("--observations", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--synthetic-seeds", type=int, default=100)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    bundle = ReferenceBundle.load(args.reference)
    nom = np.load(args.nominal_observation)
    rgb, depth, K = nom["head_rgb"], nom["head_depth"], nom["head_intrinsics"]

    report = {
        "nominal": evaluate_nominal(bundle, rgb, depth, K),
        "determinism": evaluate_determinism(bundle, rgb, depth, K),
        "synthetic_held_out": evaluate_synthetic(
            bundle, rgb, depth, K,
            n_seeds=args.synthetic_seeds, seed=args.synthetic_seed,
        ),
        "boundary_corners": evaluate_corners(bundle, rgb, depth, K),
        "support_coupled_parts": sorted(SUPPORT_COUPLED_PARTS),
    }
    if args.observations and args.results:
        report["real_seeds"] = evaluate_real_seeds(
            bundle, Path(args.observations), Path(args.results)
        )

    gates = {
        "nominal_near_zero": report["nominal"]["pass"],
        "determinism": report["determinism"]["pass"],
        "synthetic_board_mae_lt_1mm": report["synthetic_held_out"][
            "pass_board_mae_mm_lt_1"
        ],
        "synthetic_connector_mae_lt_1mm": report["synthetic_held_out"][
            "pass_connector_mae_mm_lt_1"
        ],
        "corner_board_worst_lt_2mm": report["boundary_corners"][
            "pass_worst_board_mm_lt_2"
        ],
    }
    report["gates"] = gates
    report["all_offline_gates_pass"] = all(gates.values())

    text = json.dumps(
        {k: report[k] for k in report if k != "real_seeds"},
        indent=2,
    )
    # Print a compact summary then full JSON path.
    print("Offline camera-offset gates:")
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(
        f"  synthetic board MAE "
        f"{report['synthetic_held_out']['board'].get('mae_mm', float('nan')):.3f} mm"
    )
    print(
        f"  synthetic connector MAE "
        f"{report['synthetic_held_out']['connectors'].get('mae_mm', float('nan')):.3f} mm"
    )
    if "real_seeds" in report:
        rs = report["real_seeds"]
        print(
            f"  real seeds (n={rs.get('n_files', 0)}): "
            f"board MAE {rs['board'].get('mae_mm', float('nan')):.2f} mm, "
            f"connector MAE {rs['connectors'].get('mae_mm', float('nan')):.2f} mm, "
            f"parts worst {rs['parts'].get('worst_mm', float('nan')):.2f} mm"
        )
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"wrote {out_path}")
    return 0 if report["all_offline_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
