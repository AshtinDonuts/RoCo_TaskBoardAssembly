"""Compare camera-offset estimates to post-run evaluator JSON (offline only)."""
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

from policies.camera_offset.constants import SUPPORT_COUPLED_PARTS  # noqa: E402
from policies.camera_offset.estimator import OffsetEstimator  # noqa: E402
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402

CONNECTOR_PARTS = ("usb_a", "hdmi", "pin")


def _load_offsets(results_path: Path) -> dict:
    payload = json.loads(results_path.read_text())
    meta = payload.get("xy_randomization") or payload.get("metadata", {}).get(
        "xy_randomization"
    )
    if not meta:
        raise ValueError(f"no xy_randomization in {results_path}")
    board = np.asarray(meta["board_offset"], dtype=np.float64)[:2]
    parts = {
        name: np.asarray(xy, dtype=np.float64)[:2]
        for name, xy in meta["part_offsets"].items()
    }
    return {"board": board, "parts": parts, "raw": payload}


def _stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mae_mm": float(np.mean(a) * 1e3),
        "p95_mm": float(np.percentile(a, 95) * 1e3),
        "worst_mm": float(np.max(a) * 1e3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--observations", required=True, help="dir of seed-*.npz")
    parser.add_argument("--results", required=True, help="dir of seed-*.json")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    bundle = ReferenceBundle.load(args.reference)
    obs_dir = Path(args.observations)
    res_dir = Path(args.results)
    board_err = []
    part_err = []
    connector_err = []
    per_part = {}
    conf_fail = 0
    rows = []
    for npz_path in sorted(obs_dir.glob("*.npz")):
        stem = npz_path.stem
        json_path = res_dir / f"{stem}.json"
        if not json_path.is_file():
            alt = res_dir / npz_path.name.replace(".npz", ".json")
            json_path = alt if alt.is_file() else json_path
        if not json_path.is_file():
            print(f"skip {npz_path.name}: no matching results JSON")
            continue
        snap = np.load(npz_path)
        est = OffsetEstimator(bundle)
        est.add_frame(snap["head_rgb"], snap["head_depth"], snap["head_intrinsics"])
        out = est.estimate()
        gt = _load_offsets(json_path)
        be = np.linalg.norm(out.board_xy - gt["board"])
        board_err.append(be)
        if out.board_confidence < 0.15:
            conf_fail += 1
        row = {
            "file": npz_path.name,
            "board_err_m": float(be),
            "board_confidence": float(out.board_confidence),
            "board_source": (
                out.diagnostics.get("frames", [{}])[-1].get("board_source")
            ),
        }
        for name, xy in gt["parts"].items():
            pred = out.part_offset(name)[:2]
            pe = float(np.linalg.norm(pred - xy))
            part_err.append(pe)
            per_part.setdefault(name, []).append(pe)
            row[f"{name}_err_m"] = pe
            row[f"{name}_confidence"] = float(out.part_confidence.get(name, 0.0))
            if name in CONNECTOR_PARTS:
                connector_err.append(pe)
            if name not in SUPPORT_COUPLED_PARTS and out.part_confidence.get(name, 1.0) < 0.15:
                conf_fail += 1
        rows.append(row)
        print(f"{npz_path.name}: board MAE {be*1e3:.2f} mm")

    summary = {
        "board": _stats(board_err),
        "parts": _stats(part_err),
        "connectors": _stats(connector_err),
        "per_part_mm": {k: _stats(v) for k, v in sorted(per_part.items())},
        "confidence_failures": conf_fail,
        "rows": rows,
    }
    print(json.dumps(
        {k: summary[k] for k in ("board", "parts", "connectors", "per_part_mm",
                                 "confidence_failures")},
        indent=2,
    ))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
