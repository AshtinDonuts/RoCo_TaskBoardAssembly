#!/usr/bin/env python3
"""Export head images used by OffsetEstimator from observation NPZs.

Replays each NPZ through the packaged reference + estimator (single-frame
buffer) and writes RGB/depth PNGs tagged by buffer index and the recorded
sim ``step_idx``, plus the nominal reference and a per-run manifest.

Usage:

  python3 scripts/export_offset_estimation_frames.py \\
      --observations artifacts/randomization-final-frames/observations \\
      --output artifacts/offset_estimation_frames \\
      --limit 5
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

from policies.camera_offset.estimator import OffsetEstimator  # noqa: E402
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402


def _load_obs(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    needed = ("head_rgb",)
    missing = [k for k in needed if k not in data.files]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    return {k: data[k] for k in data.files}


def _export_one(bundle: ReferenceBundle, obs_path: Path, out_dir: Path) -> dict:
    payload = _load_obs(obs_path)
    rgb = payload["head_rgb"]
    depth = payload.get("head_depth")
    K = payload.get("head_intrinsics", bundle.intrinsics)
    step = payload.get("step_idx")
    sim_step = None if step is None else int(np.asarray(step).reshape(-1)[0])

    # Offline snapshots are single frames; estimate() uses whatever is buffered.
    est = OffsetEstimator(bundle)
    est.add_frame(rgb, depth, K, sim_step_idx=sim_step)
    estimate = est.estimate()
    dest = out_dir / obs_path.stem
    est.export_buffered_frames(dest, estimate=estimate)
    summary = {
        "observation": str(obs_path),
        "export_dir": str(dest),
        "sim_step_idx": sim_step,
        "board_xy": [float(estimate.board_xy[0]), float(estimate.board_xy[1])],
        "board_confidence": float(estimate.board_confidence),
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=TASK_DIR / "policies" / "camera_reference",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "randomization-final-frames" / "observations"
        ),
    )
    parser.add_argument(
        "--nominal-observation",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "randomization-final-frames"
            / "reference" / "nominal-observation.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "offset_estimation_frames",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Max seeded NPZs to export (0 = all).")
    parser.add_argument("--include-nominal", action="store_true", default=True)
    parser.add_argument("--no-include-nominal", action="store_false",
                        dest="include_nominal")
    args = parser.parse_args()

    bundle = ReferenceBundle.load(args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []

    if args.include_nominal and args.nominal_observation.is_file():
        print(f"[export] nominal <- {args.nominal_observation}")
        summaries.append(_export_one(bundle, args.nominal_observation, args.output))

    obs_paths = sorted(args.observations.glob("seed-*.npz"))
    if args.limit and args.limit > 0:
        obs_paths = obs_paths[: int(args.limit)]
    for path in obs_paths:
        print(f"[export] {path.name}")
        summaries.append(_export_one(bundle, path, args.output))

    index_path = args.output / "index.json"
    index_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"[export] wrote {len(summaries)} run(s) -> {args.output}")
    print(f"[export] index -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
