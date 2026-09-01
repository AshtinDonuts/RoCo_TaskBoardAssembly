#!/usr/bin/env python3
"""Single-frame (no mean-pool) offset estimation + red/green annotation.

Replays the first buffered head frame from a prior multi-frame capture
directory, estimates offsets without temporal averaging, and writes the
same annotation products as the mean-pool experiment.

Usage:

  python3 scripts/run_single_frame_offset_experiment.py \\
      --source-dir artifacts/offset_estimation_frames_avg \\
      --output-dir artifacts/offset_estimation_frames_single
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

# Force single-frame path before estimating.
import policies.camera_offset.constants as C  # noqa: E402
import policies.camera_offset.estimator as E  # noqa: E402

C.AVERAGE_BUFFERED_FRAMES = False
E.AVERAGE_BUFFERED_FRAMES = False

from policies.camera_offset.estimator import OffsetEstimator  # noqa: E402
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402

# Reuse annotation helpers.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from annotate_offset_detections import (  # noqa: E402
    annotate_image,
    detected_uv,
)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def process_source_run(
    src: Path,
    dst: Path,
    bundle: ReferenceBundle,
    *,
    frame_index: int,
) -> dict:
    man = json.loads((src / "manifest.json").read_text())
    frames_meta = man["frames"]
    if not frames_meta:
        raise RuntimeError(f"{src}: no frames in manifest")
    idx = int(frame_index)
    if idx < 0:
        idx = len(frames_meta) + idx
    if idx < 0 or idx >= len(frames_meta):
        raise IndexError(
            f"{src}: frame_index={frame_index} out of range "
            f"(n={len(frames_meta)})"
        )
    fr = frames_meta[idx]
    dst.mkdir(parents=True, exist_ok=True)

    rgb = _load_rgb(src / fr["rgb"])
    depth = np.load(src / fr["depth_m"]) if fr.get("depth_m") else None
    shutil.copy2(src / "reference_rgb.png", dst / "reference_rgb.png")
    if (src / "reference_depth_m.npy").is_file():
        shutil.copy2(src / "reference_depth_m.npy", dst / "reference_depth_m.npy")
    if (src / "reference_depth_viz.png").is_file():
        shutil.copy2(
            src / "reference_depth_viz.png", dst / "reference_depth_viz.png"
        )

    est = OffsetEstimator(bundle)
    # Override readiness requirement: estimate from this one frame only.
    est.add_frame(
        rgb, depth, bundle.intrinsics, sim_step_idx=fr.get("sim_step_idx")
    )
    estimate = est.estimate(force=True)
    assert estimate.diagnostics.get("compose") == "per_frame_median"
    est.export_buffered_frames(dst, estimate=estimate)

    # Annotate using the single observation frame (not a mean composite).
    part_xy = estimate.part_xy
    board_xy = estimate.board_xy
    conf = float(estimate.board_confidence)

    nom_only = annotate_image(
        _load_rgb(dst / "reference_rgb.png"),
        bundle,
        part_xy,
        board_xy,
        title=f"{dst.name} | nominal reference",
        show_nominal=True,
        show_detected=False,
    )
    det_img = annotate_image(
        rgb,
        bundle,
        part_xy,
        board_xy,
        title=(
            f"{dst.name} | single-frame detect  "
            f"step={fr.get('sim_step_idx')} conf={conf:.2f}"
        ),
        show_nominal=True,
        show_detected=True,
    )
    gap = 8
    panel = Image.new(
        "RGB",
        (nom_only.width + det_img.width + gap,
         max(nom_only.height, det_img.height)),
        (30, 30, 30),
    )
    panel.paste(nom_only, (0, 0))
    panel.paste(det_img, (nom_only.width + gap, 0))
    det_img.save(dst / "detections_annotated.png")
    nom_only.save(dst / "nominal_annotated.png")
    panel.save(dst / "detections_panel.png")

    summary = {
        "run": dst.name,
        "mode": "single_frame",
        "compose": estimate.diagnostics.get("compose"),
        "source_dir": str(src),
        "source_frame_index": idx,
        "sim_step_idx": fr.get("sim_step_idx"),
        "board_xy": [float(board_xy[0]), float(board_xy[1])],
        "board_confidence": conf,
        "part_xy": {
            k: [float(v[0]), float(v[1])] for k, v in part_xy.items()
        },
        "detections_uv": {
            name: [float(u), float(v)]
            for name, (u, v) in (
                (n, detected_uv(bundle, n, part_xy[n])) for n in part_xy
            )
        },
        "outputs": {
            "detections_annotated": str(dst / "detections_annotated.png"),
            "nominal_annotated": str(dst / "nominal_annotated.png"),
            "detections_panel": str(dst / "detections_panel.png"),
        },
    }
    (dst / "detections_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"[single] {dst.name}: step={fr.get('sim_step_idx')} "
        f"board={summary['board_xy']} conf={conf:.3f} -> {dst}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "offset_estimation_frames_avg",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "offset_estimation_frames_single",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=TASK_DIR / "policies" / "camera_reference",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Which buffered frame to use (0=first / earliest sim step).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--gt-board",
        action="append",
        default=[],
        metavar="NAME=X,Y",
        help="Optional ground-truth board XY, e.g. seed-001=0.00024,0.00901",
    )
    args = parser.parse_args()

    gt = {}
    for item in args.gt_board:
        name, _, xy = item.partition("=")
        xs = [float(v) for v in xy.split(",")]
        gt[name] = xs

    # Defaults matching the prior live experiment logs.
    gt.setdefault("nominal", [0.0, 0.0])
    gt.setdefault("seed-001", [0.00024, 0.00901])

    bundle = ReferenceBundle.load(args.reference)
    if args.runs:
        src_dirs = [args.source_dir / n for n in args.runs]
    else:
        src_dirs = sorted(
            p for p in args.source_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").is_file()
        )
    if not src_dirs:
        raise SystemExit(f"no source runs under {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    index = []
    for src in src_dirs:
        dst = args.output_dir / src.name
        summary = process_source_run(
            src, dst, bundle, frame_index=args.frame_index
        )
        summaries.append(summary)
        board_gt = gt.get(src.name)
        err = None
        if board_gt is not None:
            est = summary["board_xy"]
            err = float(
                np.hypot(est[0] - board_gt[0], est[1] - board_gt[1])
            )
        index.append(
            {
                "run": src.name,
                "mode": "single_frame",
                "sim_step_idx": summary["sim_step_idx"],
                "compose": summary["compose"],
                "board_xy_est": summary["board_xy"],
                "board_xy_gt": board_gt,
                "board_err_m": err,
                "board_confidence": summary["board_confidence"],
            }
        )
        err_mm = None if err is None else round(err * 1000.0, 2)
        print(f"  err_mm={err_mm} gt={board_gt}")

    (args.output_dir / "detections_index.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )

    # Side-by-side comparison vs mean-pool index if present.
    mean_index = args.source_dir / "index.json"
    if mean_index.is_file():
        mean = {
            row["run"]: row for row in json.loads(mean_index.read_text())
        }
        print("\n[compare] single-frame vs mean-pool board error")
        print(f"{'run':12s} {'single_mm':>10s} {'mean_mm':>10s} {'d_mm':>8s}")
        for row in index:
            name = row["run"]
            s = row["board_err_m"]
            m = (mean.get(name) or {}).get("board_err_m")
            s_mm = None if s is None else s * 1000.0
            m_mm = None if m is None else m * 1000.0
            d = None if (s_mm is None or m_mm is None) else (s_mm - m_mm)
            print(
                f"{name:12s} "
                f"{(f'{s_mm:.2f}' if s_mm is not None else '-'):>10s} "
                f"{(f'{m_mm:.2f}' if m_mm is not None else '-'):>10s} "
                f"{(f'{d:+.2f}' if d is not None else '-'):>8s}"
            )

    print(f"\n[single] outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
