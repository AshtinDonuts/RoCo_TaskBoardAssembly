#!/usr/bin/env python3
"""Annotate mean-pooled offset detections vs nominal part centres.

Red  = nominal search centres / template boxes (from the reference bundle)
Green = detected centres after mean-pooled XY offset estimation

Usage:

  python3 scripts/annotate_offset_detections.py \\
      --runs-dir artifacts/offset_estimation_frames_avg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = REPO_ROOT / "task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from policies.camera_offset.constants import SUPPORT_COUPLED_PARTS  # noqa: E402
from policies.camera_offset.estimator import (  # noqa: E402
    OffsetEstimator,
    average_buffered_frames,
)
from policies.camera_offset.geometry import world_xy_to_pixel_delta  # noqa: E402
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402

RED = (255, 40, 40)
GREEN = (40, 220, 80)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _as_u8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(np.nanmax(arr)) if arr.size else 1.0
        if maxv <= 1.0 + 1e-6:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
    return np.asarray(np.rint(arr), dtype=np.uint8)


def _font(size: int = 14):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _draw_cross(draw: ImageDraw.ImageDraw, uv, color, size=8, width=2):
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    draw.line((u - size, v, u + size, v), fill=color, width=width)
    draw.line((u, v - size, u, v + size), fill=color, width=width)


def _draw_box(draw: ImageDraw.ImageDraw, uv, hw, color, width=2):
    u, v = float(uv[0]), float(uv[1])
    th, tw = int(hw[0]), int(hw[1])
    x0, y0 = int(round(u - tw / 2.0)), int(round(v - th / 2.0))
    x1, y1 = int(round(u + tw / 2.0)), int(round(v + th / 2.0))
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)


def _label(draw, xy, text, color, font):
    x, y = int(xy[0]), int(xy[1])
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=BLACK,
    )
    draw.text((x, y), text, fill=color, font=font)


def detected_uv(bundle: ReferenceBundle, name: str, part_xy) -> np.ndarray:
    tmpl = bundle.parts[name]
    jac = (
        tmpl.jacobian_xy_per_px
        if tmpl.jacobian_xy_per_px is not None
        else bundle.jacobian_xy_per_px
    )
    du_dv = world_xy_to_pixel_delta(part_xy, jac)
    return np.asarray(tmpl.search_center_uv, dtype=np.float64) + du_dv


def annotate_image(
    rgb,
    bundle: ReferenceBundle,
    part_xy: dict,
    board_xy,
    *,
    title: str,
    show_nominal: bool = True,
    show_detected: bool = True,
) -> Image.Image:
    img = Image.fromarray(_as_u8_rgb(rgb), mode="RGB")
    draw = ImageDraw.Draw(img)
    font = _font(13)
    font_sm = _font(11)

    # Board centre
    board_nom = np.asarray(bundle.board_center_uv, dtype=np.float64)
    if show_nominal:
        _draw_cross(draw, board_nom, RED, size=12, width=3)
        _label(draw, (board_nom[0] + 10, board_nom[1] - 18), "board nom", RED, font_sm)
    if show_detected and board_xy is not None:
        du_dv = world_xy_to_pixel_delta(board_xy, bundle.jacobian_xy_per_px)
        board_det = board_nom + du_dv
        _draw_cross(draw, board_det, GREEN, size=12, width=3)
        _label(
            draw,
            (board_det[0] + 10, board_det[1] + 6),
            "board det",
            GREEN,
            font_sm,
        )
        if show_nominal:
            draw.line(
                (
                    int(board_nom[0]), int(board_nom[1]),
                    int(board_det[0]), int(board_det[1]),
                ),
                fill=GREEN,
                width=2,
            )

    for name, tmpl in sorted(bundle.parts.items()):
        nom = np.asarray(tmpl.search_center_uv, dtype=np.float64)
        hw = tmpl.rgb.shape[:2]
        if show_nominal:
            _draw_box(draw, nom, hw, RED, width=2)
            _draw_cross(draw, nom, RED, size=6, width=2)
            _label(draw, (nom[0] - hw[1] / 2, nom[1] - hw[0] / 2 - 14),
                   f"{name}", RED, font_sm)
        if show_detected and name in part_xy:
            det = detected_uv(bundle, name, part_xy[name])
            _draw_box(draw, det, hw, GREEN, width=2)
            _draw_cross(draw, det, GREEN, size=6, width=2)
            tag = "board" if name in SUPPORT_COUPLED_PARTS else "det"
            _label(
                draw,
                (det[0] - hw[1] / 2, det[1] + hw[0] / 2 + 2),
                f"{name} {tag}",
                GREEN,
                font_sm,
            )
            if show_nominal:
                draw.line(
                    (int(nom[0]), int(nom[1]), int(det[0]), int(det[1])),
                    fill=(180, 255, 180),
                    width=1,
                )

    # Legend banner
    banner_h = 28
    draw.rectangle((0, 0, img.width, banner_h), fill=(20, 20, 20))
    _label(draw, (8, 6), f"{title}   RED=nominal   GREEN=detected",
           WHITE, font)
    return img


def _load_mean_rgb_depth(run_dir: Path):
    rgb_path = run_dir / "buffer_mean_rgb.png"
    depth_path = run_dir / "buffer_mean_depth_m.npy"
    if rgb_path.is_file() and depth_path.is_file():
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path)
        return rgb, depth
    # Rebuild mean from per-buffer frames if needed.
    man = json.loads((run_dir / "manifest.json").read_text())
    frames = []
    for fr in man["frames"]:
        rgb = np.asarray(Image.open(run_dir / fr["rgb"]).convert("RGB"))
        depth = None
        if fr.get("depth_m"):
            depth = np.load(run_dir / fr["depth_m"])
        from policies.camera_offset.estimator import BufferedFrame
        frames.append(BufferedFrame(rgb=rgb, depth=depth,
                                    sim_step_idx=fr.get("sim_step_idx")))
    composed = average_buffered_frames(frames)
    return composed.rgb, composed.depth


def process_run(run_dir: Path, bundle: ReferenceBundle, reestimate: bool) -> dict:
    rgb_mean, depth_mean = _load_mean_rgb_depth(run_dir)
    ref_rgb = np.asarray(Image.open(run_dir / "reference_rgb.png").convert("RGB"))

    if reestimate:
        est = OffsetEstimator(bundle)
        # Feed the saved buffer frames so mean-pool matches the live path.
        man = json.loads((run_dir / "manifest.json").read_text())
        from policies.camera_offset.estimator import BufferedFrame
        for fr in man["frames"]:
            rgb = np.asarray(Image.open(run_dir / fr["rgb"]).convert("RGB"))
            depth = (
                np.load(run_dir / fr["depth_m"]) if fr.get("depth_m") else None
            )
            est._frames.append(
                BufferedFrame(
                    rgb=rgb, depth=depth, sim_step_idx=fr.get("sim_step_idx")
                )
            )
        estimate = est.estimate(force=True)
        part_xy = estimate.part_xy
        board_xy = estimate.board_xy
        conf = estimate.board_confidence
        compose = estimate.diagnostics.get("compose")
    else:
        man = json.loads((run_dir / "manifest.json").read_text())
        est_block = man["estimate"]
        part_xy = {
            k: np.asarray(v, dtype=np.float64) for k, v in est_block["part_xy"].items()
        }
        board_xy = np.asarray(est_block["board_xy"], dtype=np.float64)
        conf = float(est_block["board_confidence"])
        compose = est_block.get("compose", "mean")

    nom_only = annotate_image(
        ref_rgb, bundle, part_xy, board_xy,
        title=f"{run_dir.name} | nominal reference",
        show_nominal=True,
        show_detected=False,
    )
    det_img = annotate_image(
        rgb_mean, bundle, part_xy, board_xy,
        title=f"{run_dir.name} | mean-pool detect  conf={conf:.2f}",
        show_nominal=True,
        show_detected=True,
    )
    # Side-by-side panel
    gap = 8
    panel = Image.new(
        "RGB",
        (nom_only.width + det_img.width + gap, max(nom_only.height, det_img.height)),
        (30, 30, 30),
    )
    panel.paste(nom_only, (0, 0))
    panel.paste(det_img, (nom_only.width + gap, 0))

    out_det = run_dir / "detections_annotated.png"
    out_panel = run_dir / "detections_panel.png"
    out_nom = run_dir / "nominal_annotated.png"
    det_img.save(out_det)
    nom_only.save(out_nom)
    panel.save(out_panel)

    summary = {
        "run": run_dir.name,
        "compose": compose,
        "board_xy": [float(board_xy[0]), float(board_xy[1])],
        "board_confidence": conf,
        "part_xy": {k: [float(v[0]), float(v[1])] for k, v in part_xy.items()},
        "detections_uv": {
            name: [float(u), float(v)]
            for name, (u, v) in (
                (n, detected_uv(bundle, n, part_xy[n])) for n in part_xy
            )
        },
        "nominal_uv": {
            name: [float(t.search_center_uv[0]), float(t.search_center_uv[1])]
            for name, t in bundle.parts.items()
        },
        "outputs": {
            "detections_annotated": str(out_det),
            "nominal_annotated": str(out_nom),
            "detections_panel": str(out_panel),
        },
    }
    (run_dir / "detections_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[annotate] {run_dir.name}: board={summary['board_xy']} -> {out_panel}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "offset_estimation_frames_avg",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=TASK_DIR / "policies" / "camera_reference",
    )
    parser.add_argument(
        "--reestimate",
        action="store_true",
        default=True,
        help="Re-run mean-pooled OffsetEstimator on saved buffer frames "
             "(default: on).",
    )
    parser.add_argument(
        "--no-reestimate",
        action="store_false",
        dest="reestimate",
        help="Reuse board/part XY from each run's manifest.json.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Subset of run directory names (default: all with manifest.json).",
    )
    args = parser.parse_args()

    bundle = ReferenceBundle.load(args.reference)
    run_dirs = []
    if args.runs:
        run_dirs = [args.runs_dir / name for name in args.runs]
    else:
        run_dirs = sorted(
            p for p in args.runs_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").is_file()
        )
    if not run_dirs:
        raise SystemExit(f"no runs found under {args.runs_dir}")

    summaries = [process_run(d, bundle, args.reestimate) for d in run_dirs]
    index_path = args.runs_dir / "detections_index.json"
    index_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"[annotate] index -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
