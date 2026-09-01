#!/usr/bin/env python3
"""Annotate mean-pooled transport-target offsets (pick / place).

Red  = nominal pick/place world poses projected into the head image
Green = the same targets after applying mean-pooled part/board XY offsets
        (pick ← part offset, place ← board offset — matching
        ``adjust_part_target``).

Usage:

  python3 scripts/annotate_transport_targets.py \\
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

import param_config as pc  # noqa: E402
from policies.camera_offset.constants import SUPPORT_COUPLED_PARTS  # noqa: E402
from policies.camera_offset.geometry import (  # noqa: E402
    as_xy,
    pinhole_project,
)
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402
from policies.camera_offset.targets import (  # noqa: E402
    estimated_part_offset,
)

RED = (255, 40, 40)
GREEN = (40, 220, 80)
PICK_RED = (255, 70, 70)
PICK_GREEN = (60, 255, 120)
PLACE_RED = (220, 40, 40)
PLACE_GREEN = (40, 200, 90)
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


def _font(size: int = 13):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _label(draw, xy, text, color, font):
    x, y = int(xy[0]), int(xy[1])
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=BLACK,
    )
    draw.text((x, y), text, fill=color, font=font)


def _draw_cross(draw, uv, color, size=7, width=2):
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    draw.line((u - size, v, u + size, v), fill=color, width=width)
    draw.line((u, v - size, u, v + size), fill=color, width=width)


def _draw_diamond(draw, uv, color, size=8, width=2):
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    pts = [
        (u, v - size),
        (u + size, v),
        (u, v + size),
        (u - size, v),
        (u, v - size),
    ]
    draw.line(pts, fill=color, width=width)


def _draw_circle(draw, uv, color, radius=7, width=2):
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    draw.ellipse(
        (u - radius, v - radius, u + radius, v + radius),
        outline=color,
        width=width,
    )


def project_xyz(bundle: ReferenceBundle, xyz) -> np.ndarray:
    if bundle.camera_R_world_from_cam is None or bundle.camera_t_world is None:
        raise RuntimeError(
            "reference bundle is missing camera pose; rebuild with "
            "scripts/build_camera_reference.py"
        )
    uv = pinhole_project(
        xyz,
        bundle.intrinsics,
        bundle.camera_R_world_from_cam,
        bundle.camera_t_world,
    )[0]
    if not np.isfinite(uv).all():
        raise ValueError(f"projection failed for {xyz}")
    return uv


def shift_xyz_xy(xyz, offset_xy):
    arr = np.asarray(xyz, dtype=np.float64).reshape(3).copy()
    dxy = as_xy(offset_xy)
    arr[0] += dxy[0]
    arr[1] += dxy[1]
    return arr


def load_estimate(run_dir: Path) -> tuple[np.ndarray, dict]:
    """Return (board_xy, part_xy dict) from mean-pool run artifacts."""
    summary = run_dir / "detections_summary.json"
    man = run_dir / "manifest.json"
    if summary.is_file():
        data = json.loads(summary.read_text())
        board = np.asarray(data["board_xy"], dtype=np.float64)
        parts = {
            k: np.asarray(v, dtype=np.float64) for k, v in data["part_xy"].items()
        }
        return board, parts
    if man.is_file():
        data = json.loads(man.read_text())["estimate"]
        board = np.asarray(data["board_xy"], dtype=np.float64)
        parts = {
            k: np.asarray(v, dtype=np.float64) for k, v in data["part_xy"].items()
        }
        return board, parts
    raise FileNotFoundError(f"no estimate JSON in {run_dir}")


def collect_transport_targets(part_names):
    """Nominal pick/place world poses from param_config."""
    out = {}
    for name in part_names:
        cfg = pc.get_part_config(name)
        pick = cfg.get("pick_pos")
        place = cfg.get("place_pos")
        if pick is None and place is None:
            continue
        out[name] = {
            "pick_pos": None if pick is None else np.asarray(pick, dtype=np.float64),
            "place_pos": (
                None if place is None else np.asarray(place, dtype=np.float64)
            ),
        }
    return out


def annotate_transport(
    rgb,
    bundle: ReferenceBundle,
    targets: dict,
    board_xy,
    part_xy: dict,
    *,
    title: str,
    show_nominal: bool = True,
    show_offset: bool = True,
    show_labels: bool = True,
    show_banner: bool = True,
    show_connectors: bool = True,
) -> Image.Image:
    img = Image.fromarray(_as_u8_rgb(rgb), mode="RGB")
    draw = ImageDraw.Draw(img)
    font = _font(12)
    font_sm = _font(11)

    records = []
    for name, poses in targets.items():
        part_off = estimated_part_offset(name, part_xy, board_xy)[:2]
        entry = {"name": name, "part_offset_xy": part_off.tolist(),
                 "board_offset_xy": as_xy(board_xy).tolist()}
        for kind, color_nom, color_off, marker in (
            ("pick_pos", PICK_RED, PICK_GREEN, "circle"),
            ("place_pos", PLACE_RED, PLACE_GREEN, "diamond"),
        ):
            xyz = poses.get(kind)
            if xyz is None:
                continue
            uv_nom = project_xyz(bundle, xyz)
            entry[f"{kind}_nominal_xyz"] = xyz.tolist()
            entry[f"{kind}_nominal_uv"] = [float(uv_nom[0]), float(uv_nom[1])]
            if show_nominal:
                if marker == "circle":
                    _draw_circle(draw, uv_nom, color_nom, radius=8, width=2)
                else:
                    _draw_diamond(draw, uv_nom, color_nom, size=9, width=2)
                _draw_cross(draw, uv_nom, color_nom, size=5, width=2)
                if show_labels:
                    short = "pick" if kind.startswith("pick") else "place"
                    _label(
                        draw,
                        (uv_nom[0] + 10, uv_nom[1] - 16),
                        f"{name} {short}",
                        color_nom,
                        font_sm,
                    )

            if show_offset:
                off = part_off if kind.startswith("pick") else as_xy(board_xy)
                xyz_adj = shift_xyz_xy(xyz, off)
                uv_adj = project_xyz(bundle, xyz_adj)
                entry[f"{kind}_offset_xyz"] = xyz_adj.tolist()
                entry[f"{kind}_offset_uv"] = [float(uv_adj[0]), float(uv_adj[1])]
                if marker == "circle":
                    _draw_circle(draw, uv_adj, color_off, radius=8, width=2)
                else:
                    _draw_diamond(draw, uv_adj, color_off, size=9, width=2)
                _draw_cross(draw, uv_adj, color_off, size=5, width=2)
                if show_labels:
                    short = "pick" if kind.startswith("pick") else "place"
                    tag = (
                        "board"
                        if kind.startswith("place") or name in SUPPORT_COUPLED_PARTS
                        else "part"
                    )
                    _label(
                        draw,
                        (uv_adj[0] + 10, uv_adj[1] + 4),
                        f"{name} {short} {tag}",
                        color_off,
                        font_sm,
                    )
                if show_nominal and show_connectors:
                    draw.line(
                        (
                            int(uv_nom[0]), int(uv_nom[1]),
                            int(uv_adj[0]), int(uv_adj[1]),
                        ),
                        fill=(170, 255, 170),
                        width=1,
                    )
        records.append(entry)

    if show_banner:
        banner_h = 28
        draw.rectangle((0, 0, img.width, banner_h), fill=(20, 20, 20))
        legend = []
        if show_nominal:
            legend.append("RED=nominal pick○/place◇")
        if show_offset:
            legend.append("GREEN=offset transport")
        _label(
            draw,
            (8, 6),
            f"{title}   " + "   ".join(legend),
            WHITE,
            font,
        )
    return img, records


def process_run(run_dir: Path, bundle: ReferenceBundle, targets: dict) -> dict:
    board_xy, part_xy = load_estimate(run_dir)
    mean_path = run_dir / "buffer_mean_rgb.png"
    ref_path = run_dir / "reference_rgb.png"
    if not mean_path.is_file():
        raise FileNotFoundError(f"missing {mean_path}")
    rgb_mean = np.asarray(Image.open(mean_path).convert("RGB"))
    rgb_ref = np.asarray(Image.open(ref_path).convert("RGB"))

    board_tag = f"board=({board_xy[0]:+.4f},{board_xy[1]:+.4f})"

    nom_img, _ = annotate_transport(
        rgb_ref,
        bundle,
        targets,
        board_xy,
        part_xy,
        title=f"{run_dir.name} | nominal transport targets",
        show_nominal=True,
        show_offset=False,
    )
    det_img, records = annotate_transport(
        rgb_mean,
        bundle,
        targets,
        board_xy,
        part_xy,
        title=f"{run_dir.name} | mean-pool transport offsets  {board_tag}",
        show_nominal=True,
        show_offset=True,
    )
    # Green (offset) targets alone.
    offset_only, _ = annotate_transport(
        rgb_mean,
        bundle,
        targets,
        board_xy,
        part_xy,
        title=f"{run_dir.name} | offset transport only  {board_tag}",
        show_nominal=False,
        show_offset=True,
        show_labels=True,
    )
    # Same variants without text labels / banner.
    det_nolabel, _ = annotate_transport(
        rgb_mean,
        bundle,
        targets,
        board_xy,
        part_xy,
        title="",
        show_nominal=True,
        show_offset=True,
        show_labels=False,
        show_banner=False,
    )
    offset_only_nolabel, _ = annotate_transport(
        rgb_mean,
        bundle,
        targets,
        board_xy,
        part_xy,
        title="",
        show_nominal=False,
        show_offset=True,
        show_labels=False,
        show_banner=False,
    )
    nom_nolabel, _ = annotate_transport(
        rgb_ref,
        bundle,
        targets,
        board_xy,
        part_xy,
        title="",
        show_nominal=True,
        show_offset=False,
        show_labels=False,
        show_banner=False,
    )

    gap = 8
    panel = Image.new(
        "RGB",
        (nom_img.width + det_img.width + gap, max(nom_img.height, det_img.height)),
        (30, 30, 30),
    )
    panel.paste(nom_img, (0, 0))
    panel.paste(det_img, (nom_img.width + gap, 0))

    outputs = {
        "transport_targets_annotated": run_dir / "transport_targets_annotated.png",
        "transport_targets_nominal": run_dir / "transport_targets_nominal.png",
        "transport_targets_panel": run_dir / "transport_targets_panel.png",
        "transport_targets_offset_only": run_dir / "transport_targets_offset_only.png",
        "transport_targets_annotated_nolabel": (
            run_dir / "transport_targets_annotated_nolabel.png"
        ),
        "transport_targets_offset_only_nolabel": (
            run_dir / "transport_targets_offset_only_nolabel.png"
        ),
        "transport_targets_nominal_nolabel": (
            run_dir / "transport_targets_nominal_nolabel.png"
        ),
    }
    det_img.save(outputs["transport_targets_annotated"])
    nom_img.save(outputs["transport_targets_nominal"])
    panel.save(outputs["transport_targets_panel"])
    offset_only.save(outputs["transport_targets_offset_only"])
    det_nolabel.save(outputs["transport_targets_annotated_nolabel"])
    offset_only_nolabel.save(outputs["transport_targets_offset_only_nolabel"])
    nom_nolabel.save(outputs["transport_targets_nominal_nolabel"])

    summary = {
        "run": run_dir.name,
        "compose": "mean",
        "board_xy": [float(board_xy[0]), float(board_xy[1])],
        "part_xy": {k: [float(v[0]), float(v[1])] for k, v in part_xy.items()},
        "targets": records,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    (run_dir / "transport_targets_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"[transport] {run_dir.name} -> {outputs['transport_targets_panel']}")
    print(f"            offset-only -> {outputs['transport_targets_offset_only'].name}")
    print(
        f"            nolabel -> "
        f"{outputs['transport_targets_offset_only_nolabel'].name}, "
        f"{outputs['transport_targets_annotated_nolabel'].name}"
    )
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
    parser.add_argument("--runs", nargs="*", default=None)
    args = parser.parse_args()

    bundle = ReferenceBundle.load(args.reference)
    part_names = [
        n for n in pc.part_order
        if isinstance(n, str) and not n.startswith("<")
    ]
    targets = collect_transport_targets(part_names)
    if not targets:
        raise SystemExit("no pick/place targets found in param_config")

    if args.runs:
        run_dirs = [args.runs_dir / n for n in args.runs]
    else:
        run_dirs = sorted(
            p for p in args.runs_dir.iterdir()
            if p.is_dir() and (p / "manifest.json").is_file()
        )
    if not run_dirs:
        raise SystemExit(f"no runs under {args.runs_dir}")

    summaries = [process_run(d, bundle, targets) for d in run_dirs]
    index = args.runs_dir / "transport_targets_index.json"
    index.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"[transport] index -> {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
