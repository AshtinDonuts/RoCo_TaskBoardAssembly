"""Camera-only board/part XY offset estimator."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .constants import (
    AVERAGE_BUFFERED_FRAMES,
    BOARD_CONFIDENCE_MIN,
    BOARD_CONSENSUS_MAX_M,
    DEFAULT_ROI_MARGIN_PX,
    DEPTH_DISAGREE_M,
    NCC_SCALES,
    ORB_MIN_INLIERS,
    PART_NCC_MIN,
    SUPPORT_COUPLED_PARTS,
)
from . import matching as M
from .constants import XY_MAX_M
from .geometry import clamp_xy, pixel_delta_to_world_xy, roi_half_extent_px
from .reference import ReferenceBundle

# Matches whose implied world offset exceeds the legal square by more than
# this slack cannot be the true part location; reject them instead of
# clamping them onto the boundary.
FEASIBLE_SLACK_M = 1.5e-3


@dataclass
class BufferedFrame:
    """One head RGB-D observation held for the pre-motion offset buffer."""

    rgb: np.ndarray
    depth: Optional[np.ndarray]
    sim_step_idx: Optional[int] = None


def average_buffered_frames(frames: Sequence[BufferedFrame]) -> BufferedFrame:
    """Mean-pool RGB (and nanmean depth) across the pre-motion buffer.

    Temporal averaging reduces single-frame RT denoiser / path-trace flicker
    before the composite is compared to the sharp nominal reference.
    """
    if not frames:
        raise ValueError("average_buffered_frames requires at least one frame")
    rgb_stack = np.stack(
        [np.asarray(fr.rgb, dtype=np.float64) for fr in frames], axis=0
    )
    rgb_mean = np.mean(rgb_stack, axis=0)
    depth_mean = None
    depth_list = [
        np.asarray(fr.depth, dtype=np.float64)
        for fr in frames if fr.depth is not None
    ]
    if depth_list:
        depth_stack = np.stack(depth_list, axis=0)
        with np.errstate(all="ignore"):
            depth_mean = np.nanmean(depth_stack, axis=0)
    # Tag with the last buffered sim step (buffer spans [first, last]).
    last_step = frames[-1].sim_step_idx
    return BufferedFrame(rgb=rgb_mean, depth=depth_mean, sim_step_idx=last_step)


@dataclass
class OffsetEstimate:
    board_xy: np.ndarray
    part_xy: Dict[str, np.ndarray]
    board_confidence: float
    part_confidence: Dict[str, float]
    diagnostics: dict = field(default_factory=dict)

    def part_offset(self, name: str) -> np.ndarray:
        if name in SUPPORT_COUPLED_PARTS:
            xy = self.board_xy
        else:
            xy = self.part_xy.get(name, np.zeros(2, dtype=np.float64))
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        return np.array([xy[0], xy[1], 0.0], dtype=np.float64)


class OffsetEstimator:
    """Uses only head RGB-D and packaged nominal assets."""

    def __init__(self, bundle: ReferenceBundle) -> None:
        self.bundle = bundle
        self._frames: List[BufferedFrame] = []
        self._estimate: Optional[OffsetEstimate] = None

    def reset_episode(self) -> None:
        self._frames.clear()
        self._estimate = None

    def add_frame(
        self,
        rgb,
        depth,
        intrinsics=None,
        *,
        sim_step_idx: Optional[int] = None,
    ) -> None:
        self.bundle.assert_observation_shape(rgb, depth, intrinsics)
        rgb_arr = np.asarray(rgb)
        depth_arr = None if depth is None else np.asarray(depth, dtype=np.float64)
        step = None if sim_step_idx is None else int(sim_step_idx)
        self._frames.append(
            BufferedFrame(rgb=rgb_arr.copy(), depth=depth_arr, sim_step_idx=step)
        )

    def ready(self, min_frames: Optional[int] = None) -> bool:
        need = int(self.bundle.buffer_frames if min_frames is None else min_frames)
        return len(self._frames) >= max(1, need)

    def export_buffered_frames(
        self,
        directory: Union[str, Path],
        *,
        estimate: Optional[OffsetEstimate] = None,
    ) -> Path:
        """Write buffered head frames + nominal reference used for estimation.

        Filenames encode both buffer index and the sim ``step_idx`` at which
        each frame was ingested (when known):

            buf00_simstep000012_rgb.png
            buf00_simstep000012_depth_viz.png
            reference_rgb.png
            manifest.json
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        if not self._frames:
            raise RuntimeError("OffsetEstimator has no buffered frames to export")

        est = estimate if estimate is not None else self._estimate
        frame_meta = []
        for i, fr in enumerate(self._frames):
            step_tag = (
                "unknown" if fr.sim_step_idx is None
                else f"{int(fr.sim_step_idx):06d}"
            )
            stem = f"buf{i:02d}_simstep{step_tag}"
            rgb_name = f"{stem}_rgb.png"
            depth_viz_name = f"{stem}_depth_viz.png"
            depth_npy_name = f"{stem}_depth_m.npy"
            _save_rgb_png(out / rgb_name, fr.rgb)
            entry = {
                "buffer_index": i,
                "sim_step_idx": fr.sim_step_idx,
                "rgb": rgb_name,
            }
            if fr.depth is not None:
                np.save(out / depth_npy_name, np.asarray(fr.depth, dtype=np.float64))
                _save_depth_viz_png(out / depth_viz_name, fr.depth)
                entry["depth_m"] = depth_npy_name
                entry["depth_viz"] = depth_viz_name
            frame_meta.append(entry)

        compose_meta = None
        if est is not None and est.diagnostics.get("compose") == "mean":
            composed = average_buffered_frames(self._frames)
            _save_rgb_png(out / "buffer_mean_rgb.png", composed.rgb)
            compose_meta = {
                "mode": "mean",
                "rgb": "buffer_mean_rgb.png",
                "n_frames": len(self._frames),
                "sim_step_idxs": [fr.sim_step_idx for fr in self._frames],
            }
            if composed.depth is not None:
                np.save(
                    out / "buffer_mean_depth_m.npy",
                    np.asarray(composed.depth, dtype=np.float64),
                )
                _save_depth_viz_png(out / "buffer_mean_depth_viz.png", composed.depth)
                compose_meta["depth_m"] = "buffer_mean_depth_m.npy"
                compose_meta["depth_viz"] = "buffer_mean_depth_viz.png"
            if est.diagnostics.get("composed") is not None:
                compose_meta["diagnostics"] = est.diagnostics["composed"]

        _save_rgb_png(out / "reference_rgb.png", self.bundle.rgb)
        np.save(out / "reference_depth_m.npy",
                np.asarray(self.bundle.depth, dtype=np.float64))
        _save_depth_viz_png(out / "reference_depth_viz.png", self.bundle.depth)

        manifest = {
            "buffer_frames_required": int(self.bundle.buffer_frames),
            "n_frames": len(self._frames),
            "sim_step_idxs": [fr.sim_step_idx for fr in self._frames],
            "compose": compose_meta,
            "frames": frame_meta,
            "reference": {
                "rgb": "reference_rgb.png",
                "depth_m": "reference_depth_m.npy",
                "depth_viz": "reference_depth_viz.png",
            },
        }
        if est is not None:
            manifest["estimate"] = {
                "board_xy": [float(est.board_xy[0]), float(est.board_xy[1])],
                "board_confidence": float(est.board_confidence),
                "part_xy": {
                    k: [float(v[0]), float(v[1])] for k, v in est.part_xy.items()
                },
                "part_confidence": {
                    k: float(v) for k, v in est.part_confidence.items()
                },
                "n_frames": int(est.diagnostics.get("n_frames", len(self._frames))),
                "compose": est.diagnostics.get("compose"),
            }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return out

    def estimate(self, force=False) -> OffsetEstimate:
        if self._estimate is not None and not force:
            return self._estimate
        if not self._frames:
            raise RuntimeError(
                "OffsetEstimator has no head frames. "
                "Set TASK_ENABLE_CAMERA_OUTPUT=1 and wait for rgb['head']."
            )

        if AVERAGE_BUFFERED_FRAMES:
            composed = average_buffered_frames(self._frames)
            board_xy, board_c, board_src = self._estimate_board(
                composed.rgb, composed.depth
            )
            board_xy = clamp_xy(board_xy)
            parts = {}
            confidences = {}
            per_part = {}
            for name, tmpl in self.bundle.parts.items():
                if name in SUPPORT_COUPLED_PARTS:
                    parts[name] = board_xy.copy()
                    confidences[name] = float(board_c)
                    continue
                xy, conf, src = self._estimate_part(
                    name, tmpl, composed.rgb, composed.depth
                )
                parts[name] = clamp_xy(xy)
                confidences[name] = float(conf)
                per_part[name] = {
                    "xy": [float(xy[0]), float(xy[1])],
                    "confidence": float(conf),
                    "source": src,
                }
            composed_diag = {
                "board_xy": [float(board_xy[0]), float(board_xy[1])],
                "board_confidence": float(board_c),
                "board_source": board_src,
                "parts": per_part,
                "sim_step_idxs": [fr.sim_step_idx for fr in self._frames],
            }
            self._estimate = OffsetEstimate(
                board_xy=board_xy,
                part_xy=parts,
                board_confidence=float(board_c),
                part_confidence=confidences,
                diagnostics={
                    "compose": "mean",
                    "composed": composed_diag,
                    "frames": [],
                    "n_frames": len(self._frames),
                    "sim_step_idxs": [fr.sim_step_idx for fr in self._frames],
                    "buffer_frames_required": int(self.bundle.buffer_frames),
                },
            )
            return self._estimate

        # Legacy path: match each buffer frame, then median the offsets.
        board_samples = []
        part_samples: Dict[str, List[np.ndarray]] = {
            name: [] for name in self.bundle.parts
        }
        part_conf: Dict[str, List[float]] = {name: [] for name in self.bundle.parts}
        frame_diag = []
        for fr in self._frames:
            rgb, depth = fr.rgb, fr.depth
            board_xy, board_c, board_src = self._estimate_board(rgb, depth)
            board_samples.append(board_xy)
            per_part = {}
            for name, tmpl in self.bundle.parts.items():
                if name in SUPPORT_COUPLED_PARTS:
                    continue
                xy, conf, src = self._estimate_part(name, tmpl, rgb, depth)
                part_samples[name].append(xy)
                part_conf[name].append(conf)
                per_part[name] = {
                    "xy": [float(xy[0]), float(xy[1])],
                    "confidence": float(conf),
                    "source": src,
                }
            frame_diag.append(
                {
                    "buffer_index": len(frame_diag),
                    "sim_step_idx": fr.sim_step_idx,
                    "board_xy": [float(board_xy[0]), float(board_xy[1])],
                    "board_confidence": float(board_c),
                    "board_source": board_src,
                    "parts": per_part,
                }
            )
        board = _median_xy(board_samples)
        board = clamp_xy(board)
        parts = {}
        confidences = {}
        for name in self.bundle.parts:
            if name in SUPPORT_COUPLED_PARTS:
                parts[name] = board.copy()
                confidences[name] = float(np.median([d["board_confidence"] for d in frame_diag]))
                continue
            samples = part_samples[name]
            if samples:
                parts[name] = clamp_xy(_median_xy(samples))
                confidences[name] = float(np.median(part_conf[name]))
            else:
                # Plan §4: failed estimate → nominal zero (not board offset).
                parts[name] = np.zeros(2, dtype=np.float64)
                confidences[name] = 0.0
        board_c = float(np.median([d["board_confidence"] for d in frame_diag]))
        self._estimate = OffsetEstimate(
            board_xy=board,
            part_xy=parts,
            board_confidence=board_c,
            part_confidence=confidences,
            diagnostics={
                "compose": "per_frame_median",
                "frames": frame_diag,
                "n_frames": len(self._frames),
                "sim_step_idxs": [fr.sim_step_idx for fr in self._frames],
                "buffer_frames_required": int(self.bundle.buffer_frames),
            },
        )
        return self._estimate

    def _foreground_mask(self, depth) -> np.ndarray:
        mask = np.asarray(self.bundle.board_mask, dtype=bool)
        if depth is None:
            return mask
        d = np.asarray(depth, dtype=np.float64)
        nom = np.asarray(self.bundle.depth, dtype=np.float64)
        agree = np.abs(d - nom) <= DEPTH_DISAGREE_M
        agree |= ~np.isfinite(d) | ~np.isfinite(nom)
        return mask & agree

    def _estimate_board(self, rgb, depth):
        mask = self._foreground_mask(depth)
        jac = self.bundle.jacobian_xy_per_px
        valid = []

        try:
            du_dv, score = M.ecc_translation(self.bundle.rgb, rgb, mask=mask)
            world = clamp_xy(pixel_delta_to_world_xy(du_dv, jac))
            valid.append((world, float(score), "ecc"))
        except Exception:
            pass

        try:
            orb_du, orb_score, n_inl = M.orb_translation(
                self.bundle.rgb, rgb, mask=mask, min_inliers=ORB_MIN_INLIERS
            )
            if n_inl >= ORB_MIN_INLIERS:
                world = clamp_xy(pixel_delta_to_world_xy(orb_du, jac))
                valid.append((world, float(orb_score), "orb"))
        except Exception:
            pass

        try:
            du_dv, score = M.phase_correlate(
                self.bundle.rgb, rgb, mask=mask, use_highpass=True
            )
            world = clamp_xy(pixel_delta_to_world_xy(du_dv, jac))
            valid.append((world, float(score), "phasecorr"))
        except Exception:
            pass

        if not valid:
            return np.zeros(2, dtype=np.float64), 0.0, "zero"

        by_name = {name: (world, score) for world, score, name in valid}
        if "ecc" in by_name and "orb" in by_name:
            ecc_w, ecc_s = by_name["ecc"]
            orb_w, orb_s = by_name["orb"]
            if float(np.linalg.norm(ecc_w - orb_w)) <= BOARD_CONSENSUS_MAX_M:
                world = clamp_xy(0.5 * (ecc_w + orb_w))
                return world, float(0.5 * (ecc_s + orb_s)), "ecc_orb_avg"

        # Prefer ECC when confident; otherwise median of available matchers.
        for world, score, name in valid:
            if name == "ecc" and score >= BOARD_CONFIDENCE_MIN:
                return world, float(score), name
        world = _median_xy([v[0] for v in valid])
        return clamp_xy(world), float(valid[0][1]), "median_matchers"

    def _estimate_part(self, name, tmpl, rgb, depth):
        jac = (tmpl.jacobian_xy_per_px
               if tmpl.jacobian_xy_per_px is not None
               else self.bundle.jacobian_xy_per_px)
        half_u, half_v = roi_half_extent_px(
            jac,
            tmpl.rgb.shape[:2],
            margin_px=DEFAULT_ROI_MARGIN_PX,
        )
        gray = M.to_gray(rgb)
        tmpl_gray = M.to_gray(tmpl.rgb)
        edge_img = M.sobel_magnitude(gray)
        edge_tmpl = M.sobel_magnitude(tmpl_gray)
        searches = [
            ("ncc_rgb", gray, tmpl_gray),
            ("ncc_edges", edge_img, edge_tmpl),
        ]
        if depth is not None:
            # Depth renders without the RGB denoiser, so height-map NCC
            # survives unconverged-render artifacts in the color stream.
            d = np.asarray(depth, dtype=np.float64)
            searches.append(
                ("ncc_depth", np.where(np.isfinite(d), d, 0.0),
                 np.where(np.isfinite(tmpl.depth), tmpl.depth, 0.0))
            )
        valid = []
        for src, image, template in searches:
            result = M.multiscale_ncc_search(
                image,
                template,
                template_mask=tmpl.mask,
                search_origin_uv=tmpl.search_center_uv,
                search_half=(half_u, half_v),
                scales=NCC_SCALES,
            )
            if not result["valid"]:
                continue
            match = self._best_feasible_match(result, jac, tmpl, depth)
            if match is None or match["score"] < PART_NCC_MIN:
                continue
            valid.append((match, match["residual"], src))
        if not valid:
            crop, origin = _roi_crop(
                M.to_gray(rgb), tmpl.search_center_uv, tmpl.rgb.shape[:2],
                half_u, half_v,
            )
            nom_crop, _ = _roi_crop(
                M.to_gray(self.bundle.rgb), tmpl.search_center_uv,
                tmpl.rgb.shape[:2], half_u, half_v,
            )
            if crop is not None and nom_crop is not None:
                du_dv, score = M.phase_correlate(nom_crop, crop)
                if score >= PART_NCC_MIN:
                    world = clamp_xy(pixel_delta_to_world_xy(du_dv, jac))
                    return world, float(score), "phasecorr_roi"
            return np.zeros(2, dtype=np.float64), 0.0, "zero"

        ranked = []
        for match, residual, src in valid:
            dist = float(np.hypot(match["du"], match["dv"]))
            uv = match["uv"]
            ranked.append(
                (
                    -float(match["score"]),
                    float(residual),
                    dist,
                    float(uv[1]),
                    float(uv[0]),
                    match,
                    src,
                )
            )
        ranked.sort()
        best = ranked[0]
        match, src = best[5], best[6]
        world = clamp_xy(
            pixel_delta_to_world_xy((match["du"], match["dv"]), jac)
        )
        return world, float(match["score"]), src

    def _best_feasible_match(self, result, jac, tmpl, depth):
        """Best NCC peak whose implied world offset can actually occur.

        Ties within 1e-9 of the top feasible score are broken by the fixed
        ordering from the plan: depth residual, distance from nominal,
        then lexicographic (v, u).
        """
        smap = result.get("score_map")
        origin = result.get("origin")
        if smap is None or origin is None:
            return None
        u0, v0 = origin
        th, tw = tmpl.rgb.shape[:2]
        cu, cv = (float(tmpl.search_center_uv[0]),
                  float(tmpl.search_center_uv[1]))
        ys, xs = np.mgrid[0:smap.shape[0], 0:smap.shape[1]]
        delta_u = (u0 + xs + tw / 2.0) - cu
        delta_v = (v0 + ys + th / 2.0) - cv
        jac = np.asarray(jac, dtype=np.float64).reshape(2, 2)
        wx = jac[0, 0] * delta_u + jac[0, 1] * delta_v
        wy = jac[1, 0] * delta_u + jac[1, 1] * delta_v
        limit = XY_MAX_M + FEASIBLE_SLACK_M
        feasible = (np.abs(wx) <= limit) & (np.abs(wy) <= limit)
        if not np.any(feasible):
            return None
        masked = np.where(feasible, smap, -np.inf)
        top = float(np.max(masked))
        cand_rc = np.argwhere(masked >= top - 1e-9)
        ranked = []
        for rc in cand_rc:
            y_i, x_i = int(rc[0]), int(rc[1])
            uv_centre = np.array(
                [u0 + x_i + tw / 2.0, v0 + y_i + th / 2.0], dtype=np.float64
            )
            residual = 0.0
            if depth is not None:
                residual = M.depth_residual(
                    depth, tmpl.depth, uv_centre, tmpl.mask
                )
            dist = float(np.hypot(delta_u[y_i, x_i], delta_v[y_i, x_i]))
            ranked.append((residual, dist, y_i, x_i, uv_centre))
        ranked.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
        residual, _, y_i, x_i, uv_centre = ranked[0]
        du_sub, dv_sub = M.subpixel_quadratic(smap, (y_i, x_i))
        return {
            "du": float(delta_u[y_i, x_i] + du_sub),
            "dv": float(delta_v[y_i, x_i] + dv_sub),
            "score": top,
            "uv": uv_centre + np.array([du_sub, dv_sub]),
            "residual": float(residual),
        }


def _median_xy(samples: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([np.asarray(s, dtype=np.float64).reshape(2) for s in samples])
    return np.median(stacked, axis=0).astype(np.float64)


def _roi_crop(image, centre_uv, template_hw, half_u, half_v):
    ih, iw = image.shape[:2]
    th, tw = int(template_hw[0]), int(template_hw[1])
    cu, cv = float(centre_uv[0]), float(centre_uv[1])
    x0 = int(np.floor(cu - tw / 2.0 - half_u))
    y0 = int(np.floor(cv - th / 2.0 - half_v))
    x1 = int(np.ceil(cu + tw / 2.0 + half_u))
    y1 = int(np.ceil(cv + th / 2.0 + half_v))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(iw, x1)
    y1 = min(ih, y1)
    if x1 - x0 < tw or y1 - y0 < th:
        return None, None
    return image[y0:y1, x0:x1], (x0, y0)


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


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    u8 = _as_u8_rgb(rgb)
    try:
        from PIL import Image
        Image.fromarray(u8, mode="RGB").save(path)
    except ImportError:
        # Portable fallback when Pillow is unavailable.
        np.save(path.with_suffix(".npy"), u8)


def _save_depth_viz_png(path: Path, depth: np.ndarray) -> None:
    d = np.asarray(depth, dtype=np.float64)
    finite = np.isfinite(d) & (d > 0)
    viz = np.zeros(d.shape[:2], dtype=np.uint8)
    if np.any(finite):
        lo = float(np.percentile(d[finite], 5))
        hi = float(np.percentile(d[finite], 95))
        if hi <= lo:
            hi = lo + 1e-6
        scaled = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        viz = np.where(finite, np.rint(scaled * 255.0), 0).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(viz, mode="L").save(path)
    except ImportError:
        np.save(path.with_suffix(".npy"), viz)
