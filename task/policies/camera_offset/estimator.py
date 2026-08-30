"""Camera-only board/part XY offset estimator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .constants import (
    BOARD_CONFIDENCE_MIN,
    DEFAULT_ROI_MARGIN_PX,
    DEPTH_DISAGREE_M,
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
        self._frames: List[tuple] = []
        self._estimate: Optional[OffsetEstimate] = None

    def reset_episode(self) -> None:
        self._frames.clear()
        self._estimate = None

    def add_frame(self, rgb, depth, intrinsics=None) -> None:
        self.bundle.assert_observation_shape(rgb, depth, intrinsics)
        rgb_arr = np.asarray(rgb)
        depth_arr = None if depth is None else np.asarray(depth, dtype=np.float64)
        self._frames.append((rgb_arr.copy(), depth_arr))

    def ready(self, min_frames: Optional[int] = None) -> bool:
        need = int(self.bundle.buffer_frames if min_frames is None else min_frames)
        return len(self._frames) >= max(1, need)

    def estimate(self, force=False) -> OffsetEstimate:
        if self._estimate is not None and not force:
            return self._estimate
        if not self._frames:
            raise RuntimeError(
                "OffsetEstimator has no head frames. "
                "Set TASK_ENABLE_CAMERA_OUTPUT=1 and wait for rgb['head']."
            )
        board_samples = []
        part_samples: Dict[str, List[np.ndarray]] = {
            name: [] for name in self.bundle.parts
        }
        part_conf: Dict[str, List[float]] = {name: [] for name in self.bundle.parts}
        frame_diag = []
        for rgb, depth in self._frames:
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
                parts[name] = board.copy()
                confidences[name] = 0.0
        board_c = float(np.median([d["board_confidence"] for d in frame_diag]))
        self._estimate = OffsetEstimate(
            board_xy=board,
            part_xy=parts,
            board_confidence=board_c,
            part_confidence=confidences,
            diagnostics={"frames": frame_diag, "n_frames": len(self._frames)},
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
        matchers = (
            ("ecc", lambda: M.ecc_translation(self.bundle.rgb, rgb, mask=mask)),
            ("phasecorr", lambda: M.phase_correlate(self.bundle.rgb, rgb, mask=mask)),
        )
        valid = []
        for name, fn in matchers:
            try:
                du_dv, score = fn()
            except Exception:
                continue
            world = clamp_xy(
                pixel_delta_to_world_xy(du_dv, self.bundle.jacobian_xy_per_px)
            )
            valid.append((world, float(score), name))
            if score >= BOARD_CONFIDENCE_MIN:
                return world, float(score), name
        if valid:
            # Deterministic fallback: median of matcher XY, keep first score.
            world = _median_xy([v[0] for v in valid])
            return clamp_xy(world), float(valid[0][1]), "median_matchers"
        return np.zeros(2, dtype=np.float64), 0.0, "zero"

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
            result = M.ncc_search(
                image,
                template,
                template_mask=tmpl.mask,
                search_origin_uv=tmpl.search_center_uv,
                search_half=(half_u, half_v),
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
