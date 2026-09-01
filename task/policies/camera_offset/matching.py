"""Deterministic translation matchers (NCC, phase correlation, ECC)."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple  # Sequence used by multiscale_ncc_search

import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(np.float64)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    raise ValueError(f"unsupported image shape {arr.shape}")


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float64)
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = kx.T
    padded = np.pad(g, 1, mode="edge")
    gx = (
        kx[0, 0] * padded[0:-2, 0:-2] + kx[0, 1] * padded[0:-2, 1:-1]
        + kx[0, 2] * padded[0:-2, 2:] + kx[1, 0] * padded[1:-1, 0:-2]
        + kx[1, 1] * padded[1:-1, 1:-1] + kx[1, 2] * padded[1:-1, 2:]
        + kx[2, 0] * padded[2:, 0:-2] + kx[2, 1] * padded[2:, 1:-1]
        + kx[2, 2] * padded[2:, 2:]
    )
    gy = (
        ky[0, 0] * padded[0:-2, 0:-2] + ky[0, 1] * padded[0:-2, 1:-1]
        + ky[0, 2] * padded[0:-2, 2:] + ky[1, 0] * padded[1:-1, 0:-2]
        + ky[1, 1] * padded[1:-1, 1:-1] + ky[1, 2] * padded[1:-1, 2:]
        + ky[2, 0] * padded[2:, 0:-2] + ky[2, 1] * padded[2:, 1:-1]
        + ky[2, 2] * padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _window2d(shape: Tuple[int, int]) -> np.ndarray:
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def highpass(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Deterministic box-blur high-pass used before phase correlation."""
    g = np.asarray(gray, dtype=np.float64)
    k = int(max(3, ksize))
    if k % 2 == 0:
        k += 1
    pad = np.pad(g, k // 2, mode="edge")
    # Separable cumulative sum blur (no SciPy dependency).
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    h, w = g.shape
    blur = (
        c[k:k + h, k:k + w]
        - c[0:h, k:k + w]
        - c[k:k + h, 0:w]
        + c[0:h, 0:w]
    ) / float(k * k)
    return g - blur


def phase_correlate(reference: np.ndarray, current: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    *, use_highpass: bool = True) -> Tuple[np.ndarray, float]:
    """Return ``(du, dv)`` shifting ``reference`` onto ``current`` plus peak score.

    ``du`` is +right (columns), ``dv`` is +down (rows). Uses FFT phase
    correlation with a Hanning window. When ``mask`` is set, masked-out
    pixels are replaced by the masked mean so foreground parts do not
    dominate the peak.
    """
    a = to_gray(reference)
    b = to_gray(current)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if use_highpass:
        a = highpass(a)
        b = highpass(b)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError("mask shape mismatch")
        if not np.any(m):
            return np.zeros(2, dtype=np.float64), 0.0
        fill_a = float(a[m].mean())
        fill_b = float(b[m].mean())
        a = np.where(m, a, fill_a)
        b = np.where(m, b, fill_b)
    win = _window2d(a.shape)
    a = (a - a.mean()) * win
    b = (b - b.mean()) * win
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    r = np.fft.ifft2(cross / mag).real
    peak = np.unravel_index(int(np.argmax(r)), r.shape)
    dv, du = _peak_to_signed_shift(peak, r.shape)
    du_sub, dv_sub = subpixel_quadratic(r, peak)
    score = float(r[peak] / (np.abs(r).mean() + 1e-9))
    # Negate so +du/+dv match np.roll / NCC: current content is to the
    # right/down of the reference.
    return np.array([-(du + du_sub), -(dv + dv_sub)], dtype=np.float64), score


def ncc_at_uv(image: np.ndarray, template: np.ndarray, uv_centre,
              template_mask: Optional[np.ndarray] = None) -> float:
    """Masked NCC of ``template`` placed at ``uv_centre`` in ``image``."""
    img = to_gray(image)
    tmpl = to_gray(template)
    th, tw = tmpl.shape
    cu, cv = float(uv_centre[0]), float(uv_centre[1])
    x = int(round(cu - tw / 2.0))
    y = int(round(cv - th / 2.0))
    if x < 0 or y < 0 or x + tw > img.shape[1] or y + th > img.shape[0]:
        return -1.0
    if template_mask is not None:
        m = np.asarray(template_mask, dtype=bool)
    else:
        m = np.ones(tmpl.shape, dtype=bool)
    if not np.any(m):
        return -1.0
    t = tmpl[m]
    t = t - t.mean()
    patch = img[y:y + th, x:x + tw][m]
    patch = patch - patch.mean()
    denom = (np.sqrt(float(np.dot(patch, patch)) * float(np.dot(t, t))) + 1e-12)
    return float(np.dot(patch, t) / denom)


def resize_gray(image: np.ndarray, scale: float) -> np.ndarray:
    """Nearest-neighbour resize (deterministic, no OpenCV required)."""
    img = to_gray(image)
    if abs(scale - 1.0) < 1e-9:
        return img
    h, w = img.shape
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    ys = np.clip((np.arange(nh) / scale).astype(int), 0, h - 1)
    xs = np.clip((np.arange(nw) / scale).astype(int), 0, w - 1)
    return img[ys][:, xs]


def multiscale_ncc_search(image: np.ndarray, template: np.ndarray,
                          template_mask: Optional[np.ndarray] = None,
                          search_origin_uv=None, search_half=None,
                          scales: Sequence[float] = (1.0, 0.9, 1.1),
                          *, early_score: float = 0.55) -> dict:
    """Run NCC at fixed scales; stop early when the primary scale is strong."""
    best = None
    for scale in scales:
        if abs(float(scale) - 1.0) < 1e-9:
            tmpl = to_gray(template)
            mask = template_mask
            img = to_gray(image)
            origin = search_origin_uv
            half = search_half
        else:
            tmpl = resize_gray(template, scale)
            img = to_gray(image)
            if template_mask is not None:
                mask = resize_gray(
                    np.asarray(template_mask, dtype=np.float64), scale
                ) >= 0.5
            else:
                mask = None
            if search_origin_uv is None:
                origin = None
            else:
                origin = (
                    float(search_origin_uv[0]),
                    float(search_origin_uv[1]),
                )
            half = search_half
        result = ncc_search(
            img, tmpl, template_mask=mask,
            search_origin_uv=origin, search_half=half,
        )
        if not result["valid"]:
            continue
        if best is None or result["score"] > best["score"]:
            best = result
            best["scale"] = float(scale)
        if abs(float(scale) - 1.0) < 1e-9 and result["score"] >= early_score:
            break
    if best is None:
        return {
            "du": 0.0, "dv": 0.0, "score": 0.0,
            "uv": None, "candidates": (), "valid": False,
        }
    return best


def orb_translation(reference: np.ndarray, current: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    *, min_inliers: int = 12) -> Tuple[np.ndarray, float, int]:
    """Deterministic ORB + translation-only inlier consensus.

    Returns ``(du, dv)``, inlier fraction, and inlier count. Falls back to
    zeros with score 0 when OpenCV is missing or matching fails.
    """
    try:
        import cv2
    except ImportError:
        return np.zeros(2, dtype=np.float64), 0.0, 0

    a = np.clip(to_gray(reference), 0, 255).astype(np.uint8)
    b = np.clip(to_gray(current), 0, 255).astype(np.uint8)
    a = cv2.equalizeHist(a)
    b = cv2.equalizeHist(b)
    orb = cv2.ORB_create(
        nfeatures=2000, scaleFactor=1.2, nlevels=4,
        edgeThreshold=15, patchSize=31,
    )
    input_mask = None
    if mask is not None:
        input_mask = np.asarray(mask, dtype=np.uint8) * np.uint8(255)
    k1, d1 = orb.detectAndCompute(a, input_mask)
    k2, d2 = orb.detectAndCompute(b, input_mask)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.zeros(2, dtype=np.float64), 0.0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)
    matches = [m for m in matches if m.distance < 64][:200]
    if len(matches) < 8:
        return np.zeros(2, dtype=np.float64), 0.0, len(matches)
    src = np.asarray([k1[m.queryIdx].pt for m in matches], dtype=np.float64)
    dst = np.asarray([k2[m.trainIdx].pt for m in matches], dtype=np.float64)
    diffs = dst - src
    best_inliers = 0
    best_t = np.zeros(2, dtype=np.float64)
    # Deterministic hypothesis loop over matches (no RNG).
    for i in range(len(diffs)):
        err = np.linalg.norm(diffs - diffs[i], axis=1)
        inl_mask = err < 1.5
        inl = int(np.sum(inl_mask))
        if inl > best_inliers:
            best_inliers = inl
            best_t = np.median(diffs[inl_mask], axis=0)
    if best_inliers < int(min_inliers):
        return np.zeros(2, dtype=np.float64), 0.0, best_inliers
    score = float(best_inliers) / float(max(1, len(diffs)))
    return np.asarray(best_t, dtype=np.float64), score, best_inliers


def _peak_to_signed_shift(peak_rc, shape) -> Tuple[float, float]:
    h, w = shape
    dv = float(peak_rc[0])
    du = float(peak_rc[1])
    if dv > h / 2.0:
        dv -= h
    if du > w / 2.0:
        du -= w
    return dv, du


def subpixel_quadratic(score_map: np.ndarray, peak_rc) -> Tuple[float, float]:
    """Return (du, dv) fractional refinement at an integer peak."""
    s = np.asarray(score_map, dtype=np.float64)
    r, c = int(peak_rc[0]), int(peak_rc[1])
    h, w = s.shape

    def axis(idx, size, get):
        if idx <= 0 or idx >= size - 1:
            return 0.0
        left, mid, right = get(idx - 1), get(idx), get(idx + 1)
        denom = left - 2.0 * mid + right
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (left - right) / denom

    du = axis(c, w, lambda i: s[r, i])
    dv = axis(r, h, lambda i: s[i, c])
    return float(du), float(dv)


def ncc_search(image: np.ndarray, template: np.ndarray,
               template_mask: Optional[np.ndarray] = None,
               search_origin_uv=None,
               search_half=None) -> dict:
    """Integer-pixel NCC over a bounded ROI, then quadratic sub-pixel fit.

    Returns a dict with ``du``, ``dv`` relative to the template's nominal
    placement, ``score``, ``uv`` of the match centre, and ``candidates``
    used for deterministic tie-breaking.
    """
    img = to_gray(image)
    tmpl = to_gray(template)
    th, tw = tmpl.shape
    ih, iw = img.shape
    if th < 3 or tw < 3 or th > ih or tw > iw:
        return {
            "du": 0.0, "dv": 0.0, "score": 0.0,
            "uv": None, "candidates": (), "valid": False,
        }

    if search_origin_uv is None:
        cu, cv = tw / 2.0, th / 2.0
    else:
        cu, cv = float(search_origin_uv[0]), float(search_origin_uv[1])
    if search_half is None:
        hu, hv = iw, ih
    else:
        hu, hv = int(search_half[0]), int(search_half[1])

    # search_origin is the template centre in image coordinates.
    u0 = max(0, int(np.floor(cu - tw / 2.0 - hu)))
    v0 = max(0, int(np.floor(cv - th / 2.0 - hv)))
    u1 = min(iw - tw, int(np.ceil(cu - tw / 2.0 + hu)))
    v1 = min(ih - th, int(np.ceil(cv - th / 2.0 + hv)))
    if u1 < u0 or v1 < v0:
        return {
            "du": 0.0, "dv": 0.0, "score": 0.0,
            "uv": None, "candidates": (), "valid": False,
        }

    if template_mask is not None:
        m = np.asarray(template_mask, dtype=bool)
        if m.shape != tmpl.shape:
            raise ValueError("template mask shape mismatch")
    else:
        m = np.ones(tmpl.shape, dtype=bool)
    t = tmpl[m]
    t = t - t.mean()
    t_norm = np.sqrt(float(np.dot(t, t))) + 1e-12

    scores = np.full((v1 - v0 + 1, u1 - u0 + 1), -np.inf, dtype=np.float64)
    for dv_i, y in enumerate(range(v0, v1 + 1)):
        for du_i, x in enumerate(range(u0, u1 + 1)):
            patch = img[y:y + th, x:x + tw][m]
            patch = patch - patch.mean()
            denom = (np.sqrt(float(np.dot(patch, patch))) + 1e-12) * t_norm
            scores[dv_i, du_i] = float(np.dot(patch, t) / denom)

    peak = np.unravel_index(int(np.argmax(scores)), scores.shape)
    best = float(scores[peak])
    # Collect all integer peaks within 1e-6 of the best for tie-break.
    tol = 1e-6
    cand_rc = np.argwhere(np.abs(scores - best) <= tol)
    candidates = []
    nom_u = cu - tw / 2.0
    nom_v = cv - th / 2.0
    for rc in cand_rc:
        yy, xx = int(rc[0]), int(rc[1])
        x = u0 + xx
        y = v0 + yy
        dist = float(np.hypot(x - nom_u, y - nom_v))
        candidates.append((best, dist, y, x))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    y, x = candidates[0][2], candidates[0][3]
    du_sub, dv_sub = subpixel_quadratic(scores, (y - v0, x - u0))
    match_cu = x + tw / 2.0 + du_sub
    match_cv = y + th / 2.0 + dv_sub
    return {
        "du": float(match_cu - cu),
        "dv": float(match_cv - cv),
        "score": best,
        "uv": np.array([match_cu, match_cv], dtype=np.float64),
        "candidates": tuple(candidates),
        "valid": True,
        "score_map": scores,
        "origin": (u0, v0),
    }


def depth_residual(depth: np.ndarray, depth_template: np.ndarray,
                   uv_centre, template_mask: Optional[np.ndarray] = None) -> float:
    d = np.asarray(depth, dtype=np.float64)
    t = np.asarray(depth_template, dtype=np.float64)
    th, tw = t.shape
    cu, cv = float(uv_centre[0]), float(uv_centre[1])
    x = int(round(cu - tw / 2.0))
    y = int(round(cv - th / 2.0))
    if x < 0 or y < 0 or x + tw > d.shape[1] or y + th > d.shape[0]:
        return float("inf")
    patch = d[y:y + th, x:x + tw]
    if template_mask is not None:
        m = np.asarray(template_mask, dtype=bool)
        if not np.any(m):
            return float("inf")
        diff = patch[m] - t[m]
    else:
        diff = patch - t
    finite = np.isfinite(diff)
    if not np.any(finite):
        return float("inf")
    return float(np.mean(np.abs(diff[finite])))


def ecc_translation(reference: np.ndarray, current: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """cv2 ECC translation if available, else phase correlation."""
    try:
        import cv2
    except ImportError:
        return phase_correlate(reference, current, mask=mask)

    a = to_gray(reference).astype(np.float32)
    b = to_gray(current).astype(np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    input_mask = None
    if mask is not None:
        input_mask = np.asarray(mask, dtype=np.uint8) * np.uint8(255)
    try:
        cc, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_TRANSLATION, criteria, input_mask, 1
        )
    except (cv2.error, TypeError):
        try:
            cc, warp = cv2.findTransformECC(
                a, b, warp, cv2.MOTION_TRANSLATION, criteria, input_mask
            )
        except (cv2.error, TypeError):
            return phase_correlate(reference, current, mask=mask)
    du = float(warp[0, 2])
    dv = float(warp[1, 2])
    return np.array([du, dv], dtype=np.float64), float(cc)


def pick_best_candidate(candidates: Sequence[tuple],
                        depth_residuals: Optional[Iterable[float]] = None) -> tuple:
    """Order: template score desc, depth residual, distance, (v, u)."""
    if not candidates:
        raise ValueError("no candidates")
    residuals = (
        [float("inf")] * len(candidates)
        if depth_residuals is None
        else [float(r) for r in depth_residuals]
    )
    ranked = []
    for cand, residual in zip(candidates, residuals):
        score, dist, y, x = cand
        ranked.append((-float(score), float(residual), float(dist), int(y), int(x)))
    ranked.sort()
    best = ranked[0]
    return -best[0], best[1], best[2], best[3], best[4]
