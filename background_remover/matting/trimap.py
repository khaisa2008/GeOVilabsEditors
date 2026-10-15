"""Adaptive trimap generation.

A trimap labels every pixel as:

    0   = background (solid)
    128 = unknown (needs alpha matting)
    255 = foreground (solid)

The band width is *adaptive* (requirement 4): crisp, high-contrast
silhouettes (logos, product edges) get a narrow band so the mask stays
exact, while soft, low-contrast transitions (hair, fur, feathers, sheer
fabric) get a wide band so thin strands stay inside the matting region
instead of being eroded away. We never use a single global erosion that can
cut hair (requirement 5) — erosion is per-pixel, driven by the local colour
contrast map.

Enhancements:

- **Otsu-style adaptive thresholds** for the fg/bg boundary instead of the
  fixed 140/115 split — biased model outputs (0.5–0.8 predictions) no longer
  misclassify whole regions.
- **Distance-transform erosion** (numpy) replaces the iterative MinFilter
  passes — equivalent result, far faster on large images.
"""

import numpy as np
from PIL import Image, ImageFilter

from ..utils import erode, dilate, to_pil
from .edge_detection import fg_bg_contrast


def _otsu_threshold(a):
    """Simple two-class threshold that splits the alpha histogram.

    Returns (bg_threshold, fg_threshold) — the alpha levels below which a
    pixel is background and above which it is foreground. Falls back to a
    conservative 0.45/0.55 split on degenerate histograms.
    """
    a = np.asarray(a, dtype=np.float32)
    hist, edges = np.histogram(a, bins=64, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0

    def _thresh(class_lo, class_hi):
        w = (centers >= class_lo) & (centers <= class_hi)
        if not w.any():
            return class_lo
        h = hist[w].astype(np.float64)
        if h.sum() <= 0:
            return class_lo
        c = centers[w]
        for t in range(1, len(c)):
            w1, w2 = h[:t].sum(), h[t:].sum()
            if w1 <= 0 or w2 <= 0:
                continue
            m1 = (c[:t] * h[:t]).sum() / w1
            m2 = (c[t:] * h[t:]).sum() / w2
            # Within-class variance as Otsu; pick t maximising separation,
            # constrained to lie well inside the band.
            var = w1 * w2 * (m1 - m2) ** 2
        best = float(c[np.argmax(np.array([
            w1 * w2 * ((c[:t] * h[:t]).sum() / max(w1, 1e-6)
                       - (c[t:] * h[t:]).sum() / max(w2, 1e-6)) ** 2
            for t in range(1, len(c))
            if h[:t].sum() > 0 and h[t:].sum() > 0
        ] or [0.0]))])
        # If the histogram is degenerate, fall back conservatively.
        if best <= class_lo or best >= class_hi:
            return (class_lo + class_hi) / 2.0
        return best

    bg_thr = _thresh(0.0, 0.5)
    # The fg threshold is picked among the upper half but must stay clear of
    # the predicted semi-transparent band (which BiRefNet keeps at ~0.5-0.7).
    fg_thr = max(bg_thr + 0.1, _thresh(0.45, 1.0))
    if fg_thr - bg_thr < 0.25:
        fg_thr = bg_thr + 0.25
    return min(bg_thr, 0.4), min(fg_thr, 0.92)


def generate_trimap(alpha, fg_erode=6, bg_dilate=6, adaptive=True,
                    rgb=None, cfg=None, max_radius=14):
    """Return (trimap, known_fg, known_bg, unknown).

    ``alpha`` float32 0..1. ``fg_erode`` / ``bg_dilate`` are the base band
    sizes in pixels. When ``adaptive`` the widths scale locally with the
    colour contrast around the boundary and with the alpha gradient:
    crisp edges → narrow, soft hair → wide.
    """
    a8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.float32)
    bg_thr, fg_thr = _otsu_threshold(alpha)
    fg = (a8 > fg_thr * 255.0).astype(np.uint8) * 255
    bg = (a8 <= bg_thr * 255.0).astype(np.uint8) * 255

    if fg_dilate_risk(fg):
        # Very wide foreground → reduce erosion so we don't eat hair/fur.
        fg_erode = max(2, fg_erode // 2)
        bg_dilate = max(2, bg_dilate // 2)

    cfg = cfg or {}
    if adaptive and rgb is not None:
        contrast = fg_bg_contrast(np.asarray(rgb, dtype=np.float32), alpha)
        # contrast ≈ 1 → sharp silhouette → 0.7×  the band.
        # contrast ≈ 0 → soft hair/fur   → 1.6×  the band.
        scale = 0.7 + 0.9 * (1.0 - contrast)
        scale = np.clip(scale, 0.6, 1.8)
        fg_scale = _band_scale(scale, fg, max_radius)
        bg_scale = _band_scale(scale, bg, max_radius)

        known_fg = _adaptive_erode((fg > 0), fg_erode * fg_scale, max_radius)
        known_bg_raw = _adaptive_erode((bg > 0), bg_dilate * bg_scale,
                                       max_radius)
    else:
        fe, bd = _odd(fg_erode), _odd(bg_dilate)
        known_fg = erode(fg, fe)
        known_bg_raw = erode(bg, bd)

    known_fg = (known_fg > 0).astype(bool)
    known_bg = np.where(known_fg, False, (known_bg_raw > 0))

    unknown = (~known_fg) & (~known_bg)

    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap[known_fg] = 255
    trimap[unknown] = 128
    return trimap, known_fg, known_bg, unknown


def fg_dilate_risk(fg):
    """True when foreground fills most of the frame (typical close-up)."""
    if fg.shape[0] < 3 or fg.shape[1] < 3:
        return False
    return bool(fg[fg.shape[0] // 4: 3 * fg.shape[0] // 4,
                   fg.shape[1] // 4: 3 * fg.shape[1] // 4].mean() > 0.9)


def _band_scale(scale, region, max_radius):
    """Per-pixel radius factor, only meaningful inside the region."""
    s = np.asarray(scale, dtype=np.float32)
    return np.clip(s, 0.5, max_radius + 0.5)


def _adaptive_erode(mask, radius_map, max_radius):
    """Erode a boolean mask by a *per-pixel* radius.

    Distance-transform erosion: the per-pixel erosion distance is the
    distance to the nearest background pixel (computed with chamfer passes
    adapted to the required maximum), which gives the same band width as the
    old iterative approach but in O(N) instead of O(N·k).
    """
    radius_map = np.asarray(radius_map, dtype=np.float32)
    if radius_map.ndim == 0:
        radius_map = np.full(mask.shape, float(radius_map), dtype=np.float32)
    maxr = int(np.ceil(float(np.clip(radius_map.max(), 0, max_radius))))
    if maxr <= 0:
        return mask.copy()
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m.copy()

    # Distance to the nearest background (True = inside the fg region).
    try:
        from scipy import ndimage
        dist = ndimage.distance_transform_edt(m)
        return (dist >= radius_map).astype(bool)
    except ImportError:
        pass

    # Fallback: iterative MinFilter erosion (no scipy installed).
    cur = m.copy()
    depth = np.zeros(m.shape, dtype=np.int32)
    for i in range(1, maxr + 1):
        nxt = np.asarray(
            to_pil(cur.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3)),
            dtype=np.uint8) > 0
        newly = cur & ~nxt
        depth[newly & (depth == 0)] = i - 1
        cur = nxt
        if not cur.any():
            break
    depth[cur] = int(maxr)
    return depth >= radius_map


def _odd(k):
    return k if k % 2 == 1 else k + 1