"""Boundary / edge detection for adaptive processing.

Turns the segmented alpha + the colour image into spatial maps that drive
every later pipeline stage (requirement: explicit boundary/edge step before
the adaptive trimap):

- ``alpha_gradient``  : magnitude of the mask transition → the boundary band.
- ``color_edge_map``  : luminance gradient magnitude of the photo → crisp
                        silhouette cues (a hair strand is weak per-edge, a
                        logo silhouette is sharp).
- ``fg_bg_contrast``  : mean foreground/background colour separation near the
                        transition, 0..1. High on crisp well-lit silhouettes,
                        low in hair / fur / feathers / sheer fabric.
- ``fg_similarity``   : 0..1 how close each pixel's colour is to the local
                        foreground colour. Used to *gate* dilation so flyaway
                        hair and fur are kept without widening a halo into
                        background colour.
"""

import numpy as np

from ..utils import box_blur, dilate, gaussian


def alpha_gradient(alpha, sigma=1.5):
    """0..1 normalised magnitude of the alpha transition."""
    a = np.asarray(alpha, dtype=np.float32)
    if sigma and sigma > 0:
        a = gaussian(a, sigma)
    gy, gx = np.gradient(a)
    return _normalize(np.sqrt(gx * gx + gy * gy))


def color_edge_map(rgb, sigma=1.0):
    """0..1 normalised luminance gradient magnitude of the colour image."""
    g = np.asarray(rgb, dtype=np.float32)
    if g.ndim == 3:
        lum = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    else:
        lum = g
    if sigma and sigma > 0:
        lum = gaussian(lum, sigma)
    gy, gx = np.gradient(lum)
    return _normalize(np.sqrt(gx * gx + gy * gy))


def boundary_map(alpha, sigma=2.2):
    """0..1 proximity to the mask contour (the "edge zone" of the object)."""
    return alpha_gradient(alpha, sigma=sigma)


def fg_bg_contrast(rgb, alpha, radius=6, lo=0.15, hi=0.95):
    """0..1 local colour separation between fg and bg in the transition band.

    High → a clean, well-lit silhouette (deserves a narrow trimap / crisp
    handling). Low → hair / fur / feathers / sheer fabric — the areas where a
    *wide* matting band and colour-gated dilation is required.

    Separation is judged in the perceptually uniform Lab space, so skin-on-
    skin-like-background cases (where RGB distance is misleading) are rated
    correctly.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    band = (a > lo) & (a < hi)
    if not band.any():
        return np.ones(a.shape, dtype=np.float32)

    fg_mask = (a > 0.7).astype(np.float32)
    bg_mask = (a <= 0.3).astype(np.float32)
    fw = box_blur(fg_mask, radius) + 1e-4
    bw = box_blur(bg_mask, radius) + 1e-4

    F = np.empty_like(rgb, dtype=np.float32)
    B = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        F[..., c] = box_blur(rgb[..., c] * fg_mask, radius) / fw
        B[..., c] = box_blur(rgb[..., c] * bg_mask, radius) / bw

    try:
        from ..color_ops import lab_distance
        diff = lab_distance(F, B)
    except Exception:
        diff = np.sqrt(((F - B) ** 2).sum(axis=2))
    return _normalize(diff)


def fg_similarity(rgb, alpha, radius=7):
    """0..1 colour similarity to the local foreground estimate.

    Used to gate dilation / halo suppression: pixels whose colour matches the
    subject are kept, pixels that are background-coloured are not dilated into.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    fg_mask = (a > 0.75).astype(np.float32)
    fw = box_blur(fg_mask, radius) + 1e-4
    F = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        F[..., c] = box_blur(rgb[..., c] * fg_mask, radius) / fw
    d = np.sqrt(((rgb - F) ** 2).sum(axis=2))
    # ~40 units of colour distance → half-similar; ~130+ → a different colour.
    sim = 1.0 - np.clip((d - 8.0) / 120.0, 0.0, 1.0)
    return sim.astype(np.float32)


def fg_similarity_lab(rgb, alpha, radius=7):
    """0..1 Lab-space similarity to the local foreground estimate.

    Same purpose as ``fg_similarity`` but judged in the perceptually uniform
    Lab space — dilation gates tuned by hue/chroma differences (e.g. brown fur
    against a brown-ish background that RGB would measure as "far") behave
    noticeably better for hair and fur refinement.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    fg_mask = (a > 0.75).astype(np.float32)
    fw = box_blur(fg_mask, radius) + 1e-4
    F = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        F[..., c] = box_blur(rgb[..., c] * fg_mask, radius) / fw
    try:
        from ..color_ops import lab_distance
        d = lab_distance(rgb, F)
    except Exception:
        d = np.sqrt(((rgb - F) ** 2).sum(axis=2))
    sim = 1.0 - np.clip((d - 4.0) / 60.0, 0.0, 1.0)
    return sim.astype(np.float32)


def edge_zone(alpha, thresh=0.03, dilation=4):
    """Boolean mask of the narrow band around the object contour.

    Every boundary-sensitive stage (trimap unknown band, defringe, halo
    suppression, decontamination masking) must be confined to this zone —
    otherwise the *interior* gets misinterpreted as "spill" (the local
    background estimate inside the object is the object's own colour, making
    the recovered foreground look identical to the "background").
    """
    g = alpha_gradient(alpha, sigma=1.2)
    z = g > thresh
    if dilation and dilation > 0:
        z = dilate((z.astype(np.uint8)) * 255, 2 * dilation + 1) > 0
    return z.astype(bool)


def _normalize(a):
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)