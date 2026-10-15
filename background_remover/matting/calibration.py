"""Stable alpha calibration.

The raw segmentation alpha is remapped so the mass of confident foreground
sits on ~1.0 and the mass of confident background on ~0.0 *before* any
trimap / matting / halo work (requirement 2 of the pipeline).

Why this matters: BiRefNet-type predictors often output grey-biased pixel
values (≈0.5–0.7 on soft edges, ≈0.3–0.5 on faint background fringes). Left
untouched that bias composits directly into white/grey halos. Calibration is
*gentle* near the boundary (so hair, fur and thin strands keep their alpha)
and *strong* inside the object body and far from it.
"""

import numpy as np


def estimate_modes(alpha, bins=64):
    """Locate the background and foreground alpha peaks via histogram."""
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    hist, edges = np.histogram(a, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Densest bin near the top → foreground mode; near the bottom → bg mode.
    def peak(from_top):
        if from_top:
            order = np.where(centers >= 0.6)[0]
            if len(order) == 0:
                order = np.where(centers >= 0.45)[0]
        else:
            order = np.where(centers <= 0.4)[0]
            if len(order) == 0:
                order = np.where(centers <= 0.55)[0]
        if len(order) == 0:
            order = np.arange(len(centers))
        best = order[int(np.argmax(hist[order]))]
        return float(centers[best])

    bg_mode = peak(False)
    fg_mode = peak(True)
    if fg_mode - bg_mode < 0.3:
        return 0.0, 1.0
    return bg_mode, fg_mode


def calibrate_alpha(alpha, fg_mode=None, bg_mode=None, strength=0.7,
                    edge_band=None, max_alpha=1.0):
    """Linearly remap alpha between the background and foreground modes.

    ``edge_band`` (0..1, from ``alpha_gradient``) keeps the remap *soft* on
    the boundary so hair/fur strands are not relaunched into 0 or 1. Pixels
    deep inside (``a >= max_alpha``) stay at their original value.
    """
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    lo = 0.0 if bg_mode is None else float(bg_mode)
    hi = 1.0 if fg_mode is None else float(fg_mode)
    if hi - lo < 1e-3:
        return a

    remapped = np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    # Soften the pull on the boundary: hair strands must not jump to 1/0.
    w = 1.0
    if edge_band is not None:
        edge_band = np.asarray(edge_band, dtype=np.float32)
        w = 1.0 - 0.55 * np.clip(edge_band, 0.0, 1.0)

    s = float(strength) * w
    out = a * (1.0 - s) + remapped * s
    out = np.clip(out, 0.0, 1.0)
    if max_alpha < 1.0:
        solid = a >= max_alpha
        out = np.where(solid, a, out)
    return out


def calibrate_strength_for(category, cfg):
    """Per-category calibration strength from the strategy config."""
    return float(cfg.get('calibrate_strength', 0.7))